import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Type, Union, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.agents.sub_agent_registry import get_sub_agents_by_global_type
from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.memory import MemoryStack, MemoryStackEntry
from src.prompts.template import apply_prompt_template, get_prompt_template
from src.utils.json_utils import repair_json_output
from src.utils.logger import logger
from src.utils.statistics import global_statistics
from src.prompts.central_decision import Decision, DelegateParams
from src.utils.reference_utils import global_reference_map
from src.utils.outline_parser import parse_outline, get_chapter_task_description

from ..graph.types import State

# ZX 新增 在文件顶部的导入部分添加
from src.utils.outline_parser import (
    parse_outline,
    get_next_chapter_index,
    is_all_chapters_researched,
    get_chapter_task_description
)


# from .SubAgentConfig import get_sub_agents_by_global_type


# -------------------------
# 核心枚举定义
# -------------------------
class CentralAgentAction(Enum):
    """中枢Agent动作枚举，定义系统核心决策类型"""

    THINK = "think"  # 分析当前状态并思考下一步行动
    REFLECT = "reflect"  # 反思之前的动作和结果
    SUMMARIZE = "summarize"  # 总结当前已获得的信息
    DELEGATE = "delegate"  # 委派子Agent执行专项任务
    FINISH = "finish"  # 判断任务完成并生成最终报告


# -------------------------
# 中枢Agent核心模块--中枢Agent的action
# exp: 与prompt/central_decision.py中的Decision类不同的是，那里是字符串类型，这里是枚举类型，所以要定义两次
# -------------------------
@dataclass
class CentralDecision:
    """中枢Agent决策结果数据模型"""

    action: CentralAgentAction  # 决策动作
    reasoning: str  # 决策推理过程
    params: Dict[DelegateParams, Any] = field(
        default_factory=dict
    )  # 动作参数=>delegate有参数
    instruction: Optional[str] = None  # 动作对应的指令说明
    state_updates: Optional[Dict[str, Any]] = None  # ZX 🆕 新增：状态更新


class CentralAgent:
    """
    中枢Agent核心类，负责系统整体决策与任务编排

    采用基于记忆栈的决策机制，通过状态分析动态委派子Agent执行专项任务，
    并最终整合结果生成完成报告
    """

    def __init__(self, graph_format: str = "sp"):
        self.memory_stack = MemoryStack()
        from src.agents.SubAgentManager import SubAgentManager

        self.sub_agent_manager = SubAgentManager(self)

        sub_agents = get_sub_agents_by_global_type(graph_format)
        logger.info(f"初始化中枢Agent，使用子Agent类型: {sub_agents}")

        # 初始化子Agent相关信息
        self.available_sub_agents = [agent["name"] for agent in sub_agents]
        self.sub_agents_description = ""
        for agent in sub_agents:
            self.sub_agents_description += (
                f"- **{agent['name']}**: {agent['description']}\n"
            )

        # 动作处理器映射表
        self.action_handlers = {
            CentralAgentAction.THINK: self._handle_think,
            CentralAgentAction.REFLECT: self._handle_reflect,
            CentralAgentAction.SUMMARIZE: self._handle_summarize,
            CentralAgentAction.DELEGATE: self._handle_delegate,
            CentralAgentAction.FINISH: self._handle_finish,
        }

        # 动作类型对应的指令模板
        self.action_instructions = {
            CentralAgentAction.THINK: "分析当前状态并思考下一步行动",
            CentralAgentAction.REFLECT: "反思之前的动作和结果",
            CentralAgentAction.SUMMARIZE: "总结当前已获得的信息",
            CentralAgentAction.DELEGATE: "决定委派哪个子Agent执行任务",
            CentralAgentAction.FINISH: "判断是否可以完成任务并生成最终报告",
        }

    def make_decision(
            self, state: State, config: RunnableConfig, retry_count: int = 0
    ) -> CentralDecision:
        """
        中枢Agent决策核心逻辑，分析当前状态生成决策结果

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            决策结果对象
        """
        max_retries = 3
        logger.info("中枢Agent正在进行决策...")
        start_time = datetime.now()

        # 🆕 临时测试：硬编码 强制设置 report_mode
        if "report_mode" not in state or not state.get("report_mode"):
            state["report_mode"] = "cumulative_observations"
            logger.info("🔧 临时设置 report_mode = cumulative_observations")

        # ============================================================
        # 🆕 步骤 0：最高优先级 - 检查是否需要人类交互
        # ============================================================

        need_human_interaction = state.get("need_human_interaction", False)
        human_interaction_type = state.get("human_interaction_type", "")

        logger.info(f"📊 人类交互状态检查:")
        logger.info(f"   - need_human_interaction: {need_human_interaction}")
        logger.info(f"   - human_interaction_type: {human_interaction_type}")

        if need_human_interaction:
            logger.info(f"🔴 需要人类交互，立即委派给 human agent")

            end_time = datetime.now()
            time_entry = {
                "step_name": "central_decision" + start_time.isoformat(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
            }
            global_statistics.add_time_entry(time_entry)

            return CentralDecision(
                action=CentralAgentAction.DELEGATE,
                reasoning=f"需要人类交互: {human_interaction_type}",
                params={
                    "agent_type": "human",
                    "interaction_type": human_interaction_type,
                },
                instruction="委派 Human Agent 进行人类交互",
                state_updates={
                    "need_human_interaction": False,  # 重置状态
                    "human_interaction_type": "",
                },
            )

        # ============================================================
        # 🆕 步骤 1：优先检查段落循环状态（在 LLM 决策之前）
        # ============================================================

        outline = state.get("report_outline", "")
        current_chapter_index = state.get("current_chapter_index", 0)
        chapter_stage_status = state.get("chapter_stage_status", {})
        chapter_reports = state.get("chapter_reports", {})

        # ZX 🆕 调试日志
        logger.info(f"📊 State 检查:")
        logger.info(f"   - outline exists: {bool(outline)}")
        logger.info(f"   - current_chapter_index: {current_chapter_index}")
        logger.info(f"   - chapter_stage_status: {chapter_stage_status}")
        logger.info(f"   - chapter_reports keys: {list(chapter_reports.keys()) if chapter_reports else []}")
        logger.info(f"   - report_mode: {state.get('report_mode', 'per_chapter')}")  # ZX 🆕 新增

        if outline:
            chapters = parse_outline(outline)
            total_chapters = len(chapters)

            # 检查是否还有未完成的章节
            if current_chapter_index < total_chapters:
                current_chapter = chapters[current_chapter_index]
                chapter_num_str = str(current_chapter_index + 1)
                current_stage = chapter_stage_status.get(chapter_num_str, "")

                if current_stage == "":
                    # 阶段 1：研究
                    logger.info(f"📂 段落进度: {current_chapter_index + 1}/{total_chapters} - 研究阶段")
                    logger.info(f"📝 当前段落: 第{current_chapter['number']}章 - {current_chapter['title']}")

                    task_description = get_chapter_task_description(current_chapter, current_chapter_index)

                    new_stage_status = dict(chapter_stage_status) if chapter_stage_status else {}
                    new_stage_status[chapter_num_str] = "researched"

                    logger.info(f"🔄 强制进入研究阶段: 第 {current_chapter_index + 1} 章")

                    end_time = datetime.now()
                    time_entry = {
                        "step_name": "central_decision" + start_time.isoformat(),
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "duration": (end_time - start_time).total_seconds(),
                    }
                    global_statistics.add_time_entry(time_entry)

                    return CentralDecision(
                        action=CentralAgentAction.DELEGATE,
                        reasoning=f"段落循环模式：研究第 {current_chapter_index + 1}/{total_chapters} 章 - {current_chapter['title']}",
                        params={
                            "agent_type": "researcher",
                            "task_description": task_description,
                            "chapter_index": current_chapter_index + 1,
                        },
                        instruction="委派 Researcher Agent 进行章节研究",
                        state_updates={"chapter_stage_status": new_stage_status},
                    )

                elif current_stage == "researched":
                    # 阶段 2：生成该章report & observation
                    report_mode = state.get("report_mode", "per_chapter")

                    if report_mode == "per_chapter":
                        # 原有流程：每章独立生成报告
                        logger.info(f"📂 段落进度: {current_chapter_index + 1}/{total_chapters} - 报告生成阶段")
                        logger.info(f"📝 当前段落: 第{current_chapter['number']}章 - {current_chapter['title']}")

                        new_stage_status = dict(chapter_stage_status) if chapter_stage_status else {}
                        new_stage_status[chapter_num_str] = "reported"

                        logger.info(f"🔄 强制进入报告生成阶段: 第 {current_chapter_index + 1} 章")

                        return CentralDecision(
                            action=CentralAgentAction.DELEGATE,
                            reasoning=f"段落循环模式：生成第 {current_chapter_index + 1}/{total_chapters} 章报告 - {current_chapter['title']}",
                            params={
                                "agent_type": "reporter",
                                "task_description": f"根据研究结果，撰写第{current_chapter_index + 1}章的完整内容：{current_chapter['title']}",
                                "chapter_index": current_chapter_index + 1,
                                "chapter_mode": "single",
                            },
                            instruction="委派 Reporter Agent 生成单章报告",
                            state_updates={
                                "chapter_stage_status": new_stage_status,
                                "current_chapter_index": current_chapter_index + 1,
                            },
                        )
                    elif report_mode == "cumulative_observations":
                        # 新流程：跳过单章报告，继续下一章研究
                        logger.info(f"📂 段落进度: {current_chapter_index + 1}/{total_chapters} - 研究完成，继续下一章")

                        new_stage_status = dict(chapter_stage_status) if chapter_stage_status else {}
                        new_stage_status[chapter_num_str] = "reported"

                        # 🔧 添加时间统计
                        end_time = datetime.now()
                        time_entry = {
                            "step_name": "central_decision" + start_time.isoformat(),
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                            "duration": (end_time - start_time).total_seconds(),
                        }
                        global_statistics.add_time_entry(time_entry)

                        # 🔧 关键：直接跳到下一章，不生成单章报告
                        return CentralDecision(
                            action=CentralAgentAction.THINK,
                            reasoning=f"累积模式：第 {current_chapter_index + 1} 章研究完成，准备进入下一章",
                            params={},
                            instruction="思考下一步行动",
                            state_updates={
                                "chapter_stage_status": new_stage_status,
                                "current_chapter_index": current_chapter_index + 1,
                            },
                        )
            else:
                # 所有章节已完成，根据 report_mode 决定下一步
                report_mode = state.get("report_mode", "per_chapter")

                if report_mode == "per_chapter":
                    # ==================== 原有流程：合并所有章节报告 ====================
                    logger.info(f"📊 检查合并条件:")
                    logger.info(f"   - total_chapters: {total_chapters}")
                    logger.info(f"   - chapter_reports count: {len(chapter_reports) if chapter_reports else 0}")
                    logger.info(f"   - expected keys: {[str(i) for i in range(1, total_chapters + 1)]}")

                    if chapter_reports and len(chapter_reports) == total_chapters:
                        logger.info("✅ 所有段落已完成研究和报告生成，准备合并最终报告")

                        end_time = datetime.now()
                        time_entry = {
                            "step_name": "central_decision" + start_time.isoformat(),
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                            "duration": (end_time - start_time).total_seconds(),
                        }
                        global_statistics.add_time_entry(time_entry)

                        return CentralDecision(
                            action=CentralAgentAction.DELEGATE,
                            reasoning="所有章节已完成，合并生成最终报告",
                            params={
                                "agent_type": "reporter",
                                "task_description": "合并所有章节报告，生成最终完整报告",
                                "chapter_mode": "merge",
                            },
                            instruction="委派 Reporter Agent 合并所有章节报告",
                            state_updates={"current_chapter_index": 0},
                        )
                    else:
                        logger.warning(
                            f"⚠️ 章节报告不完整: {len(chapter_reports) if chapter_reports else 0}/{total_chapters}")
                        # 打印详细信息
                        if chapter_reports:
                            for i in range(1, total_chapters + 1):
                                key = str(i)
                                if key in chapter_reports:
                                    logger.info(f"   ✅ 章节 {key}: 已生成")
                                else:
                                    logger.info(f"   ❌ 章节 {key}: 未生成")
                        else:
                            logger.warning("   chapter_reports 为空或 None")

                elif report_mode == "cumulative_observations":
                    # ==================== 新流程：使用累积的 observations 生成最终报告 ====================
                    logger.info("✅ 所有段落已完成研究，准备使用累积的 observations 生成最终报告")

                    # 检查 observations 是否有数据
                    observations = state.get("observations", [])
                    logger.info(f"📊 累积的 observations 数量: {len(observations)}")

                    if observations:
                        logger.info("✅ observations 数据充足，开始生成最终报告")

                        end_time = datetime.now()
                        time_entry = {
                            "step_name": "central_decision" + start_time.isoformat(),
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                            "duration": (end_time - start_time).total_seconds(),
                        }
                        global_statistics.add_time_entry(time_entry)

                        return CentralDecision(
                            action=CentralAgentAction.DELEGATE,
                            reasoning="累积模式：所有章节研究完成，使用累积的 observations 生成最终报告",
                            params={
                                "agent_type": "reporter",
                                "task_description": "使用所有章节的研究结果，生成最终完整报告",
                                "chapter_mode": "final_from_observations",  # 🔧 新增模式
                            },
                            instruction="委派 Reporter Agent 使用 observations 生成最终报告",
                            state_updates={"current_chapter_index": 0},
                        )
                    else:
                        logger.warning("⚠️ 没有累积的 observations，回退到传统模式")
                        # 回退到传统模式
                        pass

        # ============================================================
        # 步骤 2：以下是原有的 LLM 决策逻辑
        # ============================================================

        # 增加 SOP 部分
        DECISION_SOP_SP = """### 执行流程指南

        你正在一个具有**严格阶段约束与不可回退节点**的多智能体系统中运行。
        你的职责是**严格按照以下流程推进任务直至完成**，并遵守每个阶段的进入与退出规则。
        任何违反阶段约束的行为都被视为执行错误。

        ---

        #### 🔴 Human Agent 使用说明

        **Human Agent** 是专门负责与人类交互的子Agent。你 **必须** 在以下情况委派给它：

        1. 当任意 agent 返回时，检查 state 中的 `need_human_interaction` 字段：
           - 如果为 `true`，**必须立即** 委派给 human agent
           - 根据 `human_interaction_type` 设置正确的交互类型

        2. 交互类型说明：
           - `form_filling`: perception agent 生成表单后，需要人类填写
           - `outline_confirmation`: outline agent 生成大纲后，需要人类确认
           - `report_feedback`: reporter agent 生成报告后，需要人类反馈
           - `proactive_question`: 你判断信息不足时，主动向人类提问

        3. 🔴 **人类反馈优先级最高**：
           - 收到 human agent 返回后，**必须** 将人类反馈作为最高优先级考虑
           - **不得** 忽略或覆盖人类的明确指示
           - 在后续 delegate 指令中，**必须** 包含人类反馈的关键信息

        ---

        #### 强制性的高层执行流程

        ### 1. 感知与澄清阶段（Perception Phase，强制，第一步，且仅此一次）

        - 你 **必须** 在任务开始时首先委派给 **perception agent**
        - perception agent 生成表单后，会返回给你并标记 `need_human_interaction: true`
        - 收到此标记后，你 **必须** 立即委派给 **human agent**（设置 `interaction_type: form_filling`）
        - 🔴 人类填写的表单内容具有最高优先级，后续所有决策必须基于此
        - **一旦 perception 阶段完成并退出：**
          - 在整个任务生命周期中，**绝对禁止再次调用 perception agent**

        ---

        ### 2. 大纲构建阶段（Outline Construction Phase，强制，感知完成之后，仅一次）

        - 在人类填写表单后，你 **必须** 委派给 **outline agent**
        - outline agent 生成大纲后，会返回给你并标记 `need_human_interaction: true`
        - 收到此标记后，你 **必须** 立即委派给 **human agent**（设置 `interaction_type: outline_confirmation`）
        - 🔴 人类确认/修改的大纲具有最高优先级
        - 大纲确认后即冻结，不得重新生成

        ---

        ### 3. 推理与研究阶段（Reasoning & Research Phase，强制，大纲确认之后）

        - 在大纲被确认之后，你 **必须** 执行一个集中式的推理与研究阶段
        - 在该阶段，中枢智能体**必须**：
          - 至少调用 **Researcher agent** 一次
          - 使用可用工具、文档或外部信息源，对已确认的大纲进行验证、补充或质疑
        - 若发现信息不足，可以委派给 **human agent** 进行主动提问（设置 `interaction_type: proactive_question`）
        - **无论当前信息是否看似充分，该阶段都必须为每一个任务执行一次**

        ---

        ### 4. 内容生成阶段（Content Generation Phase，强制，最终阶段）

        - 在推理与研究阶段完成后，你 **必须** 委派给 **reporter agent**
        - reporter agent 生成报告后，会返回给你并标记 `need_human_interaction: true`
        - 收到此标记后，你 **必须** 委派给 **human agent**（设置 `interaction_type: report_feedback`）
        - 根据人类反馈决定是否需要修改报告
        - 在 reporter agent 尚未生成最终内容之前，**不得进入 FINISH 状态**

        ---

        ### 4.1 用户反馈处理循环

        - 用户反馈分为两类：
          - **风格切换**（[CHANGED_STYLE]）：直接委派 reporter agent 使用新风格重新生成报告
          - **其他修改意见**（[CONTENT_MODIFY] 等）：你需要根据修改意见的具体内容和当前上下文，自行判断应该委派哪些 agent、以什么顺序执行
        - **reporter agent 每次重新生成报告后，都会返回并标记 `need_human_interaction: true`、`human_interaction_type: "report_feedback"`**
        - 🔴 **此时你必须再次委派给 human agent**，让用户查看新报告并决定下一步操作
        - 🔴 **绝对禁止在 `need_human_interaction: true` 时选择 FINISH**
        - 这个循环可能重复多次，每次都必须经过 human agent
        - **只有当用户明确发送 [SKIP]、[END] 或 [FINISH] 反馈后，才可以进入 FINISH 状态**

        ---

        #### 执行约束与禁止行为

        - 执行顺序 **必须严格遵循**：
          **感知 → [Human] → 大纲 → [Human] → 研究 → 报告 → [Human] → 完成**
        - 🔴 当 `need_human_interaction: true` 时，**必须** 委派给 human agent，**不得跳过**
        - 🔴 **FINISH 的前置条件**：只有当 `need_human_interaction` 为 `false` 且用户已明确确认后，才允许进入 FINISH 状态

        ---

        你的目标是：
        在严格遵循上述不可回退执行流程的前提下，确保任务在结构上稳定、在人机交互上可控，并实现多智能体系统的可靠协同。
        """

        DECISION_SOP_SP_TEST = """### 执行流程指南（Execution Workflow Guidelines）

        你正在一个具有明确执行流程的多智能体系统中运行。
        你的职责是**严格按照以下流程推进任务直至完成**，仅在任务复杂度提升或信息确实缺失时，才允许插入额外步骤。

        ---

        #### 强制性的高层执行流程（Mandatory High-Level Workflow）

        ### 1. 大纲构建阶段（Outline Construction Phase，强制，规划之后）

        - 你 **必须** 先委派给 **outline agent**。
        - outline agent 负责基于已有上下文：
        - 生成新的结构化大纲，或
        - 对现有大纲进行结构性优化与修正。
        - 该阶段 **至少必须执行一次**。
        - 所有内部的大纲策略（如迭代深度、扩展、删减等）**完全由 outline agent 自主处理**。

        ---

        ### 2. 推理与研究阶段（Reasoning & Research Phase，强制，位于大纲与内容生成之间）

        - 在大纲生成之后，你 **必须** 执行一个集中式的推理阶段。
        - 在该阶段，中枢智能体（central agent）**必须**：
        - 至少调用 **Researcher agent** 一次；
        - 使用可用工具、文档或外部信息源，对大纲进行验证、补充或质疑。
        - 该阶段的核心职责包括：
        - 识别大纲中缺失、薄弱或缺乏支撑的章节；
        - 在进入内容生成之前，解决结构性歧义或不确定性问题。
        - **无论当前信息是否看似充分，该阶段都必须为每一个任务执行一次**。

        ---

        ### 3. 内容生成阶段（Content Generation Phase，强制，大纲确认之后）

        - 一旦大纲生成并被确认，你 **必须** 委派给 **reporter agent**。
        - reporter agent 必须 **严格依据已确认的大纲结构** 生成最终内容。
        - 该阶段是任务完成的 **必要条件**。

        ---

        #### 执行约束与规则（Execution Constraints and Rules）

        - 执行顺序 **必须严格遵循**：  
        **大纲构建 → 推理与研究 → 内容生成**
        - 仅当后续阶段暴露出内容结构或章节规划问题时，才允许回退至早期阶段。
        - 在任何情况下，**都不得跳过大纲构建阶段**。
        - **在 reporter agent 尚未生成最终内容之前，不得进入 FINISH 状态**。
        - 若在任何阶段发现信息不足，必须在继续之前插入适当的补充步骤。

        ---

        #### 强制研究调用规则（Mandatory Research Invocation）

        - 在 **每一次任务执行中**，Researcher agent **必须** 作为「推理与研究阶段」的一部分被调用。
        - **不得跳过、伪造或模拟该阶段**；
        - 在没有真实调用 Researcher agent 的情况下继续执行，是不被允许的。

        ---

        你的目标是：  
        在严格遵循上述执行流程的前提下，确保任务在逻辑上完整、准备充分，并实现多智能体之间的高效、协调执行。
        """

        # 这个似乎要改其他地方，反正后面用不上，不要了
        graph_format = config["configurable"]["graph_format"]
        if graph_format == "sp_xxqg":
            state["sop"] = DECISION_SOP_SP
            logger.info(f"使用 SP 的 SOP")
        if graph_format == "sp_test":
            state["sop"] = DECISION_SOP_SP_TEST
            logger.info(f"使用 SP 的 SOP")
        else:
            state["sop"] = None
            logger.info(f"不使用 SOP")

        # 构建决策prompt
        messages = self._build_decision_prompt(state, config)
        logger.debug(f"决策prompt: {messages}")

        # 获取LLM决策并处理异常
        try:
            llm = get_llm_by_type(
                AGENT_LLM_MAP.get("central_agent", "default")
            ).with_structured_output(
                Decision,
                method="json_mode",
            )
            response = llm.invoke(messages)

            # 解析决策结果
            action = CentralAgentAction(response.action)
            reasoning = response.reasoning
            params = response.params or {}
            instruction = response.instruction or self.action_instructions.get(
                action, ""
            )

            if state.get("locale") == None:
                locale = response.locale or "zh-CN"
                state["locale"] = locale

            logger.info(f"决策结果: {response}")
            end_time = datetime.now()
            time_entry = {
                "step_name": "central decision" + start_time.isoformat(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
            }
            global_statistics.add_time_entry(time_entry)

            return CentralDecision(
                action=action,
                reasoning=reasoning,
                params=params,
                instruction=instruction,
                state_updates=None,
            )

        except Exception as e:
            import traceback

            logger.error(
                f"决策解析失败:  (尝试 {retry_count + 1}/{max_retries}): {str(e)}"
            )
            logger.error("详细错误信息：\n" + traceback.format_exc())
            if retry_count < max_retries - 1:
                return self.make_decision(state, config, retry_count + 1)
            # 异常情况下返回默认决策
            end_time = datetime.now()
            time_entry = {
                "step_name": "central_decision" + start_time.isoformat(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
            }
            global_statistics.add_time_entry(time_entry)
            return CentralDecision(
                action=CentralAgentAction.THINK,
                reasoning="决策解析失败，默认选择思考动作",
                params={},
                instruction=self.action_instructions[CentralAgentAction.THINK],
            )

    def _build_decision_prompt(
        self,
        state: State,
        config: RunnableConfig,
    ) -> List[Union[AIMessage, HumanMessage]]:
        """
        构建中枢Agent决策提示词，使用统一的prompt模板

        Args:
            context: 决策上下文（已包含所有关键参数）
            config: 运行配置
            action_options: 可用动作选项

        Returns:
            格式化的提示词消息列表
        """
        messages_history = state.get("messages", [])
        SOP = state.get("sop", None)
        converted_messages = []
        for msg in messages_history:
            if isinstance(msg, (HumanMessage, AIMessage)):
                converted_messages.append(
                    {
                        "role": msg.type,
                        "content": msg.content,
                        "additional_kwargs": getattr(msg, "additional_kwargs", {}),
                    }
                )
            else:
                converted_messages.append(msg)

        # 提取用户反馈相关的 state 变量（具体渲染逻辑已迁移到 central_agent.md 模板）
        need_human_interaction = state.get("need_human_interaction", False)
        human_interaction_type = state.get("human_interaction_type", "")
        hitl_feedback = state.get("hitl_feedback", "")

        context = {
            "available_actions": [action.value for action in CentralAgentAction],
            "available_sub_agents": self.available_sub_agents,
            "sub_agents_description": self.sub_agents_description,
            "current_action": "decision",
            "messages_history": converted_messages,
            "locale": state.get("locale", "zh-CN"),  # 确保locale被传递到模板
            "hitl_feedback": hitl_feedback,
            "SOP": SOP,
            "need_human_interaction": need_human_interaction,
            "human_interaction_type": human_interaction_type,
        }
        action_options = list(CentralAgentAction)
        # 加载正确的模板名称并合并动作选项
        context_with_actions = {
            **context,
            **config,
            "available_actions": ", ".join([a.value for a in action_options]),
            "SOP": SOP,
        }

        return apply_prompt_template(
            "central_agent", state, extra_context=context_with_actions
        )

    def execute_action(
            self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """
        执行决策动作，调度对应的动作处理器

        Args:
            decision: 决策结果
            state: 当前系统状态
            config: 运行配置

        Returns:
            动作执行结果Command对象
        """
        handler = self.action_handlers.get(decision.action)
        if not handler:
            error_msg = f"未知动作: {decision.action}"
            logger.error(error_msg)
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"错误：未知动作: {decision.action}",
                            name="central_error",
                        )
                    ],
                    "locale": state.get("locale"),
                    "current_node": "central_agent",
                    "memory_stack": self.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

        # 执行 handler
        result = handler(decision, state, config)

        # 🆕 如果决策中有状态更新，创建新的 Command 对象
        if hasattr(decision, 'state_updates') and decision.state_updates:
            # 合并状态更新
            merged_update = {}
            if result.update:
                merged_update.update(result.update)  # 先复制原有的 update
            merged_update.update(decision.state_updates)  # 再让 state_updates 覆盖

            # 🆕 对于 delegate 动作，添加用户可见的消息
            if decision.action == CentralAgentAction.DELEGATE:
                params = decision.params if isinstance(decision.params, dict) else {}
                agent_type = params.get("agent_type", "")
                task_description = params.get("task_description", "")

                # 根据不同的 agent_type 生成不同的消息
                if agent_type == "researcher":
                    user_message = f"🔍 正在进行研究：{task_description[:50]}..." if len(
                        task_description) > 50 else f"🔍 正在进行研究：{task_description}"
                elif agent_type == "reporter":
                    chapter_mode = params.get("chapter_mode", "")
                    if chapter_mode == "single":
                        chapter_index = params.get("chapter_index", 1)
                        user_message = f"📝 正在生成第 {chapter_index} 章内容..."
                    elif chapter_mode == "merge":
                        user_message = "📄 正在合并所有章节，生成最终报告..."
                    else:
                        user_message = f"📝 正在生成报告..."
                elif agent_type == "human":
                    user_message = "👤 等待用户反馈..."
                else:
                    user_message = f"⚡ 正在执行任务..."

                # 添加用户可见的消息
                if "messages" not in merged_update:
                    merged_update["messages"] = []
                merged_update["messages"].append(
                    AIMessage(content=user_message, name="central_agent")
                )

            logger.debug(f"状态更新已合并: {decision.state_updates}")

            # 创建新的 Command 对象
            return Command(
                update=merged_update,
                goto=result.goto,
            )

        # 🆕 对于没有 state_updates 的 delegate 动作，也添加用户可见的消息
        if decision.action == CentralAgentAction.DELEGATE:
            params = decision.params if isinstance(decision.params, dict) else {}
            agent_type = params.get("agent_type", "")
            task_description = params.get("task_description", "")

            # 根据不同的 agent_type 生成不同的消息
            if agent_type == "researcher":
                user_message = f"🔍 正在进行研究：{task_description[:50]}..." if len(
                    task_description) > 50 else f"🔍 正在进行研究：{task_description}"
            elif agent_type == "reporter":
                chapter_mode = params.get("chapter_mode", "")
                if chapter_mode == "single":
                    chapter_index = params.get("chapter_index", 1)
                    user_message = f"📝 正在生成第 {chapter_index} 章内容..."
                elif chapter_mode == "merge":
                    user_message = "📄 正在合并所有章节，生成最终报告..."
                else:
                    user_message = f"📝 正在生成报告..."
            elif agent_type == "human":
                user_message = "👤 等待用户反馈..."
            else:
                user_message = f"⚡ 正在执行任务..."

            # 添加用户可见的消息
            if result.update:
                if "messages" not in result.update:
                    result.update["messages"] = []
                result.update["messages"].append(
                    AIMessage(content=user_message, name="central_agent")
                )

        return result

    def _handle_think(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理思考动作，分析当前状态生成下一步计划"""
        logger.info("中枢Agent正在思考...")
        start_time = datetime.now()
        context = {
            "current_action": "think",
            "current_progress": state.get("observations", []),
            "decision_reasoning": decision.reasoning,
            "instruction": decision.instruction,
            "locale": state.get("locale", "zh-CN"),  # 确保locale被传递到模板
        }

        # 应用统一的决策提示模板
        messages = apply_prompt_template("central_agent", state, extra_context=context)

        llm = get_llm_by_type(AGENT_LLM_MAP.get("central_agent", "default"))
        response = llm.invoke(messages)

        # 记录思考过程到记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="think",
            content=response.content,
        )
        self.memory_stack.push(memory_entry)

        logger.info(f"central_think: {response.content}")
        end_time = datetime.now()
        time_entry = {
            "step_name": "central_think" + start_time.isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
        }
        global_statistics.add_time_entry(time_entry)
        return Command(
            update={
                "messages": [AIMessage(content=response.content, name="central_think")],
                "current_node": "central_agent",
                "memory_stack": json.dumps(
                    [entry.to_dict() for entry in self.memory_stack.get_all()]
                ),
                "locale": state.get("locale"),
            },
            goto="central_agent",
        )

    def _handle_reflect(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理反思动作，评估之前的步骤并清理记忆栈"""
        logger.info("中枢Agent正在反思...")
        start_time = datetime.now()

        # 获取反思目标和上下文
        # recent_memory = self.memory_stack.get_recent(5)  # 获取最近5条记忆

        context = {
            "current_action": "reflect",
            "decision_reasoning": decision.reasoning,
            "instruction": decision.instruction,
            "locale": state.get("locale", "zh-CN"),  # 确保locale被传递到模板
        }

        # 应用反思提示模板
        messages = apply_prompt_template("central_agent", state, extra_context=context)

        llm = get_llm_by_type(AGENT_LLM_MAP.get("central_agent", "default"))
        response = llm.invoke(messages)

        # 解析反思结果的JSON
        try:
            reflection_data = json.loads(repair_json_output(response.content))
            analysis = reflection_data.get("analysis", "反思分析")
            pop_count = reflection_data.get("pop_count", 0)
            reasoning = reflection_data.get("reasoning", "反思完成")

            # 验证pop_count是有效数字
            if not isinstance(pop_count, int) or pop_count < 0:
                logger.warning(f"无效的pop_count: {pop_count}，设置为0")
                pop_count = 0

        except Exception as e:
            logger.error(f"反思结果解析失败: {e}")
            analysis = response.content
            pop_count = 0
            reasoning = "JSON解析失败，保持现有记忆栈"

        logger.debug(f"reflect决定清理{pop_count}条消息")
        # 执行记忆栈清理
        removed_items = []
        if pop_count > 0:
            reflection_content = (
                f"反思分析: {analysis}\n"
                f"反思原因: {reasoning}\n"
                f"清理了 {pop_count} 条记忆。"
            )

            memory_entry = MemoryStackEntry(
                timestamp=datetime.now().isoformat(),
                action="reflect",
                content=reflection_content,
            )

            self.memory_stack.push_with_pop(memory_entry, pop_count)

            removed_items = self.memory_stack.pop(pop_count)

            logger.info(f"成功从记忆栈中移除了 {pop_count} 项记忆")
            # logger.info(
            #     f"从记忆栈中移除了 {len(removed_items)} 项: {[item.action for item in removed_items]}"
            # )
        else:
            logger.info("不移除任何记忆栈项目")

        logger.info(f"central_reflect: {analysis}")
        end_time = datetime.now()
        time_entry = {
            "step_name": "central_reflect" + start_time.isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
        }
        global_statistics.add_time_entry(time_entry)
        return Command(
            update={
                "messages": [AIMessage(content=analysis, name="central_reflect")],
                "reflection": {
                    "analysis": analysis,
                    "pop_count": len(removed_items),
                    "reasoning": reasoning,
                    "removed_items": removed_items,
                },
                "current_node": "central_agent",
                "memory_stack": json.dumps(
                    [entry.to_dict() for entry in self.memory_stack.get_all()]
                ),
                "locale": state.get("locale"),
            },
            goto="central_agent",
        )

    def _handle_summarize(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理总结动作，归纳当前已获得的信息"""
        logger.info("中枢Agent正在总结...")
        start_time = datetime.now()

        context = {
            "current_action": "summarize",
            "summarization_focus": decision.reasoning,
            "instruction": decision.instruction,
            "locale": state.get("locale", "zh-CN"),  # 确保locale被传递到模板
        }

        # 打印上下文用于调试
        logger.debug(
            f"Summarize context: {json.dumps(context, ensure_ascii=False, indent=2)}"
        )

        # 应用统一的总结提示模板
        messages = apply_prompt_template("central_agent", state, extra_context=context)

        llm = get_llm_by_type(AGENT_LLM_MAP.get("central_agent", "default"))
        response = llm.invoke(messages)

        # 更新记忆栈，替换最新的总结结果
        new_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="summarize",
            content=context.get("summarization_focus", ""),
            result={"summary_result": response.content},
        )

        # logger.info("NEW_ENTRY", new_entry)
        # logger.info("*"*100)

        self.memory_stack.push_with_pop(new_entry)

        # logger.info(f"central_summarize: {response.content}")
        end_time = datetime.now()
        time_entry = {
            "step_name": "central_summarize" + start_time.isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
        }
        global_statistics.add_time_entry(time_entry)
        return Command(
            update={
                "messages": [
                    AIMessage(content=response.content, name="central_summarize")
                ],
                "summary": response.content,
                "current_node": "central_agent",
                "memory_stack": json.dumps(
                    [entry.to_dict() for entry in self.memory_stack.get_all()]
                ),
                "locale": state.get("locale"),
            },
            goto="central_agent",
        )

    def _handle_delegate(
            self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理委派动作，调度子Agent执行专项任务"""
        # 检查 decision.params 的类型并提取参数
        if isinstance(decision.params, dict):
            agent_type = decision.params.get("agent_type")
            task_description = decision.params.get("task_description", "")
            params_dict = decision.params
        else:
            agent_type = decision.params.agent_type
            task_description = decision.params.task_description or ""
            if hasattr(decision.params, "model_dump"):  # Pydantic v2
                params_dict = decision.params.model_dump()
            elif hasattr(decision.params, "dict"):  # Pydantic v1
                params_dict = decision.params.dict()
            else:
                params_dict = {}

        # 验证子Agent类型有效性
        if not agent_type or agent_type not in self.available_sub_agents:
            error_msg = f"无效的子Agent类型: {agent_type}，可用类型: {self.available_sub_agents}"
            logger.error(f"central_error: {error_msg}")
            return Command(
                update={
                    "messages": [AIMessage(content=error_msg, name="central_error")],
                    "current_node": "central_agent",
                },
                goto="central_agent",
            )

        logger.info(f"中枢Agent委派 {agent_type} 执行任务: {task_description}")

        # 记录委派动作到记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type=agent_type,
            content=f"委派任务: {task_description}",
        )
        self.memory_stack.push(memory_entry)

        # 构建子Agent执行上下文
        delegation_context = {
            "task_description": task_description,
            "agent_type": agent_type,
            "memory_context": self.memory_stack.get_summary(include_full_history=True),
            "original_query": state.get("user_query", ""),
        }

        # 🔧 风格处理逻辑（只保留一次）
        hitl_feedback = state.get("hitl_feedback", "")
        if hitl_feedback and "[CHANGED_STYLE]" in hitl_feedback:
            # 🔧 风格切换：解析新风格
            new_style = hitl_feedback.split("[CHANGED_STYLE]")[1].strip().split()[0] if \
                hitl_feedback.split("[CHANGED_STYLE]")[1].strip().split() else hitl_feedback.split("[CHANGED_STYLE]")[
                1].strip()

            # 将新风格添加到 delegation_context
            delegation_context["style"] = new_style
            delegation_context["original_report"] = state.get("final_report", "")

            logger.info(f"🎨 风格切换被触发！")
            logger.info(f"   - 新风格: {new_style}")
            logger.info(f"   - delegation_context['style']: {delegation_context.get('style')}")
        else:
            # 正常获取风格
            user_selected_style = state.get("user_selected_style", "")

            if not user_selected_style:
                # 尝试从 user_query 或 original_query 中解析风格要求
                user_query = state.get("user_query", "") or state.get("original_query", "")
                original_query = state.get("original_query", "")
                all_query_text = f"{user_query} {original_query}"

                import re

                # 方法1：提取【风格要求】后面的内容
                style_match = re.search(r'【风格要求】\s*([\s\S]*?)(?=\n\n|【|$)', all_query_text)
                if style_match:
                    style_text = style_match.group(1).strip()
                    # 提取前3行作为风格描述
                    style_lines = [line.strip() for line in style_text.split('\n') if line.strip()]
                    if style_lines:
                        user_selected_style = ' '.join(style_lines[:3])
                        logger.info(f"🎨 从【风格要求】中提取到风格: {user_selected_style[:100]}")

            # 设置最终风格
            final_style = user_selected_style or "政策研究报告"
            delegation_context["style"] = final_style
            logger.info(f"📊 最终使用的风格: {final_style[:100] if len(final_style) > 100 else final_style}")

        # ZX 🆕 将 params 中的所有字段合并到 delegation_context
        for key, value in params_dict.items():
            if key not in ["task_description", "agent_type"] and value is not None:
                delegation_context[key] = value

        # ZX 🆕 添加调试日志
        logger.info(f"📊 delegation_context keys: {list(delegation_context.keys())}")
        logger.info(f"📊 chapter_mode: {delegation_context.get('chapter_mode')}")
        logger.info(f"📊 chapter_index: {delegation_context.get('chapter_index')}")

        # 处理内容修改导致的 reporter 委派
        hitl_feedback = state.get("hitl_feedback", "")
        clear_hitl_feedback = False
        if (
                agent_type == "reporter"
                and isinstance(hitl_feedback, str)
                and hitl_feedback.upper().startswith("[CONTENT_MODIFY]")
        ):
            clear_hitl_feedback = True
            modify_request = hitl_feedback[len("[CONTENT_MODIFY]"):].strip()
            if modify_request:
                delegation_context["content_modify_request"] = modify_request
            delegation_context["skip_hitl_feedback"] = True

        logger.info(f"central_delegate: 委派{agent_type}执行: {task_description}")

        # 获取当前段落索引
        current_chapter_index = state.get("current_chapter_index", 0)

        # ZX 🆕 保留子 Agent 设置的交互状态
        need_human_interaction = state.get("need_human_interaction", False)
        human_interaction_type = state.get("human_interaction_type", "")

        logger.info(f"📊 保留交互状态:")
        logger.info(f"   - need_human_interaction: {need_human_interaction}")
        logger.info(f"   - human_interaction_type: {human_interaction_type}")

        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"委派{agent_type}执行: {task_description}",
                        name="central_delegate",
                    )
                ],
                "delegation_context": delegation_context,
                "current_node": "central_agent",
                "memory_stack": json.dumps(
                    [entry.to_dict() for entry in self.memory_stack.get_all()]
                ),
                "locale": state.get("locale"),
                "current_chapter_index": current_chapter_index,
                # ZX 🆕 保留交互状态
                "need_human_interaction": need_human_interaction,
                "human_interaction_type": human_interaction_type,
                **({"hitl_feedback": ""} if clear_hitl_feedback else {}),
            },
            goto=agent_type,
        )

    def _handle_finish(
            self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理完成动作，生成最终报告并结束任务"""
        logger.info("中枢Agent完成任务...")

        final_report = state.get("final_report", None)

        # 🆕 新增：检查用户是否已明确确认
        hitl_feedback = state.get("hitl_feedback", "")
        user_confirmed = (
                hitl_feedback and
                str(hitl_feedback).upper().startswith(("[SKIP]", "[END]", "[FINISH]"))
        )

        # 🆕 关键修改：优先检查用户确认状态
        if not final_report:
            if user_confirmed:
                # 用户已确认但没有报告 -> 记录警告并强制结束
                logger.warning("用户已确认完成，但未找到最终报告。强制结束任务以避免死循环。")
                logger.warning(f"hitl_feedback: {hitl_feedback}")

                # 强制结束，避免死循环
                return Command(
                    update={
                        "messages": [
                            AIMessage(
                                content="用户已确认完成，任务结束",
                                name="central_agent",
                            )
                        ],
                        "current_node": "central_agent",
                    },
                    goto="zip_data",  # 直接结束
                )
            else:
                # 用户未确认且没有报告 -> 正常委派 reporter
                logger.info("未找到最终报告，委派Reporter Agent生成报告...")

                # 记录委派动作到记忆栈
                memory_entry = MemoryStackEntry(
                    timestamp=datetime.now().isoformat(),
                    action="delegate",
                    agent_type="reporter",
                    content="未生成最终报告，委派Reporter Agent生成最终报告",
                )
                self.memory_stack.push(memory_entry)

                # 构建Reporter执行上下文
                delegation_context = {
                    "task_description": "根据所有收集到的信息生成完整的最终报告",
                    "agent_type": "reporter",
                    "memory_context": self.memory_stack.get_summary(
                        include_full_history=True
                    ),
                    "original_query": state.get("user_query", ""),
                    "report_type": "final_report",
                    "execution_history": [
                        entry.to_dict() for entry in self.memory_stack.get_all()
                    ],
                }

                logger.info("central_delegate_reporter: 委派Reporter Agent生成最终报告")
                return Command(
                    update={
                        "messages": [
                            AIMessage(
                                content="委派Reporter Agent生成最终报告",
                                name="central_delegate_reporter",
                            )
                        ],
                        "delegation_context": delegation_context,
                        "current_node": "central_agent",
                        "memory_stack": json.dumps(
                            [entry.to_dict() for entry in self.memory_stack.get_all()]
                        ),
                        "pending_finish": True,  # 标记等待报告完成后再finish
                    },
                    goto="reporter",
                )

        # 有报告，正常结束流程
        logger.info(f"final_report: {final_report}")

        session_id = config["configurable"]["thread_id"]
        # global_reference_map.save_session(session_id)
        # 构建执行摘要（包含完整记忆栈历史）
        execution_summary = {
            "user_query": state.get("user_query", "未知查询"),
            "execution_history": [
                entry.to_dict() for entry in self.memory_stack.get_all()
            ],
            "final_report": final_report,
            "research": global_reference_map.get_session_ref_map(
                session_id
            ),  # state.get("data_collections", []),
            "completion_time": datetime.now().isoformat(),
            "statistics": global_statistics.get_statistics(),
        }

        # 保存执行摘要到文件
        os.makedirs("./reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./reports/execution_report_{timestamp}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(execution_summary, f, ensure_ascii=False, indent=4)
            report_msg = f"任务完成，报告已保存: {filename}"
        except Exception as e:
            logger.error(f"报告保存失败: {str(e)}")
            report_msg = f"任务完成，但报告保存失败: {str(e)}"
            execution_summary["error"] = str(e)

        logger.info(report_msg)
        logger.info(global_statistics.get_statistics())
        return Command(
            goto="zip_data",  # 结束执行
        )
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

from ..graph.types import State

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

        # 增加 SOP 部分，用于加入 decision 模块
        # SOP改成中文，SOP应该要的是抽象的。不能写是outline，replanner，具体谁来生成是让 CentralAgent 自己找
        DECISION_SOP_SP = """### 执行流程指南（Execution Workflow Guidelines）

        你正在一个具有**严格阶段约束与不可回退节点**的多智能体系统中运行。
        你的职责是**严格按照以下流程推进任务直至完成**，并遵守每个阶段的进入与退出规则。
        任何违反阶段约束的行为都被视为执行错误。

        ---

        #### 🔴 Human Agent 使用说明（Critical）

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

        #### 强制性的高层执行流程（Mandatory High-Level Workflow）

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
        - 在该阶段，中枢智能体（central agent）**必须**：
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

        ### 4.1 用户反馈处理循环（User Feedback Loop，关键补充）

        - 用户反馈分为两类：
          - **风格切换**（[CHANGED_STYLE]）：直接委派 reporter agent 使用新风格重新生成报告
          - **其他修改意见**（[CONTENT_MODIFY] 等）：你需要根据修改意见的具体内容和当前上下文，自行判断应该委派哪些 agent、以什么顺序执行。例如：
            - 如果修改意见涉及补充信息或搜索更多资料，可以先委派 researcher，再委派 reporter
            - 如果修改意见仅涉及措辞或结构调整，可以直接委派 reporter
            - 无论经过多少中间步骤，最终都必须由 reporter 重新生成报告
        - **reporter agent 每次重新生成报告后，都会返回并标记 `need_human_interaction: true`、`human_interaction_type: "report_feedback"`**
        - 🔴 **此时你必须再次委派给 human agent**，让用户查看新报告并决定下一步操作
        - 🔴 **绝对禁止在 `need_human_interaction: true` 时选择 FINISH**——这会导致用户永远看不到重新生成的报告
        - 这个循环可能重复多次，每次都必须经过 human agent
        - **只有当用户明确发送 [SKIP]、[END] 或 [FINISH] 反馈后，才可以进入 FINISH 状态**

        ---

        #### 主动提问机制（Proactive Questioning）

        在任何阶段，如果你判断当前信息不足以继续执行任务，可以委派给 **human agent** 进行主动提问：

        ```json
        {
          "action": "delegate",
          "reasoning": "当前信息不足，需要向用户询问具体问题",
          "params": {
            "agent_type": "human",
            "task_description": "向用户询问关于XXX的具体信息",
            "interaction_type": "proactive_question",
            "question": "你需要问的具体问题"
          },
          "instruction": "委派给 Human Agent 进行主动提问"
        }
        ```

        ---

        #### DELEGATE to Human Agent 示例

        当收到 `need_human_interaction: true` 时，必须这样委派：

        ```json
        {
          "action": "delegate",
          "reasoning": "Perception agent 已生成表单，需要人类填写后才能继续",
          "params": {
            "agent_type": "human",
            "task_description": "请人类填写表单",
            "interaction_type": "form_filling"
          },
          "instruction": "委派给 Human Agent 收集人类输入"
        }
        ```

        ---

        #### 执行约束与禁止行为（Hard Constraints & Prohibited Actions）

        - 执行顺序 **必须严格遵循**：
          **感知 → [Human] → 大纲 → [Human] → 研究 → 报告 → [Human] → (反馈循环: [根据反馈内容自行决定中间步骤] → 报告 → [Human] →) → 完成**
        - 🔴 当 `need_human_interaction: true` 时，**必须** 委派给 human agent，**不得跳过**，**不得选择 FINISH 或其他任何动作**
        - 🔴 **FINISH 的前置条件**：只有当 `need_human_interaction` 为 `false` 且用户已明确确认（发送 [SKIP]/[END]/[FINISH]）后，才允许进入 FINISH 状态
        - perception 阶段与 outline 阶段：
          - **均为一次性阶段**
          - **均不可重复、不可回退、不可重新进入**

        ---

        #### 强制研究调用规则（Mandatory Research Invocation）

        - 在 **每一次任务执行中**，Researcher agent **必须** 作为「推理与研究阶段」的一部分被真实调用
        - **不得跳过、伪造或模拟该阶段**
        - 在未真实调用 Researcher agent 的情况下继续执行，是被明确禁止的

        ---

        你的目标是：
        在严格遵循上述不可回退执行流程的前提下，确保任务在结构上稳定、在人机交互上可控，并实现多智能体系统的可靠协同。
        """

        # 这个似乎要改其他地方，反正后面用不上，不要了
        # graph_format=config["configurable"]["graph_format"]
        # if graph_format=="sp_xxqg":
        #     state["sop"] = DECISION_SOP_SP
        #     logger.info(f"使用 SP 的 SOP")
        # else:
        #     state["sop"] = None
        #     logger.info(f"不使用 SOP")
        state["sop"] = DECISION_SOP_SP
        logger.info(f"使用 SP 的 SOP")

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
                # 将 locale 添加到 state
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

        return handler(decision, state, config)

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
        agent_type = decision.params.agent_type
        task_description = decision.params.task_description
        # agent_type = decision.agent_type
        # task_description = decision.task_description or "未指定任务"

        # 验证子Agent类型有效性
        if not agent_type or agent_type not in self.available_sub_agents:
            error_msg = (
                f"无效的子Agent类型: {agent_type}，可用类型: "
                f"{self.available_sub_agents}"
            )
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

        # 构建子Agent执行上下文（包含记忆栈摘要）
        delegation_context = {
            "task_description": task_description,
            "agent_type": agent_type,
            "memory_context": self.memory_stack.get_summary(include_full_history=True),
            "original_query": state.get("user_query", ""),
        }
        # 若为内容修改导致的 reporter 委派，清理 hitl_feedback 以避免 reporter 反复处理同一条反馈
        hitl_feedback = state.get("hitl_feedback", "")
        clear_hitl_feedback = False
        if (
            agent_type == "reporter"
            and isinstance(hitl_feedback, str)
            and hitl_feedback.upper().startswith("[CONTENT_MODIFY]")
        ):
            clear_hitl_feedback = True
            modify_request = hitl_feedback[len("[CONTENT_MODIFY]") :].strip()
            if modify_request:
                delegation_context["content_modify_request"] = modify_request
            delegation_context["skip_hitl_feedback"] = True

        # 传递 decision.params 中的额外字段（如 interaction_type, question 等）
        # 这对于 Human Agent 来说是必需的
        if hasattr(decision.params, "model_dump"):  # Pydantic v2
            params_dict = decision.params.model_dump()
            for key, value in params_dict.items():
                if key not in delegation_context and value is not None:
                    delegation_context[key] = value
        elif hasattr(decision.params, "dict"):  # Pydantic v1
            params_dict = decision.params.dict()
            for key, value in params_dict.items():
                if key not in delegation_context and value is not None:
                    delegation_context[key] = value
        elif isinstance(decision.params, dict):
            for key, value in decision.params.items():
                if key not in delegation_context and value is not None:
                    delegation_context[key] = value

        logger.info(f"central_delegate: 委派{agent_type}执行: {task_description}")
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
        if not final_report:
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
        logger.info(f"final_report: {final_report}")

        # 构建执行摘要（包含完整记忆栈历史）
        execution_summary = {
            "user_query": state.get("user_query", "未知查询"),
            "execution_history": [
                entry.to_dict() for entry in self.memory_stack.get_all()
            ],
            "final_report": final_report,
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

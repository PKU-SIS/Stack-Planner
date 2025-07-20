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

        # 智能反射触发条件（满足任一即建议）
        reflect_conditions = []

        # 条件1：关键节点检查（子任务完成且结果重要）
        if state.get("subtask_completed") and self._is_critical_result(state):
            reflect_conditions.append("关键子任务完成需质量验证")

        # 条件2：错误率阈值（最近3步错误率>25%）
        if self._error_rate() > 0.25:
            reflect_conditions.append(f"高错误率({self._error_rate()*100}%)需分析")

        # 条件3：信息复杂度（结果长度>500字符或含复杂结构）
        if self._is_complex_content(state.get("last_result", "")):
            reflect_conditions.append("复杂内容需深度验证")

        # 条件4：路径决策点（多分支任务）
        if "OR" in state.get("user_query", "") or "multiple" in state.get("plan", ""):
            reflect_conditions.append("多路径决策需优化")

        # 动态决策：有条件时优先但不强制
        if reflect_conditions:
            logger.info(f"建议REFLECT: {', '.join(reflect_conditions)}")
            state["suggest_reflect"] = {
                "conditions": reflect_conditions,
                "priority": len(reflect_conditions),  # 条件越多优先级越高
            }

        # 添加信息冗余检查 - 当记忆栈相似条目过多时
        if self._has_redundant_entries(threshold=3):  # 新增方法
            logger.info("检测到冗余信息，优先触发REFLECT")
            return CentralDecision(
                action=CentralAgentAction.REFLECT,
                reasoning="记忆栈中存在相似条目，需要清理冗余",
                instruction="识别并移除重复信息",
            )

        # 构建决策prompt
        messages = self._build_decision_prompt(state, config)
        # logger.debug(f"决策prompt: {messages}")

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

        # Add previous action check
        previous_action = ""
        if self.memory_stack.peek():
            previous_action = self.memory_stack.peek().action

        context = {
            "available_actions": [action.value for action in CentralAgentAction],
            "available_sub_agents": self.available_sub_agents,
            "sub_agents_description": self.sub_agents_description,
            "current_action": "decision",
            "messages_history": converted_messages,
            "previous_action": previous_action,  # 新加一个previous_action，防止在遇到无法解决的问题时，一直重复reflect
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
                "subtask_completed": False,
                "is_critical_task": "报告" in task_description,  # 动态标记关键任务
            },
            goto=agent_type,
        )

    # 在子代理返回结果时标记完成
    def handle_subagent_result(self, result):
        self.memory_stack.push(result)
        return {"subtask_completed": True}  # 仅设置标记，不强制Reflect

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

    def _has_redundant_entries(self, threshold=3) -> bool:
        """检查最近条目是否高度相似"""
        recent_entries = self.memory_stack.get_recent(5)
        if len(recent_entries) < threshold:
            return False

        # 简单相似度检测（实际可替换为文本相似度算法）
        contents = [entry.content[:50] for entry in recent_entries]
        return len(set(contents)) < threshold

    def _is_critical_result(self, state) -> bool:
        """检查结果是否关键（需验证）"""
        critical_keywords = ["结论", "报告", "建议", "方案", "分析"]
        return any(kw in state.get("last_result", "") for kw in critical_keywords)

    def _error_rate(self) -> float:
        """计算最近错误率"""
        recent = self.memory_stack.get_recent(5)
        if not recent:
            return 0.0
        errors = sum(1 for e in recent if e.action == "error")
        return errors / len(recent)

    def _is_complex_content(self, content: str) -> bool:
        """检查内容复杂度"""
        if len(content) > 500:
            return True
        if "```" in content or "|" in content:  # 含代码/表格
            return True
        return False

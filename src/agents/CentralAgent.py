import json
import os
from typing import Annotated, Literal, Dict, List, Optional, Any, Union, Type, cast
from dataclasses import dataclass, field
from enum import Enum
import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command


from src.utils.logger import logger
from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.prompts.template import apply_prompt_template, get_prompt_template
from src.utils.json_utils import repair_json_output
from src.memory import MemoryStack, MemoryStackEntry

from ..graph.types import State


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


class SubAgentType(Enum):
    """子Agent类型枚举，定义可委派的专项Agent"""

    RESEARCHER = "researcher"  # 负责信息检索与研究
    CODER = "coder"  # 负责代码生成与执行
    REPORTER = "reporter"  # 负责结果整理与报告生成


# -------------------------
# 中枢Agent核心模块--中枢Agent的action
# -------------------------
@dataclass
class CentralDecision:
    """中枢Agent决策结果数据模型"""

    action: CentralAgentAction  # 决策动作
    reasoning: str  # 决策推理过程
    params: Dict[str, Any] = field(default_factory=dict)  # 动作参数=>delegate有参数
    instruction: Optional[str] = None  # 动作对应的指令说明


# TODO: 总结和反思动作没有总结到精华，导致需要重新research，又陷入了死循环
# TODO: prompt的参数和传入的参数没有对齐，需要尽快统一传入的参数
# 上下文没有传进agent
# 为什么只有思考和决策两个个动作
class CentralAgent:
    """
    中枢Agent核心类，负责系统整体决策与任务编排

    采用基于记忆栈的决策机制，通过状态分析动态委派子Agent执行专项任务，
    并最终整合结果生成完成报告
    """

    def __init__(self):
        self.memory_stack = MemoryStack()
        from src.agents.SubAgentManager import SubAgentManager

        self.sub_agent_manager = SubAgentManager(self)

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

    def make_decision(self, state: State, config: RunnableConfig) -> CentralDecision:
        """
        中枢Agent决策核心逻辑，分析当前状态生成决策结果

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            决策结果对象
        """
        logger.info("中枢Agent正在进行决策...")

        # 构建决策上下文
        context = self._build_decision_context(state)

        # 构建决策提示
        action_options = list(CentralAgentAction)
        messages = self._build_decision_prompt(context, config, action_options)

        # print(messages[0].content[:500])

        # 获取LLM决策并处理异常
        try:
            llm = get_llm_by_type(AGENT_LLM_MAP.get("central_agent", "default"))
            response = llm.invoke(messages)

            # 解析决策结果
            decision_data = json.loads(repair_json_output(response.content))
            print(f"决策结果: {decision_data}")
            action = CentralAgentAction(decision_data["action"])
            reasoning = decision_data.get("reasoning", "")
            params = decision_data.get("params", {})
            instruction = self.action_instructions.get(action, "")

            return CentralDecision(
                action=action,
                reasoning=reasoning,
                params=params,
                instruction=instruction,
            )

        except Exception as e:
            logger.error(f"决策解析失败: {str(e)}")
            # 异常情况下返回默认决策
            return CentralDecision(
                action=CentralAgentAction.THINK,
                reasoning="决策解析失败，默认选择思考动作",
                params={},
                instruction=self.action_instructions[CentralAgentAction.THINK],
            )

    def _build_decision_context(self, state: State) -> Dict[str, Any]:
        """
        构建决策上下文，包含系统当前状态的完整信息

        Args:
            state: 当前系统状态

        Returns:
            决策上下文字典
        """
        # 转换messages_history中的消息对象
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

        return {
            "user_query": state.get("user_query", ""),
            "current_node": state.get("current_node", "central_agent"),
            "memory_history": self.memory_stack.get_summary(include_full_history=True),
            "available_actions": [action.value for action in CentralAgentAction],
            "available_sub_agents": [agent.value for agent in SubAgentType],
            # 这个参数似乎没啥必要，先删了试试
            # "task_completed": state.get("task_completed", False),
            "recent_observations": state.get("observations", [])[-3:],
            "messages_history": converted_messages,  # 使用转换后的消息
        }

    def _build_decision_prompt(
        self,
        context: Dict[str, Any],
        config: RunnableConfig,
        action_options: List[CentralAgentAction],
    ) -> List[Union[AIMessage, HumanMessage]]:
        """
        构建中枢Agent决策提示词，使用统一的prompt模板

        Args:
            context: 决策上下文
            config: 运行配置
            action_options: 可用动作选项

        Returns:
            格式化的提示词消息列表
        """
        # 加载正确的模板名称("central_agent"修正之前的"center_agent")

        # 格式化prompt内容
        context_with_actions = {
            **context,
            "available_actions": ", ".join([a.value for a in action_options]),
        }
        formatted_prompt = get_prompt_template("central_agent", context_with_actions)

        # logger.debug(f"Formatted prompt: {formatted_prompt.split("### Decision Requirements", 1)[0]}")

        # # 打印上下文用于调试
        # logger.debug(
        #     f"Decision context: {json.dumps(context, ensure_ascii=False, indent=2)}"
        # )

        return [HumanMessage(content=formatted_prompt)]

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

        context = {
            "user_query": state.get("user_query", ""),
            "current_node": state.get("current_node", "central_agent"),
            "memory_history": [
                entry.to_dict() for entry in self.memory_stack.get_all()
            ],
            "current_progress": state.get("observations", []),
            "decision_reasoning": decision.reasoning,
            "instruction": decision.instruction,
        }
        # print(config)

        # 应用统一的决策提示模板
        # TODO：非Configrable上下文使用extra_context（这里添加了类）
        messages = apply_prompt_template("central_agent", state, extra_context=context)

        llm = get_llm_by_type(AGENT_LLM_MAP.get("central_agent", "default"))
        response = llm.invoke(messages)

        # 记录思考过程到记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.datetime.now().isoformat(),
            action="think",
            content=response.content,
        )
        self.memory_stack.push(memory_entry)

        return Command(
            update={
                "messages": [AIMessage(content=response.content, name="central_think")],
                "current_node": "central_agent",
                "memory_stack": self.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    def _handle_reflect(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理反思动作，评估之前的动作执行效果"""
        logger.info("中枢Agent正在反思...")

        # 获取最近动作用于反思
        previous_actions = self.memory_stack.get_recent(1)

        context = {
            "user_query": state.get("user_query", ""),
            "current_node": state.get("current_node", "central_agent"),
            "recent_actions": [entry.to_dict() for entry in previous_actions],
            "current_progress": state.get("observations", []),
            "original_query": state.get("user_query", ""),
            "reflection_target": decision.reasoning,
            "instruction": decision.instruction,
            "need_reflect_context": self.memory_stack.get_recent(),
        }

        # 应用统一的反思提示模板
        messages = apply_prompt_template("central_agent", state, extra_context=context)

        llm = get_llm_by_type(AGENT_LLM_MAP.get("central_agent", "default"))
        response = llm.invoke(messages)

        # 更新记忆栈，替换最新的反思结果
        new_entry = MemoryStackEntry(
            timestamp=datetime.datetime.now().isoformat(),
            action="reflect",
            content=response.content,
        )
        self.memory_stack.push_with_pop(new_entry)

        return Command(
            update={
                "messages": [
                    AIMessage(content=response.content, name="central_reflect")
                ],
                "reflection": response.content,
                "current_node": "central_agent",
                "memory_stack": self.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    def _handle_summarize(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理总结动作，归纳当前已获得的信息"""
        logger.info("中枢Agent正在总结...")

        context = {
            "user_query": state.get("user_query", ""),
            "current_node": state.get("current_node", "central_agent"),
            "memory_history": [
                entry.to_dict() for entry in self.memory_stack.get_all()
            ],
            "summarization_focus": decision.reasoning,
            "instruction": decision.instruction,
            "need_summary_context": self.memory_stack.get_recent(),
        }

        # 应用统一的总结提示模板
        messages = apply_prompt_template("central_agent", state, extra_context=context)

        llm = get_llm_by_type(AGENT_LLM_MAP.get("central_agent", "default"))
        response = llm.invoke(messages)

        # 更新记忆栈，替换最新的总结结果
        new_entry = MemoryStackEntry(
            timestamp=datetime.datetime.now().isoformat(),
            action="summarize",
            content=response.content,
            result={"summary_result": response.content},
        )
        self.memory_stack.push_with_pop(new_entry)

        return Command(
            update={
                "messages": [
                    AIMessage(content=response.content, name="central_summarize")
                ],
                "summary": response.content,
                "current_node": "central_agent",
                "memory_stack": self.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    def _handle_delegate(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理委派动作，调度子Agent执行专项任务"""
        agent_type = decision.params.get("agent_type")
        task_description = decision.params.get("task_description", "未指定任务")

        # 验证子Agent类型有效性
        if not agent_type or agent_type not in [agent.value for agent in SubAgentType]:
            error_msg = (
                f"无效的子Agent类型: {agent_type}，可用类型: "
                f"{[agent.value for agent in SubAgentType]}"
            )
            logger.error(error_msg)
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
            timestamp=datetime.datetime.now().isoformat(),
            action="delegate",
            agent_type=agent_type,
            content=f"委派任务: {task_description}",
        )
        self.memory_stack.push(memory_entry)

        # 构建子Agent执行上下文
        delegation_context = {
            "task_description": task_description,
            "agent_type": agent_type,
            "memory_context": self.memory_stack.get_summary(),
            "original_query": state.get("user_query", ""),
        }

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
                "memory_stack": self.memory_stack.to_dict(),
            },
            goto=agent_type,
        )

    def _handle_finish(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理完成动作，生成最终报告并结束任务"""
        logger.info("中枢Agent完成任务...")

        final_report = state.get("final_report", "任务已完成，但未生成详细报告")

        # 构建执行摘要
        execution_summary = {
            "user_query": state.get("user_query", "未知查询"),
            "execution_history": [
                entry.to_dict() for entry in self.memory_stack.get_all()
            ],
            "final_report": final_report,
            "completion_time": datetime.datetime.now().isoformat(),
        }

        # 保存执行摘要到文件
        os.makedirs("./reports", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

        # 记录完成动作到记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.datetime.now().isoformat(),
            action="finish",
            content=report_msg,
            result={"execution_summary": execution_summary, "report_file": filename},
        )
        self.memory_stack.push(memory_entry)

        return Command(
            update={
                "messages": [AIMessage(content=report_msg, name="central_finish")],
                # "task_completed": True,
                "execution_summary": execution_summary,
                "current_node": "central_agent",
            },
            goto="__end__",
        )

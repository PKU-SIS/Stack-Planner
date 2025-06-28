# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
from src.utils.logger import logger
import os
from typing import Annotated, Literal, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.CommonReactAgent import CommonReactAgent
from src.agents.CoderAgent import CoderAgent
from src.agents.ResearcherAgent import ResearcherAgent
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
    search_docs_tool,
)

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan, StepType
from src.prompts.template import apply_prompt_template, get_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine


# 定义中枢Agent动作类型
class CentralAgentAction(Enum):
    THINK = "think"  # 思考下一步行动
    REFLECT = "reflect"  # 反思当前状态
    SUMMARIZE = "summarize"  # 总结已有信息
    DELEGATE = "delegate"  # 委派子Agent
    FINISH = "finish"  # 完成任务


# 定义可用的子Agent类型
class SubAgentType(Enum):
    RESEARCHER = "researcher"
    CODER = "coder"
    REPORTER = "reporter"


# 记忆栈条目
@dataclass
class MemoryStackEntry:
    timestamp: str
    action: str
    agent_type: Optional[str] = None  # central or sub-agent
    content: str = ""
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "agent_type": self.agent_type,
            "content": self.content,
            "result": self.result,
        }


# 记忆栈管理器
class MemoryStack:
    def __init__(self, max_size: int = 50):
        self.stack: List[MemoryStackEntry] = []
        self.max_size = max_size

    def push(self, entry: MemoryStackEntry):
        """添加新条目到栈顶"""
        self.stack.append(entry)
        self._maintain_stack_size()

    def push_with_pop(self, entry: MemoryStackEntry):
        """先弹出栈顶，再推入新条目 - 用于反思和总结动作"""
        if self.stack:
            self.stack.pop()
        self.stack.append(entry)
        self._maintain_stack_size()

    def peek(self) -> Optional[MemoryStackEntry]:
        """查看栈顶条目但不弹出"""
        if self.stack:
            return self.stack[-1]
        return None

    def get_recent(self, count: int = 5) -> List[MemoryStackEntry]:
        """获取最近的N个条目"""
        return self.stack[-count:] if len(self.stack) >= count else self.stack

    def get_all(self) -> List[MemoryStackEntry]:
        """获取所有条目"""
        return self.stack.copy()

    def get_summary(self, include_full_history: bool = False) -> str:
        """获取记忆栈摘要，支持获取全部历史信息"""
        if not self.stack:
            return "记忆栈为空"

        if include_full_history:
            # 返回完整历史信息供大模型使用
            return json.dumps(
                [entry.to_dict() for entry in self.stack], ensure_ascii=False, indent=2
            )

        # 返回最近操作摘要
        recent_entries = self.get_recent(3)
        summary_parts = []
        for entry in recent_entries:
            action_desc = (
                f"{entry.action}({entry.agent_type})"
                if entry.agent_type
                else entry.action
            )
            summary_parts.append(
                f"[{entry.timestamp[:19]}] {action_desc}: {entry.content[:100]}..."
            )

        return "最近执行:\n" + "\n".join(summary_parts)

    def to_dict(self) -> List[Dict[str, Any]]:
        """转换为字典格式"""
        return [entry.to_dict() for entry in self.stack]

    def size(self) -> int:
        """获取栈大小"""
        return len(self.stack)

    def is_empty(self) -> bool:
        """检查栈是否为空"""
        return len(self.stack) == 0

    def _maintain_stack_size(self):
        """保持栈大小限制"""
        if len(self.stack) > self.max_size:
            self.stack.pop(0)


class SubAgentManager:
    """子Agent管理器"""

    def __init__(self, central_agent: "CentralAgent"):
        self.central_agent = central_agent

    async def execute_researcher(self, state: State, config: RunnableConfig) -> Command:
        """执行研究Agent"""
        logger.info("研究Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "未知研究任务")

        # 配置研究工具
        tools = [get_web_search_tool(10), crawl_tool, search_docs_tool]
        retriever_tool = get_retriever_tool(state.get("resources", []))
        if retriever_tool:
            tools.insert(0, retriever_tool)

        # 创建研究Agent
        research_agent = ResearcherAgent(
            config=config, agent_type="researcher", default_tools=tools
        )

        # 执行研究
        try:
            result = await research_agent.execute_agent_step(state)
        except Exception as e:
            logger.error(f"研究Agent执行失败: {e}")
            return Command(
                update={
                    "messages": [
                        AIMessage(content=f"研究任务失败: {str(e)}", name="researcher")
                    ],
                    "current_node": "central_agent",
                    "memory_stack": self.central_agent.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.datetime.now().isoformat(),
            action="delegate",
            agent_type="researcher",
            content=f"研究任务: {task_description}",
            result={"observations": state.get("observations", [])},
        )
        self.central_agent.memory_stack.push(memory_entry)

        return Command(
            update={
                "messages": [
                    AIMessage(content="研究任务完成，返回中枢Agent", name="researcher")
                ],
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    async def execute_coder(self, state: State, config: RunnableConfig) -> Command:
        """执行编码Agent"""
        logger.info("编码Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "未知编码任务")

        # 创建编码Agent
        code_agent = CoderAgent(
            config=config, agent_type="coder", default_tools=[python_repl_tool]
        )

        # 执行编码
        try:
            result = await code_agent.execute_agent_step(state)
        except Exception as e:
            logger.error(f"编码Agent执行失败: {e}")
            return Command(
                update={
                    "messages": [
                        AIMessage(content=f"编码任务失败: {str(e)}", name="coder")
                    ],
                    "current_node": "central_agent",
                    "memory_stack": self.central_agent.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.datetime.now().isoformat(),
            action="delegate",
            agent_type="coder",
            content=f"编码任务: {task_description}",
            result={"observations": state.get("observations", [])},
        )
        self.central_agent.memory_stack.push(memory_entry)

        return Command(
            update={
                "messages": [
                    AIMessage(content="编码任务完成，返回中枢Agent", name="coder")
                ],
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    def execute_reporter(self, state: State, config: RunnableConfig) -> Command:
        """执行报告Agent"""
        logger.info("报告Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "生成最终报告")

        # 收集所有信息
        context = {
            "user_query": state.get("user_query", ""),
            "memory_history": self.central_agent.memory_stack.get_all(),
            "task_description": task_description,
        }

        # 生成报告（使用子Agent独立的prompt）
        try:
            messages = apply_prompt_template(
                "reporter", context, Configuration.from_runnable_config(config)
            )
            llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "default"))
            response = llm.invoke(messages)
            final_report = response.content
        except Exception as e:
            logger.error(f"报告Agent执行失败: {e}")
            final_report = f"报告生成失败: {str(e)}"

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.datetime.now().isoformat(),
            action="delegate",
            agent_type="reporter",
            content=f"报告任务: {task_description}",
            result={"final_report": final_report},
        )
        self.central_agent.memory_stack.push(memory_entry)

        return Command(
            update={
                "messages": [
                    AIMessage(content="报告生成完成，返回中枢Agent", name="reporter")
                ],
                "final_report": final_report,
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )


# 中枢Agent Action决策结果
@dataclass
class CentralDecision:
    action: CentralAgentAction
    reasoning: str
    params: Dict[str, Any] = field(default_factory=dict)
    instruction: Optional[str] = None  # 新增：动作对应的指令


class CentralAgent:
    """中枢Agent - 负责动态编排和决策"""

    def __init__(self):
        self.memory_stack = MemoryStack()
        self.sub_agent_manager = SubAgentManager(self)

        # 注册动作处理器
        self.action_handlers = {
            CentralAgentAction.THINK: self._handle_think,
            CentralAgentAction.REFLECT: self._handle_reflect,
            CentralAgentAction.SUMMARIZE: self._handle_summarize,
            CentralAgentAction.DELEGATE: self._handle_delegate,
            CentralAgentAction.FINISH: self._handle_finish,
        }

        # 动作类型对应的指令模板（统一使用central_agent prompt）
        self.action_instructions = {
            CentralAgentAction.THINK: "分析当前状态并思考下一步行动",
            CentralAgentAction.REFLECT: "反思之前的动作和结果",
            CentralAgentAction.SUMMARIZE: "总结当前已获得的信息",
            CentralAgentAction.DELEGATE: "决定委派哪个子Agent执行任务",
            CentralAgentAction.FINISH: "判断是否可以完成任务并生成最终报告",
        }

    def make_decision(self, state: State, config: RunnableConfig) -> CentralDecision:
        """中枢Agent决策逻辑"""
        logger.info("中枢Agent正在进行决策...")

        # 构建决策上下文
        context = self._build_decision_context(state)

        # 构建决策提示（使用统一的central_agent prompt）
        action_options = list(CentralAgentAction)
        messages = self._build_decision_prompt(context, config, action_options)

        # 获取LLM决策
        try:
            llm = get_llm_by_type(AGENT_LLM_MAP.get("center_agent", "default"))
            response = llm.invoke(messages)

            # 解析决策结果
            decision_data = json.loads(repair_json_output(response.content))

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
            logger.error(f"决策解析失败: {e}")
            # 默认决策：思考
            return CentralDecision(
                action=CentralAgentAction.THINK,
                reasoning="决策解析失败，默认选择思考动作",
                params={},
                instruction=self.action_instructions[CentralAgentAction.THINK],
            )

    def _build_decision_context(self, state: State) -> Dict[str, Any]:
        """构建决策上下文（包含完整记忆栈）"""
        return {
            "user_query": state.get("user_query", ""),
            "current_node": state.get("current_node", "central_agent"),
            "memory_summary": self.memory_stack.get_summary(include_full_history=True),
            "available_actions": [action.value for action in CentralAgentAction],
            "available_sub_agents": [agent.value for agent in SubAgentType],
            "task_completed": state.get("task_completed", False),
            "recent_observations": state.get("observations", [])[-3:],
            "messages_history": state.get("messages", [])[-3:],
        }

    def _build_decision_prompt(
        self,
        context: Dict[str, Any],
        config: RunnableConfig,
        action_options: List[CentralAgentAction],
    ) -> List[Union[AIMessage, HumanMessage]]:
        """构建统一的中枢Agent决策提示词"""
        # 修正模板名称：从 "center_agent" 改为 "central_agent"
        prompt_template = get_prompt_template("center_agent")

        print(context)

        # 格式化prompt（移除json_example变量）
        context_with_actions = {
            **context,
            "available_actions": ", ".join([a.value for a in action_options]),
        }
        formatted_prompt = prompt_template.format(**context_with_actions)
        return [HumanMessage(content=formatted_prompt)]

    def execute_action(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """执行决策动作（使用统一的prompt逻辑）"""
        handler = self.action_handlers.get(decision.action)
        if not handler:
            logger.error(f"未知动作: {decision.action}")
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
        """处理思考动作（使用统一prompt）"""
        logger.info("中枢Agent正在思考...")

        context = {
            "memory_history": self.memory_stack.get_all(),
            "user_query": state.get("user_query", ""),
            "current_progress": state.get("observations", []),
            "decision_reasoning": decision.reasoning,
            "instruction": decision.instruction,  # 传递统一指令
        }

        # 使用统一的central_agent prompt（通过instruction参数区分动作）
        messages = apply_prompt_template(
            "center_agent", context, Configuration.from_runnable_config(config)
        )

        llm = get_llm_by_type(AGENT_LLM_MAP.get("center_agent", "default"))
        response = llm.invoke(messages)

        # 记录到记忆栈
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
        """处理反思动作（使用统一prompt）"""
        logger.info("中枢Agent正在反思...")

        # 获取之前的动作信息用于反思
        previous_actions = self.memory_stack.get_recent(1)

        context = {
            "recent_actions": [entry.to_dict() for entry in previous_actions],
            "current_progress": state.get("observations", []),
            "original_query": state.get("user_query", ""),
            "reflection_target": decision.reasoning,
            "instruction": decision.instruction,  # 传递统一指令
        }

        # 使用统一的central_agent prompt
        messages = apply_prompt_template(
            "center_agent", context, Configuration.from_runnable_config(config)
        )

        llm = get_llm_by_type(AGENT_LLM_MAP.get("center_agent", "default"))
        response = llm.invoke(messages)

        # 先弹出再推入，更新记忆栈
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
        """处理总结动作（使用统一prompt）"""
        logger.info("中枢Agent正在总结...")

        context = {
            "memory_history": self.memory_stack.get_all(),
            "summarization_focus": decision.reasoning,
            "instruction": decision.instruction,  # 传递统一指令
        }

        # 使用统一的central_agent prompt
        messages = apply_prompt_template(
            "center_agent", context, Configuration.from_runnable_config(config)
        )

        llm = get_llm_by_type(AGENT_LLM_MAP.get("center_agent", "default"))
        response = llm.invoke(messages)

        # 先弹出再推入，更新记忆栈
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
        """处理委派动作"""
        agent_type = decision.params.get("agent_type")
        task_description = decision.params.get("task_description", "未指定任务")

        if not agent_type or agent_type not in [agent.value for agent in SubAgentType]:
            error_msg = f"无效的子Agent类型: {agent_type}，可用类型: {[agent.value for agent in SubAgentType]}"
            logger.error(error_msg)
            return Command(
                update={
                    "messages": [AIMessage(content=error_msg, name="central_error")],
                    "current_node": "central_agent",
                },
                goto="central_agent",
            )

        logger.info(f"中枢Agent委派 {agent_type} 执行任务: {task_description}")

        # 记录到记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.datetime.now().isoformat(),
            action="delegate",
            agent_type=agent_type,
            content=f"委派任务: {task_description}",
        )
        self.memory_stack.push(memory_entry)

        # 准备子Agent执行上下文
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
        """处理完成动作"""
        logger.info("中枢Agent完成任务...")

        final_report = state.get("final_report", "任务已完成，但未生成详细报告")

        execution_summary = {
            "user_query": state.get("user_query", "未知查询"),
            "execution_history": self.memory_stack.get_all(),
            "final_report": final_report,
            "completion_time": datetime.datetime.now().isoformat(),
        }

        # 保存到文件
        os.makedirs("./reports", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./reports/execution_report_{timestamp}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(execution_summary, f, ensure_ascii=False, indent=4)
            report_msg = f"任务完成，报告已保存: {filename}"
        except Exception as e:
            logger.error(f"报告保存失败: {e}")
            report_msg = f"任务完成，但报告保存失败: {str(e)}"
            execution_summary["error"] = str(e)

        logger.info(report_msg)

        # 记录到记忆栈
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
                "task_completed": True,
                "execution_summary": execution_summary,
                "current_node": "central_agent",
            },
            goto="__end__",
        )


global_central_agent = CentralAgent()
sub_agent_manager = SubAgentManager(global_central_agent)


# 节点处理函数
async def central_agent_node(state: State, config: RunnableConfig) -> Command:
    """中枢Agent节点 - 动态决策和编排"""
    logger.info("中枢Agent节点激活")

    # 进行决策
    decision = global_central_agent.make_decision(state, config)

    # 执行决策动作
    return global_central_agent.execute_action(decision, state, config)


async def researcher_node(state: State, config: RunnableConfig) -> Command:
    """研究Agent节点"""
    return await sub_agent_manager.execute_researcher(state, config)


async def coder_node(state: State, config: RunnableConfig) -> Command:
    """编码Agent节点"""
    return await sub_agent_manager.execute_coder(state, config)


def reporter_node(state: State, config: RunnableConfig) -> Command:
    """报告Agent节点"""
    return sub_agent_manager.execute_reporter(state, config)


def build_multi_agent_graph():
    """构建多Agent系统状态图"""
    # 修复导入问题
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(State)

    # 添加节点
    builder.add_node("central_agent", central_agent_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reporter", reporter_node)

    # 设置起始节点
    builder.add_edge(START, "central_agent")

    # 中枢Agent可以跳转到任何子Agent
    builder.add_edge("central_agent", "researcher")
    builder.add_edge("central_agent", "coder")
    builder.add_edge("central_agent", "reporter")

    # 所有子Agent完成后必须返回中枢Agent
    builder.add_edge("researcher", "central_agent")
    builder.add_edge("coder", "central_agent")
    builder.add_edge("reporter", "central_agent")

    # 中枢Agent可以结束流程
    builder.add_edge("central_agent", END)

    return builder.compile()


# 生成最终的多Agent系统图
graph = build_multi_agent_graph()

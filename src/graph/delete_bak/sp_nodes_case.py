# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import os
import asyncio
from typing import Annotated, Literal, Dict, Any, Optional, List, Callable, Union
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from copy import deepcopy

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END

from src.agents.CommonReactAgent import CommonReactAgent
from src.agents.CoderAgent import CoderAgent
from StackPlanner.src.agents.ResearcherAgent_SP import ResearcherAgentSP
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
    search_docs_tool,
)
from src.utils.logger import logger
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from ..types import State
from ...config import SELECTED_SEARCH_ENGINE, SearchEngine


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
    agent_type: Optional[str] = None
    content: str = ""
    result: Optional[Dict[str, Any]] = None
    reflection: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "agent_type": self.agent_type,  # central or sub-agent
            "content": self.content,
        }


# 记忆栈管理器
class MemoryStack:
    def __init__(self, max_size: int = 50):
        self.stack: List[MemoryStackEntry] = []
        self.max_size = max_size

    def push(self, entry: MemoryStackEntry):
        """添加新条目到栈顶"""
        self.stack.append(entry)
        # 保持栈大小限制
        if len(self.stack) > self.max_size:
            self.stack.pop(0)

    def pop(self) -> Optional[MemoryStackEntry]:
        """弹出栈顶条目"""
        if self.stack:
            return self.stack.pop()
        return None

    def peek(self) -> Optional[MemoryStackEntry]:
        """查看栈顶条目但不弹出"""
        if self.stack:
            return self.stack[-1]
        return None

    def push_with_pop(self, entry: MemoryStackEntry):
        """先弹出栈顶，再推入新条目 - 用于反思和总结动作"""
        if self.stack:
            popped = self.stack.pop()
            # 保留一些信息到新条目中
            if entry.result is None:
                entry.result = {}

        self.stack.append(entry)

        # 保持栈大小限制
        if len(self.stack) > self.max_size:
            self.stack.pop(0)

    def get_recent(self, count: int = 5) -> List[MemoryStackEntry]:
        """获取最近的N个条目"""
        return self.stack[-count:] if len(self.stack) >= count else self.stack

    def get_all(self) -> List[MemoryStackEntry]:
        """获取所有条目"""
        return self.stack.copy()

    def get_summary(self) -> str:  # FIXME HERE
        """获取记忆栈摘要"""
        if not self.stack:
            return "记忆栈为空"

        recent_entries = self.get_recent(3)
        summary_parts = []
        for entry in recent_entries:
            summary_parts.append(
                f"[{entry.timestamp}] {entry.action}: {entry.content[:100]}..."
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


# 中枢Agent决策结果
@dataclass
class CentralDecision:
    action: CentralAgentAction
    reasoning: str
    params: Dict[str, Any] = field(default_factory=dict)


class CentralAgent:
    """中枢Agent - 负责动态编排和决策"""

    def __init__(self):
        self.memory_stack = MemoryStack()

        # 注册动作处理器
        self.action_handlers = {
            CentralAgentAction.THINK: self._handle_think,
            CentralAgentAction.REFLECT: self._handle_reflect,
            CentralAgentAction.SUMMARIZE: self._handle_summarize,
            CentralAgentAction.DELEGATE: self._handle_delegate,
            CentralAgentAction.FINISH: self._handle_finish,
        }

    def make_decision(self, state: State, config: RunnableConfig) -> CentralDecision:
        """中枢Agent决策逻辑"""
        logger.info("中枢Agent正在进行决策...")

        # 构建决策上下文
        context = self._build_decision_context(state)

        # 构建决策提示
        messages = self._build_decision_prompt(context, config)

        # 获取LLM决策
        llm = get_llm_by_type(
            AGENT_LLM_MAP.get("center_agent", "default")
        )  # FIXME check here
        response = llm.invoke(messages)

        try:
            # 解析决策结果
            decision_data = json.loads(repair_json_output(response.content))

            action = CentralAgentAction(decision_data["action"])
            reasoning = decision_data.get("reasoning", "")
            params = decision_data.get("params", {})

            return CentralDecision(action=action, reasoning=reasoning, params=params)

        except Exception as e:
            logger.error(f"决策解析失败: {e}")
            # 默认决策：思考
            return CentralDecision(
                action=CentralAgentAction.THINK,
                reasoning="决策解析失败，默认选择思考动作",
                params={},
            )

    def _build_decision_context(self, state: State) -> Dict[str, Any]:
        """构建决策上下文"""
        return {
            "user_query": state.get("user_query", ""),
            "current_node": state.get("current_node", ""),
            "messages": state.get("messages", []),
            "observations": state.get("observations", []),
            "memory_summary": self.memory_stack.get_summary(),
            "available_actions": [action.value for action in CentralAgentAction],
            "available_sub_agents": [agent.value for agent in SubAgentType],
            "task_completed": state.get("task_completed", False),
            "final_report": state.get("final_report"),
        }

    def _build_decision_prompt(
        self, context: Dict[str, Any], config: RunnableConfig
    ) -> List[BaseMessage]:
        """构建决策提示"""
        system_prompt = """你是一个中枢Agent，负责动态编排多Agent系统的执行流程。

你的职责：
1. 分析当前状态和上下文
2. 从以下5个动作中选择最合适的一个：
   - think: 深入思考和推理当前情况
   - reflect: 反思之前的执行结果和决策
   - summarize: 总结当前已获得的信息
   - delegate: 委派子Agent执行具体任务
   - finish: 完成整个任务流程

你的决策原则：
- 优先思考(think)来理解任务需求
- 需要信息时委派research agent
- 需要编码时委派coder agent  
- 需要生成报告时委派reporter agent
- 执行复杂任务前先反思(reflect)之前的进展
- 信息太多时进行总结(summarize)
- 任务完成后选择结束(finish)

请以JSON格式返回决策：
{
    "action": "选择的动作",
    "reasoning": "选择理由",
    "params": {
        "agent_type": "如果是delegate动作，指定子agent类型",
        "task_description": "具体任务描述",
        "other_params": "其他参数"
    }
}"""

        messages = [
            HumanMessage(content=system_prompt),
            HumanMessage(
                content=f"当前上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
            ),
        ]

        return messages

    def execute_action(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """执行决策动作"""
        logger.info(f"执行动作: {decision.action.value}, 理由: {decision.reasoning}")

        # 创建记忆条目
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action=decision.action.value,
            content=decision.reasoning,
        )

        # 根据动作类型决定记忆栈操作方式
        if decision.action in [
            CentralAgentAction.REFLECT,
            CentralAgentAction.SUMMARIZE,
        ]:
            # 反思和总结：先pop再push（替换上一步）
            self.memory_stack.push_with_pop(memory_entry)
            logger.info(f"{decision.action.value}动作：替换了上一步的记忆条目")
        else:
            # 其他动作：直接push
            self.memory_stack.push(memory_entry)
            logger.info(f"{decision.action.value}动作：添加了新的记忆条目")

        # 更新状态中的记忆栈
        state["memory_stack"] = self.memory_stack.to_dict()

        # 执行对应的动作处理器
        return self.action_handlers[decision.action](decision, state, config)

    def _handle_think(
        self, decision: CentralDecision, state: State, config: RunnableConfig
    ) -> Command:
        """处理思考动作"""
        logger.info("中枢Agent正在思考...")

        context = {
            "user_query": state.get("user_query", ""),
            "current_situation": decision.reasoning,
            "memory_summary": self.memory_stack.get_summary(),
            "available_info": state.get("observations", []),
        }

        messages = apply_prompt_template(
            "central_think", context, Configuration.from_runnable_config(config)
        )

        llm = get_llm_by_type(AGENT_LLM_MAP.get("center_agent", "default"))
        response = llm.invoke(messages)

        # 更新当前记忆栈顶部条目的结果
        if not self.memory_stack.is_empty():
            current_entry = self.memory_stack.peek()
            if current_entry:
                current_entry.result = {"thought_result": response.content}

        logger.info(f"central_think: {response.content}")
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
        """处理反思动作"""
        logger.info("中枢Agent正在反思...")

        # 获取之前的动作信息用于反思
        previous_actions = self.memory_stack.get_recent(3)

        context = {
            "recent_actions": [entry.to_dict() for entry in previous_actions],
            "current_progress": state.get("observations", []),
            "original_query": state.get("user_query", ""),
            "reflection_target": decision.reasoning,
        }

        messages = apply_prompt_template(
            "central_reflect", context, Configuration.from_runnable_config(config)
        )

        llm = get_llm_by_type(AGENT_LLM_MAP.get("center_agent", "default"))
        response = llm.invoke(messages)

        # 更新当前记忆栈顶部条目的结果和反思
        if not self.memory_stack.is_empty():
            current_entry = self.memory_stack.peek()
            if current_entry:
                current_entry.result = {"reflection_result": response.content}
                current_entry.reflection = response.content

        logger.info(f"central_reflect: {response.content}")
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
        """处理总结动作"""
        logger.info("中枢Agent正在总结...")

        observations = state.get("observations", [])
        messages = state.get("messages", [])

        context = {
            "observations": observations,
            "messages": [
                msg.content if hasattr(msg, "content") else str(msg) for msg in messages
            ],
            "memory_history": self.memory_stack.get_all(),
            "summarization_focus": decision.reasoning,
        }

        messages = apply_prompt_template(
            "central_summarize", context, Configuration.from_runnable_config(config)
        )

        llm = get_llm_by_type(AGENT_LLM_MAP.get("center_agent", "default"))
        response = llm.invoke(messages)

        # 更新当前记忆栈顶部条目的结果
        if not self.memory_stack.is_empty():
            current_entry = self.memory_stack.peek()
            if current_entry:
                current_entry.result = {"summary_result": response.content}

        logger.info(f"central_summarize: {response.content}")
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
        task_description = decision.params.get("task_description", "")

        logger.info(f"中枢Agent委派 {agent_type} 执行任务: {task_description}")

        if not agent_type or agent_type not in [agent.value for agent in SubAgentType]:
            logger.error(f"无效的子Agent类型: {agent_type}")
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"无效的子Agent类型: {agent_type}",
                            name="central_error",
                        )
                    ],
                    "current_node": "central_agent",
                },
                goto="central_agent",
            )

        # 更新当前记忆栈顶部条目
        if not self.memory_stack.is_empty():
            current_entry = self.memory_stack.peek()
            if current_entry:
                current_entry.agent_type = agent_type
                current_entry.result = {"delegated_task": task_description}

        # 准备子Agent执行上下文
        delegation_context = {
            "task_description": task_description,
            "agent_type": agent_type,
            "memory_context": self.memory_stack.get_summary(),
            "original_query": state.get("user_query", ""),
        }

        logger.info(f"central_delegate: 委派 {agent_type} 执行任务: {task_description}")
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
            "user_query": state.get("user_query"),
            "execution_history": self.memory_stack.to_dict(),
            "final_report": final_report,
            "completion_time": datetime.now().isoformat(),
        }

        # 保存到文件
        os.makedirs("./reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./reports/execution_report_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(execution_summary, f, ensure_ascii=False, indent=4)

        logger.info(f"执行报告已保存: {filename}")

        # 更新当前记忆栈顶部条目
        if not self.memory_stack.is_empty():
            current_entry = self.memory_stack.peek()
            if current_entry:
                current_entry.result = {
                    "execution_summary": execution_summary,
                    "report_file": filename,
                }

        logger.info(f"central_finish: 任务完成，报告已保存: {filename}")
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"任务完成，报告已保存: {filename}",
                        name="central_finish",
                    )
                ],
                "task_completed": True,
                "execution_summary": execution_summary,
                "current_node": "central_agent",
            },
            goto="__end__",
        )


class SubAgentManager:
    """子Agent管理器"""

    def __init__(self, central_agent: CentralAgent):
        self.central_agent = central_agent

    async def execute_researcher(self, state: State, config: RunnableConfig) -> Command:
        """执行研究Agent"""
        logger.info("研究Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "")

        # 配置研究工具
        tools = [get_web_search_tool(10), crawl_tool, search_docs_tool]
        retriever_tool = get_retriever_tool(state.get("resources", []))
        if retriever_tool:
            tools.insert(0, retriever_tool)

        # 创建研究Agent
        research_agent = ResearcherAgentSP(
            config=config, agent_type="researcher", default_tools=tools
        )

        # 执行研究
        result = await research_agent.execute_agent_step(state)

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="researcher_completed",
            agent_type="researcher",
            content=f"研究任务完成: {task_description}",
            result={"observations": state.get("observations", [])},
        )
        self.central_agent.memory_stack.push(memory_entry)

        # 确保返回中枢Agent
        logger.info("Researcher Agent完成，返回中枢Agent")
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
        task_description = delegation_context.get("task_description", "")

        # 创建编码Agent
        code_agent = CoderAgent(
            config=config, agent_type="coder", default_tools=[python_repl_tool]
        )

        # 执行编码
        result = await code_agent.execute_agent_step(state)

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="coder_completed",
            agent_type="coder",
            content=f"编码任务完成: {task_description}",
            result={"observations": state.get("observations", [])},
        )
        self.central_agent.memory_stack.push(memory_entry)

        # 确保返回中枢Agent
        logger.info("Coder Agent完成，返回中枢Agent")
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
            "observations": state.get("observations", []),
            "memory_history": self.central_agent.memory_stack.to_dict(),
            "messages": state.get("messages", []),
            "task_description": task_description,
        }

        # 生成报告
        messages = apply_prompt_template(
            "reporter", context, Configuration.from_runnable_config(config)
        )

        llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "default"))
        response = llm.invoke(messages)
        final_report = response.content

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="reporter_completed",
            agent_type="reporter",
            content=f"报告生成完成: {task_description}",
            result={"final_report": final_report},
        )
        self.central_agent.memory_stack.push(memory_entry)

        # 确保返回中枢Agent
        logger.info("Reporter Agent完成，返回中枢Agent")
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


# 全局中枢Agent实例
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

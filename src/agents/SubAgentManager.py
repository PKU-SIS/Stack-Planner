from src.llms.llm import get_llm_by_type
from ..graph.types import State
from langchain_core.runnables import RunnableConfig
import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.agents.CoderAgent import CoderAgent
from src.agents.ResearcherAgent import ResearcherAgent
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
    search_docs_tool,
)
from src.utils.logger import logger
from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.prompts.template import apply_prompt_template
from src.memory import MemoryStack, MemoryStackEntry
from src.agents.CentralAgent import CentralAgent

from ..graph.types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine


# -------------------------
# 子Agent管理模块
# TODO: check sub-agent bugs
# -------------------------
class SubAgentManager:
    """子Agent管理器，负责创建和执行各类专项子Agent"""

    def __init__(self, central_agent: "CentralAgent"):
        self.central_agent = central_agent

    async def execute_researcher(self, state: State, config: RunnableConfig) -> Command:
        """
        执行研究Agent，负责信息检索与分析

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("研究Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "未知研究任务")

        # 配置研究工具链
        tools = [get_web_search_tool(10), crawl_tool, search_docs_tool]
        retriever_tool = get_retriever_tool(state.get("resources", []))
        if retriever_tool:
            tools.insert(0, retriever_tool)

        # 实例化研究Agent
        research_agent = ResearcherAgent(
            config=config, agent_type="researcher", default_tools=tools
        )

        # 执行研究任务并处理异常
        try:
            await research_agent.execute_agent_step(state)
        except Exception as e:
            logger.error(f"研究Agent执行失败: {str(e)}")
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
        """
        执行编码Agent，负责代码生成与执行

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("编码Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "未知编码任务")

        # 实例化编码Agent
        code_agent = CoderAgent(
            config=config, agent_type="coder", default_tools=[python_repl_tool]
        )

        # 执行编码任务并处理异常
        try:
            await code_agent.execute_agent_step(state)
        except Exception as e:
            logger.error(f"编码Agent执行失败: {str(e)}")
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
        """
        执行报告Agent，负责结果整理与报告生成

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("报告Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "生成最终报告")

        # 收集报告生成所需上下文
        context = {
            "user_query": state.get("user_query", ""),
            "memory_history": self.central_agent.memory_stack.get_all(),
            "task_description": task_description,
        }

        # 生成报告并处理异常
        final_report = "报告生成失败: 未知错误"
        try:
            messages = apply_prompt_template("reporter", context, state)
            llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "default"))
            response = llm.invoke(messages)
            final_report = response.content
        except Exception as e:
            logger.error(f"报告Agent执行失败: {str(e)}")
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

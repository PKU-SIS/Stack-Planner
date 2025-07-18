from src.llms.llm import get_llm_by_type
from ..graph.types import State
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.agents.CoderAgent import CoderAgent
from src.agents.ResearcherAgent_SP import ResearcherAgentSP
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
from src.utils.statistics import global_statistics, timed_step


# -------------------------
# 子Agent管理模块
# TODO: check sub-agent bugs
# TODO: 搜索太多时会超过输入限制或者缓冲区溢出，需要限制搜索到的内容长度或者做一个简单的摘要
# TODO: 需要处理搜索敏感词（以“985大学最多的五个城市”为例，AI就无法处理信息，返回Error）
# -------------------------
class SubAgentManager:
    """子Agent管理器，负责创建和执行各类专项子Agent"""

    def __init__(self, central_agent: "CentralAgent"):
        self.central_agent = central_agent

    @timed_step("execute_researcher")
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
        research_agent = ResearcherAgentSP(
            config=config, agent_type="researcher", default_tools=tools
        )

        # 执行研究任务并处理异常
        try:
            result_command = await research_agent.execute_agent_step(state)

            # 从结果中提取数据用于记忆栈
            result_observations = []
            result_data_collections = []

            if result_command and result_command.update:
                result_observations = result_command.update.get("observations", [])
                result_data_collections = result_command.update.get(
                    "data_collections", []
                )

        except Exception as e:
            logger.error(f"Researcher Agent执行失败: {str(e)}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"研究任务失败: {str(e)}", name="researcher"
                        )
                    ],
                    "current_node": "central_agent",
                    "memory_stack": self.central_agent.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="researcher",
            content=f"研究任务: {task_description}",
            result={
                "observations": result_observations,
                # "data_collections": result_data_collections,
            },
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("研究任务完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content="研究任务完成，返回中枢Agent", name="researcher"
                    )
                ],
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    @timed_step("execute_xxqg_researcher")
    async def execute_xxqg_researcher(
        self, state: State, config: RunnableConfig
    ) -> Command:
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
        tools = [search_docs_tool]

        # 实例化研究Agent
        research_agent = ResearcherAgentSP(
            config=config, agent_type="researcher_xxqg", default_tools=tools
        )

        # 执行研究任务并处理异常
        try:
            result_command = await research_agent.execute_agent_step(state)

            # 从结果中提取数据用于记忆栈
            result_observations = []
            result_data_collections = []

            if result_command and result_command.update:
                result_observations = result_command.update.get("observations", [])
                result_data_collections = result_command.update.get(
                    "data_collections", []
                )

        except Exception as e:
            logger.error(f"研究Agent执行失败: {str(e)}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"研究任务失败: {str(e)}", name="researcher"
                        )
                    ],
                    "current_node": "central_agent",
                    "memory_stack": self.central_agent.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="researcher",
            content=f"研究任务: {task_description}",
            result={
                "observations": result_observations,
                # "data_collections": result_data_collections,
            },
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("研究任务完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content="研究任务完成，返回中枢Agent", name="researcher"
                    )
                ],
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    @timed_step("execute_coder")
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
            result_command = await code_agent.execute_agent_step(state)
            # 从结果中提取数据用于记忆栈
            result_observations = []
            if result_command and result_command.update:
                result_observations = result_command.update.get("observations", [])
        except Exception as e:
            logger.error(f"编码Agent执行失败: {str(e)}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=f"编码任务失败: {str(e)}", name="coder")
                    ],
                    "current_node": "central_agent",
                    "memory_stack": self.central_agent.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="coder",
            content=f"编码任务: {task_description}",
            result={"observations": result_observations},
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("编码任务完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="编码任务完成，返回中枢Agent", name="coder")
                ],
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    @timed_step("execute_reporter")
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
            messages = apply_prompt_template(
                "reporter", state, extra_context=context
            )  # 修复：参数顺序
            llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "default"))
            response = llm.invoke(messages)
            final_report = response.content
        except Exception as e:
            logger.error(f"报告Agent执行失败: {str(e)}")
            final_report = f"报告生成失败: {str(e)}"

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="reporter",
            content=f"报告任务: {task_description}",
            result={"final_report": final_report},
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("报告生成完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="报告生成完成，返回中枢Agent", name="reporter")
                ],
                "final_report": final_report,
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    @timed_step("execute_xxqg_reporter")
    def execute_xxqg_reporter(self, state: State, config: RunnableConfig) -> Command:
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
            messages = apply_prompt_template(
                "reporter_xxqg", state, extra_context=context
            )  # 修复：参数顺序
            llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "default"))
            response = llm.invoke(messages)
            final_report = response.content
        except Exception as e:
            logger.error(f"报告Agent执行失败: {str(e)}")
            final_report = f"报告生成失败: {str(e)}"

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="reporter",
            content=f"报告任务: {task_description}",
            result={"final_report": final_report},
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("报告生成完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="报告生成完成，返回中枢Agent", name="reporter")
                ],
                "final_report": final_report,
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    @timed_step("execute_sp_planner")
    def execute_sp_planner(self, state: State, config: RunnableConfig) -> Command:
        """
        执行任务拆解Agent，负责将复杂任务拆解为可管理的子任务

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("任务拆解Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get(
            "task_description",
            state.get("user_query", "") + "\n将用户的任务拆解成2-5个子任务",
        )

        # 收集任务拆解所需上下文
        context = {
            "user_query": state.get("user_query", ""),
            "memory_history": [],  # self.central_agent.memory_stack.get_all(),
            "task_description": task_description,
        }

        # 生成任务拆解并处理异常
        replan_result = "任务拆解失败: 未知错误"
        try:
            messages = apply_prompt_template(
                "replanner", state, extra_context=context
            )  # 修复：参数顺序
            llm = get_llm_by_type(AGENT_LLM_MAP.get("replanner", "default"))
            response = llm.invoke(messages)
            replan_result = response.content
            replan_result = (
                replan_result.replace("```json", "").replace("```", "").strip()
            )

            logger.debug(f"任务拆解结果: {replan_result}")

            # 解析LLM返回的任务拆解结果
            import json

            try:
                response_json = json.loads(replan_result)
                if isinstance(response_json, list):
                    response_json = {"DAG": response_json}
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                response_json = {"DAG": [(input, input)]}
            if isinstance(response_json["DAG"], list):
                new_dag = []
                for item in response_json["DAG"]:
                    if isinstance(item, dict):
                        pairs = list(item.items())
                        new_dag.append(
                            (pairs[0][1], pairs[1][1])
                            if len(pairs) > 1
                            else (pairs[0][1], pairs[0][1])
                        )
                    elif isinstance(item, list) and len(item) > 1:
                        new_dag.append((item[0], item[1]))
                    else:
                        new_dag.append((item, item))
                response_json["DAG"] = new_dag

            from src.utils.graph_utils import Graph

            graph = Graph()
            graph.load_dag_from_json(response_json)
            sorted_nodes = graph.topological_sort()
            # Generate a unique ID for each input using a hash
            input_id = hash(input)
            # replan_result = {"id":input_id,"plans":[{node_id: graph.nodes[node_id].question} for node_id in sorted_nodes],"status":["uncomplete" for node_id in sorted_nodes]}
            replan_result = {
                "id": input_id,
                "plans": [
                    {node_id: graph.nodes[node_id].question} for node_id in sorted_nodes
                ],
            }
        except Exception as e:
            logger.error(f"任务拆解Agent执行失败: {str(e)}")
            replan_result = f"任务拆解失败: {str(e)}"

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="replanner",
            content=f"任务拆解: {task_description}",
            result={"replan_result": replan_result},
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("任务拆解完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="任务拆解完成，返回中枢Agent", name="planner")
                ],
                "replan_result": replan_result,
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

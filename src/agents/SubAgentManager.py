from src.llms.llm import get_llm_by_type
from ..graph.types import State
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from src.agents.CoderAgent import CoderAgent
from src.agents.ResearcherAgent_SP import ResearcherAgentSP
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
    search_docs_tool,
)
from src.utils.json_utils import repair_json_output
from src.utils.logger import logger
from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.prompts.template import apply_prompt_template
from src.memory import MemoryStack, MemoryStackEntry
from src.agents.CentralAgent import CentralAgent
from src.tools.get_docs_info import search_docs
from src.tools.bocha_search.web_search_en import web_search
from src.factstruct import (
    run_factstruct_stage1,
    outline_node_to_markdown,
    memory_to_dict,
)

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

            logger.info(f"data_collections_in subagent:{result_data_collections}")

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
                "data_collections": result_data_collections,
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
                "data_collections": result_data_collections,
            },
            goto="central_agent",
        )

    @timed_step("execute_web_researcher")
    async def execute_web_researcher(
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
        logger.info("Web Agent开始执行...")
        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "未知研究任务")

        # 配置研究工具链
        # tools = [search_docs_tool]
        tools = [get_web_search_tool(10)]
        
        # 实例化研究Agent
        research_agent = ResearcherAgentSP(
            config=config, agent_type="researcher_web", default_tools=tools
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
                "data_collections": result_data_collections,
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

        data_collections = state.get("data_collections", [])
        logger.info(
            f"report agent: data_collections:{data_collections}"
        )  # NOTE: data_collections可以在这里取

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
            data_collections = state.get("data_collections", [])
            data_joined = "\n\n".join(data_collections)
            user_query_str = state.get("user_query", "")
            user_dst_str = state.get("user_dst", "")
            report_outline_str = state.get("report_outline", "用户未提供大纲")

            reporter_content = (
                f"##User Query\n\n{user_query_str}\n\n"
                f"##用户约束\n\n{user_dst_str}\n\n"
                f"##报告大纲\n{report_outline_str}\n\n"
                "Below are data collected in previous tasks:\n\n"
                f"{data_joined}"
            )

            messages.append(HumanMessage(content=reporter_content))

            logger.debug(f"Reporter messages: {messages}")
            llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "default"))
            response = llm.invoke(messages)
            final_report = response.content
        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
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

    @timed_step("execute_xxqg_reporter_factstruct")
    def execute_xxqg_reporter_factstruct(
        self, state: State, config: RunnableConfig
    ) -> Command:
        """
        执行报告Agent（使用 FactStruct Stage 2）

        基于 FactStruct Stage 1 生成的大纲和 Memory，为每个叶子节点
        分别生成内容，最终合并为完整报告。

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("报告Agent开始执行（FactStruct Stage 2）...")

        factstruct_outline = state.get("factstruct_outline")
        factstruct_memory = state.get("factstruct_memory")

        if not factstruct_outline or not factstruct_memory:
            logger.warning(
                "FactStruct 数据缺失，回退到传统 Reporter 方法"
            )
            return self.execute_xxqg_reporter(state, config)

        user_query = state.get("user_query", "")

        final_report = "报告生成失败: 未知错误"
        try:
            from src.factstruct import run_factstruct_stage2
            from src.config.agents import AGENT_LLM_MAP

            final_report = run_factstruct_stage2(
                outline_dict=factstruct_outline,
                memory_dict=factstruct_memory,
                user_query=user_query,
                llm_type=AGENT_LLM_MAP.get("reporter_factstruct", "basic"),
                locale=state.get("locale", "zh-CN"),
            )

            logger.info(
                f"FactStruct Stage 2 报告生成完成: {len(final_report)} 个字符"
            )

        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
            logger.error(f"FactStruct Stage 2 报告生成失败: {str(e)}")
            final_report = f"报告生成失败: {str(e)}"

        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="reporter",
            content="报告任务: 使用 FactStruct Stage 2 生成报告",
            result={"final_report": final_report},
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("报告生成完成（FactStruct Stage 2），返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content="报告生成完成（FactStruct Stage 2），返回中枢Agent",
                        name="reporter",
                    )
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

    @timed_step("execute_perception")
    async def execute_perception(self, state: State, config: RunnableConfig) -> Command:
        user_query = state.get("user_query", "")
        # check if the plan is auto accepted
        perception_llm = get_llm_by_type(AGENT_LLM_MAP.get("perception", "default"))
        auto_accepted_plan = state.get("auto_accepted_plan", False)
        skip_perception = state.get("skip_perception", False)
        
        if skip_perception:
            logger.info("跳过感知层，直接进入大纲生成")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="感知层已跳过",
                            name="perception",
                        )
                    ],
                    "user_dst": "",
                    "current_node": "perception",
                    "wait_for_user": False,
                },
                goto="outline",
            )

        if auto_accepted_plan:
            try:
                messages = apply_prompt_template("perception", state) + [
                    HumanMessage(f"##User Query\n\n{user_query}\n\n")
                ]
                response = perception_llm.invoke(messages)
                dst_question = response.content
                dst_question = repair_json_output(dst_question)
                logger.info(f"感知层完成，生成DST问题: {dst_question}")
                state["wait_for_user"] = True
            except Exception as e:
                logger.error(f"感知层执行失败: {str(e)}")

            feedback = interrupt(
                "Please Fill the Question.[DST]" + dst_question + "[/DST]"
            )

            # if the feedback is not accepted, return the planner node
            if feedback and str(feedback).upper().startswith("[FILLED_QUESTION]"):
                messages = apply_prompt_template("perception", state) + [
                    HumanMessage(
                        f"##User Query\n\n{user_query}\n\n##希望用户回答的问题\n\n{dst_question}\n\n##用户回答的结果\n\n{feedback}\n\n"
                    )
                ]
                response = perception_llm.invoke(messages)
                summary = response.content
                logger.info(f"感知层完成，收集用户反馈: {summary}")

                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"感知层完成，收集用户反馈: {summary}",
                                name="perception",
                            )
                        ],
                        "user_dst": summary,
                        "current_node": "perception",
                        "wait_for_user": False,
                    },
                    goto="outline",
                )
            elif feedback and str(feedback).upper().startswith("[SKIP]"):
                logger.info("DST question is skipped by user.")
                messages.append(
                    AIMessage(content=f"##LLM DST Question\n\n{dst_question}\n\n")
                )
                messages.append(
                    HumanMessage(
                        content=f"用户跳过了回答，你可以根据自己的理解进行总结\n\n"
                    )
                )
                response = perception_llm.invoke(messages)
                summary = response.content
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content="DST question is skipped by user.",
                                name="perception",
                            )
                        ],
                        "user_dst": summary,
                        "current_node": "perception",
                        "wait_for_user": False,
                    },
                    goto="outline",
                )
            else:
                raise TypeError(f"Interrupt value of {feedback} is not supported.")

    @timed_step("execute_outline")
    async def execute_outline(self, state: State, config: RunnableConfig) -> Command:
        user_query = state.get("user_query", "")
        # check if the plan is auto accepted
        outline_llm = get_llm_by_type(AGENT_LLM_MAP.get("outline", "default"))
        wait_stage = state.get("wait_stage", "")
        if wait_stage != "outline":
            # bg_investigation = search_docs(user_query, top_k=5)
            bg_investigation = web_search(user_query, top_k=5)
            user_dst = state.get("user_dst", "")
            try:
                messages = [
                    HumanMessage(
                        f"##用户原始问题\n\n{user_query}\n\n##用户补充需求\n\n{user_dst}\n\n##可能用到的相关数据\n\n{bg_investigation}\n\n"
                    )
                ] + apply_prompt_template("outline", state)
                response = outline_llm.invoke(messages)
                outline_response = response.content
                outline_response = repair_json_output(outline_response)
                logger.info(f"大纲生成完成: {outline_response}")

            except Exception as e:
                logger.error(f"大纲生成执行失败: {str(e)}")
                # 返回最简单的默认大纲
                import json

                outline_response = json.dumps(
                    {"title": user_query, "children": []}, ensure_ascii=False
                )


            outline_confirmed = outline_response.strip()
            logger.info(f"大纲自动确认: {outline_confirmed}")

            return Command(
                update={
                    "messages": [
                        HumanMessage(content=f"大纲确认: {outline_confirmed}", name="outline")
                    ],
                    "report_outline": outline_confirmed,
                    "current_node": "outline",
                },
                goto="central_agent",
            )


    @timed_step("execute_outline_factstruct")
    async def execute_outline_factstruct(self, state: State, config: RunnableConfig) -> Command:
        """
        执行大纲生成（使用 FactStruct Stage 1 Batch-MAB 算法）

        使用批量-信息觅食多臂老虎机算法动态生成和优化大纲结构。
        """
        user_query = state.get("user_query", "")
        user_dst = state.get("user_dst", "")
        auto_accepted_plan = state.get("auto_accepted_plan", False)

        if auto_accepted_plan:
            factstruct_outline_dict = None
            factstruct_memory_dict = None

            # 使用 FactStruct Stage 1 生成大纲
            try:
                logger.info("开始使用 FactStruct Stage 1 生成大纲...")

                # 构建完整查询（包含用户补充需求）
                full_query = user_query
                if user_dst:
                    full_query = f"{user_query}\n\n用户补充需求：{user_dst}"

                # 运行 Batch-MAB 算法
                # 注意：这里使用较小的迭代次数和批量大小以加快响应速度
                # 生产环境可以根据需要调整这些参数
                outline_root, memory = run_factstruct_stage1(
                    query=full_query,
                    max_iterations=state.get(
                        "factstruct_max_iterations", 20
                    ),  # 默认 10 次迭代
                    batch_size=state.get("factstruct_batch_size", 5),  # 默认批量大小 3
                )

                # 转换为 Markdown 格式（完整大纲，不限制深度）
                outline_response = outline_node_to_markdown(
                    outline_root, max_depth=None, include_root=True
                )

                # 保存到 state（供 FactStruct Stage 2 使用）
                from src.factstruct import outline_node_to_dict, memory_to_dict

                factstruct_outline_dict = outline_node_to_dict(outline_root)
                factstruct_memory_dict = memory_to_dict(memory)

                logger.info(
                    f"FactStruct Stage 1 大纲生成完成: "
                    f"{len(outline_root.get_all_nodes())} 个节点, "
                    f"{len(memory.documents)} 个文档"
                )

            except Exception as e:
                import traceback

                logger.error(f"FactStruct Stage 1 执行失败: {str(e)}")
                logger.error(f"详细错误:\n{traceback.format_exc()}")

                # Fallback: 使用传统方法生成大纲
                logger.warning("回退到传统大纲生成方法...")
                outline_llm = get_llm_by_type(AGENT_LLM_MAP.get("outline", "default"))
                # bg_investigation = search_docs(user_query, top_k=5)
                bg_investigation = web_search(user_query, top_k=5)
                try:
                    messages = [
                        HumanMessage(
                            f"##用户原始问题\n\n{user_query}\n\n##用户补充需求\n\n{user_dst}\n\n##可能用到的相关数据\n\n{bg_investigation}\n\n"
                        )
                    ] + apply_prompt_template("outline", state)
                    response = outline_llm.invoke(messages)
                    outline_response = response.content
                    outline_response = repair_json_output(outline_response)
                    logger.info(f"传统方法大纲生成完成: {outline_response}")
                except Exception as fallback_error:
                    logger.error(f"传统方法也失败: {str(fallback_error)}")
                    # 返回最简单的默认大纲
                    import json

                    outline_response = json.dumps(
                        {"title": user_query, "children": []}, ensure_ascii=False
                    )


            outline_confirmed = outline_response.strip()
            logger.info(f"大纲自动确认: {outline_confirmed}")

            return Command(
                update={
                    "messages": [
                        HumanMessage(content=f"大纲确认: {outline_confirmed}", name="outline")
                    ],
                    "report_outline": outline_confirmed,
                    "factstruct_outline": factstruct_outline_dict,
                    "factstruct_memory": factstruct_memory_dict,
                    "current_node": "outline",
                },
                goto="central_agent",
            )
            #不要确认了
            # feedback = interrupt(
            #     "Please Confirm or Edit the Outline.[OUTLINE]"
            #     + outline_response
            #     + "[/OUTLINE]"
            # )

            # # if the feedback is not accepted, return the planner node
            # if feedback and str(feedback).upper().startswith("[CONFIRMED_OUTLINE]"):
            #     outline_confirmed = feedback[len("[CONFIRMED_OUTLINE]") :].strip()
            #     logger.info(f"大纲确认: {outline_confirmed}")

            #     return Command(
            #         update={
            #             "messages": [
            #                 HumanMessage(
            #                     content=f"大纲确认: {outline_confirmed}", name="outline"
            #                 )
            #             ],
            #             "report_outline": outline_confirmed,
            #             "current_node": "outline",
            #         },
            #         goto="central_agent",
            #     )
            # elif feedback and str(feedback).upper().startswith("[SKIP]"):
            #     outline_confirmed = feedback[len("[SKIP]") :].strip()
            #     logger.info(f"大纲确认: {outline_confirmed}")

            #     return Command(
            #         update={
            #             "messages": [
            #                 HumanMessage(
            #                     content=f"大纲确认: {outline_confirmed}", name="outline"
            #                 )
            #             ],
            #             "report_outline": outline_confirmed,
            #             "current_node": "outline",
            #         },
            #         goto="central_agent",
            #     )
            # else:
            #     raise TypeError(f"Interrupt value of {feedback} is not supported.")

from src.llms.llm import get_llm_by_type
from ..graph.types import State
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt
from sentence_transformers import CrossEncoder

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
    filter_content_by_relevant_docs,
    mark_content_with_support,
    repair_unknown_citations
)
from src.factstruct.outline_node import OutlineNode

from ..graph.types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine
from src.utils.statistics import global_statistics, timed_step
import re
from typing import Dict, Any
import json
from src.utils.reference_utils import global_reference_map, process_final_report
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
                "observations": result_observations,
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
                "observations": result_observations,
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
        logger.info("Web研究任务完成，返回中枢Agent")
        logger.info(f"state:{state}")
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
                "observations": result_observations,
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
        logger.info(f"state:{state}")
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
            observations = state.get("observations", [])

            messages.append(
                HumanMessage(
                    f"##User Query\n\n{state.get('user_query', '')}\n\n##用户约束\n\n{state.get("user_dst","")}\n\n##报告大纲{state.get('report_outline','用户未提供大纲')}\n\nBelow are information collected in previous tasks:\n\n{"\n\n".join(observations)}"
                )
            )        
            # messages.append(
            #     HumanMessage(
            #         f"##User Query\n\n{state.get('user_query', '')}\n\n##用户约束\n\n{state.get("user_dst","")}\n\n##报告大纲{state.get('report_outline','用户未提供大纲')}\n\nBelow are information collected in previous tasks:\n\n{"\n\n".join(data_collections)}"
            #     )
            # )        
            logger.debug(f"Reporter messages: {messages}")
            llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "default"))
            response = llm.invoke(messages)
            final_report = response.content
            #可以在这个地方加一个对final_report的处理
            

            
            
            session_id = config["configurable"]["thread_id"]
            reference_map=global_reference_map.get_session_ref_map(session_id)
            # logger.info(f"before reference_map:{reference_map}")
            # logger.info(f"before final_report :{final_report}")
            final_report = process_final_report(final_report, reference_map)
            # logger.info(f"after final_report :{final_report}")


            #增加引用检查部分
            logger.info(f"引用检查")
            # logger.info(f"state:{state}")
            logger.info(f"observations:{observations}")
            # logger.info(f"data_collections:{data_collections}")
            logger.info(f"final_report:{final_report}")
            semantic_cls = CrossEncoder("/data1/Yangzb/Model/StructBert/cross-encoder/nli-deberta-v3-small")
            #这个是判断引用和句子的关系
            supported = filter_content_by_relevant_docs(
                content=final_report,
                relevant_docs=reference_map,
                semantic_cls=semantic_cls
            )
            logger.info(f"supported :{supported}")
            
            #这个是把关系应用到生成文章上
            new_content = mark_content_with_support(
                content=final_report,
                nli_results=supported
            )
            logger.info(f"new_content :{new_content}")
            
            #这个是把错误引用进行处理的
            final_report=repair_unknown_citations(
                content=new_content,
                relevant_docs=reference_map,
                semantic_cls=semantic_cls
            )
            logger.info(f"final_report :{final_report}")
            
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
            
            #可以在这个地方加一个对final_report的处理
            session_id = config["configurable"]["thread_id"]
            reference_map=global_reference_map.get_session_ref_map(session_id)
            logger.info(f"before reference_map:{reference_map}")
            logger.info(f"before final_report :{final_report}")
            final_report = process_final_report(final_report, reference_map)
            logger.info(f"after final_report :{final_report}")
            
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


    @timed_step("execute_human_feedback")
    async def execute_human_feedback(self, state: State, config: RunnableConfig) -> Command:
        stage = state.get("wait_stage", "perception")
        if stage == "perception":
            dst_question = state.get("dst_question", "")
            feedback = interrupt(
                    "Please Fill the Question.[DST]" + dst_question + "[/DST]"
                )
            logger.info(f"用户反馈的DST问题: {feedback}. goto perception node again.")
            return Command(
                update={
                    "hitl_feedback": feedback,
                    "current_node": "human_feedback",
                },
                goto="perception",
            )
        elif stage == "outline":
            outline = state.get("report_outline", "")
            feedback = interrupt(
                    "Please Confirm or Edit the Outline.[OUTLINE]"
                    + outline
                    + "[/OUTLINE]"
                )
            logger.info(f"用户反馈的大纲: {feedback}. goto outline node again.")
            return Command(
                update={
                    "hitl_feedback": feedback,
                    "current_node": "human_feedback",
                },
                goto="outline",
            )


    @timed_step("execute_human_feedback")
    async def execute_human_feedback(self, state: State, config: RunnableConfig) -> Command:
        stage = state.get("wait_stage", "perception")
        if stage == "perception":
            dst_question = state.get("dst_question", "")
            feedback = interrupt(
                    "Please Fill the Question.[DST]" + dst_question + "[/DST]"
                )
            logger.info(f"用户反馈的DST问题: {feedback}. goto perception node again.")
            return Command(
                update={
                    "hitl_feedback": feedback,
                    "current_node": "human_feedback",
                },
                goto="perception",
            )
        elif stage == "outline":
            outline = state.get("report_outline", "")
            feedback = interrupt(
                    "Please Confirm or Edit the Outline.[OUTLINE]"
                    + outline
                    + "[/OUTLINE]"
                )
            logger.info(f"用户反馈的大纲: {feedback}. goto outline node again.")
            return Command(
                update={
                    "hitl_feedback": feedback,
                    "current_node": "human_feedback",
                },
                goto="outline",
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
                # messages = apply_prompt_template("perception", state) + [
                #     HumanMessage(f"##User Query\n\n{user_query}\n\n")
                # ]
                messages = apply_prompt_template("perception", state)

                # logger.debug("messages"+str(messages))
                response = perception_llm.invoke(messages)
                dst_question = response.content
                # logger.debug("dst_question"+str(dst_question))
                dst_question = repair_json_output(dst_question)
                logger.info(f"感知层完成，生成DST问题: {dst_question}")
                return Command(
                    update={
                        "dst_question": dst_question,
                        "wait_stage": "perception",
                        "current_node": "perception",
                    },
                    goto="human_feedback",
                )
            except Exception as e:
                logger.error(f"感知层执行失败: {str(e)}")

        if wait_stage == "perception":
            feedback = state.get("hitl_feedback", "")
            dst_question = state.get("dst_question", "")
            # if the feedback is not accepted, return the planner node
            if feedback and str(feedback).upper().startswith("[FILLED_QUESTION]"):
                messages = apply_prompt_template("perception", state) + [
                    HumanMessage(
                        f"##User Query\n\n{user_query}\n\n##希望用户回答的问题\n\n{dst_question}\n\n##用户回答的结果\n\n{feedback}\n\n"
                    )
                ]
                # logger.debug("messages"+str(messages))
                # exit()
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
                        "wait_stage": "",
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
                        "wait_stage": "",
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
            #bg_investigation = search_docs(user_query, top_k=5)
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
                        "factstruct_max_iterations",  4
                    ),  # 默认 10 次迭代
                    batch_size=state.get("factstruct_batch_size", 2),  # 默认批量大小 3
                    config=config,
                )

                # 转换为 Markdown 格式（完整大纲，不限制深度）
                outline_response = outline_node_to_markdown(
                    outline_root, max_depth=None, include_root=True
                )
                

                # 如果用户指定了字数限制，执行字数规划
                total_word_limit = state.get("total_word_limit", 5000)
                if total_word_limit > 0:
                    logger.info(f"检测到字数限制 {total_word_limit}，开始字数规划...")
                    outline_root = self.execute_word_planning(
                        outline_root, total_word_limit
                    )
                    # 更新大纲文本，包含字数信息
                    outline_response = outline_root.to_text_tree(
                        include_word_limit=True
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


    @timed_step("execute_word_planning")
    def execute_word_planning(
        self, outline_root: OutlineNode, total_word_limit: int
    ) -> OutlineNode:
        """
        执行字数规划，为大纲中的每个叶子节点分配字数配额

        Args:
            outline_root: 大纲根节点
            total_word_limit: 用户指定的总字数限制

        Returns:
            更新了字数配额的大纲根节点
        """
        import json

        logger.info(f"开始字数规划，总字数限制: {total_word_limit}")

        # 构建大纲结构信息供LLM分析
        def build_outline_info(node: OutlineNode, depth: int = 0) -> list:
            nodes_info = []
            nodes_info.append(
                {
                    "id": node.id,
                    "title": node.title,
                    "depth": depth,
                    "is_leaf": node.is_leaf(),
                }
            )
            for child in node.children:
                nodes_info.extend(build_outline_info(child, depth + 1))
            return nodes_info

        outline_info = build_outline_info(outline_root)
        leaf_nodes = [n for n in outline_info if n["is_leaf"]]

        # 构建LLM请求
        outline_text = outline_root.to_text_tree()
        prompt_content = f"""请为以下报告大纲分配字数。

        ## 大纲结构
        {outline_text}

        ## 叶子节点列表
        {json.dumps(leaf_nodes, ensure_ascii=False, indent=2)}

        ## 总字数限制
        {total_word_limit} 字

        请根据每个叶子节点的重要性和内容复杂度，智能分配字数配额。
        你必须只输出一个合法的 JSON 对象。禁止输出任何解释、说明、注释、标题或额外文本。如果输出包含非 JSON 内容，将被视为错误。
        """

        try:
            messages = apply_prompt_template("word_planner", {"messages": []}) + [
                HumanMessage(content=prompt_content)
            ]
            llm = get_llm_by_type(AGENT_LLM_MAP.get("outline", "default"))
            response = llm.invoke(messages)
            result = response.content

            # 解析JSON结果
            logger.info(f"result:{result}")
            # result = result.replace("```json", "").replace("```", "").strip()

            match = re.search(r"\{[\s\S]*\}", result)
            if not match:
                raise ValueError("No JSON object found in LLM output")

            allocations = json.loads(match.group(0))

            # 将字数配额写入节点
            for alloc in allocations.get("allocations", []):
                node_id = alloc.get("node_id")
                word_limit = alloc.get("word_limit", 0)
                node = outline_root.find_node_by_id(node_id)
                if node:
                    node.word_limit = word_limit
                    logger.debug(
                        f"节点 {node_id} ({node.title}) 分配字数: {word_limit}"
                    )

            # 自底向上计算非叶子节点的字数
            def update_parent_word_limits(node: OutlineNode) -> int:
                if node.is_leaf():
                    return node.word_limit
                total = sum(update_parent_word_limits(child) for child in node.children)
                node.word_limit = total
                return total

            update_parent_word_limits(outline_root)
            logger.info(f"字数规划完成，根节点总字数: {outline_root.word_limit}")

        except Exception as e:
            logger.error(f"字数规划失败: {str(e)}")
            # Fallback: 平均分配
            leaf_nodes_obj = outline_root.get_leaf_nodes()
            avg_words = total_word_limit // len(leaf_nodes_obj) if leaf_nodes_obj else 0
            for node in leaf_nodes_obj:
                node.word_limit = avg_words
            logger.warning(f"使用平均分配策略，每个叶子节点: {avg_words} 字")

        logger.info(f"outline_root:{outline_root}")
        # exit()
        return outline_root

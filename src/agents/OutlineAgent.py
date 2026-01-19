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
# from src.prompts.central_decision import Decision, DelegateParams
from src.utils.reference_utils import global_reference_map
from ..graph.types import State
# from .SubAgentConfig import get_sub_agents_by_global_type


# -------------------------
# 核心枚举定义
# -------------------------
class OutlineTool(Enum):
    """Outline Agent 可执行的结构性工具"""

    INITIALIZATION = "initialization"
    EXPANDATION = "expandation"
    REDUCTION = "reduction"
    REFLECT = "reflect"
    FINISH = "finish"



# -------------------------
# 中枢Agent核心模块--中枢Agent的action
# exp: 与prompt/Outline_decision.py中的Decision类不同的是，那里是字符串类型，这里是枚举类型，所以要定义两次
# -------------------------
@dataclass
class OutlineToolDecision:
    """Outline Agent 的决策结果"""

    tool: OutlineTool                 # 使用的工具
    reasoning: str                    # 为什么这么做
    params: Optional[Dict[str, Any]]  # 工具参数（各 tool 自己解释）



#不知道为啥要两个，那就实现成两个吧
from pydantic import BaseModel

class OutlineToolDecision_Base(BaseModel):
    tool: Literal[
        "initialization",
        "expandation",
        "reduction",
        "reflect",
        "finish",
    ]
    reasoning: str
    params: Optional[Dict[str, Any]] = None



class OutlineAgent:
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

    def __init__(
        self,
        initial_query: str,
        central_guidance: str | None = None,
        factstruct_outline: str | None = None,
        state: State | None = None,
    ):
        # --- Core task signal ---
        self.initial_query = initial_query
        # --- High-level planning signals ---
        self.central_guidance = central_guidance
        self.replan_result = replan_result

        # --- Current outline state ---
        self.factstruct_outline = state.get("factstruct_outline")
        self.factstruct_memory = state.get("factstruct_memory")
        self.feedback = state.get("feedback")
        self.total_word_limit = state.get("total_word_limit")




    def make_decision(
        self, state: State, config: RunnableConfig, retry_count: int = 0
    ) -> OutlineToolDecision:
        """
        中枢Agent决策核心逻辑，分析当前状态生成决策结果

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            决策结果对象
        """
        max_retries = 3
        logger.info("Outline Agent进行决策...")
        start_time = datetime.now()

        # 构建决策prompt
        messages = self._build_decision_prompt(state, config)
        logger.debug(f"outline 决策prompt: {messages}")


        try:
            llm = get_llm_by_type(
                AGENT_LLM_MAP.get("outline_agent", "default")
            ).with_structured_output(
                OutlineToolDecision_Base,   # ✅ 给 LLM 用的 schema
                method="json_mode",
            )

            response: OutlineToolDecision_Base = llm.invoke(messages)

            logger.info(f"Outline 决策结果(raw): {response}")

            # ✅ 从 LLM 协议对象 → 系统内部对象
            decision = OutlineToolDecision(
                tool=response.tool,
                reasoning=response.reasoning,
                params=response.params,
            )

            end_time = datetime.now()
            global_statistics.add_time_entry(
                {
                    "step_name": "outline_decision",
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                }
            )

            return decision

        except Exception as e:
            import traceback

            logger.error(
                f"Outline 决策解析失败 (尝试 {retry_count + 1}/{max_retries}): {e}"
            )
            logger.error(traceback.format_exc())

            if retry_count < max_retries - 1:
                return self.make_decision(state, config, retry_count + 1)

            # 🚨 兜底：强制 finish，防止系统卡死
            return OutlineToolDecision(
                tool="finish",
                reasoning="Outline decision parsing failed repeatedly, forcing termination.",
                params=None,
            )


    def _build_decision_prompt(
        self,
        state: State,
        config: RunnableConfig,
    ) -> List[Union[AIMessage, HumanMessage]]:
        """
        构建 Outline Agent 的决策 prompt
        """

        context = {
            # 必须项
            "user_query": state.get("initial_query"),

            # 可选项（prompt 里有 if 判断）
            "central_guidance": state.get("central_guidance"),
            "factstruct_outline": state.get("factstruct_outline"),
            "total_word_limit": state.get("total_word_limit"),
            "feedback": state.get("feedback"),
            "SOP": state.get("sop"),

            # 语言
            "locale": state.get("locale", "zh-CN"),
        }

        # 合并 config（如需要）
        context = {**context, **config}

        return apply_prompt_template(
            "outline_decision",   # ✅ 对应 src/prompts/outline_decision.md
            state,
            extra_context=context,
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
        
        session_id = config["configurable"]["thread_id"]
        # global_reference_map.save_session(session_id)
        # 构建执行摘要（包含完整记忆栈历史）
        execution_summary = {
            "user_query": state.get("user_query", "未知查询"),
            "execution_history": [
                entry.to_dict() for entry in self.memory_stack.get_all()
            ],
            "final_report": final_report,
            "research": global_reference_map.get_session_ref_map(session_id),#state.get("data_collections", []),
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
    outline_node_to_dict,
    memory_to_dict,
    filter_content_by_relevant_docs,
    mark_content_with_support,
    repair_unknown_citations
)

from src.factstruct import outline_node_to_dict, memory_to_dict
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
        执行大纲子Agent（FactStruct Stage 1）

        基于用户问题和已确认的任务规划，生成或调整报告的大纲结构，
        并为后续 FactStruct Stage 2 提供结构化 Outline 与 Memory。
        """
        logger.info("大纲Agent开始执行（FactStruct Stage 1）...")

        user_query = state.get("user_query", "")
        user_dst = state.get("user_dst", "")
        factstruct_outline_dict = state.get("factstruct_outline", None)#如果有的话，后续更改到时候再修
        factstruct_memory_dict = state.get("factstruct_memory",None)
        #提取的是 guideline
        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "未知研究任务")
        outline_response = "大纲生成失败: 未知错误"
        
        #这玩意是人工确认 human node的，感觉没啥用，FactStruct 如果配上 Human feedback 才需要这个
        # auto_accepted_plan = state.get("auto_accepted_plan", False)
        # if not auto_accepted_plan:
        #     logger.warning("任务规划未确认，Outline Agent 不执行")
        #     return Command(
        #         update={
        #             "messages": [
        #                 HumanMessage(
        #                     content="任务规划尚未确认，跳过大纲生成",
        #                     name="outline",
        #                 )
        #             ],
        #             "current_node": "central_agent",
        #         },
        #         goto="central_agent",
        #     )

        try:
            replan_result= state.get("replan_result", None)
            full_query = user_query
            if user_dst:
                full_query = f"{user_query}\n\n用户补充需求：{user_dst}"

            # 创建大纲
            # 扩展大纲
            # 删减大纲
            # 字数控制反馈
            # 使用FactStruct自己的 LLM 来做这个事情。
            
            outline_root, memory = run_factstruct_stage1(
                query=full_query,
                max_iterations=state.get("factstruct_max_iterations", 4),
                batch_size=state.get("factstruct_batch_size", 2),
                task_description=task_description,
                replan_result=replan_result,
                config=config,
            )

            # 大纲的字数匹配
            total_word_limit = state.get("total_word_limit", 5000)
            if total_word_limit > 0:
                logger.info(f"检测到字数限制 {total_word_limit}，执行字数规划...")
                outline_root = self.execute_word_planning(
                    outline_root, total_word_limit
                )
                outline_response = outline_root.to_text_tree(
                    include_word_limit=True
                )
            else:
                outline_response = outline_node_to_markdown(
                    outline_root, max_depth=None, include_root=True
                )

            factstruct_outline_dict = outline_node_to_dict(outline_root)
            factstruct_memory_dict = memory_to_dict(memory)

            logger.info(
                f"FactStruct Stage 1 完成: "
                f"{len(outline_root.get_all_nodes())} 个节点"
            )

        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
            logger.error(f"FactStruct Stage 1 执行失败: {str(e)}")

            outline_response = f"大纲生成失败（FactStruct Stage 1）: {str(e)}"

        # === 写入 central agent memory stack ===
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="outline",
            content="大纲任务: 使用 FactStruct Stage 1 生成或调整报告大纲",
            result={
                "outline": outline_response,
                "factstruct_outline": factstruct_outline_dict,
            },
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("大纲生成完成（FactStruct Stage 1），返回中枢Agent")

        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content="大纲生成完成（FactStruct Stage 1），返回中枢Agent",
                        name="outline",
                    )
                ],
                "report_outline": outline_response,
                "factstruct_outline": factstruct_outline_dict,
                "factstruct_memory": factstruct_memory_dict,
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
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

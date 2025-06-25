# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import os
import asyncio
from typing import Annotated, Literal, Dict, Any, Optional, List, Callable, Union
from enum import Enum
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
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
from src.utils.logger import logger
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan, Step, StepType
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine


# 定义中枢Agent动作类型
class CentralAgentAction(Enum):
    REFLECT = "reflect"  # 反思当前状态
    SUMMARIZE = "summarize"  # 总结已有信息
    THINK = "think"  # 思考下一步行动
    DELEGATE = "delegate"  # 委派子Agent
    FINISH = "finish"  # 完成任务


# 定义可用的子Agent类型
class SubAgentType(Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    REPORTER = "reporter"
    BACKGROUND_INVESTIGATOR = "background_investigator"


# 定义动作处理器接口
ActionHandler = Callable[[State, RunnableConfig], Command]


class CentralAgent:
    def __init__(self):
        # 注册可用的子Agent
        self.available_agents = {
            SubAgentType.PLANNER: self._planner_agent,
            SubAgentType.RESEARCHER: self._researcher_agent,
            SubAgentType.CODER: self._coder_agent,
            SubAgentType.REPORTER: self._reporter_agent,
            SubAgentType.BACKGROUND_INVESTIGATOR: self._background_investigator_agent,
        }

        # 注册中枢Agent动作处理器
        self.action_handlers: Dict[CentralAgentAction, ActionHandler] = {
            CentralAgentAction.REFLECT: self._handle_reflect,
            CentralAgentAction.SUMMARIZE: self._handle_summarize,
            CentralAgentAction.THINK: self._handle_think,
            CentralAgentAction.DELEGATE: self._handle_delegate,
            CentralAgentAction.FINISH: self._handle_finish,
        }

    def _planner_agent(self, state: State, config: RunnableConfig) -> Command:
        """规划子Agent - 生成任务执行计划"""
        logger.info("规划子Agent生成计划中...")
        configurable = Configuration.from_runnable_config(config)
        plan_iterations = state.get("plan_iterations", 0)

        # 应用规划提示模板，传递完整上下文
        context = {
            "state": state,
            "plan_iterations": plan_iterations,
            "background_investigation_results": state.get(
                "background_investigation_results", ""
            ),
            "messages": state.get("messages", []),
            "user_query": state.get("user_query", ""),
            "current_plan": state.get("current_plan", {}),
        }

        messages = apply_prompt_template("planner", context, configurable)

        # 整合背景调查结果（如果有）
        if (
            plan_iterations == 0
            and state.get("enable_background_investigation")
            and state.get("background_investigation_results")
        ):
            messages += [
                {
                    "role": "user",
                    "content": (
                        "background investigation results of user query:\n"
                        + state["background_investigation_results"]
                        + "\n"
                    ),
                }
            ]

        # 获取LLM实例
        if AGENT_LLM_MAP["planner"] == "basic":
            llm = get_llm_by_type(AGENT_LLM_MAP["planner"]).with_structured_output(
                Plan,
                method="json_mode",
            )
        else:
            llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

        # 处理计划迭代限制
        if plan_iterations >= configurable.max_plan_iterations:
            return Command(
                update={"plan_iterations": plan_iterations, "current_node": "planner"},
                goto="sp_center_agent",  # 执行完毕返回中枢
            )

        # 获取LLM响应
        full_response = ""
        if AGENT_LLM_MAP["planner"] == "basic":
            response = llm.invoke(messages)
            full_response = response.model_dump_json(indent=4, exclude_none=True)
        else:
            response = llm.stream(messages)
            for chunk in response:
                full_response += chunk.content

        try:
            curr_plan = json.loads(repair_json_output(full_response))
        except json.JSONDecodeError:
            logger.warning("Planner response is not a valid JSON")
            plan_iterations += 1
            return Command(
                update={"plan_iterations": plan_iterations, "current_node": "planner"},
                goto="sp_center_agent",  # 执行完毕返回中枢
            )

        plan_iterations += 1
        has_enough_context = curr_plan.get("has_enough_context", False)

        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": curr_plan,
                "plan_iterations": plan_iterations,
                "has_enough_context": has_enough_context,
                "current_node": "planner",
            },
            goto="sp_center_agent",  # 执行完毕返回中枢
        )

    async def _researcher_agent(self, state: State, config: RunnableConfig) -> Command:
        """研究子Agent - 执行信息检索和分析"""
        logger.info("研究子Agent研究中...")
        configurable = Configuration.from_runnable_config(config)

        # 配置研究工具
        tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
        retriever_tool = get_retriever_tool(state.get("resources", []))
        if retriever_tool:
            tools.insert(0, retriever_tool)

        tools = [search_docs_tool]

        # 应用研究提示模板，传递完整上下文
        context = {
            "state": state,
            "messages": state.get("messages", []),
            "current_plan": state.get("current_plan", {}),
            "available_tools": [tool.name for tool in tools],
        }

        # 创建并执行研究Agent
        research_agent = ResearcherAgent(
            config=config, agent_type="researcher", default_tools=tools
        )

        # 等待异步方法完成
        result = await research_agent.execute_agent_step(state)

        # 确保返回中枢Agent
        if result.goto != "sp_center_agent":
            result = result._replace(goto="sp_center_agent")

        return result

    async def _coder_agent(self, state: State, config: RunnableConfig) -> Command:
        """编码子Agent - 执行代码编写和调试"""
        logger.info("编码子Agent编码中...")

        # 应用编码提示模板，传递完整上下文
        context = {
            "state": state,
            "messages": state.get("messages", []),
            "current_plan": state.get("current_plan", {}),
            "observations": state.get("observations", []),
        }

        # 创建并执行编码Agent
        code_agent = CoderAgent(
            config=config, agent_type="coder", default_tools=[python_repl_tool]
        )

        # 确保有计划可执行
        current_plan = state.get("current_plan")
        if (
            not current_plan
            or not hasattr(current_plan, "steps")
            or not current_plan.steps
        ):
            logger.warning("执行编码子Agent时缺少有效计划，创建临时计划")
            # 创建临时计划
            temp_plan = Plan(
                title="编码任务",
                thought="执行代码编写和调试",
                steps=[
                    Step(
                        title="编写代码",
                        description="根据需求编写代码",
                        step_type=StepType.CODING,
                    )
                ],
            )
            state["current_plan"] = temp_plan

        # 等待异步方法完成
        result = await code_agent.execute_agent_step(state)

        # 确保返回中枢Agent
        if result.goto != "sp_center_agent":
            result = result._replace(goto="sp_center_agent")

        return result

    def _reporter_agent(self, state: State) -> Command:
        """报告子Agent - 生成最终报告"""
        logger.info("报告子Agent生成报告中...")
        current_plan = state.get("current_plan")

        if not current_plan:
            return Command(
                update={"current_node": "reporter"},
                goto="sp_center_agent",  # 执行完毕返回中枢
            )

        # 准备报告输入，传递完整上下文
        context = {
            "state": state,
            "messages": state.get("messages", [])
            + [
                HumanMessage(
                    f"# Research Requirements\n\n## Task\n\n{current_plan.get('title', 'No Title')}\n\n## Description\n\n{current_plan.get('thought', 'No Description')}"
                )
            ],
            "locale": state.get("locale", "en-US"),
            "observations": state.get("observations", []),
            "current_plan": current_plan,
            "resources": state.get("resources", []),
        }

        # 应用报告提示模板
        invoke_messages = apply_prompt_template("reporter", context)

        # 添加格式提醒
        invoke_messages.append(
            HumanMessage(
                content="IMPORTANT: Structure your report with Key Points, Overview, Detailed Analysis, and Key Citations.\n"
                "Use markdown tables for data presentation. Cite sources at the end using - [Title](URL).",
                name="system",
            )
        )

        # 添加观察结果
        for observation in state.get("observations", []):
            invoke_messages.append(
                HumanMessage(
                    content=f"Observations: {observation}",
                    name="observation",
                )
            )

        # 生成报告
        response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
        response_content = response.content

        return Command(
            update={"final_report": response_content, "current_node": "reporter"},
            goto="sp_center_agent",  # 执行完毕返回中枢
        )

    def _background_investigator_agent(
        self, state: State, config: RunnableConfig
    ) -> Command:
        """背景调查子Agent - 获取任务相关背景信息"""
        logger.info("背景调查子Agent运行中...")
        configurable = Configuration.from_runnable_config(config)

        # 应用背景调查提示模板，传递完整上下文
        context = {
            "state": state,
            "messages": state.get("messages", []),
            "user_query": state.get("user_query", "unknown query"),
        }

        query = (
            state["messages"][-1].content if state.get("messages") else "unknown query"
        )

        try:
            background_investigation_results = search_docs_tool.invoke(query)
        except Exception as e:
            logger.error(f"搜索工具调用失败: {e}")
            background_investigation_results = []

        return Command(
            update={
                "background_investigation_results": json.dumps(
                    background_investigation_results, ensure_ascii=False
                ),
                "current_node": "background_investigator",
            },
            goto="sp_center_agent",  # 执行完毕返回中枢
        )

    def _handle_reflect(self, state: State, config: RunnableConfig) -> Command:
        """处理反思动作 - 评估当前状态和进展"""
        logger.info("中枢Agent正在反思当前状态...")

        # 构建反思提示，传递完整上下文
        context = {
            "state": state,
            "current_node": state.get("current_node", "无"),
            "node_history": state.get("node_history", []),
            "current_plan": state.get("current_plan", {}),
            "messages": state.get("messages", []),
            "action_history": state.get("action_history", []),
        }

        messages = apply_prompt_template(
            "central_reflect", context, Configuration.from_runnable_config(config)
        )

        # 添加当前状态信息
        messages.append(
            {
                "role": "system",
                "content": f"当前执行节点: {state.get('current_node', '无')}\n"
                f"历史节点: {state.get('node_history', [])}\n"
                f"当前计划: {json.dumps(state.get('current_plan', {}), default=str, ensure_ascii=False)}",
            }
        )

        # 获取LLM反思结果
        llm = get_llm_by_type(AGENT_LLM_MAP["center_agent"])
        full_response = ""
        response = llm.stream(messages)
        for chunk in response:
            full_response += chunk.content

        # 更新反思结果
        state["reflection"] = full_response

        return Command(
            update={
                "messages": [
                    AIMessage(content=full_response, name="central_reflection")
                ],
                "reflection": full_response,
                "current_node": "sp_center_agent",
            },
            goto="sp_center_agent",
        )

    def _handle_summarize(self, state: State, config: RunnableConfig) -> Command:
        """处理总结动作 - 整合已有信息"""
        logger.info("中枢Agent正在总结已有信息...")

        # 获取历史观察结果
        observations = state.get("observations", [])

        # 构建总结提示，传递完整上下文
        context = {
            "state": state,
            "observations": observations,
            "messages": state.get("messages", []),
            "current_plan": state.get("current_plan", {}),
        }

        messages = apply_prompt_template(
            "central_summarize", context, Configuration.from_runnable_config(config)
        )

        # 添加观察结果
        messages.append(
            {
                "role": "user",
                "content": f"Summarize the following observations:\n\n{json.dumps(observations, ensure_ascii=False)}",
            }
        )

        # 获取LLM总结结果
        llm = get_llm_by_type(AGENT_LLM_MAP["center_agent"])
        response = llm.invoke(messages)
        summary = response.content

        return Command(
            update={
                "messages": [AIMessage(content=summary, name="central_summary")],
                "summary": summary,
                "current_node": "sp_center_agent",
            },
            goto="sp_center_agent",
        )

    def _handle_think(self, state: State, config: RunnableConfig) -> Command:
        """处理思考动作 - 决定下一步行动"""
        logger.info("中枢Agent正在思考下一步行动...")

        # 构建思考提示，传递完整上下文
        context = {
            "state": state,
            "messages": state.get("messages", []),
            "reflection": state.get("reflection", "No reflection"),
            "summary": state.get("summary", "No summary"),
            "current_node": state.get("current_node", "sp_center_agent"),
            "action_history": state.get("action_history", []),
            "current_plan": state.get("current_plan", {}),
            "observations": state.get("observations", []),
        }

        messages = apply_prompt_template(
            "central_think", context, Configuration.from_runnable_config(config)
        )

        # 添加当前状态、反思和总结
        messages.append(
            {
                "role": "system",
                "content": f"当前状态: {json.dumps(state, default=str, ensure_ascii=False)}\n"
                f"反思结果: {state.get('reflection', 'No reflection')}\n"
                f"总结内容: {state.get('summary', 'No summary')}",
            }
        )

        # 获取LLM思考结果（JSON格式的决策）
        llm = get_llm_by_type(AGENT_LLM_MAP["center_agent"])
        full_response = ""
        response = llm.stream(messages)
        for chunk in response:
            full_response += chunk.content

        try:
            # 解析决策JSON
            decision = json.loads(repair_json_output(full_response))

            # 验证决策格式
            required_fields = ["next_action", "action_params"]
            for field in required_fields:
                if field not in decision:
                    raise ValueError(f"Missing field '{field}' in decision")

            # 检查委派决策中的子Agent类型
            if decision["next_action"] == "delegate":
                agent_type = decision.get("action_params", {}).get("agent_type")
                if not agent_type or agent_type not in [
                    agent.value for agent in SubAgentType
                ]:
                    raise ValueError(f"Invalid sub-agent type: {agent_type}")

            # 保存决策结果
            state["next_decision"] = decision

            return Command(
                update={
                    "messages": [
                        AIMessage(content=full_response, name="central_decision")
                    ],
                    "next_decision": decision,
                    "current_node": "sp_center_agent",
                },
                goto="sp_center_agent",
            )
        except Exception as e:
            logger.error(f"决策解析失败: {e}")
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"Decision parsing failed: {str(e)}",
                            name="central_error",
                        )
                    ],
                    "current_node": "sp_center_agent",
                },
                goto="sp_center_agent",
            )

    def _handle_delegate(self, state: State, config: RunnableConfig) -> Command:
        """处理委派动作 - 调用子Agent执行任务"""
        logger.info("中枢Agent正在委派子Agent...")

        # 获取决策结果
        decision = state.get("next_decision")
        if not decision:
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content="No decision found for delegation",
                            name="central_error",
                        )
                    ],
                    "current_node": "sp_center_agent",
                },
                goto="sp_center_agent",
            )

        # 获取要委派的子Agent类型
        agent_type = decision.get("action_params", {}).get("agent_type")
        if not agent_type:
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content="Missing 'agent_type' in delegation decision",
                            name="central_error",
                        )
                    ],
                    "current_node": "sp_center_agent",
                },
                goto="sp_center_agent",
            )

        try:
            # 转换为枚举类型
            agent_enum = SubAgentType(agent_type)
        except ValueError:
            available_agents_list = ", ".join([agent.value for agent in SubAgentType])
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"Invalid agent type: {agent_type}. Available types: {available_agents_list}",
                            name="central_error",
                        )
                    ],
                    "current_node": "sp_center_agent",
                },
                goto="sp_center_agent",
            )

        # 获取子Agent参数
        agent_params = decision.get("action_params", {}).get("agent_params", {})

        # 更新状态，添加委派信息
        state["delegation"] = {
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat(),
            "params": agent_params,
        }

        # 调用子Agent
        agent_fn = self.available_agents[agent_enum]

        # 确保研究和编码Agent有临时计划
        if agent_enum in [SubAgentType.RESEARCHER, SubAgentType.CODER]:
            # 使用委派参数中的任务信息，或提供默认值
            task_title = agent_params.get("title", f"{agent_type.capitalize()} Task")
            task_description = agent_params.get(
                "description", f"Execute {agent_type} task"
            )

            # 创建正确类型的Plan对象
            temp_plan = Plan(
                title=task_title,
                thought=task_description,
                steps=[
                    Step(
                        title=task_title,
                        description=task_description,
                        step_type=(
                            StepType.RESEARCH
                            if agent_enum == SubAgentType.RESEARCHER
                            else StepType.CODING
                        ),
                    )
                ],
            )

            # 确保计划被设置
            state["current_plan"] = temp_plan
            logger.info(f"为{agent_type}代理设置临时计划: {task_title}")

        # 同步调用子Agent
        if agent_enum in [SubAgentType.RESEARCHER, SubAgentType.CODER]:
            # 异步子Agent需要异步调用
            return asyncio.run(agent_fn(state, config))
        else:
            # 同步子Agent直接调用
            return agent_fn(state, config)

    def _handle_finish(self, state: State, config: RunnableConfig) -> Command:
        """处理完成动作 - 结束任务并保存结果"""
        logger.info("中枢Agent正在完成任务...")

        # 获取最终报告
        final_report = state.get("final_report", "No report generated")

        # 保存结果
        os.makedirs("./reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./reports/report_{timestamp}.json"

        # 构建保存数据
        data = {
            "user_query": state.get("user_query"),
            "plan": state.get("current_plan"),
            "final_report": final_report,
            "execution_history": state.get("node_history", []),
            "action_history": state.get("action_history", []),
            "timestamp": timestamp,
        }

        # 写入文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"结果已保存至: {filename}")

        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"Task completed. Results saved to {filename}",
                        name="central_finish",
                    )
                ],
                "current_node": "sp_center_agent",
                "task_completed": True,
            },
            goto="__end__",  # 结束流程
        )

    def execute_action(
        self, action: CentralAgentAction, state: State, config: RunnableConfig
    ) -> Command:
        """执行指定动作"""
        if action not in self.action_handlers:
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"Unsupported action: {action.value}",
                            name="central_error",
                        )
                    ],
                    "current_node": "sp_center_agent",
                },
                goto="sp_center_agent",
            )

        # 记录动作历史
        action_history = state.get("action_history", [])
        action_history.append(
            {"action": action.value, "timestamp": datetime.now().isoformat()}
        )
        state["action_history"] = action_history

        # 执行动作处理函数
        return self.action_handlers[action](state, config)


# 子Agent节点处理函数
async def planner_agent_node(state: State, config: RunnableConfig) -> Command:
    """规划子Agent节点处理函数"""
    central_agent = CentralAgent()
    return central_agent._planner_agent(state, config)


async def researcher_agent_node(state: State, config: RunnableConfig) -> Command:
    """研究子Agent节点处理函数"""
    central_agent = CentralAgent()
    return await central_agent._researcher_agent(state, config)


async def coder_agent_node(state: State, config: RunnableConfig) -> Command:
    """编码子Agent节点处理函数"""
    central_agent = CentralAgent()
    return await central_agent._coder_agent(state, config)


async def reporter_agent_node(state: State, config: RunnableConfig) -> Command:
    """报告子Agent节点处理函数"""
    central_agent = CentralAgent()
    return central_agent._reporter_agent(state)


async def background_investigator_agent_node(
    state: State, config: RunnableConfig
) -> Command:
    """背景调查子Agent节点处理函数"""
    central_agent = CentralAgent()
    return central_agent._background_investigator_agent(state, config)


# 中枢Agent节点
async def sp_center_agent_node(state: State, config: RunnableConfig) -> Command:
    """中枢Agent动态编排节点，接收决策并执行相应动作"""
    logger.info(
        f"中枢Agent正在分析状态，当前节点: {state.get('current_node', '初始状态')}"
    )

    # 初始化中枢Agent
    central_agent = CentralAgent()

    # 如果是首次调用，从思考开始
    if not state.get("action_history"):
        return central_agent.execute_action(CentralAgentAction.THINK, state, config)

    # 获取最新决策
    decision = state.get("next_decision")
    if not decision:
        # 如果没有决策，再次思考
        return central_agent.execute_action(CentralAgentAction.THINK, state, config)

    # 根据决策执行相应动作
    next_action = decision.get("next_action")
    if not next_action:
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="Invalid decision: missing 'next_action'",
                        name="central_error",
                    )
                ],
                "current_node": "sp_center_agent",
            },
            goto="sp_center_agent",
        )

    try:
        # 映射到动作类型
        action = CentralAgentAction(next_action)
        return central_agent.execute_action(action, state, config)
    except ValueError:
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"Invalid action: {next_action}", name="central_error"
                    )
                ],
                "current_node": "sp_center_agent",
            },
            goto="sp_center_agent",
        )


# 构建状态图
def build_graph_central_agent():
    """构建以中枢Agent为核心的动态编排状态图"""
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(State)
    builder.add_edge(START, "sp_center_agent")  # 起始节点指向中枢Agent

    # 添加中枢Agent节点
    builder.add_node("sp_center_agent", sp_center_agent_node)

    # 注册所有子Agent节点
    sub_agents = {
        "planner": planner_agent_node,
        "researcher": researcher_agent_node,
        "coder": coder_agent_node,
        "reporter": reporter_agent_node,
        "background_investigator": background_investigator_agent_node,
    }

    for agent_name, agent_handler in sub_agents.items():
        builder.add_node(agent_name, agent_handler)
        builder.add_edge("sp_center_agent", agent_name)  # 从中枢到子Agent
        builder.add_edge(agent_name, "sp_center_agent")  # 从子Agent返回中枢

    # 完成节点指向结束
    builder.add_edge("sp_center_agent", "__end__")

    return builder.compile()


# 生成动态编排状态图
graph = build_graph_central_agent()

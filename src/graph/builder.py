# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .types import State

# from .nodes import (
#     coordinator_node,
#     planner_node,
#     reporter_node,
#     research_team_node,
#     researcher_node,
#     coder_node,
#     human_feedback_node,
#     background_investigation_node,
#     sp_planner_node,
#     speech_node,
#     zip_data,
#     sp_center_agent_node,
# )

from .sp_nodes import *

# 定义可用的子Agent名称
NODE_NAMES = [
    "planner",
    "researcher",
    "coder",
    "reporter",
    "background_investigator",
    "coordinator",
    "research_team",
    "zip_data",
]


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_edge("reporter", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


def build_graph_sp():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", sp_planner_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_edge("reporter", END)
    return builder.compile()


def build_graph_xxqg():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", sp_planner_node)
    builder.add_node("reporter", speech_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("zip_data", zip_data)
    builder.add_edge("reporter", "zip_data")
    builder.add_edge("zip_data", END)
    return builder.compile()


# 简化的状态图构建函数
def build_graph_central_agent():
    """构建以中枢Agent为核心的动态编排状态图"""

    builder = StateGraph(State)
    builder.add_edge(START, "sp_center_agent")  # 起始节点指向中枢Agent

    # 添加中枢Agent节点
    builder.add_node("sp_center_agent", sp_center_agent_node)

    # 中枢Agent可以跳转到所有子节点
    for agent_name in [agent.value for agent in SubAgentType]:
        builder.add_edge("sp_center_agent", agent_name)
        builder.add_edge(agent_name, "sp_center_agent")  # 子节点执行后返回中枢

    # 完成节点指向结束
    builder.add_edge("sp_center_agent", END)

    return builder.compile()


# graph = build_graph_xxqg()


# -------------------------
# 状态图构建
# -------------------------
def build_multi_agent_graph():
    """
    构建多Agent系统状态图，定义系统状态转移逻辑

    Returns:
        编译后的状态图对象
    """
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(State)

    # 添加center planner agent
    builder.add_node("central_agent", central_agent_node)

    # 添加sub agent
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reporter", reporter_node)

    # 定义状态转移
    builder.add_edge(START, "central_agent")
    # builder.add_edge("central_agent", "researcher")
    # builder.add_edge("central_agent", "coder")
    # builder.add_edge("central_agent", "reporter")

    # builder.add_edge("researcher", "central_agent")
    # builder.add_edge("coder", "central_agent")
    # builder.add_edge("reporter", "central_agent")

    builder.add_edge("central_agent", END)

    return builder.compile()


# 生成最终的多Agent系统图
graph = build_multi_agent_graph()

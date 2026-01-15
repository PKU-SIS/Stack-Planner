# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.utils.logger import logger
from .types import State

from .nodes import (
    coordinator_node,
    planner_node,
    reporter_node,
    research_team_node,
    researcher_node,
    researcher_sp_node,
    coder_node,
    human_feedback_node,
    background_investigation_node,
    sp_planner_node,
    speech_node,
    zip_data,
    coordinator_xxqg_node,
    reporter_xxqg_node,
    researcher_xxqg_node,
)

from .sp_nodes import central_agent_node, perception_node, outline_node,human_feedback_node
from src.agents.sub_agent_registry import get_sub_agents_by_global_type


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
    builder.add_node("researcher", researcher_sp_node)
    builder.add_node("coder", coder_node)
    builder.add_edge("reporter", END)
    return builder.compile()


def build_graph_xxqg():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_xxqg_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", sp_planner_node)
    builder.add_node("reporter", reporter_xxqg_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_xxqg_node)
    builder.add_node("coder", coder_node)
    builder.add_node("zip_data", zip_data)
    builder.add_edge("reporter", "zip_data")
    builder.add_edge("zip_data", END)
    return builder.compile()


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
    sub_agents = get_sub_agents_by_global_type("sp")

    for sub_agent in sub_agents:
        builder.add_node(sub_agent["name"], sub_agent["node"])
    # builder.add_node("researcher", sp_researcher_node)
    # builder.add_node("coder", sp_coder_node)
    # builder.add_node("reporter", sp_reporter_node)

    # 定义状态转移
    builder.add_edge(START, "central_agent")

    builder.add_edge("central_agent", END)

    return builder.compile()


def get_next_perception(state: State) -> str:
    wait_stage = state.get("wait_stage", "")
    if wait_stage == "perception":
        return "human_feedback"
    else:
        return "outline"

def get_next_outline(state: State) -> str:
    wait_stage = state.get("wait_stage", "")
    if wait_stage == "outline":
        return "human_feedback"
    else:
        return "central_agent"

def get_next_feedback(state: State) -> str:
    wait_stage = state.get("wait_stage", "")
    if wait_stage == "perception":
        return "perception"
    elif wait_stage == "outline":
        return "outline"
    else:
        return "central_agent"

def _build_graph_sp_xxqg():
    """
    构建多Agent系统状态图，定义系统状态转移逻辑

    Returns:
        编译后的状态图对象
    """
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(State)

    # 添加center planner agent
    builder.add_node("perception", perception_node)
    builder.add_node("central_agent", central_agent_node)
    builder.add_node("outline", outline_node)
    builder.add_node("human_feedback", human_feedback_node)

    # 添加sub agent
    sub_agents = get_sub_agents_by_global_type("sp_xxqg")

    for sub_agent in sub_agents:
        builder.add_node(sub_agent["name"], sub_agent["node"])
    # builder.add_node("researcher", sp_xxqg_researcher_node)
    # builder.add_node("coder", sp_coder_node)
    # builder.add_node("reporter", sp_xxqg_reporter_node)

    # 下面这些暂时没有算sub agent
    builder.add_node("zip_data", zip_data)

    # 感知层，包括search before plan、human in the loop
    builder.add_edge(START, "perception")
    # builder.add_conditional_edge(
    #     "perception",
    #     get_next_perception
    # )
    # builder.add_conditional_edge(
    #     "outline",
    #     get_next_outline
    # )
    # builder.add_conditional_edge(
    #     "human_feedback",
    #     get_next_feedback
    # )

    builder.add_edge("central_agent", "zip_data")

    # 后处理部分
    builder.add_edge("zip_data", END)

    return builder


# 生成最终的多Agent系统图
base_graph = build_graph()
sp_graph = build_multi_agent_graph()
xxqg_graph = build_graph_xxqg()


def build_graph_with_memory_from_builder(builder):
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    return builder.compile(checkpointer=memory)


sp_xxqg_graph_builder = _build_graph_sp_xxqg()


_GRAPH_BUILDER_CLASS_MAP = {
    "base": None,
    "sp": None,
    "xxqg": None,
    "sp_xxqg": sp_xxqg_graph_builder,
}

_GRAPH_CLASS_MAP = {
    "base": {"memory": None, "no_memory": base_graph},
    "sp": {"memory": None, "no_memory": sp_graph},
    "xxqg": {"memory": None, "no_memory": xxqg_graph},
    "sp_xxqg": {"memory": None, "no_memory": sp_xxqg_graph_builder.compile()},
}


def get_graph_by_format(graph_format: str, with_memory: bool = False):
    """
    根据图格式获取状态图实例

    Args:
        graph_format (str): 图格式标识
        with_memory (bool): 是否启用记忆功能

    Returns:
        状态图实例
    """
    if graph_format not in _GRAPH_BUILDER_CLASS_MAP:
        raise ValueError(f"Unsupported graph format: {graph_format}")

    graph_builder = _GRAPH_BUILDER_CLASS_MAP[graph_format]
    if with_memory:
        if graph_format != "sp_xxqg":
            logger.error("Memory功能目前仅支持 sp_xxqg 图格式")
            return _GRAPH_CLASS_MAP[graph_format]["no_memory"]
        else:
            if _GRAPH_CLASS_MAP[graph_format]["memory"] is None:
                _GRAPH_CLASS_MAP[graph_format]["memory"] = (
                    build_graph_with_memory_from_builder(graph_builder)
                )
            return _GRAPH_CLASS_MAP[graph_format]["memory"]
    else:
        return _GRAPH_CLASS_MAP[graph_format]["no_memory"]

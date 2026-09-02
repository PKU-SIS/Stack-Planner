"""Dedicated StackPlanner graph for multi-turn text-to-SQL tasks."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .sp_nodes import central_agent_node, sql_agent_node, sql_conclusion_node
from .types import State


def build_sql_graph() -> StateGraph:
    """Build Central -> SQL draft -> SQL conclusion orchestration."""
    builder = StateGraph(State)
    builder.add_node("central_agent", central_agent_node)
    builder.add_node("sql_agent", sql_agent_node)
    builder.add_node("sql_conclusion", sql_conclusion_node)
    builder.add_edge(START, "central_agent")
    builder.add_edge("central_agent", END)
    return builder

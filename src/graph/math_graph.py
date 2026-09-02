"""Dedicated StackPlanner graph for multi-turn mathematical tasks."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .sp_nodes import central_agent_node, coder_node, conclusion_node
from .types import State


def build_math_graph() -> StateGraph:
    """Build Central -> optional Coder -> Conclusion Math orchestration."""
    builder = StateGraph(State)
    builder.add_node("central_agent", central_agent_node)
    builder.add_node("coder", coder_node)
    builder.add_node("conclusion", conclusion_node)
    builder.add_edge(START, "central_agent")
    # Command-based routing handles coder/conclusion. This fallback terminates
    # only if Central explicitly returns without a destination.
    builder.add_edge("central_agent", END)
    return builder

"""Task-specific StackPlanner profiles.

Each task family owns its graph, SOP, allowed sub-agents, terminal agent, and
memory policy. Task families use separate profiles rather than branching inside
another task's graph.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskGraphProfile:
    graph_format: str
    task_family: str
    sop_name: str
    sub_agents: tuple[tuple[str, str], ...]
    terminal_agent: str
    supports_checkpoint_memory: bool = True


MATH_PROFILE = TaskGraphProfile(
    graph_format="sp_math",
    task_family="math",
    sop_name="math",
    sub_agents=(
        ("coder", "Verify non-trivial arithmetic or symbolic calculations"),
        ("conclusion", "Produce a concise, verified mathematical conclusion"),
    ),
    terminal_agent="conclusion",
)

SQL_PROFILE = TaskGraphProfile(
    graph_format="sp_sql",
    task_family="sql",
    sop_name="sql",
    sub_agents=(
        ("sql_agent", "Draft a SQLite query from the active conversational intent"),
        ("sql_conclusion", "Return the final query in one SQL code block"),
    ),
    terminal_agent="sql_conclusion",
)


TASK_GRAPH_PROFILES = {
    MATH_PROFILE.graph_format: MATH_PROFILE,
    SQL_PROFILE.graph_format: SQL_PROFILE,
}


def get_task_graph_profile(graph_format: str) -> TaskGraphProfile | None:
    return TASK_GRAPH_PROFILES.get(graph_format)

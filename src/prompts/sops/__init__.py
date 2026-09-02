"""Load task-family SOPs used by specialized StackPlanner graphs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.graph.task_profiles import get_task_graph_profile


@lru_cache(maxsize=None)
def load_task_sop(graph_format: str) -> str | None:
    profile = get_task_graph_profile(graph_format)
    if profile is None:
        return None
    path = Path(__file__).with_name(f"{profile.sop_name}.md")
    if not path.is_file():
        raise FileNotFoundError(
            f"SOP for graph format {graph_format!r} does not exist: {path}"
        )
    sop = path.read_text(encoding="utf-8").strip()
    if not sop:
        raise ValueError(f"SOP for graph format {graph_format!r} is empty")
    return sop

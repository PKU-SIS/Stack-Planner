# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import MessagesState

from src.prompts.planner_model import Plan
from src.rag import Resource
from typing import Any


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Runtime Variables
    locale: str = "en-US"
    observations: list[str] = []
    data_collections:list[Any] = []
    resources: list[Resource] = []
    plan_iterations: int = 0
    current_plan: Plan | str = None
    user_query: str = ""
    final_report: str = ""
    auto_accepted_plan: bool = False
    enable_background_investigation: bool = True
    background_investigation_results: str = None


from langgraph.graph import MessagesState

from src.prompts.planner_model import Plan
from src.rag import Resource
from typing import Any


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Runtime Variables
    locale: str = "zh-CN"
    observations: list[str] = []
    data_collections: list[Any] = []
    resources: list[Resource] = []
    plan_iterations: int = 0
    current_plan: Plan | str = None
    user_query: str = ""
    final_report: str = ""
    replan_result: str = ""
    auto_accepted_plan: bool = False
    enable_background_investigation: bool = True
    background_investigation_results: str = None
    user_dst: str = None
    wait_for_user: bool = False
    report_outline: str = None

    delegation_context: dict = None
    current_node: str = None
    memory_stack: str = None

    # FactStruct Stage 1 相关字段
    factstruct_outline: Any = None  # OutlineNode 根节点
    factstruct_memory: Any = None  # Memory 实例（需要序列化时转换为 dict）

# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

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
    wait_stage: str = ""
    dst_question: str = None
    hitl_feedback: str = ""
    report_outline: str = None

    delegation_context: dict = None
    current_node: str = None
    memory_stack: str = None
    current_style: str = ""  # 当前报告风格，用于风格切换功能
    original_report: str = ""  # 首次生成的报告，用于风格切换时保持引用一致性

    # Human Agent 相关字段
    need_human_interaction: bool = False  # 是否需要人类交互
    human_interaction_type: str = (
        ""  # 人类交互类型: form_filling, outline_confirmation, report_feedback, proactive_question
    )

    # ZB V1.1相关字段
    sop: str = None

    # ZX 新增：段落研究进度追踪
    current_chapter_index: int = 0  # 当前研究到哪个段落（从0开始）

    # ZX 🆕 修改：将 chapter_research_status 改为 chapter_stage_status，含义更清晰
    # 记录每章当前阶段："" | "researched" | "reported"
    # 示例：{"1": "reported", "2": "researched", "3": "", ...}
    chapter_stage_status: dict = {}

    # ZX 🆕 新增：存储每章生成的报告内容
    # 示例：{"1": "第1章的完整内容...", "2": "第2章的完整内容...", ...}
    chapter_reports: dict = {}

    # ZX 🆕 新增：报告生成模式
    # "per_chapter": 每章独立生成报告，最后合并（原有流程）
    # "cumulative_observations": 每章累积 observations，最后一次性生成报告（新流程）
    report_mode: str = "cumulative_observations"  # "per_chapter" | "cumulative_observations"

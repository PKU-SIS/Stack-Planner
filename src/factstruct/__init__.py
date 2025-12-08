"""
FactStruct Stage 1: 动态大纲生成与优化 (Batch-MAB 框架)

本模块实现了基于批量-信息觅食多臂老虎机（Batch-IF-MAB）的动态大纲生成与优化系统。
"""

from .outline_node import OutlineNode
from .document import FactStructDocument
from .memory import Memory
from .reward_calculator import RewardCalculator
from .batch_mab import BatchMAB
from .embedder import Embedder
from .llm_wrapper import FactStructLLMWrapper
from .integration import (
    run_factstruct_stage1,
    run_factstruct_stage2,
    create_search_engine_adapter,
    outline_node_to_text,
    outline_node_to_markdown,
    outline_node_to_json,
    memory_to_dict,
    outline_node_to_dict,
    dict_to_outline_node,
    dict_to_memory,
)

__all__ = [
    "OutlineNode",
    "FactStructDocument",
    "Memory",
    "RewardCalculator",
    "BatchMAB",
    "Embedder",
    "FactStructLLMWrapper",
    "run_factstruct_stage1",
    "run_factstruct_stage2",
    "create_search_engine_adapter",
    "outline_node_to_text",
    "outline_node_to_markdown",
    "outline_node_to_json",
    "memory_to_dict",
    "outline_node_to_dict",
    "dict_to_outline_node",
    "dict_to_memory",
]

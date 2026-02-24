import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Type, Union, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.agents.sub_agent_registry import get_sub_agents_by_global_type
from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.memory import MemoryStack, MemoryStackEntry
from src.prompts.template import apply_prompt_template, get_prompt_template
from src.utils.json_utils import repair_json_output
from src.utils.logger import logger
from src.utils.statistics import global_statistics
from src.utils.reference_utils import global_reference_map
from ..graph.types import State
from src.factstruct import (
    FactStructLLMWrapper,
    BatchMAB,
    OutlineNode,
    create_search_engine_adapter,
    Embedder,
    outline_node_to_dict,
    memory_to_dict,
    Memory,
    outline_node_to_markdown,
)
import re
from src.memory import (
    MemoryStack,
    MemoryStackEntry,
)  # 加一个 Memory 吧，要不 decision 做不了

from src.utils.statistics import global_statistics, timed_step


# -------------------------
# 核心枚举定义
# -------------------------
class OutlineTool(Enum):
    INITIALIZATION = "initialization"
    EXPANDATION = "expandation"
    REDUCTION = "compression"
    UPDATE = "update"
    FINISH = "finish"


# -------------------------
# 中枢Agent核心模块--中枢Agent的action
# exp: 与prompt/Outline_decision.py中的Decision类不同的是，那里是字符串类型，这里是枚举类型，所以要定义两次
# -------------------------
@dataclass
class OutlineToolDecision:
    """Outline Agent 的决策结果"""

    tool: OutlineTool  # 使用的工具
    reasoning: str  # 为什么这么做
    params: Optional[Dict[str, Any]]  # 工具参数（各 tool 自己解释）


# 不知道为啥要两个，那就实现成两个吧
from pydantic import BaseModel


class OutlineToolDecision_Base(BaseModel):
    tool: Literal[
        "initialization",
        "expandation",
        "compression",
        "update",
        "finish",
    ]
    reasoning: str
    params: Optional[Dict[str, Any]] = None


class OutlineAgent:
    """
    大纲Agent核心类，负责大纲决策与任务编排
    """

    def __init__(
        self,
        initial_query: str,
        central_guidance: str | None = None,
        state: State | None = None,
        llm=None,  # 这三个就是 None
        search_engine=None,
        embedder=None,
        max_trys: int = 5,  # 这两个是默认参数
    ):
        # 处理上游输入信息
        # --- Core task signal ---
        self.initial_query = initial_query
        # --- High-level planning signals ---
        self.central_guidance = central_guidance
        logger.info(f"self.central_guidance是否正确存储{self.central_guidance}")
        self.replan_result = state.get("replan_result")
        self.total_word_limit = state.get("total_word_limit", 5000)

        # 处理当前状态，这玩意是不是应该放到 decision 里面也要啊
        # --- Current outline state ---
        self.factstruct_outline = state.get("factstruct_outline")
        self.factstruct_memory = state.get("factstruct_memory")
        # if self.factstruct_memory==None:#初始化一下，后面再考虑重复调用，先跑起来再说,把 init 改为调用 batchmabMemroy 封装到里面
        #     self.factstruct_memory = Memory(embedding_dim=embedder.get_embedding_dim())

        self.outline_feedback = state.get("outline_feedback")
        self.max_trys = max_trys

        # === Search Engine ===
        self.search_engine = search_engine or create_search_engine_adapter()

        # === Embedder（重资源，只初始化一次）===
        self.embedder = embedder or Embedder(
            model_name="../../Model/MiniLM/all-MiniLM-L6-v2"
        )

        # === LLM ===
        if llm is None:
            llm_type = AGENT_LLM_MAP.get("outline", "basic")
            self.llm = get_llm_by_type(llm_type)
        else:
            self.llm = llm

        self.llm_wrapper = FactStructLLMWrapper(self.llm)

        # --- Batch MAB（核心）---
        self.batch_mab = BatchMAB(
            llm_wrapper=self.llm_wrapper,
            embedder=self.embedder,
            search_engine=self.search_engine,
            max_iterations=4,
            memory=self.factstruct_memory,
            batch_size=2,
        )

        # 记录做了啥的 Memroy stack,memory stack只能给中枢智能体用，感觉不太行。
        self.memory_stack = []

        # --- Tool handlers ---
        self.tool_handlers = {
            "initialization": self._tool_initialization,
            "expandation": self._tool_expansion,
            "compression": self._tool_compress,  # 新增
            "update": self._tool_update,  # 新增
            "finish": self._tool_finish,
        }

    async def execute(self, state: State, config: RunnableConfig) -> Command:
        """
        OutlineAgent 的主循环：
        decision → tool → state update → until finish
        """
        logger.info("OutlineAgent 执行开始")
        # if self.factstruct_memory:
        #     logger.info(f"self.factstruct_memory.documents{self.factstruct_memory.documents}")
        # 万一有报错信息
        last_decision = None
        last_error = None

        for step in range(self.max_trys):
            logger.info(
                f"OutlineAgent Step {step + 1}/{self.max_trys}"
            )  # 一共迭代了多少步

            try:
                # === 1. 决策 ===
                decision = self.make_decision(state, config)
                last_decision = decision
                self.memory_stack.append(decision)
                logger.info(
                    f"OutlineAgent Decision: {decision.tool} | {decision.reasoning}"
                )

                # === 2. 执行工具 ===
                # command = await self.execute_tool(decision, state, config)
                command = self.execute_tool(decision, state, config)

                # === 3. 合并 state ===
                if command and command.update:
                    state.update(command.update)

                # === 4. 是否完成 ===
                if decision.tool == "finish":
                    logger.info("OutlineAgent 收到 finish 指令，退出循环")
                    break

            except Exception as e:
                import traceback

                logger.error("OutlineAgent 执行异常")
                logger.error(traceback.format_exc())
                last_error = str(e)
                break

        # =========================
        # === 统一结果整理阶段 ===
        # =========================
        factstruct_outline = state.get("factstruct_outline")
        factstruct_memory = state.get("factstruct_memory")
        report_outline = state.get("report_outline")
        if not report_outline:  # 没有大纲的情况下要反馈
            report_outline = "大纲未完成生成（OutlineAgent 未正常 finish）"
            state["report_outline"] = report_outline

        return Command(
            update={
                "factstruct_outline": state.get("factstruct_outline"),
                "factstruct_memory": state.get("factstruct_memory"),
                "report_outline": state.get("report_outline"),
            }
        )

    def make_decision(
        self, state: State, config: RunnableConfig, retry_count: int = 0
    ) -> OutlineToolDecision:
        """
        中枢Agent决策核心逻辑，分析当前状态生成决策结果

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            决策结果对象
        """
        max_retries = 3
        logger.info("Outline Agent进行决策...")
        start_time = datetime.now()
        # 整理当前状态
        decision_state = self._compute_decision_state(state)

        # 构建决策prompt
        messages = self._build_decision_prompt(state, config, decision_state)
        # messages = self._build_decision_prompt(state, config)
        logger.debug(f"outline 决策prompt: {messages}")

        try:
            llm = get_llm_by_type(
                AGENT_LLM_MAP.get("outline", "default")
            ).with_structured_output(
                OutlineToolDecision_Base,  # ✅ 给 LLM 用的 schema
                method="json_mode",
            )

            response: OutlineToolDecision_Base = llm.invoke(messages)

            logger.info(f"Outline 决策结果(raw): {response}")

            # ✅ 从 LLM 协议对象 → 系统内部对象
            decision = OutlineToolDecision(
                tool=response.tool,
                reasoning=response.reasoning,
                params=response.params,
            )

            end_time = datetime.now()
            global_statistics.add_time_entry(
                {
                    "step_name": "outline_decision",
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                }
            )

            return decision

        except Exception as e:
            import traceback

            logger.error(
                f"Outline 决策解析失败 (尝试 {retry_count + 1}/{max_retries}): {e}"
            )
            logger.error(traceback.format_exc())

            if retry_count < max_retries - 1:
                return self.make_decision(state, config, retry_count + 1)

            # 🚨 兜底：强制 finish，防止系统卡死
            return OutlineToolDecision(
                tool="finish",
                reasoning="Outline decision parsing failed repeatedly, forcing termination.",
                params=None,
            )

    def _compute_decision_state(self, state: State) -> Dict[str, Any]:
        outline = state.get("factstruct_outline")
        outline_exists = outline is not None

        # ---------- 基础结构信息 ----------
        if outline_exists:
            if isinstance(outline, dict):
                outline = dict_to_outline_node(outline)
            leaf_nodes = outline.get_leaf_nodes()
            leaf_count = len(leaf_nodes)
            max_depth = max(node.get_depth() for node in outline.get_all_nodes())
            total_planned_words = getattr(outline, "word_limit", 0)
        else:
            leaf_nodes = []
            leaf_count = 0
            max_depth = 0
            total_planned_words = 0

        total_word_limit = state.get("total_word_limit", 5000)

        has_expandation = False
        for e in self.memory_stack or []:
            tool_name = e.tool.value if hasattr(e.tool, "value") else e.tool
            if tool_name == "expandation":
                has_expandation = True
                break

        # ---------- 字数分布分析 ----------
        small_nodes = []
        large_nodes = []
        logger.info(f"leaf_nodes:{leaf_nodes}")
        for node in leaf_nodes:
            wc = getattr(node, "word_limit", 0)
            logger.info(f"node{node}")
            logger.info(f"wc,{wc}")
            if wc < 300:
                small_nodes.append(node)
            elif wc > 600:
                large_nodes.append(node)
        small_ratio = len(small_nodes) / leaf_count if leaf_count > 0 else 0

        # ---------- 文档分布分析 ----------
        # 找出所有缺文献的叶子节点
        uncovered_leaf_nodes = [
            node
            for node in leaf_nodes
            if not getattr(self.factstruct_memory, "node_to_docs", {}).get(node.id)
        ]
        logger.info(f"uncovered_leaf_nodes{uncovered_leaf_nodes}")
        # 叶子节点覆盖率（可选，用于简单统计）
        leaf_coverage_ratio = (
            1 - len(uncovered_leaf_nodes) / len(leaf_nodes) if leaf_nodes else 0
        )
        logger.info(f"leaf_coverage_ratio{leaf_coverage_ratio}")

        # ----------- 决策建议，根据 prompt 规则生成下一步建议 -------------
        if not outline_exists:
            suggestion = (
                f"当前 outline_exists={outline_exists}，尚未生成任何大纲结构，"
                "无法进行字数与结构评估，需要先初始化大纲，应调用 initialization 工具。"
            )

        elif len(large_nodes) > 1:
            suggestion = (
                f"当前 outline_exists={outline_exists}，存在 {len(large_nodes)} 个叶子节点字数超过 600，"
                "说明部分章节负担过重，需要进一步拆分细化章节结构，"
                "应调用 expandation 工具。"
            )

        elif small_ratio >= 1 / 3:
            suggestion = (
                f"当前 outline_exists={outline_exists}，leaf_node_count={leaf_count}，"
                f"其中有 {len(small_nodes)} 个叶子节点字数小于 300（占比约 {small_ratio:.0%}），"
                "说明当前大纲结构过于零散、内容承载不足，整体规划不合理。"
                "应调用 compression 工具，对【同一父节点下】字数 < 300 的部分叶子节点进行合并压缩，"
                "并通过可能的叶子节点列表来控制压缩强度。"
            )

        elif leaf_coverage_ratio < 0.9:
            # 有具体未覆盖的节点 → update
            missing_ids = [node.id for node in uncovered_leaf_nodes]
            suggestion = (
                f"存在 {len(uncovered_leaf_nodes)} 个叶子节点缺文献，"
                f"节点ID={missing_ids}缺少文献，请你选择其中的属于相同父节点的叶子节点，优化这些节点或微调结构。"
                "应调用 update 工具"
            )
        elif (
            total_planned_words >= 0.8 * total_word_limit
            and total_planned_words <= 1.2 * total_word_limit
        ):
            suggestion = (
                f"当前 outline_exists={outline_exists}，叶子节点字数均落在合理区间（300–600），"
                f"且大纲规划总字数为 {total_planned_words}，与目标字数 {total_word_limit} 基本一致，"
                "说明大纲结构和字数规划合理，可以结束大纲构建，应调用 finish 工具。"
            )
        else:
            suggestion = (
                f"当前 outline_exists={outline_exists}，大纲结构已存在，但字数或结构尚未达到最优状态，"
                "需要通过小幅扩展进一步平衡章节粒度与内容覆盖，应调用 expandation 工具。"
            )

        return {
            "outline_exists": outline_exists,
            "max_depth": max_depth,
            "leaf_node_count": leaf_count,
            "estimated_words": total_planned_words,  # ← 真实字数
            "total_word_limit": total_word_limit,
            "uncovered_leaf_nodes": uncovered_leaf_nodes,  # 具体哪些节点没覆盖
            "leaf_coverage_ratio": leaf_coverage_ratio,  # 简单统计，可供参考
            "has_expandation_history": has_expandation,
            "next_step_suggestion": suggestion,
        }

    def _build_decision_prompt(
        self,
        state: State,
        config: RunnableConfig,
        decision_state: dict,
    ) -> List[Union[AIMessage, HumanMessage]]:
        """
        构建 Outline Agent 的决策 prompt
        """

        history_decision = (
            [
                f"工具：{e.tool.value if hasattr(e.tool, 'value') else e.tool} | 推理：{e.reasoning} | 参数：{None if isinstance(e.params, list) and len(e.params) == 0 else e.params}"
                for e in self.memory_stack
            ]
            if self.memory_stack
            else None
        )

        logger.info(f"history_decision记录{history_decision}")

        # 改一下大纲的格式吧
        factstruct_outline = state.get("factstruct_outline", None)
        if factstruct_outline:
            outline_response = factstruct_outline.to_text_tree(include_word_limit=True)
            logger.info(f"是否可以做成功格式转化outline_response{outline_response}")
        else:
            outline_response = None
        context = {
            # 必须项
            "user_query": state.get("initial_query"),
            # 可选项（prompt 里有 if 判断）
            "central_guidance": self.central_guidance,
            "decision_state": decision_state,  # 这个是直接传给了 Prompt
            "factstruct_outline": outline_response,
            "total_word_limit": state.get("total_word_limit", 5000),
            "outline_feedback": state.get("outline_feedback"),
            "history_decision": history_decision,
        }

        # 合并 config（如需要）
        context = {**context, **config}
        logger.info(f"Context:\n{context}")

        return apply_prompt_template(
            "outline_decision",  # ✅ 对应 src/prompts/outline_decision.md
            state,
            extra_context=context,
        )

    def execute_tool(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        根据 Outline Agent 的 tool 决策，调用对应的 outline 工具函数
        """

        tool_name = decision.tool
        handler = self.tool_handlers.get(tool_name)

        if not handler:
            error_msg = f"未知 outline tool: {tool_name}"
            logger.error(error_msg)
            # 这个地方要不要这么跳，还不一定，还需要再看一看
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"错误：未知 outline tool: {tool_name}",
                            name="outline_error",
                        )
                    ],
                    "current_node": "central_agent",
                },
                goto="central_agent",
            )

        logger.info(f"Outline Agent 执行工具: {tool_name}, params={decision.params}")

        return handler(
            decision=decision,
            state=state,
            config=config,
        )

    def _tool_initialization(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        初始化大纲结构（initialization tool）
        """
        logger.info("Outline Tool: initialization")

        initial_query = state["initial_query"]
        central_guidance = self.central_guidance  # state.get("central_guidance")
        # replan_result = state.get("replan_result",None)
        replan_result = None
        factstruct_outline = state.get("factstruct_outline")
        # initial_docs = state.get("initial_docs") #暂时不要
        initial_docs = state.get("data_collections", None)
        # observation_text = state.get("observation", None)
        # initial_docs = None

        # 提取decision当中的参数
        params = decision.params or {}
        instruction = params.get("instruction", None)

        # 已有 outline，不应再次初始化
        if factstruct_outline is not None:
            logger.warning("Outline already exists, skip initialization.")
            return Command(
                update={
                    "outline_feedback": "Initialization skipped: outline already exists."
                },
            )

        try:
            outline_root, memory, initial_docs = self.batch_mab.run_initialization(
                user_query=initial_query,
                central_guidance=central_guidance,
                replan_result=replan_result,
                instruction=instruction,
                initial_docs=initial_docs,
                config=config,
            )
            # logger.info(f"initialization memory.documents{memory.documents}")

            # 字数规划
            total_word_limit = state.get("total_word_limit", 5000)
            if total_word_limit > 0:
                logger.info(f"检测到字数限制 {total_word_limit}，执行字数规划...")
                outline_root = self.execute_word_planning(
                    outline_root, total_word_limit
                )
                # outline_response = outline_root.to_text_tree(

            # --- Step 7: 更新状态并回到 outline agent ---
            return Command(
                update={
                    "factstruct_outline": outline_root,
                    "factstruct_memory": memory,
                    "initial_docs": initial_docs,
                    "outline_feedback": "Outline initialized successfully.",
                },
            )
        except Exception as e:
            import traceback

            logger.error("Outline initialization failed")
            logger.error(traceback.format_exc())

            return Command(
                update={"outline_feedback": f"Outline initialization failed: {str(e)}"}
            )

    def _tool_expansion(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        扩展现有大纲（Batch-MAB 驱动）
        """
        logger.info(f"Outline Tool: expansion | reasoning={decision.reasoning}")

        outline_root = state.get("factstruct_outline")
        memory = state.get("factstruct_memory")
        initial_query = state["initial_query"]
        logger.info(f"用于写代码的case outline_root{outline_root}")
        logger.info(f"用于写代码的case memory{memory}")
        # 错误排查，防止没初始化
        if outline_root is None or memory is None:
            logger.warning("Expansion skipped: outline or memory missing")
            return Command(
                update={
                    "outline_feedback": "Expansion skipped: outline or memory missing."
                }
            )

        # 提取 decision 参数
        params = decision.params or {}
        max_iterations = params.get(
            "max_iterations",
            state.get("factstruct_max_iterations", 4),
        )
        batch_size = params.get(
            "batch_size",
            state.get("factstruct_batch_size", 2),
        )
        logger.info(
            f"Expansion params resolved: max_iterations={max_iterations}, batch_size={batch_size}"
        )

        try:
            # === 调用算法层 ===
            outline_root, memory = self.batch_mab.run_expansion(
                user_query=initial_query,
                outline_root=outline_root,
                memory=memory,
                max_iterations=max_iterations,
                batch_size=batch_size,
                config=config,
            )

            logger.info(
                f"Outline expanded: {len(outline_root.get_all_nodes())} nodes total"
            )

            # 字数规划
            total_word_limit = state.get("total_word_limit", 5000)
            if total_word_limit > 0:
                logger.info(f"检测到字数限制 {total_word_limit}，执行字数规划...")
                outline_root = self.execute_word_planning(
                    outline_root, total_word_limit
                )
                # outline_response = outline_root.to_text_tree(
            # logger.info(f"expansion memory.documents{memory.documents}")

            # 最后返回
            logger.info(
                f"FactStruct Stage 1 完成: "
                f"{len(outline_root.get_all_nodes())} 个节点"
            )

            return Command(
                update={
                    "factstruct_outline": outline_root,
                    "factstruct_memory": memory,
                    "outline_feedback": "Outline expansion completed successfully.",
                }
            )

        except Exception as e:
            import traceback

            logger.error("Outline expansion failed")
            logger.error(traceback.format_exc())

            return Command(
                update={"outline_feedback": f"Outline expansion failed: {str(e)}"}
            )

    def _tool_compress(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        收缩现有大纲（压缩 / 合并节点）
        """
        logger.info(f"Outline Tool: compress | reasoning={decision.reasoning}")

        initial_query = state["initial_query"]
        outline_root = state.get("factstruct_outline")
        memory = state.get("factstruct_memory")

        if outline_root is None or memory is None:
            logger.warning("Compress skipped: outline or memory missing")
            return Command(
                update={
                    "outline_feedback": "Compress skipped: outline or memory missing."
                }
            )

        # 提取 decision 参数
        params = decision.params or {}

        merge_candidates = params.get("merge_candidates", [])
        # max_merges = params.get("max_merges", 1)
        # target_leaf_count = params.get("target_leaf_count",2)

        merge_candidates_raw = merge_candidates
        resolved = []

        if not merge_candidates_raw:
            merge_candidates = resolved

        for item in merge_candidates_raw:
            # 已经是 OutlineNode
            if isinstance(item, OutlineNode):
                resolved.append(item)
                continue

            # ID（int / str）
            node_id = str(item)
            node = outline_root.find_node_by_id(node_id)
            if node:
                resolved.append(node)
            else:
                logger.warning(f"Merge candidate id '{node_id}' not found in outline")

        merge_candidates = resolved

        logger.info(
            "Compress params resolved: "
            f"merge_candidates={len(merge_candidates)}, "
            # f"max_merges={max_merges}, "
            # f"target_leaf_count={target_leaf_count}"
        )

        try:
            # 调用 batch_mab 压缩算法（后续实现）
            outline_root, memory = self.batch_mab.run_compression(
                user_query=initial_query,
                outline_root=outline_root,
                memory=memory,
                merge_candidates=merge_candidates,
                # max_merges=max_merges,
                # target_leaf_count=target_leaf_count,
                config=config,
            )

            # 字数规划（可选）
            total_word_limit = state.get("total_word_limit", 5000)
            if total_word_limit > 0:
                outline_root = self.execute_word_planning(
                    outline_root, total_word_limit
                )

            logger.info(
                f"Outline compressed: {len(outline_root.get_all_nodes())} nodes total"
            )

            return Command(
                update={
                    "factstruct_outline": outline_root,
                    "factstruct_memory": memory,
                    "outline_feedback": "Outline compression completed successfully.",
                }
            )

        except Exception as e:
            import traceback

            logger.error("Outline compression failed")
            logger.error(traceback.format_exc())
            return Command(
                update={"outline_feedback": f"Outline compression failed: {str(e)}"}
            )

    def _tool_update(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        更新 / 微调现有大纲（等价变换或添加文献覆盖）
        """
        logger.info(f"Outline Tool: update | reasoning={decision.reasoning}")
        initial_query = state["initial_query"]
        outline_root = state.get("factstruct_outline")
        memory = state.get("factstruct_memory")

        if outline_root is None or memory is None:
            logger.warning("Update skipped: outline or memory missing")
            return Command(
                update={
                    "outline_feedback": "Update skipped: outline or memory missing."
                }
            )

        # 提取 decision 参数（可选）
        params = decision.params or {}
        # max_iterations = params.get("max_iterations", state.get("factstruct_max_iterations", 2))
        # batch_size = params.get("batch_size", state.get("factstruct_batch_size", 2))
        # uncovered_leaf_nodes = params.get("uncovered_leaf_nodes", [])  # 需要微调的叶子节点
        # logger.info(f"Update params resolved: max_iterations={max_iterations}, batch_size={batch_size}, uncovered_leaf_nodes={uncovered_leaf_nodes}")
        update_candidates = params.get("update_candidates", "无指令")

        # ================================
        # 0️⃣ 解析 update_candidates
        # ================================

        resolved = []

        for item in update_candidates or []:
            # 已经是 OutlineNode
            if isinstance(item, OutlineNode):
                resolved.append(item)
                continue

            # ID（int / str）
            node_id = str(item)
            node = outline_root.find_node_by_id(node_id)

            if node:
                resolved.append(node)
            else:
                logger.warning(f"Update candidate id '{node_id}' not found in outline")

        update_candidates = resolved

        logger.info(
            f"Update params resolved: update_candidates={len(update_candidates)}"
        )

        try:
            # 调用 batch_mab 更新算法（后续实现）
            outline_root, memory = self.batch_mab.run_update(
                user_query=initial_query,
                outline_root=outline_root,
                memory=memory,
                update_candidates=update_candidates,
                config=config,
            )

            # 字数规划（可选）
            total_word_limit = state.get("total_word_limit", 5000)
            if total_word_limit > 0:
                outline_root = self.execute_word_planning(
                    outline_root, total_word_limit
                )

            logger.info(
                f"Outline updated: {len(outline_root.get_all_nodes())} nodes total"
            )

            return Command(
                update={
                    "factstruct_outline": outline_root,
                    "factstruct_memory": memory,
                    "outline_feedback": "Outline update completed successfully.",
                }
            )

        except Exception as e:
            import traceback

            logger.error("Outline update failed")
            logger.error(traceback.format_exc())
            return Command(
                update={"outline_feedback": f"Outline update failed: {str(e)}"}
            )

    def _tool_finish(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        完成 OutlineAgent 执行，生成总结性反馈返回给 CentralAgent
        """
        logger.info(f"Outline Tool: finish | reasoning={decision.reasoning}")

        outline_root = state.get("factstruct_outline")
        memory = state.get("factstruct_memory")
        total_word_limit = state.get("total_word_limit", 5000)

        # === 兜底处理 ===
        if outline_root is None:
            logger.warning("Finish called but outline is missing")

            return Command(
                update={
                    "report_outline": "大纲未成功生成（OutlineAgent 在 finish 前缺失 outline）",
                    "outline_feedback": "Finish reached without valid outline.",
                }
            )

        # === 基本结构信息 ===
        node_count = len(outline_root.get_all_nodes())
        leaf_count = len(outline_root.get_leaf_nodes())

        # === 可读大纲输出（给 Central / Reporter 用）===
        try:
            outline_text = outline_root.to_text_tree(
                include_word_limit=bool(total_word_limit)
            )
        except Exception:
            outline_text = outline_root.to_text_tree()

        # === 构建给 LLM 的总结 prompt（可选，但很有价值）===
        try:
            llm = get_llm_by_type(AGENT_LLM_MAP.get("outline", "default"))

            context_lines = []

            if state.get("initial_query"):
                context_lines.append(f"用户原始问题：{state.get('initial_query')}")

            if self.central_guidance:
                context_lines.append(f"中枢策略指导：{self.central_guidance}")

            if total_word_limit:
                context_lines.append(f"目标总字数限制：{total_word_limit}")

            context_block = (
                "\n".join(context_lines) if context_lines else "（无额外上下文）"
            )

            summary_prompt = [
                {
                    "role": "system",
                    "content": (
                        "你是一个研究大纲评估助手，"
                        "你的任务不是生成内容，而是判断当前大纲是否已经达到"
                        "可以进入正式内容生成阶段的结构成熟度。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context_block}\n\n"
                        f"=== 当前大纲结构指标 ===\n"
                        f"- 节点总数: {node_count}\n"
                        f"- 叶子节点数: {leaf_count}\n\n"
                        f"=== 当前大纲 ===\n"
                        f"{outline_text}\n\n"
                        "请基于【用户问题】和【中枢策略指导】，判断该大纲：\n"
                        "1. 是否覆盖了用户问题的主要方面\n"
                        "2. 结构层级是否清晰、粒度是否合适\n"
                        "3. 是否适合在当前字数限制下展开正文\n\n"
                        "请给出一句简洁的总结性判断（不超过 2 句话），"
                        "用于上游 Agent 的决策参考。"
                    ),
                },
            ]

            logger.info(f"outline agent 的 finish 的 prompt:{summary_prompt}")
            llm_response = llm.invoke(summary_prompt)
            finish_summary = llm_response.content.strip()

        except Exception as e:
            logger.warning(f"Finish summary LLM failed: {e}")
            finish_summary = "大纲结构已生成，节点层级完整，可进入内容生成阶段。"

        # === 写回 state（这是 CentralAgent 最关心的部分）===
        return Command(
            update={
                "report_outline": outline_text,  # ✅ 给 reporter / central
                "outline_feedback": finish_summary,  # ✅ 决策级总结
            }
        )

    @timed_step("execute_word_planning")
    def execute_word_planning(
        self, outline_root: OutlineNode, total_word_limit: int
    ) -> OutlineNode:
        """
        执行字数规划，为大纲中的每个叶子节点分配字数配额

        Args:
            outline_root: 大纲根节点
            total_word_limit: 用户指定的总字数限制

        Returns:
            更新了字数配额的大纲根节点
        """
        import json

        logger.info(f"开始字数规划，总字数限制: {total_word_limit}")

        # 构建大纲结构信息供LLM分析
        def build_outline_info(node: OutlineNode, depth: int = 0) -> list:
            nodes_info = []
            nodes_info.append(
                {
                    "id": node.id,
                    "title": node.title,
                    "depth": depth,
                    "is_leaf": node.is_leaf(),
                }
            )
            for child in node.children:
                nodes_info.extend(build_outline_info(child, depth + 1))
            return nodes_info

        outline_info = build_outline_info(outline_root)
        leaf_nodes = [n for n in outline_info if n["is_leaf"]]

        # 构建LLM请求
        outline_text = outline_root.to_text_tree()
        prompt_content = f"""请为以下报告大纲分配字数。

        ## 大纲结构
        {outline_text}

        ## 叶子节点列表
        {json.dumps(leaf_nodes, ensure_ascii=False, indent=2)}

        ## 总字数限制
        {total_word_limit} 字

        请根据每个叶子节点的重要性和内容复杂度，智能分配字数配额。
        你必须只输出一个合法的 JSON 对象。禁止输出任何解释、说明、注释、标题或额外文本。如果输出包含非 JSON 内容，将被视为错误。
        """

        try:
            messages = apply_prompt_template("word_planner", {"messages": []}) + [
                HumanMessage(content=prompt_content)
            ]
            llm = get_llm_by_type(AGENT_LLM_MAP.get("outline", "default"))
            response = llm.invoke(messages)
            result = response.content

            # 解析JSON结果
            logger.info(f"result:{result}")
            # result = result.replace("```json", "").replace("```", "").strip()

            match = re.search(r"\{[\s\S]*\}", result)
            if not match:
                raise ValueError("No JSON object found in LLM output")

            allocations = json.loads(match.group(0))

            # 将字数配额写入节点
            for alloc in allocations.get("allocations", []):
                node_id = alloc.get("node_id")
                word_limit = alloc.get("word_limit", 0)
                node = outline_root.find_node_by_id(node_id)
                if node:
                    node.word_limit = word_limit
                    logger.debug(
                        f"节点 {node_id} ({node.title}) 分配字数: {word_limit}"
                    )

            # 自底向上计算非叶子节点的字数
            def update_parent_word_limits(node: OutlineNode) -> int:
                if node.is_leaf():
                    return node.word_limit
                total = sum(update_parent_word_limits(child) for child in node.children)
                node.word_limit = total
                return total

            update_parent_word_limits(outline_root)
            logger.info(f"字数规划完成，根节点总字数: {outline_root.word_limit}")

        except Exception as e:
            logger.error(f"字数规划失败: {str(e)}")
            # Fallback: 平均分配
            leaf_nodes_obj = outline_root.get_leaf_nodes()
            avg_words = total_word_limit // len(leaf_nodes_obj) if leaf_nodes_obj else 0
            for node in leaf_nodes_obj:
                node.word_limit = avg_words
            logger.warning(f"使用平均分配策略，每个叶子节点: {avg_words} 字")

        logger.info(f"outline_root:{outline_root}")
        # exit()
        return outline_root

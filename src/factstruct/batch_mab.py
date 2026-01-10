"""
Batch-MAB: 批量-信息觅食多臂老虎机主算法

实现了 Stage 1 的核心算法：动态大纲生成与优化。
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import logger
from .outline_node import OutlineNode
from .document import FactStructDocument
from .memory import Memory
from .embedder import Embedder
from .reward_calculator import RewardCalculator
from .llm_wrapper import FactStructLLMWrapper
from langchain_core.runnables import RunnableConfig

class BatchMAB:
    """
    批量-信息觅食多臂老虎机（Batch-IF-MAB）算法

    核心思想：
        1. 使用 UCB1 策略选择 Top-K 个最值得探索的叶子节点
        2. 批量生成查询、批量检索、批量计算奖励
        3. 批量修纲（单次 LLM 调用）

    优势：
        - 相比原 IF-MAB 方案，大幅降低 LLM 调用次数
        - 保持自适应探索策略
    """

    def __init__(
        self,
        llm_wrapper: FactStructLLMWrapper,
        embedder: Embedder,
        search_engine: Callable[[str, int], List[FactStructDocument]],
        reward_calculator: Optional[RewardCalculator] = None,
        memory: Optional[Memory] = None,
        max_iterations: int = 20,
        batch_size: int = 5,
        reward_weights: Tuple[float, float] = (0.7, 0.3),  # (w_rel, w_nov)
    ):
        """
        初始化 Batch-MAB

        参数:
            llm_wrapper: LLM 方法包装器
            embedder: 文档嵌入生成器
            search_engine: 搜索引擎函数，签名 (query: str, k: int) -> List[FactStructDocument]
            reward_calculator: 奖励计算器（可选，默认创建）
            memory: 记忆模块（可选，默认创建）
            max_iterations: 最大迭代次数 T_max（默认 20）
            batch_size: 批量大小 K（默认 5）
            reward_weights: 奖励权重 (w_rel, w_nov)（默认 (0.7, 0.3)）
        """
        self.llm_wrapper = llm_wrapper
        self.embedder = embedder
        self.search_engine = search_engine
        self.max_iterations = max_iterations
        self.batch_size = batch_size

        # 初始化奖励计算器
        if reward_calculator is None:
            self.reward_calculator = RewardCalculator(
                w_rel=reward_weights[0],
                w_nov=reward_weights[1],
            )
        else:
            self.reward_calculator = reward_calculator

        # 初始化记忆模块
        if memory is None:
            self.memory = Memory(embedding_dim=embedder.get_embedding_dim())
        else:
            self.memory = memory

        logger.info(
            f"Initialized BatchMAB: max_iterations={max_iterations}, "
            f"batch_size={batch_size}, embedding_dim={embedder.get_embedding_dim()}"
        )

    def run(
        self,
        initial_query: str,
        initial_docs: Optional[List[FactStructDocument]] = None,
        config:RunnableConfig=None,
    ) -> Tuple[OutlineNode, Memory]:
        """
        运行 Batch-MAB 算法

        参数:
            initial_query: 初始查询
            initial_docs: 初始文档列表（可选，如果不提供则自动检索）

        返回:
            (outline_root, memory): 最终大纲根节点和记忆模块
        """
        logger.info(f"Starting Batch-MAB with query: {initial_query}")

        # --- 初始化阶段 ---
        # 1. 初始检索与大纲生成
        if initial_docs is None:
            logger.info("Performing initial search...")
            initial_docs = self.search_engine(initial_query, k=5,config=config)

        # 嵌入初始文档
        initial_docs_with_embed = self.embedder.embed_docs(initial_docs)

        # 存储到记忆库
        self.memory.store_docs(initial_docs_with_embed)

        # 生成初始大纲（LLM Call #1）
        logger.info("Generating initial outline...")
        outline_root = self.llm_wrapper.generate_initial_outline(
            initial_query,
            initial_docs_with_embed,
        )

        # 将初始文档映射到根节点
        self.memory.map_node_to_docs(outline_root.id, initial_docs_with_embed)

        # 打印初始大纲
        try:
            from .integration import outline_node_to_markdown

            initial_outline_markdown = outline_node_to_markdown(
                outline_root, max_depth=None, include_root=True
            )
            logger.info(f"Initial outline:\n{initial_outline_markdown}")
        except Exception as e:
            logger.warning(f"Failed to format initial outline for logging: {e}")
            logger.info(f"Initial outline:\n{outline_root.to_text_tree()}")

        # 总迭代次数计数器
        t = 0

        # --- 迭代循环（Batch-MAB 过程）---
        num_rounds = math.ceil(self.max_iterations / self.batch_size)
        logger.info(
            f"Starting {num_rounds} rounds of batch optimization "
            f"(max_iterations={self.max_iterations}, batch_size={self.batch_size}, "
            f"total iterations={self.max_iterations})"
        )

        for round_num in range(num_rounds):
            logger.info(f"--- Round {round_num + 1}/{num_rounds} ---")

            # 1. 获取当前所有可行动的"手臂"（叶子节点）
            current_leaf_nodes = outline_root.get_leaf_nodes()

            if not current_leaf_nodes:
                logger.info("No leaf nodes available, terminating early.")
                break

            logger.info(f"Found {len(current_leaf_nodes)} leaf nodes")

            # 2. UCB 策略选择 Top-K 手臂
            selected_nodes = self._select_top_k_nodes(current_leaf_nodes, t)

            if not selected_nodes:
                logger.info("No nodes selected, terminating.")
                break

            logger.info(
                f"Selected {len(selected_nodes)} nodes: "
                f"{[node.title for node in selected_nodes]}"
            )

            # 3. "批量拉动摇臂"（执行检索）
            # (LLM Call #Round*2)
            logger.info("Batch generating queries...")
            queries = self.llm_wrapper.batch_generate_queries(selected_nodes)

            # 并行执行检索（按照 proposal 要求实现真正的并行检索）
            logger.info(f"Performing parallel search for {len(queries)} queries...")
            new_docs_list = self._parallel_search(queries, k=3,config=config)

            # 预处理新文档（嵌入）
            new_docs_list_with_embed = []
            for docs in new_docs_list:
                docs_with_embed = self.embedder.embed_docs(docs)
                new_docs_list_with_embed.append(docs_with_embed)

            # 4. 批量计算并记录"奖励"
            all_embeddings = self.memory.get_all_doc_embeddings()
            node_doc_pairs_for_refine = []

            for i, node in enumerate(selected_nodes):
                t += 1  # 增加全局迭代计数器
                new_docs = new_docs_list_with_embed[i]

                # 计算奖励
                # 生成节点嵌入：使用节点标题和父节点上下文
                node_text = node.title
                parent_context = node.get_parent_context()
                if parent_context:
                    # 如果有父节点上下文，将其包含在节点文本中
                    node_text = f"{parent_context} > {node.title}"

                # 使用 embedder 生成节点嵌入
                node_embedding = self.embedder.embed_text(node_text)

                reward, breakdown = self.reward_calculator.calculate_reward(
                    new_docs=new_docs,
                    node_title=node.title,
                    all_doc_embeddings=all_embeddings,
                    node_embedding=node_embedding,
                )

                logger.info(
                    f"Node '{node.title}': reward={reward:.4f} "
                    f"(rel={breakdown['relevance']:.4f}, nov={breakdown['novelty']:.4f})"
                )

                # 更新 MAB 状态
                node.reward_history.append(reward)
                node.pull_count += 1

                # 准备用于 LLM 修订的数据
                node_doc_pairs_for_refine.append((node, new_docs))

                # 5. 更新记忆库
                self.memory.store_docs(new_docs)
                self.memory.map_node_to_docs(node.id, new_docs)

            # 6. (关键) LLM 批量更新大纲
            # (LLM Call #Round*2 + 1)
            if node_doc_pairs_for_refine:
                logger.info("Batch refining outline...")
                outline_root, expanded_nodes_list, new_node_doc_mapping = (
                    self.llm_wrapper.batch_refine_outline(
                        outline_root,
                        node_doc_pairs_for_refine,
                        memory=self.memory,  # 传递 memory 以获取累积文档
                    )
                )

                # --- (已修正) 关键的状态继承步骤 ---
                # 遍历那些刚刚被扩展的节点 (从叶子节点变成了内部节点)
                # expanded_nodes_list 格式: [(parent_node, [new_child_node_1, ...]), ...]
                for parent_node, new_children in expanded_nodes_list:
                    if not new_children:
                        continue

                    # 将父节点的 MAB 状态复制给所有新生成的子节点
                    for child in new_children:
                        child.pull_count = parent_node.pull_count
                        # 必须创建 reward_history 的副本
                        child.reward_history = list(parent_node.reward_history)

                    logger.info(
                        f"MAB state inherited: parent '{parent_node.title}' "
                        f"(pull_count={parent_node.pull_count}, "
                        f"rewards={len(parent_node.reward_history)}) -> "
                        f"{len(new_children)} children"
                    )
                # --- 状态继承结束 ---

                # --- 新增：为新子节点存储引文映射 ---
                # new_node_doc_mapping 格式: {node_id: [doc1, doc2, ...]}
                for node_id, docs in new_node_doc_mapping.items():
                    self.memory.map_node_to_docs(node_id, docs)
                    logger.debug(
                        f"New child node '{node_id}' mapped to {len(docs)} documents"
                    )

                if new_node_doc_mapping:
                    logger.info(
                        f"Citation mapping stored for {len(new_node_doc_mapping)} new child nodes"
                    )
                # --- 引文映射存储结束 ---

                logger.info(
                    f"Outline refined: {len(outline_root.get_all_nodes())} total nodes, "
                    f"{len(outline_root.get_leaf_nodes())} leaf nodes"
                )

            # 打印当前大纲（每次迭代结束）
            try:
                # 延迟导入避免循环导入
                from .integration import outline_node_to_markdown

                current_outline_markdown = outline_node_to_markdown(
                    outline_root, max_depth=None, include_root=True
                )
                logger.info(
                    f"Current outline after round {round_num + 1}:\n{current_outline_markdown}"
                )
            except Exception as e:
                logger.warning(f"Failed to format outline for logging: {e}")
                # 降级方案：使用简单的文本树
                logger.info(
                    f"Current outline after round {round_num + 1}:\n{outline_root.to_text_tree()}"
                )

            # 检查是否达到最大迭代次数
            if t >= self.max_iterations:
                logger.info(
                    f"Reached max iterations ({self.max_iterations}), terminating."
                )
                break

        logger.info(
            f"Batch-MAB completed: {t} iterations, "
            f"{len(outline_root.get_all_nodes())} nodes in final outline"
        )

        # 可视化大纲树与引文映射关系
        try:
            from .integration import visualize_outline_with_citations

            visualize_outline_with_citations(
                outline_root,
                self.memory,
                output_path="outline_with_citations",  # 生成 outline_with_citations.png
                print_text=True,
            )
        except Exception as e:
            logger.warning(f"无法可视化大纲树: {e}")

        return outline_root, self.memory

    def _select_top_k_nodes(
        self,
        leaf_nodes: List[OutlineNode],
        current_t: int,
    ) -> List[OutlineNode]:
        """
        使用 UCB1 策略选择 Top-K 节点

        参数:
            leaf_nodes: 所有叶子节点
            current_t: 当前总迭代次数

        返回:
            选中的 Top-K 节点列表
        """
        ucb_scores = []

        for node in leaf_nodes:
            t_current = current_t + 1  # UCB 公式中的 t

            if node.pull_count == 0:
                # 未探索过的节点，给予无限大的探索奖励
                ucb_score = float("inf")
            else:
                # UCB1 公式
                avg_reward = node.avg_reward()
                exploration_bonus = math.sqrt(2 * math.log(t_current) / node.pull_count)
                ucb_score = avg_reward + exploration_bonus

            ucb_scores.append((ucb_score, node))

        # 排序并选出 Top-K
        ucb_scores.sort(key=lambda x: x[0], reverse=True)
        selected = [node for score, node in ucb_scores[: self.batch_size]]

        return selected

    def _parallel_search(
        self,
        queries: List[str],
        k: int = 3,
        config:RunnableConfig=None,
    ) -> List[List[FactStructDocument]]:
        """
        并行执行检索

        参数:
            queries: 查询列表
            k: 每个查询返回的文档数量

        返回:
            文档列表的列表，与 queries 一一对应
        """
        if not queries:
            return []

        # 使用线程池执行并行检索
        # 检索通常是 I/O 密集型任务，使用 ThreadPoolExecutor 比较合适
        results = [None] * len(queries)  # 预分配结果列表，保持顺序

        with ThreadPoolExecutor(
            max_workers=min(len(queries), self.batch_size)
        ) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.search_engine, query, k,config): i
                for i, query in enumerate(queries)
            }

            # 收集结果（按完成顺序，但保持原始索引）
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    docs = future.result()
                    results[index] = docs
                except Exception as e:
                    logger.error(
                        f"Search failed for query {index} ({queries[index]}): {e}"
                    )
                    results[index] = []  # 失败时返回空列表

        return results

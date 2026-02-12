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
from ..graph.types import State
import re
import json
import traceback
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
        central_guidance=None,
        replan_result=None,
        factstruct_outline=None,
        factstruct_memory=None,
        config:RunnableConfig=None,
    ) -> Tuple[OutlineNode, Memory]:
        
        # """
        # 运行 Batch-MAB 算法
        # 参数:
        #     initial_query: 初始查询
        #     initial_docs: 初始文档列表（可选，如果不提供则自动检索）
        # 返回:
        #     (outline_root, memory): 最终大纲根节点和记忆模块
        # """
        
        logger.info(f"Starting Batch-MAB with query: {initial_query}")

        #初始化
        if factstruct_outline==None:
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
                central_guidance,#感觉plan_text就是第一次的 feedback
                replan_result=replan_result,
            )
            logger.info(f"outline_root:{outline_root}")
            logger.info(f"outline_root.id:{outline_root.id}")
            # 将初始文档映射到根节点
            self.memory.map_node_to_docs(outline_root.id, initial_docs_with_embed)
            logger.info(f"self.memory:{self.memory}")
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
            logger.info(f"current_leaf_nodes{current_leaf_nodes}")
            if not current_leaf_nodes:
                logger.info("No leaf nodes available, terminating early.")
                break

            logger.info(f"Found {len(current_leaf_nodes)} leaf nodes")

            # 2. UCB 策略选择 Top-K 手臂
            selected_nodes = self._select_top_k_nodes(current_leaf_nodes, t)
            logger.info(f"selected_nodes{selected_nodes}")
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
                logger.info(f"node{node}")
                logger.info(f"new_docs{new_docs}")
                node_doc_pairs_for_refine.append((node, new_docs))

                # 5. 更新记忆库
                self.memory.store_docs(new_docs)
                self.memory.map_node_to_docs(node.id, new_docs)

            # 6. (关键) LLM 批量更新大纲
            # (LLM Call #Round*2 + 1)
            logger.info(f"node_doc_pairs_for_refine{node_doc_pairs_for_refine}")
            logger.info(f"outline_root{outline_root}")
            logger.info(f"self.memory{self.memory}")
            if node_doc_pairs_for_refine:
                logger.info("Batch refining outline...")
                outline_root, expanded_nodes_list, new_node_doc_mapping = (
                    self.llm_wrapper.batch_refine_outline(
                        outline_root,
                        node_doc_pairs_for_refine,
                        memory=self.memory,  # 传递 memory 以获取累积文档
                    )
                )
                logger.info(f"expanded_nodes_list{expanded_nodes_list}")
                
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


    def run_initialization(
        self,
        query: str,
        central_guidance=None,
        replan_result=None,
        instruction=None,
        initial_docs=None,
        k: int = 5,
        config: RunnableConfig=None,
    ):
        """
        初始化 FactStruct：检索 → 向量化 → 存储 → 生成初始大纲
        """
        logger.info(f"Starting Batch-MAB initialization with query: {query}")

        # --- Step 1: 检索 ---
        if not initial_docs:
            logger.info("Performing initial search...")
            initial_docs = self.search_engine(query, k=k, config=config)
        else:
            logger.info("Use existing")
            if not all(isinstance(doc, FactStructDocument) for doc in initial_docs):
                logger.info(f"initial_docs before:{initial_docs}")
                logger.info(f"Type of initial_docs: {type(initial_docs)}")
                initial_docs = self.wrap_raw_docs_to_factstruct(initial_docs)
                logger.info(f"initial_docs after:{initial_docs}")


        # --- Step 2: 向量化 ---
        initial_docs_with_embed = self.embedder.embed_docs(initial_docs)

        # --- Step 3: 存入 memory ---
        self.memory.store_docs(initial_docs_with_embed)

        # --- Step 4: 生成初始大纲 ---
        logger.info("Generating initial outline...")
        outline_root = self.llm_wrapper.generate_initial_outline(
            query=query,
            docs=initial_docs_with_embed,
            central_guidance=central_guidance,
            replan_result=replan_result,
            instruction=instruction,
        )

        logger.info(f"Generated outline root id: {outline_root.id}")

        # --- Step 5: 文档绑定 ---
        self.memory.map_node_to_docs(outline_root.id, initial_docs_with_embed)
        logger.info(f"self.memory.node_to_docs{self.memory.node_to_docs}")
        
        #感觉这个地方需要思考一下，初始化的文档是归谁的，现在是放到了 Root 上
        #我觉得确实不能给子节点，否则文档覆盖率这个东西就不太行了
        #初始化的节点是否需要给上，奖励如果不给奖励的话，一开始的大纲扩展就只会在第一层进行扩展
        #那这么说，finish 是不是要多一个 保证大家的reward 都不是零才好结束。
        #expansion的 reward 更新包括两个地方，一个是检索文档的更新，一个是子节点的生成
        #compassion的 reward的更新包括两个地方，一个是select node需要计算 reward但是不更新。 另一个是新的节点的生成，这个应该是用parent 的 node 就可以了
        #updat 的 reward的更新包括两个地方，一个是
        return outline_root, self.memory, initial_docs



    def run_expansion(
        self,
        outline_root,
        memory,
        max_iterations: int,
        batch_size: int,
        config: RunnableConfig,
    ):
        """
        执行 Batch-MAB 驱动的大纲扩展
        """
       # 总迭代次数计数器
        t = 0
        self.max_iterations=max_iterations
        self.batch_size=batch_size
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
            logger.info(f"current_leaf_nodes{current_leaf_nodes}")
            if not current_leaf_nodes:
                logger.info("No leaf nodes available, terminating early.")
                break

            logger.info(f"Found {len(current_leaf_nodes)} leaf nodes")

            # 2. UCB 策略选择 Top-K 手臂
            selected_nodes = self._select_top_k_nodes(current_leaf_nodes, t)
            logger.info(f"selected_nodes{selected_nodes}")
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
                # logger.info(f"node{node}")
                # logger.info(f"new_docs{new_docs}")
                node_doc_pairs_for_refine.append((node, new_docs))

                # 5. 更新记忆库
                self.memory.store_docs(new_docs)
                self.memory.map_node_to_docs(node.id, new_docs)

            # 6. (关键) LLM 批量更新大纲
            # (LLM Call #Round*2 + 1)
            # logger.info(f"node_doc_pairs_for_refine{node_doc_pairs_for_refine}")
            logger.info(f"outline_root{outline_root}")
            logger.info(f"self.memory.node_to_docs{self.memory.node_to_docs}")
            if node_doc_pairs_for_refine:
                logger.info("Batch refining outline...")
                outline_root, expanded_nodes_list, new_node_doc_mapping = (
                    self.llm_wrapper.batch_refine_outline(
                        outline_root,
                        node_doc_pairs_for_refine,
                        memory=self.memory,  # 传递 memory 以获取累积文档
                    )
                )
                logger.info(f"expanded_nodes_list{expanded_nodes_list}")
                
                # --- (已修正) 关键的状态继承步骤 ---
                # 遍历那些刚刚被扩展的节点 (从叶子节点变成了内部节点)
                # expanded_nodes_list 格式: [(parent_node, [new_child_node_1, ...]), ...]
                for parent_node, new_children in expanded_nodes_list:
                    if not new_children:
                        continue

                    # 将父节点的 MAB 状态复制给所有新生成的子节点
                    # 给子节点提供新的 reward
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

    def wrap_raw_docs_to_factstruct(
        self,
        raw_docs,
        source_type="web",
    ):
        wrapped = []

        if (
            isinstance(raw_docs, list)
            and len(raw_docs) == 1
            and isinstance(raw_docs[0], str)
            and "tool_name" in raw_docs[0]
            and "web_search" in raw_docs[0]
        ):
            s = raw_docs[0]
            try:
                # 1️⃣ 提取 content='[...]' 部分（非贪婪匹配）
                content_match = re.search(r"content='(.*?)' name=", s, re.DOTALL)
                if content_match:
                    content_str = content_match.group(1)
                    # 2️⃣ 替换转义双引号
                    content_str = content_str.replace('\\"', '"')
                    # 3️⃣ 转成 list（异常捕获）
                    try:
                        content_list = json.loads(content_str)
                    except json.JSONDecodeError:
                        # 如果 JSONDecodeError，尝试解析到最后一个 JSON 对象
                        # 这里简单方法：用 eval 安全子集
                        import ast
                        content_list = ast.literal_eval(content_str)

                    # 4️⃣ 找到 raw_results
                    raw_docs = []
                    for item in content_list:
                        if isinstance(item, dict) and "raw_results" in item:
                            raw_docs = item["raw_results"]
                            break
                else:
                    raw_docs = []

            except Exception as e:
                # 改用单参数输出，避免 logger 报错
                logger.error(f"文档转换失败: {e}")
                raw_docs = []

        # 封装 FactStructDocument
        for i, d in enumerate(raw_docs):
            if not isinstance(d, dict):
                continue

            text = d.get("snippet", "")
            text = re.sub(r"[。.]{2,}", "", text).strip()
            if not text:
                continue

            doc = FactStructDocument(
                id=f"doc_{hash(text)}_{i}",
                cite_id=i + 1,
                text=text,
                source_type=source_type,
                timestamp=datetime.now(),
                url=d.get("link"),
                title=d.get("title"),
            )
            wrapped.append(doc)

        return wrapped
    



    def run_compression(
        self,
        outline_root,
        memory,
        merge_candidates: List["OutlineNode"],
        max_merges: int, #不做迭代了。
        target_leaf_count: int, #这个给到提示词即可。
        config: RunnableConfig,
    ):
        """
        执行大纲压缩（Batch-MAB 风格，每次选择一个父节点进行结构合并）
        """

        logger.info("Running outline compression")
        t = 0  # 压缩次数计数
        compression_node_number=0

        # 1️⃣ 按父节点分组（从叶子节点回溯）
        parent_to_children = {}

        for node in merge_candidates:
            if node.parent:
                parent = node.parent
                parent_to_children.setdefault(parent, []).append(node)

        print(f"可压缩父节点数量: {len(parent_to_children)}")
        print("父节点列表:")
        for parent, children in parent_to_children.items():
            print(
                f"- Parent: {parent.title} (id={parent.id}) "
                f"children_count={len(children)}"
            )

        parents = list(parent_to_children.keys())  # 现在是 OutlineNode 对象



        # 2️⃣ Batch-MAB 主循环（每轮只压一个 parent）
        while t < max_merges:

            ucb_scores = []
            for parent_iter in parents:
                # 用 id 在最新树中重新定位节点
                current_parent = outline_root.find_node_by_id(parent_iter.id)

                if current_parent is None:
                    logger.warning(f"Parent {parent.id} not found in new tree, skipping")
                    continue
                parent=current_parent
                children = parent_to_children.get(parent, [])
                
                # 子节点相关性（是否适合压）
                cohesion = self.compute_children_cohesion(parent, children)
                print("parent",parent,"cohesion",cohesion)
                # exploration / exploitation
                t_current = t + 1
                if parent.pull_count == 0:
                    exploration = float("inf")
                    avg_reward = 0.0
                else:
                    avg_reward = parent.avg_reward()
                    exploration = math.sqrt(2 * math.log(t_current) / parent.pull_count)

                # ✅ Compression 专用 UCB
                ucb_score = cohesion - (avg_reward + exploration)
                ucb_scores.append((ucb_score, parent))

            if not ucb_scores:
                logger.info("No parents scored, stopping compression")
                break

            # 3️⃣ 选择最“安全可压”的父节点
            ucb_scores.sort(key=lambda x: x[0], reverse=True)
            parent = ucb_scores[0][1]
            children = parent_to_children.get(parent, [])

            logger.info(
                f"Compressing under parent '{parent.title}' "
                f"(children={len(children)})"
            )

            try:
                # 4️⃣ 调用 LLM 做结构压缩
                print("compress_under_parent的parent",parent)
                outline_root, compressed_nodes_list, new_node_doc_mapping, merged_node_mapping = (
                    self.llm_wrapper.compress_under_parent(
                        outline_root=outline_root,
                        parent_node=parent,
                        child_nodes=children,
                        memory=memory,
                    )
                )
                print("compressed_nodes_list",compressed_nodes_list)
                print("new_node_doc_mapping",new_node_doc_mapping)
                print("merged_node_mapping",merged_node_mapping)
                # 5️⃣ 状态继承（完全对齐 expansion）
                for parent_node, new_children in compressed_nodes_list:
                    for child in new_children:
                        child.pull_count = parent_node.pull_count
                        child.reward_history = list(parent_node.reward_history)


                # --- 6️⃣ reward 更新和Memory 更新（Compression 专用逻辑）---
                parent.pull_count += 1
                logger.info(
                    f"Compression success under '{parent.title}', "
                    f"reward={parent.reward_history}"
                )
                # merged_node_mapping: { new_node_id: [old_node_id1, old_node_id2, ...] }


                #先删除，后增加，被修改的都是 new_memory
                # new_memory=memory
                import copy
                new_memory = copy.deepcopy(memory)
                # compression_node_number=
                # --- 删除被压缩节点的文档映射（非常重要）---
                for old_node_ids in merged_node_mapping.values():
                    for old_id in old_node_ids:
                        if old_id in new_memory.node_to_docs:
                            del new_memory.node_to_docs[old_id]
                            logger.debug(
                                f"Removed document mapping for compressed node '{old_id}'"
                            )
                
                # --- 增加新节点的文档映射（非常重要）---
                for new_node_id, old_node_ids in merged_node_mapping.items():
                    merged_docs = []

                    for old_id in old_node_ids:
                        docs = memory.get_docs_by_node(old_id)
                        # print("docs",docs)
                        merged_docs.extend(docs)
                    # merged_docs 是 FactStructDocument 对象列表
                    merged_docs = list({doc.id: doc for doc in merged_docs}.values())
                    # print("merged_docs",merged_docs)
                    if merged_docs:
                        new_memory.map_node_to_docs(new_node_id, merged_docs)

                

            except Exception as e:
                # 打印完整 traceback
                tb_str = traceback.format_exc()
                logger.error(
                    f"Compression failed under parent '{parent.title}': {e}\n"
                    f"Traceback:\n{tb_str}"
                )
                parents.remove(parent)
                continue

            t += 1
            memory=new_memory
            outline_root=outline_root

            # 7️⃣ 检查终止条件，这个地方是不对的
            current_leaf_count = len(outline_root.get_leaf_nodes())
            logger.info(f"Current leaf count: {current_leaf_count}")
            if current_leaf_count <= target_leaf_count:
                logger.info("Target leaf count reached, stopping compression")
                break

        logger.info(
            f"Compression finished: merges={t}, "
            f"total_nodes={len(outline_root.get_all_nodes())}"
        )
        return outline_root, new_memory


    def compute_children_cohesion(
        self,
        parent: "OutlineNode",
        children: List["OutlineNode"],
    ) -> float:
        """
        计算子节点之间的语义内聚度（平均 pairwise cosine similarity）

        cohesion ∈ [-1, 1]（通常在 [0, 1]）
        越大表示这些子节点越应该被合并
        """

        n = len(children)
        if n < 2:
            return 0.0

        # 1️⃣ 构造子节点 embeddings（带父上下文，保证语义空间一致）
        child_embeddings = []
        parent_context = parent.title  # 只用 parent，不再引入 parent.parent

        for child in children:
            node_text = f"{parent_context} > {child.title}"
            emb = self.embedder.embed_text(node_text)
            child_embeddings.append(emb)

        if len(child_embeddings) < 2:
            return 0.0

        # 2️⃣ 计算 pairwise cosine similarity
        total_sim = 0.0
        pair_count = 0

        for i in range(len(child_embeddings)):
            for j in range(i + 1, len(child_embeddings)):
                sim = cosine_similarity(
                    child_embeddings[i],
                    child_embeddings[j],
                )
                total_sim += sim
                pair_count += 1

        if pair_count == 0:
            return 0.0

        cohesion = total_sim / pair_count
        return float(cohesion)



def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


if __name__ == "__main__":
    """
    测试 Batch-MAB run_compression 功能
    场景：压缩 “中性粒细胞在脑缺血急性期的作用” 下的 4 个子节点
    """

    from datetime import datetime
    from src.factstruct.memory import Memory
    from src.factstruct.document import FactStructDocument
    from src.factstruct.outline_node import OutlineNode
    from src.factstruct.llm_wrapper import FactStructLLMWrapper
    from src.llms.llm import get_llm_by_type
    from src.config.agents import AGENT_LLM_MAP
    from langchain_core.runnables import RunnableConfig
    from src.factstruct.embedder import Embedder
    from src.factstruct.batch_mab import BatchMAB
    # from src.search.adapter import create_search_engine_adapter  # 如果有的话
    
    print("========== START BATCH-MAB COMPRESSION DEBUG ==========")

    # -----------------------
    # 1️⃣ 构造 Memory
    # -----------------------
    memory = Memory()

    # 测试文档
    doc1 = FactStructDocument(
        id="doc_1", cite_id="CIT001", source_type="journal",
        title="中性粒细胞募集机制研究",
        text="急性期中性粒细胞通过趋化因子被募集到缺血区域。",
        embedding=None, timestamp=datetime.now()
    )
    doc2 = FactStructDocument(
        id="doc_2", cite_id="CIT002", source_type="journal",
        title="血脑屏障破坏机制",
        text="促炎因子释放导致血脑屏障通透性增加。",
        embedding=None, timestamp=datetime.now()
    )
    doc3 = FactStructDocument(
        id="doc_3", cite_id="CIT003", source_type="journal",
        title="炎症与神经损伤",
        text="炎症反应加剧脑水肿与神经损伤。",
        embedding=None, timestamp=datetime.now()
    )

    # -----------------------
    # 2️⃣ 构造 Outline
    # -----------------------
    # root = OutlineNode(id="node_0", title="中性粒细胞在脑缺血中的作用", pull_count=2, reward_history=[0.8, 0.9], word_limit=500)
    # acute = OutlineNode(id="node_1", title="中性粒细胞在脑缺血急性期的作用", pull_count=1, reward_history=[0.7], word_limit=300)
    
    # n2 = OutlineNode(id="node_2", title="中性粒细胞的募集与激活机制", pull_count=0, reward_history=[], word_limit=100)
    # n3 = OutlineNode(id="node_3", title="促炎因子释放与血-脑屏障破坏", pull_count=0, reward_history=[], word_limit=100)
    # n4 = OutlineNode(id="node_4", title="炎症反应对脑水肿与神经损伤的影响", pull_count=0, reward_history=[], word_limit=100)
    # n5 = OutlineNode(id="node_5", title="中性粒细胞在神经修复中的潜在作用", pull_count=0, reward_history=[], word_limit=100)

    # acute.add_child(n2)
    # acute.add_child(n3)
    # acute.add_child(n4)
    # acute.add_child(n5)
    # root.add_child(acute)
    
    # -----------------------
    # 2️⃣ 构造 Outline（多 parent 测试）
    # -----------------------

    root = OutlineNode(
        id="node_0",
        title="中性粒细胞在脑缺血中的作用",
        pull_count=0,
        reward_history=[],
        word_limit=500
    )

    # ===== Parent A（高 reward，不应该被选）=====
    acute_A = OutlineNode(id="node_1",
        title="中性粒细胞在脑缺血急性炎症阶段的分子机制",
        # pull_count=5,reward_history=[0.9, 0.88, 0.92, 0.91, 0.87],
        pull_count=1,reward_history=[0.1],
        word_limit=300
    )

    A1 = OutlineNode(id="node_2",
        title="中性粒细胞募集的趋化因子调控机制",
        pull_count=0,reward_history=[]
    )

    A2 = OutlineNode(id="node_3",
        title="中性粒细胞激活后的促炎信号级联反应",
        pull_count=0,reward_history=[]
    )

    A3 = OutlineNode(id="node_4",
        title="中性粒细胞诱导的血脑屏障通透性改变",
        pull_count=0,reward_history=[]
    )

    acute_A.add_child(A1)
    acute_A.add_child(A2)
    acute_A.add_child(A3)

    # ===== Parent B（低 reward，应该被选）=====
    acute_B = OutlineNode(id="node_5",
        title="急性缺血性脑卒中中中性粒细胞介导的炎症损伤",
        pull_count=1,reward_history=[0.1],word_limit=300
    )

    B1 = OutlineNode(id="node_6",
        title="中性粒细胞介导的急性炎症反应机制",
        pull_count=0,reward_history=[]
    )

    B2 = OutlineNode(id="node_7",
        title="急性期中性粒细胞释放炎症因子的机制",
        pull_count=0,reward_history=[]
    )

    B3 = OutlineNode(id="node_8",
        title="中性粒细胞在急性脑缺血炎症级联中的作用",
        pull_count=0,reward_history=[]
    )


    acute_B.add_child(B1)
    acute_B.add_child(B2)
    acute_B.add_child(B3)

    root.add_child(acute_A)
    root.add_child(acute_B)




    # # -----------------------
    # # 3️⃣ 构建节点-文档映射
    # # -----------------------
    # memory.map_node_to_docs("node_2", [doc1])
    # memory.map_node_to_docs("node_3", [doc2])
    # memory.map_node_to_docs("node_4", [doc3])
    # -----------------------
    # 3️⃣ 构建节点-文档映射（增强版）
    # -----------------------

    docs = []
    docs = [
        FactStructDocument(
            id="doc_1",
            cite_id="CIT001",
            source_type="journal",
            title="中性粒细胞在脑缺血损伤中作用的研究进展",
            text="综述中性粒细胞在急性脑缺血中的炎症机制...",
            embedding=None,
            timestamp=datetime.now(),
        ),
        FactStructDocument(
            id="doc_2",
            cite_id="CIT002",
            source_type="journal",
            title="急性缺血性脑卒中中炎症级联反应机制",
            text="炎症因子释放与脑组织损伤密切相关...",
            embedding=None,
            timestamp=datetime.now(),
        ),
        FactStructDocument(
            id="doc_3",
            cite_id="CIT003",
            source_type="journal",
            title="血脑屏障破坏与中性粒细胞浸润关系研究",
            text="BBB通透性变化促进炎症细胞进入脑组织...",
            embedding=None,
            timestamp=datetime.now(),
        ),
        FactStructDocument(
            id="doc_4",
            cite_id="CIT004",
            source_type="journal",
            title="急性脑卒中后免疫细胞动态变化研究",
            text="中性粒细胞在早期炎症反应中占主导地位...",
            embedding=None,
            timestamp=datetime.now(),
        ),
    ]


    # Parent A
    memory.map_node_to_docs("node_2", [docs[0]])
    memory.map_node_to_docs("node_3", [docs[1]])
    memory.map_node_to_docs("node_4", [docs[2]])

    # Parent B
    memory.map_node_to_docs("node_6", [docs[0], docs[3]])
    memory.map_node_to_docs("node_7", [docs[1]])
    memory.map_node_to_docs("node_8", [docs[0], docs[1]])



    # -----------------------
    # 4️⃣ 初始化 LLM Wrapper
    # -----------------------
    llm_type = AGENT_LLM_MAP.get("outline", "basic")
    llm = get_llm_by_type(llm_type)
    wrapper = FactStructLLMWrapper(llm)


    # -----------------------
    # 5️⃣ 初始化 Embedder + LLM + BatchMAB（仿真实系统）
    # -----------------------



    # === Embedder（真实模型）===
    embedder = Embedder(model_name="../../Model/MiniLM/all-MiniLM-L6-v2")

    # === Search Engine（如果不想真的搜，可以给 dummy）===
    search_engine = lambda q, k: []   # 测试压缩不需要搜索

    # === LLM ===
    llm_type = AGENT_LLM_MAP.get("outline", "basic")
    llm = get_llm_by_type(llm_type)

    # === Wrapper ===
    wrapper = FactStructLLMWrapper(llm)
    print("\n--- BEFORE COMPRESSION ---")
    print(root.to_text_tree(include_word_limit=True, include_mab_state=True))
    print("Memory:", memory.node_to_docs)
    wrapper._inherit_mab_state_for_existing_nodes(old_root=None, new_root=root)
    
    # === Batch-MAB ===
    batch_mab = BatchMAB(
        llm_wrapper=wrapper,
        embedder=embedder,
        search_engine=search_engine,
        max_iterations=4,
        memory=memory,        # ⭐ 关键：把你构造的 memory 传进去
        batch_size=2,
    )



    # -----------------------
    # 6️⃣ 执行 run_compression
    # -----------------------
    # merge_candidates = [n3, n4, n5]
    merge_candidates = [A1, A2, A3, B1, B2, B3]

    # merge_candidates = [n3, n4]
    # merge_candidates = [n2, n3]
    # merge_candidates = [n2]
    max_merges = 2
    target_leaf_count = 2
    config = RunnableConfig()

    new_root, new_memory = batch_mab.run_compression(
        outline_root=root,
        memory=memory,
        merge_candidates=merge_candidates,
        max_merges=max_merges,
        target_leaf_count=target_leaf_count,
        config=config
    )

    # -----------------------
    # 7️⃣ 输出结果
    # -----------------------
    print("\n--- AFTER COMPRESSION ---")
    print(new_root.to_text_tree(include_word_limit=True, include_mab_state=True))
    print("Memory:", new_memory.node_to_docs)

    # 验证 parent-child 链路
    def validate_tree(node):
        for child in node.children:
            assert child.parent == node, f"Parent pointer broken at {child.title}"
            validate_tree(child)

    validate_tree(new_root)
    print("\n✅ Tree structure valid.")
    print("========== END DEBUG ==========")





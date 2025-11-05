"""
RewardCalculator: 奖励计算模块

实现了 MAB 算法的奖励函数，包含相关性（Relevance）和新颖性（Novelty）计算。
注意：根据要求，已移除质量（Quality）项。
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import logger
from .document import FactStructDocument


class RewardCalculator:
    """
    奖励计算器

    计算 MAB 算法的奖励函数：
        r = w_rel * Relevance + w_nov * Novelty

    其中：
        - Relevance: 文档与查询节点的语义相关性
        - Novelty: 文档与全局文档池的差异度（避免信息冗余）
    """

    def __init__(
        self,
        w_rel: float = 0.7,
        w_nov: float = 0.3,
    ):
        """
        初始化奖励计算器

        参数:
            w_rel: 相关性权重（默认 0.7）
            w_nov: 新颖性权重（默认 0.3）

        注意:
            w_rel + w_nov 应该等于 1.0（已移除 Quality 项）
        """
        if abs(w_rel + w_nov - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {w_rel + w_nov}, not 1.0. Normalizing...")
            total = w_rel + w_nov
            w_rel = w_rel / total
            w_nov = w_nov / total

        self.w_rel = w_rel
        self.w_nov = w_nov

    def calculate_reward(
        self,
        new_docs: List[FactStructDocument],
        node_title: str,
        all_doc_embeddings: np.ndarray,
        node_embedding: np.ndarray = None,
    ) -> Tuple[float, Dict]:
        """
        计算奖励值

        参数:
            new_docs: 新检索到的文档列表
            node_title: 节点标题（用于生成节点嵌入，如果 node_embedding 未提供）
            all_doc_embeddings: 全局文档池的所有嵌入向量，形状为 (N, embedding_dim)
            node_embedding: 节点的嵌入向量（可选，如果未提供则使用文档平均嵌入）

        返回:
            (reward, breakdown): 奖励值和详细分解
        """
        if not new_docs:
            return 0.0, {
                "relevance": 0.0,
                "novelty": 0.0,
                "reward": 0.0,
            }

        # 计算相关性（Relevance）
        relevance = self._calculate_relevance(new_docs, node_embedding)

        # 计算新颖性（Novelty）
        novelty = self._calculate_novelty(new_docs, all_doc_embeddings)

        # 加权求和
        reward = self.w_rel * relevance + self.w_nov * novelty

        breakdown = {
            "relevance": relevance,
            "novelty": novelty,
            "reward": reward,
        }

        return reward, breakdown

    def _calculate_relevance(
        self,
        docs: List[FactStructDocument],
        node_embedding: np.ndarray = None,
    ) -> float:
        """
        计算相关性（Relevance）

        如果提供了 node_embedding，则计算文档与节点的相似度。
        否则，计算文档之间的平均相似度（文档集合的内聚性）。

        参数:
            docs: 文档列表
            node_embedding: 节点嵌入向量（可选）

        返回:
            相关性分数（0-1）
        """
        if not docs:
            return 0.0

        # 过滤出有 embedding 的文档
        docs_with_embedding = [d for d in docs if d.embedding is not None]
        if not docs_with_embedding:
            logger.warning("No documents with embeddings for relevance calculation")
            return 0.0

        embeddings = np.array([d.embedding for d in docs_with_embedding])

        if node_embedding is not None:
            # 计算文档与节点的平均相似度
            node_embedding = np.array(node_embedding, dtype=np.float32)
            if node_embedding.ndim == 1:
                node_embedding = node_embedding.reshape(1, -1)

            # 归一化
            node_norm = node_embedding / np.linalg.norm(node_embedding)
            doc_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # 计算余弦相似度
            similarities = np.dot(doc_norms, node_norm.T).flatten()
            relevance = float(np.mean(similarities))
        else:
            # 计算文档之间的平均相似度（内聚性）
            if len(embeddings) == 1:
                relevance = 1.0
            else:
                # 归一化
                doc_norms = embeddings / np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )
                # 计算两两相似度矩阵
                similarity_matrix = cosine_similarity(doc_norms)
                # 取上三角矩阵的平均值（排除对角线）
                n = len(docs_with_embedding)
                relevance = float(
                    np.sum(similarity_matrix[np.triu_indices(n, k=1)])
                    / (n * (n - 1) / 2)
                )

        # 确保返回值在 [0, 1] 范围内
        return max(0.0, min(1.0, relevance))

    def _calculate_novelty(
        self,
        new_docs: List[FactStructDocument],
        all_doc_embeddings: np.ndarray,
    ) -> float:
        """
        计算新颖性（Novelty）

        新颖性定义为新文档与全局文档池的平均差异度。
        如果全局文档池为空，返回 1.0（完全新颖）。

        参数:
            new_docs: 新文档列表
            all_doc_embeddings: 全局文档池的所有嵌入向量

        返回:
            新颖性分数（0-1），1.0 表示完全新颖
        """
        if not new_docs:
            return 0.0

        # 过滤出有 embedding 的文档
        docs_with_embedding = [d for d in new_docs if d.embedding is not None]
        if not docs_with_embedding:
            logger.warning("No documents with embeddings for novelty calculation")
            return 0.0

        new_embeddings = np.array([d.embedding for d in docs_with_embedding])

        # 如果全局文档池为空，返回完全新颖
        if len(all_doc_embeddings) == 0:
            return 1.0

        # 归一化
        new_norms = new_embeddings / np.linalg.norm(
            new_embeddings, axis=1, keepdims=True
        )
        all_norms = all_doc_embeddings / np.linalg.norm(
            all_doc_embeddings, axis=1, keepdims=True
        )

        # 计算每个新文档与全局文档池的最大相似度
        max_similarities = []
        for new_norm in new_norms:
            similarities = np.dot(all_norms, new_norm.T).flatten()
            max_sim = float(np.max(similarities))
            max_similarities.append(max_sim)

        # 新颖性 = 1 - 平均最大相似度
        avg_max_sim = np.mean(max_similarities)
        novelty = 1.0 - avg_max_sim

        # 确保返回值在 [0, 1] 范围内
        return max(0.0, min(1.0, novelty))

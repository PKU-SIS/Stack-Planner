"""
Memory: 文档存储与索引模块

实现了文档的存储、向量索引（FAISS）和节点-文档映射功能。
"""

import numpy as np
from typing import List, Dict, Set, Optional
from datetime import datetime

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.utils.logger import logger
from .document import FactStructDocument


class Memory:
    """
    文档存储与索引模块

    功能:
        1. 存储文档并维护去重
        2. 使用 FAISS 进行向量索引（加速相似度搜索）
        3. 维护节点-文档映射关系
        4. 提供文档检索接口
    """

    def __init__(self, embedding_dim: int = 384):
        """
        初始化 Memory

        参数:
            embedding_dim: 向量嵌入维度（默认 384，对应 sentence-transformers 的 all-MiniLM-L6-v2）
        """
        self.embedding_dim = embedding_dim
        self.documents: Dict[str, FactStructDocument] = {}  # id -> Document
        self.node_to_docs: Dict[str, Set[str]] = {}  # node_id -> set of doc_ids
        self.doc_ids_list: List[str] = []  # 保持插入顺序，用于 FAISS 索引映射

        # 初始化 FAISS 索引
        if FAISS_AVAILABLE:
            # 使用 L2 距离的平面索引（适合小到中等规模数据）
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            self.index = None
            logger.warning(
                "FAISS not available. Falling back to brute-force similarity search. "
                "Install faiss-cpu or faiss-gpu for better performance."
            )

    def store_docs(self, docs: List[FactStructDocument]):
        """
        存储文档列表（自动去重）

        参数:
            docs: 要存储的文档列表
        """
        new_count = 0
        for doc in docs:
            if doc.id not in self.documents:
                self.documents[doc.id] = doc
                self.doc_ids_list.append(doc.id)
                new_count += 1

                # 如果有 embedding，添加到 FAISS 索引
                if doc.embedding is not None and self.index is not None:
                    # 确保 embedding 是正确形状的 numpy 数组
                    embedding = np.array(doc.embedding, dtype=np.float32)
                    if embedding.ndim == 1:
                        embedding = embedding.reshape(1, -1)

                    # 确保维度匹配
                    if embedding.shape[1] == self.embedding_dim:
                        self.index.add(embedding)
                    else:
                        logger.warning(
                            f"Document {doc.id} embedding dimension {embedding.shape[1]} "
                            f"does not match expected {self.embedding_dim}"
                        )

        if new_count > 0:
            logger.info(
                f"Stored {new_count} new documents (total: {len(self.documents)})"
            )

    def map_node_to_docs(self, node_id: str, docs: List[FactStructDocument]):
        """
        将文档映射到节点

        参数:
            node_id: 节点 ID
            docs: 与该节点相关的文档列表
        """
        if node_id not in self.node_to_docs:
            self.node_to_docs[node_id] = set()

        doc_ids = {doc.id for doc in docs}
        self.node_to_docs[node_id].update(doc_ids)

        # 确保文档已存储
        self.store_docs(docs)

    def get_docs_by_node(self, node_id: str) -> List[FactStructDocument]:
        """
        获取与节点相关的所有文档

        参数:
            node_id: 节点 ID

        返回:
            文档列表
        """
        doc_ids = self.node_to_docs.get(node_id, set())
        return [
            self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents
        ]

    def get_all_doc_embeddings(self) -> np.ndarray:
        """
        获取所有文档的嵌入向量（用于 Novelty 计算）

        返回:
            numpy 数组，形状为 (N, embedding_dim)，N 为文档数量
        """
        embeddings = []
        for doc_id in self.doc_ids_list:
            doc = self.documents.get(doc_id)
            if doc and doc.embedding is not None:
                embedding = np.array(doc.embedding, dtype=np.float32)
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                embeddings.append(embedding[0])

        if not embeddings:
            return np.array([]).reshape(0, self.embedding_dim)

        return np.array(embeddings)

    def retrieve_by_citation_id(self, citation_id: str) -> List[FactStructDocument]:
        """
        根据引用 ID 检索文档（用于 Stage 4 一致性检测）

        参数:
            citation_id: 引用 ID（可以是文档 ID 或节点 ID）

        返回:
            文档列表
        """
        # 先尝试作为文档 ID
        if citation_id in self.documents:
            return [self.documents[citation_id]]

        # 再尝试作为节点 ID
        return self.get_docs_by_node(citation_id)

    def search_similar_docs(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[FactStructDocument]:
        """
        使用 FAISS 索引搜索相似文档（如果 FAISS 不可用，则使用暴力搜索）

        参数:
            query_embedding: 查询向量（形状为 (embedding_dim,) 或 (1, embedding_dim)）
            top_k: 返回最相似的文档数量

        返回:
            最相似的文档列表
        """
        if not self.documents:
            return []

        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.embedding_dim:
            logger.error(
                f"Query embedding dimension {query_embedding.shape[1]} "
                f"does not match expected {self.embedding_dim}"
            )
            return []

        if self.index is not None and self.index.ntotal > 0:
            # 使用 FAISS 索引搜索
            distances, indices = self.index.search(
                query_embedding, min(top_k, self.index.ntotal)
            )

            results = []
            for idx in indices[0]:
                if idx < len(self.doc_ids_list):
                    doc_id = self.doc_ids_list[idx]
                    if doc_id in self.documents:
                        results.append(self.documents[doc_id])

            return results
        else:
            # 暴力搜索（FAISS 不可用或索引为空）
            all_embeddings = self.get_all_doc_embeddings()
            if len(all_embeddings) == 0:
                return []

            # 计算余弦相似度
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = all_embeddings / np.linalg.norm(
                all_embeddings, axis=1, keepdims=True
            )
            similarities = np.dot(doc_norms, query_norm.T).flatten()

            # 获取 top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                doc_id = self.doc_ids_list[idx]
                if doc_id in self.documents:
                    results.append(self.documents[doc_id])

            return results

    def get_statistics(self) -> Dict:
        """
        获取存储统计信息

        返回:
            包含统计信息的字典
        """
        return {
            "total_documents": len(self.documents),
            "total_nodes": len(self.node_to_docs),
            "index_size": self.index.ntotal if self.index is not None else 0,
            "faiss_available": FAISS_AVAILABLE and self.index is not None,
        }

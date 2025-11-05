"""
Embedder: 文档嵌入生成模块

使用 sentence-transformers 生成文档的向量嵌入。
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from src.utils.logger import logger
from .document import FactStructDocument


class Embedder:
    """
    文档嵌入生成器

    使用预训练的 sentence-transformers 模型将文本转换为向量嵌入。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化嵌入器

        参数:
            model_name: sentence-transformers 模型名称
                - "all-MiniLM-L6-v2": 默认模型，速度快，维度 384
                - "all-mpnet-base-v2": 更高质量，维度 768
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Initialized Embedder with model '{model_name}', dimension: {self.embedding_dim}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Embedder: {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """
        为单个文本生成嵌入向量

        参数:
            text: 输入文本

        返回:
            嵌入向量（numpy 数组）
        """
        if not text or not text.strip():
            # 返回零向量
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def embed_docs(self, docs: List[FactStructDocument]) -> List[FactStructDocument]:
        """
        为文档列表生成嵌入向量（批量处理，如果文档已有 embedding 则跳过）

        参数:
            docs: 文档列表

        返回:
            已添加 embedding 的文档列表（同一对象，原地修改）
        """
        # 过滤出需要嵌入的文档
        texts_to_embed = []
        indices_to_embed = []

        for i, doc in enumerate(docs):
            if doc.embedding is None:
                texts_to_embed.append(doc.text)
                indices_to_embed.append(i)

        if not texts_to_embed:
            return docs

        # 批量嵌入
        try:
            embeddings = self.model.encode(
                texts_to_embed,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # 将嵌入添加到文档
            for idx, embedding in zip(indices_to_embed, embeddings):
                docs[idx].embedding = embedding.astype(np.float32)

            logger.info(f"Embedded {len(texts_to_embed)} documents")
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            # 为失败的文档设置零向量
            for idx in indices_to_embed:
                docs[idx].embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        return docs

    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.embedding_dim

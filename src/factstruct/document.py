"""
FactStruct Document 数据结构

定义了用于 Stage 1 的文档数据结构，包含嵌入向量（embedding）和时间戳等元数据。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np


@dataclass
class FactStructDocument:
    """
    FactStruct 文档数据结构

    属性:
        id: 文档唯一标识符
        text: 文档文本内容
        source_type: 来源类型（如 "academic_paper", "blog", "social_media"）
        timestamp: 文档时间戳
        embedding: 文档的向量嵌入（可选，用于语义相似度计算）
        url: 文档URL（可选）
        title: 文档标题（可选）
    """

    id: str
    cite_id: str
    text: str
    source_type: str
    timestamp: datetime
    embedding: Optional[np.ndarray] = None
    url: Optional[str] = None
    title: Optional[str] = None

    def __post_init__(self):
        """验证数据有效性"""
        if not self.id:
            raise ValueError("Document id cannot be empty")
        if not self.text:
            raise ValueError("Document text cannot be empty")
        if not self.source_type:
            raise ValueError("Document source_type cannot be empty")

    def to_dict(self) -> dict:
        """转换为字典格式（用于序列化，不包含 embedding）"""
        return {
            "id": self.id,
            "cite_id": self.cite_id,
            "text": self.text,
            "source_type": self.source_type,
            "timestamp": self.timestamp.isoformat(),
            "url": self.url,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FactStructDocument":
        """从字典恢复 Document 实例"""
        return cls(
            id=data["id"],
            cite_id=data["cite_id"],
            text=data["text"],
            source_type=data["source_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            url=data.get("url"),
            title=data.get("title"),
        )

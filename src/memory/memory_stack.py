from src.memory.memory_stack_entry import MemoryStackEntry
from typing import Annotated, Literal, Dict, List, Optional, Any
import json
from src.utils.logger import logger


# -------------------------
# 记忆管理模块
# -------------------------
class MemoryStack:
    """记忆栈管理器，负责存储和管理系统执行历史"""

    def __init__(self, max_size: int = 50, horizon_topk=1):
        self.stack: List[MemoryStackEntry] = []
        self.max_size = max_size
        self.horizon_topk = horizon_topk

    def push(self, entry: MemoryStackEntry) -> None:
        """添加新条目到栈顶，并维护栈大小限制"""
        self.stack.append(entry)
        self._maintain_stack_size()

    def push_with_pop(self, entry: MemoryStackEntry, topk=1) -> None:
        """先弹出栈顶再推入新条目，用于更新最新记忆"""
        if self.stack:
            for i in range(topk):
                self.stack.pop()
        self.push(entry)

    def peek(self) -> Optional[MemoryStackEntry]:
        """查看栈顶条目但不弹出，返回None如果栈为空"""
        return self.stack[-1] if self.stack else None

    def get_recent(self, count: int = 5) -> List[MemoryStackEntry]:
        """获取最近的N个条目，不足时返回全部"""
        count = self.horizon_topk
        return self.stack[-count:] if len(self.stack) >= count else self.stack.copy()

    def get_all(self) -> List[MemoryStackEntry]:
        """获取所有记忆条目，返回副本避免修改原始数据"""
        return self.stack.copy()

    def get_summary(self, include_full_history: bool = False) -> str:
        """
        获取记忆栈摘要，支持返回完整历史或最近操作摘要

        Args:
            include_full_history: 是否返回完整历史，默认为False返回最近操作
        """
        if not self.stack:
            return "记忆栈为空"

        if include_full_history:
            # 返回完整历史信息供大模型使用
            return json.dumps(
                [entry.to_dict() for entry in self.stack], ensure_ascii=False, indent=2
            )

        # 生成最近操作摘要
        recent_entries = self.get_recent(3)
        summary_parts = []
        for entry in recent_entries:
            action_desc = (
                f"{entry.action}({entry.agent_type})"
                if entry.agent_type
                else entry.action
            )
            content_preview = (
                entry.content[:100] + "..."
                if len(entry.content) > 100
                else entry.content
            )
            summary_parts.append(
                f"[{entry.timestamp[:19]}] {action_desc}: {content_preview}"
            )

        return "最近执行:\n" + "\n".join(summary_parts)

    def to_dict(self) -> List[Dict[str, Any]]:
        """将记忆栈转换为字典列表，便于序列化存储"""
        return [entry.to_dict() for entry in self.stack]

    def load_from_dict(self, data) -> None:
        """从字典列表或 JSON 字符串恢复记忆栈"""
        self.stack = []
        if not data:
            return
        # 如果是字符串，先解析为列表
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"无法解析 memory_stack 字符串: {data[:100]}...")
                return
        if not isinstance(data, list):
            logger.warning(f"memory_stack 数据格式不正确: {type(data)}")
            return
        for entry_dict in data:
            entry = MemoryStackEntry(
                timestamp=entry_dict.get("timestamp", ""),
                action=entry_dict.get("action", ""),
                agent_type=entry_dict.get("agent_type"),
                content=entry_dict.get("content", ""),
                result=entry_dict.get("result"),
            )
            self.stack.append(entry)

    def size(self) -> int:
        """获取当前记忆栈大小"""
        return len(self.stack)

    def is_empty(self) -> bool:
        """检查记忆栈是否为空"""
        return len(self.stack) == 0

    def _maintain_stack_size(self) -> None:
        """维护栈大小在限制范围内，超出时移除最早的条目"""
        if len(self.stack) > self.max_size:
            self.stack.pop(0)

    def pop(self, count: int = 1) -> List[MemoryStackEntry]:
        """
        从栈顶弹出指定数量的条目
        """
        removed_items = []
        if count <= 0:
            return removed_items
        for _ in range(min(count, len(self.stack))):
            top_element = self.peek()
            removed_items.append(top_element)
            self.stack.pop()
        return removed_items

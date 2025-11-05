"""
OutlineNode: 大纲树节点数据结构

实现了大纲的树形结构，支持 MAB（多臂老虎机）算法的状态追踪。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OutlineNode:
    """
    大纲树节点

    属性:
        id: 节点唯一标识符
        title: 节点标题
        parent: 父节点（根节点为 None）
        children: 子节点列表
        pull_count: 该节点被"拉动"（检索）的次数（用于 MAB 算法）
        reward_history: 该节点获得的奖励历史记录
    """

    id: str
    title: str
    parent: Optional["OutlineNode"] = None
    children: List["OutlineNode"] = field(default_factory=list)
    pull_count: int = 0
    reward_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        """初始化后处理：确保父节点关系正确"""
        # 确保子节点的父节点指向当前节点
        for child in self.children:
            if child.parent != self:
                child.parent = self

    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """判断是否为根节点"""
        return self.parent is None

    def get_leaf_nodes(self) -> List["OutlineNode"]:
        """
        获取当前节点子树中的所有叶子节点

        返回:
            叶子节点列表
        """
        if self.is_leaf():
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_nodes())
        return leaves

    def get_parent_context(self) -> str:
        """
        获取父节点上下文（用于生成查询时提供上下文信息）

        返回:
            从根节点到父节点的路径字符串
        """
        if self.parent is None:
            return ""

        context_parts = []
        current = self.parent
        while current is not None:
            context_parts.insert(0, current.title)
            current = current.parent

        return " > ".join(context_parts)

    def avg_reward(self) -> float:
        """
        计算平均奖励（用于 MAB 的 Exploitation 部分）

        返回:
            平均奖励值，如果没有历史记录则返回 0.0
        """
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    def to_text_tree(self, indent: int = 0) -> str:
        """
        将节点树转换为文本格式（用于 LLM 提示）

        参数:
            indent: 缩进级别

        返回:
            文本格式的大纲树
        """
        prefix = "  " * indent
        result = f"{prefix}- {self.title}\n"

        for child in self.children:
            result += child.to_text_tree(indent + 1)

        return result

    def find_node_by_id(self, node_id: str) -> Optional["OutlineNode"]:
        """
        根据 ID 查找节点（递归搜索）

        参数:
            node_id: 要查找的节点 ID

        返回:
            找到的节点，如果不存在则返回 None
        """
        if self.id == node_id:
            return self

        for child in self.children:
            found = child.find_node_by_id(node_id)
            if found:
                return found

        return None

    def add_child(self, child: "OutlineNode"):
        """
        添加子节点

        参数:
            child: 要添加的子节点
        """
        child.parent = self
        self.children.append(child)

    def get_depth(self) -> int:
        """
        获取节点深度（根节点深度为 0）

        返回:
            节点深度
        """
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1

    def get_all_nodes(self) -> List["OutlineNode"]:
        """
        获取当前节点子树中的所有节点（包括自身）

        返回:
            所有节点的列表
        """
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

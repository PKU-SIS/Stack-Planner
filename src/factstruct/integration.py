"""
FactStruct Stage 1 集成模块

提供了与现有系统集成的便捷接口。
"""

from typing import List, Optional, Callable, Tuple
from langchain_core.language_models import BaseChatModel

from src.utils.logger import logger
from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.tools.get_docs_info import search_docs
from .batch_mab import BatchMAB
from .embedder import Embedder
from .llm_wrapper import FactStructLLMWrapper
from .document import FactStructDocument
from .outline_node import OutlineNode
from .memory import Memory


def create_search_engine_adapter(
    search_func: Callable = None,
) -> Callable[[str, int], List[FactStructDocument]]:
    """
    创建搜索引擎适配器

    将现有的 search_docs 函数适配为 FactStruct 需要的格式。

    参数:
        search_func: 搜索函数，签名 (question: str, top_k: int) -> List[dict]
                    如果不提供，使用默认的 search_docs

    返回:
        适配后的搜索函数，签名 (query: str, k: int) -> List[FactStructDocument]
    """
    if search_func is None:
        search_func = search_docs

    def adapter(query: str, k: int) -> List[FactStructDocument]:
        """
        适配后的搜索函数

        参数:
            query: 搜索查询
            k: 返回文档数量

        返回:
            FactStructDocument 列表
        """
        from datetime import datetime

        # 调用原始搜索函数
        results = search_func(query, top_k=k)

        # 转换为 FactStructDocument
        documents = []
        for i, result in enumerate(results):
            doc_id = f"doc_{hash(result.get('content', ''))}_{i}"
            doc = FactStructDocument(
                id=doc_id,
                text=result.get("content", ""),
                source_type=result.get("source", "unknown"),
                timestamp=datetime.now(),  # 如果没有时间戳，使用当前时间
                url=None,
                title=None,
            )
            documents.append(doc)

        return documents

    return adapter


def run_factstruct_stage1(
    query: str,
    llm: Optional[BaseChatModel] = None,
    max_iterations: int = 20,
    batch_size: int = 5,
    initial_docs: Optional[List[FactStructDocument]] = None,
    search_engine: Optional[Callable] = None,
) -> Tuple[OutlineNode, Memory]:
    """
    运行 FactStruct Stage 1（便捷接口）

    参数:
        query: 用户查询
        llm: LLM 实例（可选，默认使用 "outline" 类型的 LLM）
        max_iterations: 最大迭代次数（默认 20）
        batch_size: 批量大小（默认 5）
        initial_docs: 初始文档列表（可选）
        search_engine: 搜索引擎函数（可选，默认使用 search_docs）

    返回:
        (outline_root, memory): 最终大纲根节点和记忆模块
    """
    # 初始化组件
    if llm is None:
        # 使用 AGENT_LLM_MAP 获取 outline 对应的 LLM 类型（映射到 "basic"）
        llm_type = AGENT_LLM_MAP.get("outline", "basic")
        llm = get_llm_by_type(llm_type)

    if search_engine is None:
        search_engine = create_search_engine_adapter()

    embedder = Embedder()
    llm_wrapper = FactStructLLMWrapper(llm)

    # 创建 Batch-MAB 实例
    batch_mab = BatchMAB(
        llm_wrapper=llm_wrapper,
        embedder=embedder,
        search_engine=search_engine,
        max_iterations=max_iterations,
        batch_size=batch_size,
    )

    # 运行算法
    outline_root, memory = batch_mab.run(
        initial_query=query,
        initial_docs=initial_docs,
    )

    return outline_root, memory


def outline_node_to_text(outline_root: OutlineNode) -> str:
    """
    将 OutlineNode 转换为文本格式（用于保存到 State）

    参数:
        outline_root: 大纲根节点

    返回:
        文本格式的大纲
    """
    return outline_root.to_text_tree()


def outline_node_to_markdown(
    outline_root: OutlineNode,
    max_depth: Optional[int] = None,
    include_root: bool = True,
) -> str:
    """
    将 OutlineNode 转换为 Markdown 格式

    参数:
        outline_root: 大纲根节点
        max_depth: 最大层级深度（None 表示不限制深度，打印完整大纲）
                  - 根节点为第1层
                  - 如果 include_root=True，max_depth=3 表示根+2层子节点
        include_root: 是否包含根节点（默认True）

    返回:
        Markdown 格式的大纲字符串
    """

    def node_to_markdown(
        node: OutlineNode, current_level: int = 1, parent_indent: str = ""
    ) -> str:
        """
        递归将节点转换为Markdown格式

        参数:
            node: 当前节点
            current_level: 当前层级（1表示根节点，2表示第一层子节点，以此类推）
            parent_indent: 父节点的缩进字符串
        """
        # 检查深度限制
        if max_depth is not None and current_level > max_depth:
            return ""

        result = ""

        # 根节点特殊处理
        if current_level == 1 and include_root:
            result = f"- {node.title}\n"
            # 根节点的子节点应该有2个空格缩进
            child_indent = "  "
        else:
            # 非根节点：缩进 = 父节点缩进 + 2个空格
            child_indent = parent_indent + "  "

        # 处理子节点
        for child in node.children:
            # 计算当前子节点的缩进
            # 如果是根节点的子节点（level 2），缩进是2个空格
            # 如果是子节点的子节点，缩进是父节点缩进 + 2个空格
            result += f"{child_indent}- {child.title}\n"

            # 递归处理子节点的子节点
            if max_depth is None or current_level + 1 <= max_depth:
                child_markdown = node_to_markdown(
                    child, current_level + 1, child_indent
                )
                result += child_markdown

        return result

    markdown_text = node_to_markdown(outline_root, 1)
    return markdown_text.strip()


def outline_node_to_json(outline_root: OutlineNode) -> str:
    """
    将 OutlineNode 转换为 JSON 格式（兼容现有的 outline 格式）

    参数:
        outline_root: 大纲根节点

    返回:
        JSON 字符串格式的大纲
    """

    def node_to_dict(node: OutlineNode) -> dict:
        """递归将节点转换为字典"""
        result = {"title": node.title, "children": []}
        for child in node.children:
            result["children"].append(node_to_dict(child))
        return result

    import json

    outline_dict = node_to_dict(outline_root)
    return json.dumps(outline_dict, ensure_ascii=False, indent=2)


def memory_to_dict(memory: Memory) -> dict:
    """
    将 Memory 实例转换为字典（用于序列化到 State）

    注意：embedding 信息会被丢弃，只保留文档元数据。

    参数:
        memory: Memory 实例

    返回:
        字典格式的内存数据
    """
    return {
        "total_documents": len(memory.documents),
        "node_to_docs": {
            node_id: list(doc_ids) for node_id, doc_ids in memory.node_to_docs.items()
        },
        "documents": {
            doc_id: doc.to_dict() for doc_id, doc in memory.documents.items()
        },
    }


def visualize_outline_with_citations(
    outline_root: OutlineNode,
    memory: Memory,
    output_path: Optional[str] = None,
    print_text: bool = True,
) -> str:
    """
    可视化大纲树及其引文映射关系。

    参数:
        outline_root: 大纲根节点
        memory: Memory 实例，包含节点到文档的映射
        output_path: 图片输出路径（可选，需要安装 graphviz）
        print_text: 是否打印文本格式

    返回:
        文本格式的大纲树（带引文映射）
    """
    lines = []
    lines.append("=" * 80)
    lines.append("大纲树与引文映射关系")
    lines.append("=" * 80)

    def format_node(node: OutlineNode, indent: int = 0) -> None:
        """递归格式化节点"""
        prefix = "  " * indent

        # 获取该节点的引文
        doc_ids = memory.node_to_docs.get(node.id, set())
        docs = [
            memory.documents.get(doc_id)
            for doc_id in doc_ids
            if doc_id in memory.documents
        ]

        # 节点标题
        citation_count = len(docs)
        if citation_count > 0:
            lines.append(
                f"{prefix}├─ {node.title} [ID: {node.id}] 📚 {citation_count} 篇引文"
            )
        else:
            lines.append(f"{prefix}├─ {node.title} [ID: {node.id}] ⚠️ 无引文")

        # 显示引文详情（截断显示）
        for i, doc in enumerate(docs[:3]):  # 最多显示3篇
            if doc:
                doc_title = (doc.title or doc.text[:50] + "...") if doc.text else "未知"
                lines.append(f"{prefix}│    └─ 📄 [{i+1}] {doc_title[:60]}")

        if len(docs) > 3:
            lines.append(f"{prefix}│    └─ ... 还有 {len(docs) - 3} 篇引文")

        # 递归处理子节点
        for child in node.children:
            format_node(child, indent + 1)

    format_node(outline_root)

    # 统计信息
    all_nodes = outline_root.get_all_nodes()
    nodes_with_citations = sum(
        1
        for n in all_nodes
        if n.id in memory.node_to_docs and memory.node_to_docs[n.id]
    )

    lines.append("")
    lines.append("=" * 80)
    lines.append("统计信息")
    lines.append("=" * 80)
    lines.append(f"总节点数: {len(all_nodes)}")
    lines.append(f"有引文的节点数: {nodes_with_citations}")
    lines.append(f"无引文的节点数: {len(all_nodes) - nodes_with_citations}")
    lines.append(f"引文覆盖率: {nodes_with_citations / len(all_nodes) * 100:.1f}%")
    lines.append(f"总文档数: {len(memory.documents)}")
    lines.append("=" * 80)

    text_output = "\n".join(lines)

    if print_text:
        logger.info(f"\n{text_output}")

    # 尝试生成 graphviz 图片
    if output_path:
        try:
            _generate_graphviz_image(outline_root, memory, output_path)
            logger.info(f"大纲可视化图片已保存到: {output_path}")
        except Exception as e:
            logger.warning(f"无法生成 graphviz 图片: {e}")

    return text_output


def _generate_graphviz_image(
    outline_root: OutlineNode,
    memory: Memory,
    output_path: str,
) -> None:
    """
    使用 graphviz 生成大纲树可视化图片。

    参数:
        outline_root: 大纲根节点
        memory: Memory 实例
        output_path: 输出路径（不含扩展名）
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("需要安装 graphviz: pip install graphviz")

    dot = Digraph(comment="Outline Tree with Citations")
    dot.attr(rankdir="TB", splines="ortho")
    dot.attr("node", shape="box", style="rounded,filled", fontname="SimHei")

    def add_node(node: OutlineNode) -> None:
        """递归添加节点"""
        doc_ids = memory.node_to_docs.get(node.id, set())
        doc_count = len(doc_ids)

        # 根据引文数量设置颜色
        if doc_count == 0:
            color = "#ffcccc"  # 红色 - 无引文
        elif doc_count <= 2:
            color = "#ffffcc"  # 黄色 - 少量引文
        else:
            color = "#ccffcc"  # 绿色 - 有引文

        # 截断标题
        title = node.title[:30] + "..." if len(node.title) > 30 else node.title
        label = f"{title}\\n📚 {doc_count} docs"

        dot.node(node.id, label, fillcolor=color)

        for child in node.children:
            add_node(child)
            dot.edge(node.id, child.id)

    add_node(outline_root)

    # 保存图片
    dot.render(output_path, format="png", cleanup=True)

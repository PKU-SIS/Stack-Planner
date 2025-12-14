"""
FactStruct Stage 1 集成模块

提供了与现有系统集成的便捷接口。
"""

from typing import List, Optional, Callable, Tuple
from langchain_core.language_models import BaseChatModel
import traceback


from src.utils.logger import logger
from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.tools.get_docs_info import search_docs
from src.tools.bocha_search.web_search_en import web_search
from .batch_mab import BatchMAB
from .embedder import Embedder
from .llm_wrapper import FactStructLLMWrapper
from .document import FactStructDocument
from .outline_node import OutlineNode
from .memory import Memory
from datetime import datetime
from src.utils.reference_utils import global_reference_map
from langchain_core.runnables import RunnableConfig

import re
from collections import defaultdict
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from sentence_transformers import CrossEncoder
from .cite_verify import filter_content_by_relevant_docs,mark_content_with_support,repair_unknown_citations
def create_search_engine_adapter(
    search_func: Callable = None,
) -> Callable[[str, int,RunnableConfig], List[FactStructDocument]]:
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
        #这个地方要改成网络搜索
        # search_func = search_docs
        search_func = web_search
    def adapter(query: str, k: int, config:RunnableConfig=None) -> List[FactStructDocument]:
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
        logger.info(f"results:{results}")
        ids = None
        if config!=None:
            session_id = config["configurable"]["thread_id"]
            # logger.info(f"config:{config}")
            ids = global_reference_map.add_references(session_id, results)
        else:
            # logger.debug("config为None，无法存储 reference_map")
            logger.debug("config为None，无法存储 reference_map\n" + "".join(traceback.format_stack()))        
        if not ids:
            # 没有 ids，说明 config=None 或 add_references 失败
            # 直接 fallback：用 enumerate 的顺序作为临时 id
            ids = list(range(1, len(results) + 1))
        ids, sorted_results = zip(*sorted(zip(ids, results), key=lambda x: x[0]))


        
        # 转换为 FactStructDocument
        # documents = []
        # for i, result in enumerate(results):
        #     doc_id = f"doc_{hash(result.get('content', ''))}_{i}"
        #     doc = FactStructDocument(
        #         id=doc_id,
        #         cite_id=ids[i] if ids is not None and i < len(ids) else None,
        #         text=result.get("content", ""),
        #         source_type=result.get("source", "unknown"),
        #         timestamp=datetime.now(),  # 如果没有时间戳，使用当前时间
        #         url=result.get("url", None),
        #         title=result.get("title", None),
        #     )
        #     documents.append(doc)
        documents = []
        
        for cite_id, result in zip(ids, sorted_results):
            doc_id = f"doc_{hash(result.get('content', ''))}_{cite_id}"
            doc = FactStructDocument(
                id=doc_id,            # 直接使用 reference id（排序后的）
                cite_id=cite_id,            # cite_id 同 doc_id
                text=result.get("content", ""),
                source_type=result.get("source", "unknown"),
                timestamp=datetime.now(),
                url=result.get("url", None),
                title=result.get("title", None),
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
    config: RunnableConfig=None,
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

    # embedder = Embedder()
    embedder = Embedder(model_name="../../Model/MiniLM/all-MiniLM-L6-v2") 
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
        config=config,
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


def outline_node_to_dict(node: OutlineNode) -> dict:
    """
    将 OutlineNode 完整转换为字典（保留所有字段，包括 MAB 状态）

    参数:
        node: OutlineNode 实例

    返回:
        字典格式的节点数据（可递归包含子节点）
    """
    return {
        "id": node.id,
        "title": node.title,
        "pull_count": node.pull_count,
        "reward_history": node.reward_history,
        "children": [outline_node_to_dict(child) for child in node.children],
    }


def dict_to_outline_node(data: dict, parent: Optional[OutlineNode] = None) -> OutlineNode:
    """
    从字典恢复 OutlineNode（递归构建子树）

    参数:
        data: 节点字典数据
        parent: 父节点（可选）

    返回:
        OutlineNode 实例
    """
    node = OutlineNode(
        id=data["id"],
        title=data["title"],
        parent=parent,
        children=[],
        pull_count=data.get("pull_count", 0),
        reward_history=data.get("reward_history", []),
    )

    for child_data in data.get("children", []):
        child = dict_to_outline_node(child_data, parent=node)
        node.children.append(child)

    return node


def dict_to_memory(data: dict) -> Memory:
    """
    从字典恢复 Memory 实例

    参数:
        data: 内存字典数据

    返回:
        Memory 实例
    """
    from .memory import Memory
    from .document import FactStructDocument

    memory = Memory(embedding_dim=384)

    for doc_id, doc_data in data.get("documents", {}).items():
        doc = FactStructDocument.from_dict(doc_data)
        memory.documents[doc_id] = doc

    for node_id, doc_ids in data.get("node_to_docs", {}).items():
        memory.node_to_docs[node_id] = set(doc_ids)

    return memory

def run_factstruct_stage2(
    outline_dict: dict,
    memory_dict: dict,
    user_query: str,
    llm_type: str = "basic",
    locale: str = "zh-CN",
) -> str:
    """
    FactStruct Stage 2: 基于大纲的递归分段文本生成

    采用深度优先遍历策略，递归生成报告：
    1. 从根节点开始递归遍历整棵大纲树
    2. 遇到叶子节点：
       - 从 Memory 中直接获取 Stage 1 关联的文档（使用节点-文档映射）
       - 使用 LLM 生成该节点的段落内容
       - 添加到报告中
    3. 中间节点：添加标题，继续递归子节点

    参数:
        outline_dict: OutlineNode 序列化字典
        memory_dict: Memory 序列化字典
        user_query: 用户原始查询
        llm_type: LLM 类型（默认 "basic"）
        locale: 语言区域设置（默认 "zh-CN"），用于 prompt 模板

    返回:
        完整的 Markdown 格式报告
    """
    from src.llms.llm import get_llm_by_type
    from src.prompts.template import apply_prompt_template

    logger.info(f"开始 FactStruct Stage 2: 基于大纲分段生成内容...")

    outline_root = dict_to_outline_node(outline_dict)
    memory = dict_to_memory(memory_dict)

    # 生成完整大纲的 Markdown 表示
    full_outline = outline_node_to_markdown(outline_root, max_depth=None, include_root=True)

    llm = get_llm_by_type(llm_type)
    report_parts = []
    path_stack = [[]]

    #初始化 NLI 模型
    nli_model_path="/data1/Yangzb/Model/nlp_structbert_nli_chinese-tiny"
    # semantic_cls = pipeline(Tasks.nli,nli_model_path,model_revision='master')
    semantic_cls = CrossEncoder("/data1/Yangzb/Model/StructBert/cross-encoder/nli-deberta-v3-small")


    def get_progress_context(stack, will_complete_chapters: list, next_chapter: str):
        context_lines = []

        for i, level_nodes in enumerate(stack):
            indent = "  " * i
            current_node_title = level_nodes[-1]
            completed_siblings = level_nodes[:-1]
            if completed_siblings:
                siblings_str = "、".join(completed_siblings)
                context_lines.append(f"{indent}其中已完成{siblings_str}，")
            context_lines.append(f"{indent}正在完成{current_node_title}")

        if will_complete_chapters:
            chapters_str = "、".join([f"「{title}」" for title in will_complete_chapters])
            context_lines.append(f"\n完成当前章节后，以下父章节也将完成：{chapters_str}")

        if next_chapter:
            context_lines.append(f"接下来将开始：{next_chapter}")
        else:
            context_lines.append("至此整篇文章将全部完成")

        return "\n".join(context_lines)

    def generate(node: OutlineNode, level: int = 1, will_complete_chapters: list = None, next_chapter: str = None,semantic_cls=None):
        logger.debug(f"正在生成子章节: {node.title}（ID: {node.id}）")

        path_stack[-1].append(node.title)

        if level <= 6:
            report_parts.append(f"{'#' * level} {node.title}\n")

        if node.is_leaf():
            relevant_docs = memory.get_docs_by_node(node.id)

            if not relevant_docs:
                logger.warning(
                    f"节点 '{node.title}' (ID: {node.id}) 未找到关联文档"
                )
                relevant_docs_text = "（无相关资料）"
            else:
                # logger.debug(
                #     f"获取到 {len(relevant_docs)} 个 Stage 1 关联文档"
                # )
                relevant_docs_text = "\n\n".join(
                    [
                        f"[{doc.cite_id}] 来源: {doc.source_type}\n{doc.text[:500]}..."
                        for idx, doc in enumerate(relevant_docs)
                    ]
                )
                # logger.info(f"relevant_docs_text :{relevant_docs_text }")
                logger.info(f"relevant_docs :{relevant_docs}")
            progress_context = get_progress_context(path_stack, will_complete_chapters, next_chapter)

            completed_content = "".join(report_parts).strip()
            if not completed_content:
                completed_content = "（尚未生成任何内容）"

            temp_state = {
                "messages": [],
                "user_query": user_query,
                "full_outline": full_outline,
                "progress_context": progress_context,
                "completed_content": completed_content,
                "reference_materials": relevant_docs_text,
                "locale": locale,
            }

            try:
                messages = apply_prompt_template(
                    "reporter_factstruct",
                    temp_state,
                    extra_context={
                        "user_query": user_query,
                        "full_outline": full_outline,
                        "progress_context": progress_context,
                        "completed_content": completed_content,
                        "reference_materials": relevant_docs_text,
                        "locale": locale,
                    }
                )
                response = llm.invoke(messages)
                content = response.content.strip()
                report_parts.append(f"{content}\n")
                logger.debug(f"  生成了 {len(content)} 个字符")
                
                #如果没文档就不做引用检查了，后面再考虑上文的引用
                if not relevant_docs:
                    logger.warning(
                        f"节点 '{node.title}' (ID: {node.id}) 未找到关联文档，不进行引用检查"
                    )
                else:
                    logger.info(f"content :{content}")
                    #这个是判断引用和句子的关系
                    supported = filter_content_by_relevant_docs(
                        content=content,
                        relevant_docs=relevant_docs,
                        semantic_cls=semantic_cls
                    )
                    logger.info(f"supported :{supported}")
                    
                    #这个是把关系应用到生成文章上
                    new_content = mark_content_with_support(
                        content=content,
                        nli_results=supported
                    )
                    logger.info(f"new_content :{new_content}")
                    
                    #这个是把错误引用进行处理的
                    content=repair_unknown_citations(
                        content=new_content,
                        relevant_docs=relevant_docs,
                        semantic_cls=semantic_cls
                    )
                    logger.info(f"content :{content}")
                    
            except Exception as e:
                logger.error(f"  生成失败: {str(e)}")

        if node.children:
            path_stack.append([])
            for i, child in enumerate(node.children):
                if (i == len(node.children) - 1):
                    child_will_complete = will_complete_chapters + [node.title]
                    child_next_chapter = next_chapter
                else:
                    child_will_complete = []
                    child_next_chapter = node.children[i + 1].title
                generate(child, level + 1, child_will_complete, child_next_chapter,semantic_cls=semantic_cls)
            path_stack.pop()

    generate(outline_root, level=1, will_complete_chapters=[], next_chapter=None,semantic_cls=semantic_cls)

    final_report = "\n".join(report_parts)

    logger.info(
        f"FactStruct Stage 2 完成: 生成了 {len(final_report)} 个字符的报告"
    )

    return final_report

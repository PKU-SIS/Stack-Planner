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
from langchain_core.messages import HumanMessage, SystemMessage
import re
from collections import defaultdict
import json

# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from sentence_transformers import CrossEncoder
from .cite_verify import (
    filter_content_by_relevant_docs,
    mark_content_with_support,
    repair_unknown_citations,
)


def create_search_engine_adapter(
    search_func: Callable = None,
) -> Callable[[str, int, RunnableConfig], List[FactStructDocument]]:
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
        # 这个地方要改成网络搜索
        # search_func = search_docs
        search_func = web_search

    def adapter(
        query: str, k: int, config: RunnableConfig = None
    ) -> List[FactStructDocument]:
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
        if config != None:
            session_id = config["configurable"]["thread_id"]
            # logger.info(f"config:{config}")
            ids = global_reference_map.add_references(session_id, results)
        else:
            # logger.debug("config为None，无法存储 reference_map")
            logger.debug(
                "config为None，无法存储 reference_map\n"
                + "".join(traceback.format_stack())
            )
        if not ids:
            # 没有 ids，说明 config=None 或 add_references 失败
            # 直接 fallback：用 enumerate 的顺序作为临时 id
            ids = list(range(1, len(results) + 1))
        ids, sorted_results = zip(*sorted(zip(ids, results), key=lambda x: x[0]))

        # 转换为 FactStructDocument
        documents = []
        for cite_id, result in zip(ids, sorted_results):
            doc_id = f"doc_{hash(result.get('content', ''))}_{cite_id}"
            doc = FactStructDocument(
                id=doc_id,  # 直接使用 reference id（排序后的）
                cite_id=cite_id,  # cite_id 同 doc_id
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
    task_description=None,
    replan_result=None,
    factstruct_outline=None,
    factstruct_memory=None,
    initial_docs: Optional[List[FactStructDocument]] = None,
    search_engine: Optional[Callable] = None,
    config: RunnableConfig = None,
) -> Tuple[OutlineNode, Memory]:
    # """
    # 运行 FactStruct Stage 1（便捷接口）

    # 参数:
    #     query: 用户查询
    #     llm: LLM 实例（可选，默认使用 "outline" 类型的 LLM）
    #     max_iterations: 最大迭代次数（默认 20）
    #     batch_size: 批量大小（默认 5）
    #     initial_docs: 初始文档列表（可选）
    #     search_engine: 搜索引擎函数（可选，默认使用 search_docs）

    # 返回:
    #     (outline_root, memory): 最终大纲根节点和记忆模块
    # """
    # 初始化组件
    if llm is None:
        # 使用 AGENT_LLM_MAP 获取 outline 对应的 LLM 类型（映射到 "basic"）
        llm_type = AGENT_LLM_MAP.get("outline", "basic")
        llm = get_llm_by_type(llm_type)

    if search_engine is None:
        search_engine = create_search_engine_adapter()

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
    central_guidance = json.dumps(
        task_description,
        ensure_ascii=False,
        indent=2,
    )
    replan_result = json.dumps(
        replan_result,
        ensure_ascii=False,
        indent=2,
    )
    logger.info(f"central_guidance{central_guidance}")
    outline_root, memory = batch_mab.run(
        initial_query=query,
        initial_docs=initial_docs,
        central_guidance=central_guidance,
        replan_result=replan_result,
        factstruct_outline=factstruct_outline,
        factstruct_memory=factstruct_memory,
        config=config,
    )
    logger.info(f"outline_root:{outline_root}")
    logger.info(f"memory:{memory}")
    # exit()
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
        "word_limit": node.word_limit,
        "children": [outline_node_to_dict(child) for child in node.children],
    }


def dict_to_outline_node(
    data: dict, parent: Optional[OutlineNode] = None
) -> OutlineNode:
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
        word_limit=data.get("word_limit", 0),
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
    full_outline = outline_node_to_markdown(
        outline_root, max_depth=None, include_root=True
    )

    llm = get_llm_by_type(llm_type)
    report_parts = []
    path_stack = [[]]

    # 初始化 NLI 模型
    nli_model_path = "/data1/Yangzb/Model/nlp_structbert_nli_chinese-tiny"
    # semantic_cls = pipeline(Tasks.nli,nli_model_path,model_revision='master')
    semantic_cls = CrossEncoder(
        "/data1/Yangzb/Model/StructBert/cross-encoder/nli-deberta-v3-small"
    )

    def get_progress_context(stack, will_complete_chapters: list, next_chapter: str):
        context_lines = []

        context_lines.append("当前文章写作进度：")

        for i, level_nodes in enumerate(stack):
            indent = "  " * i
            current_node_title = level_nodes[-1]
            completed_siblings = level_nodes[:-1]
            if completed_siblings:
                siblings_str = "、".join(completed_siblings)
                context_lines.append(f"{indent}其中已完成{siblings_str}，")
            context_lines.append(f"{indent}正在完成{current_node_title}")

        if will_complete_chapters:
            chapters_str = "、".join(
                [f"「{title}」" for title in will_complete_chapters]
            )
            context_lines.append(
                f"\n完成当前章节后，以下父章节也将完成：{chapters_str}"
            )

        if next_chapter:
            context_lines.append(f"当前章节完成后的下一个章节为：{next_chapter}")
        else:
            context_lines.append("当前章节完成后整篇文章将全部完成")

        return "\n".join(context_lines)

    def generate(
        node: OutlineNode,
        level: int = 1,
        will_complete_chapters: list = None,
        next_chapter: str = None,
        semantic_cls=None,
    ):
        logger.debug(f"正在生成子章节: {node.title}（ID: {node.id}）")

        path_stack[-1].append(node.title)

        if level <= 6:
            report_parts.append(f"{'#' * level} {node.title}\n")

        if node.is_leaf():
            # 记录用户query
            logger.info(
                f"----=====USER_QUERY_START=====----{user_query}----=====USER_QUERY_END=====----"
            )
            # 记录大纲
            logger.info(
                f"----=====OUTLINE_START=====----{full_outline}----=====OUTLINE_END=====----"
            )
            # 记录当前节点
            logger.info(
                f"----=====NODE_ID_START=====----节点:  {node.title} (ID: {node.id}) ----=====NODE_ID_END=====----"
            )
            # 记录父节点
            if node.parent:
                logger.info(
                    f"----=====PARENT_NODE_ID_START=====----节点:  {node.parent.title} (ID: {node.parent.id}) ----=====PARENT_NODE_ID_END=====----"
                )
            else:
                logger.info(
                    f"----=====PARENT_NODE_ID_START=====----节点:  None (ID: None) ----=====PARENT_NODE_ID_END=====----"
                )
            # relevant_docs = []
            # current = node
            # while current is not None:
            #     docs = memory.get_docs_by_node(current.id)
            #     relevant_docs.extend(docs)
            #     current = current.parent
            relevant_docs = []
            seen_ids = set()
            current = node
            while current is not None:
                docs = memory.get_docs_by_node(current.id)
                for doc in docs:
                    if doc.cite_id not in seen_ids:
                        relevant_docs.append(doc)
                        seen_ids.add(doc.cite_id)
                current = current.parent
            # 进行文档凝练的自由创作
            # 不考虑上下文连贯，不考虑大纲，不考虑 report 自己的 prompt，不考虑文本限制
            # 处理文档
            if not relevant_docs:
                logger.warning(f"节点 '{node.title}' (ID: {node.id}) 未找到关联文档")
                relevant_docs_text = "（无相关资料）"
            else:
                relevant_docs_text = "\n\n".join(
                    [
                        f"引用号:[{doc.cite_id}] 来源: {doc.title}\n"
                        f"{(doc.observation if getattr(doc, 'observation', None) else (doc.text[:500] if doc.text else ''))}..."
                        for doc in relevant_docs
                    ]
                )

                # logger.info(f"relevant_docs :{relevant_docs}")
            progress_context = get_progress_context(
                path_stack, will_complete_chapters, next_chapter
            )

            # prompt = f"""
            # 你是一个严谨的学术助手。

            # 根据以下文献资料，围绕主题“{node.title}”整合所有事实信息。

            # 要求：
            # 1. 每个句子必须有文献支持
            # 2. 每个陈述后必须使用数字型引用格式，如【1】【2】【3】
            # 3. 引用编号必须对应下方文献资料的顺序编号
            # 4. 不允许编造信息
            # 5. 不做风格润色，只做事实整合

            # 文献资料（按编号顺序排列）：
            # {relevant_docs_text}

            # 请输出整合后的事实草稿：
            # """

            # prompt = f"""
            # 你是一个严谨的学术助手。

            # ## 输入
            # - 用户原始查询：
            # {user_query}

            # - 当前节点：
            # {node.title}

            # - 参考文献（按编号顺序排列）：
            # {relevant_docs_text}

            # ## 任务
            # 请根据参考文献**整合生成当前节点的事实草稿**。内容必须满足以下要求：

            # 1. **对齐用户查询**
            # - 所有生成内容必须直接回应用户原始查询的相关部分。
            # - 不得扩展到与 query 无关的方向。
            # - 每个段落只需覆盖当前节点所承担的职责。

            # 2. **文献支撑与引用**
            # - 每条陈述必须有文献支撑，不允许编造。
            # - 引用编号必须使用【1】【2】【3】格式，并严格对应文献顺序。
            # - 避免生成重复或空引用。

            # 3. **内容完整性与结构**
            # - 尽可能整合文献中的关键事实。
            # - 避免遗漏与节点职责相关的重要信息。
            # - 不进行风格润色、扩写或主观评价，仅输出事实。

            # 4. **输出要求**
            # - 直接输出整合后的段落文本，不要包含其他解释性文字或总结。
            # - 每个段落尽量独立、可读，保持信息清晰。

            # ## 输出
            # 请生成“{node.title}”对应的事实整合段落文本。"""

            prompt = f'''你是一名专业的高级研究分析师，负责撰写一份深度调研报告的特定章节。你的目标是产出**高信息密度、硬核事实驱动**的内容，坚决杜绝空洞的术语堆砌。
            
            ### 一、 核心写作准则（防范“虚、散、重”）

            1. **实体化锚点（拒绝空泛）**：
            * 禁止使用“相关团队”、“某著名机构”、“多种算法”等模糊词汇。若资料包含具体机构、团队、数据库、年份或论文标题，必须准确呈现。
            * **强制要求**：若资料中包含具体的实验室名称、带头人姓名、软件版本、数据库代号、论文标题、具体年份或特定城市，**必须**完整准确地呈现在正文中。


            2. **量化证据链（拒绝定性描述）**：
            * 禁止使用“显著提升”、“效果良好”、“规模巨大”等形容词。
            * **强制要求**：若资料包含原始数值，必须呈现。若未提供具体数据，明确标注“未提供相关具体数据”，不得推测。

            3. **结构化对比（拒绝信息堆砌）**：
            * 当多个对象在同一维度可横向比较时，使用 Markdown 表格。
            * 若重点在机制解释或因果链条，使用段落分析。
            * 避免为单一对象单独建立表格。

            4. **因果逻辑闭环（提升深度洞察）**：
            * 在描述“挑战”或“局限性”时，应该建立逻辑链条。例如：`数据标注成本高 -> 导致样本量稀缺 -> 进而限制了模型的外推预测能力【2】【4】`。


            ### 二、 写作规范与格式

            1. 格式要求：
            - 使用规范的 Markdown 语法
            - **重要：不得添加任何标题（# ## ###），仅撰写段落内容**
            - 章节标题由系统自动生成
            - 呈现对比数据、统计结果、功能或选项时可以使用表格，但是不要每一次都使用表格
            - 正文内不包含行内引用标注
            - 追踪信息来源，保持正文简洁易读
            - 专业概念要解释


            2. **直接切入，无元描述**：
            * 严禁使用“本节主要介绍...”、“如前所述...”、“综上所述...”等过渡性废话。
            * 第一句话必须是该章节的核心事实或关键结论。主题要聚焦，结构要凝练严密，不要重复
            * 要突出重点，可以有表格。


            3. **引用规范（行内分散式）**：
            * 采用“事实陈述【编号】”格式。引用编号必须分散在句中或句末，严禁在一大段话最后堆砌一连串编号。
            * 每条陈述必须有文献支撑，不允许编造。  
            * 引用编号必须使用【1】【2】【3】格式，并严格对应文献顺序。  
            * 避免生成重复或空引用。


            4. **缺失信息处理**：
            * 若参考资料中缺失关键信息（如具体的数据库规模或产业化时间线），请明确标注“未提供相关具体数据”，不得进行主观臆测。

            ### 三、 数据完整性与表格模板

            涉及对比分析时，请参考以下逻辑结构：
            - 使用 Markdown 表格呈现对比数据、统计信息、功能或选项
            - 必须包含清晰的表头行，注明列名
            - 合理对齐列（文字左对齐，数字右对齐）
            - 表格简洁，聚焦关键信息
            - 使用规范 Markdown 表格语法：

            | 表头1 | 表头2 | 表头3 |
            |-------|-------|-------|
            | 数据1 | 数据2 | 数据3 |
            | 数据4 | 数据5 | 数据6 |

            ---

            ### 四、 任务上下文（由系统自动填充）

            **1. 原始用户查询：**
            { user_query }

            **2. 当前章节位置：**

            * 完整大纲：{ full_outline }
            * 当前节点：{node.title}

            **写作规范**：
            - 按照当前进度指示的章节撰写内容
            - 参考完整大纲，明确自身在全文结构中的位置
            - **不撰写结论性结尾**，完成本节内容即可


            **3. 基于证据的草稿/素材（核心干货来源）：**
            {relevant_docs_text}

            ## 输出
            请生成“{node.title}”对应的事实整合段落文本。"""
            **记住**：你的任务是**仅撰写当前章节的新增内容**，以上内容仅供参考。'''
            draft_prompt = [HumanMessage(content=prompt)]
            # step1 input
            # logger.info(f"草稿messages:{draft_prompt}")
            draft_response = llm.invoke(draft_prompt)
            draft_content = draft_response.content.strip()

            # 记录文档
            logger.info(
                f"----=====SUPPORT_DOCS_START=====----{relevant_docs}----=====SUPPORT_DOCS_END=====----"
            )
            # 记录step1 input
            logger.info(
                f"----=====STEP1_INPUT_START=====----{draft_prompt}----=====STEP1_INPUT_END=====----"
            )
            # 记录step1 output
            logger.info(
                f"----=====STEP1_OUTPUT_START=====----{draft_content}----=====STEP1_OUTPUT_END=====----"
            )

            # 处理字数限制
            word_limit = None  # 是零就不处理
            logger.info(f"node{node}")
            logger.info(
                f"node.word_limit = {node.word_limit}, type = {type(node.word_limit)}"
            )
            # logger.info(f"relevant_docs{relevant_docs}")

            if (
                isinstance(node.word_limit, int) and node.word_limit > 0
            ):  # 是正整数就处理
                word_limit = node.word_limit

            completed_content = "".join(report_parts).strip()
            if not completed_content:
                completed_content = "（尚未生成任何内容）"

            temp_state = {
                "messages": [],
                "user_query": user_query,
                "full_outline": full_outline,
                "progress_context": progress_context,
                "completed_content": completed_content,
                # "reference_materials": relevant_docs_text,
                "draft_content": draft_content,
                "locale": locale,
                "word_limit": word_limit,  # 词数限制
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
                        # "reference_materials": relevant_docs_text,
                        "draft_content": draft_content,
                        "locale": locale,
                        "word_limit": word_limit,  # 词数限制
                    },
                )

                # response = llm.invoke(messages)
                # content = response.content.strip()
                content = draft_content  # naive版本来个直接拼的，看看效果如何。

                report_parts.append(f"{content}\n")
                logger.debug(f"  生成了 {len(content)} 个字符")
                # logger.info(f"正文messages:{messages}")
                # logger.info(
                #     f"迁移输入开始开始开始标志标志标志{messages}迁移输入结束结束结束标志标志标志"
                # )
                # logger.info(
                #     f"迁移输出开始开始开始标志标志标志{content}迁移输出结束结束结束标志标志标志"
                # )
                # 记录step2 input
                logger.info(
                    f"----=====STEP2_INPUT_START=====----{messages}----=====STEP2_INPUT_END=====----"
                )
                # 记录step2 output
                logger.info(
                    f"----=====STEP2_OUTPUT_START=====----{content}----=====STEP2_OUTPUT_END=====----"
                )

                # 如果没文档就不做引用检查了，后面再考虑上文的引用
                # if not relevant_docs:
                #     logger.warning(
                #         f"节点 '{node.title}' (ID: {node.id}) 未找到关联文档，不进行引用检查"
                #     )
                # else:
                #     logger.info(f"content :{content}")
                #     logger.info(f"relevant_docs:{relevant_docs}")
                #     # 这个是判断引用和句子的关系
                #     supported = filter_content_by_relevant_docs(
                #         content=content,
                #         relevant_docs=relevant_docs,
                #         semantic_cls=semantic_cls,
                #     )
                #     logger.info(f"supported :{supported}")

                #     # 这个是把关系应用到生成文章上
                #     new_content = mark_content_with_support(
                #         content=content, nli_results=supported
                #     )
                #     logger.info(f"new_content :{new_content}")

                #     # 这个是把错误引用进行处理的
                #     content = repair_unknown_citations(
                #         content=new_content,
                #         relevant_docs=relevant_docs,
                #         semantic_cls=semantic_cls,
                #     )
                #     logger.info(f"content :{content}")

            except Exception as e:
                logger.error(f"  生成失败: {str(e)}")

        if node.children:
            path_stack.append([])
            for i, child in enumerate(node.children):
                if i == len(node.children) - 1:
                    child_will_complete = will_complete_chapters + [node.title]
                    child_next_chapter = next_chapter
                else:
                    child_will_complete = []
                    child_next_chapter = node.children[i + 1].title
                generate(
                    child,
                    level + 1,
                    child_will_complete,
                    child_next_chapter,
                    semantic_cls=semantic_cls,
                )
            path_stack.pop()

    generate(
        outline_root,
        level=1,
        will_complete_chapters=[],
        next_chapter=None,
        semantic_cls=semantic_cls,
    )

    final_report = "\n".join(report_parts)

    logger.info(f"FactStruct Stage 2 完成: 生成了 {len(final_report)} 个字符的报告")

    return final_report


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


if __name__ == "__main__":
    """
    测试 FactStruct Stage 2:
    基于大纲递归生成 Markdown 报告
    """

    from datetime import datetime

    print("========== START FACTSTRUCT STAGE2 DEBUG ==========")

    # =====================================================
    # 1️⃣ 构造 Memory
    # =====================================================

    memory = Memory()

    docs = [
        FactStructDocument(
            id="doc_1",
            cite_id="CIT001",
            source_type="journal",
            title="中性粒细胞募集机制",
            text="急性脑缺血后，中性粒细胞通过趋化因子被募集至缺血区域。",
            embedding=None,
            timestamp=datetime.now(),
        ),
        FactStructDocument(
            id="doc_2",
            cite_id="CIT002",
            source_type="journal",
            title="炎症因子释放机制",
            text="中性粒细胞释放IL-1β和TNF-α，加剧炎症反应。",
            embedding=None,
            timestamp=datetime.now(),
        ),
        FactStructDocument(
            id="doc_3",
            cite_id="CIT003",
            source_type="journal",
            title=" 根节点的",
            text="根节点，根节点，根节点",
            embedding=None,
            timestamp=datetime.now(),
        ),
    ]

    # =====================================================
    # 2️⃣ 构造 Outline（树结构）
    # =====================================================

    root = OutlineNode(id="node_0", title="中性粒细胞在脑缺血中的作用", word_limit=800)

    acute = OutlineNode(id="node_1", title="急性期炎症机制", word_limit=400)

    mech1 = OutlineNode(id="node_2", title="中性粒细胞募集机制", word_limit=200)

    mech2 = OutlineNode(id="node_3", title="炎症因子释放机制", word_limit=200)

    acute.add_child(mech1)
    acute.add_child(mech2)
    root.add_child(acute)

    print("\n--- OUTLINE TREE ---")
    print(root.to_text_tree(include_word_limit=True, include_mab_state=True))

    # =====================================================
    # 3️⃣ 构建 Memory 映射
    # =====================================================
    memory.map_node_to_docs("node_0", [docs[2]])
    memory.map_node_to_docs("node_1", [docs[0]])
    memory.map_node_to_docs("node_3", [docs[1]])

    print("\n--- MEMORY MAPPING ---")
    print(memory.node_to_docs)

    # =====================================================
    # 4️⃣ 序列化（模拟真实系统）
    # =====================================================

    outline_dict = outline_node_to_dict(root)
    memory_dict = memory_to_dict(memory)

    print("\n--- SERIALIZED OUTLINE ---")
    print(outline_dict)

    print("\n--- SERIALIZED MEMORY ---")
    print(memory_dict)

    # =====================================================
    # 5️⃣ 运行 Stage 2
    # =====================================================

    user_query = "请系统阐述中性粒细胞在脑缺血急性期的炎症作用机制"

    print("\n========== RUNNING STAGE 2 ==========")

    final_report = run_factstruct_stage2(
        outline_dict=outline_dict,
        memory_dict=memory_dict,
        user_query=user_query,
        llm_type="basic",
        locale="zh-CN",
    )

    # =====================================================
    # 6️⃣ 输出结果
    # =====================================================

    print("\n========== FINAL REPORT ==========")
    print(final_report)

    print("\n========== END STAGE2 DEBUG ==========")

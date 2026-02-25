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
            completed_content = "".join(report_parts).strip()
            if not completed_content:
                completed_content = "（尚未生成任何内容）"
            # 处理字数限制
            word_limit = None  # 是零就不处理
            logger.info(f"node{node}")
            logger.info(
                f"node.word_limit = {node.word_limit}, type = {type(node.word_limit)}"
            )

            if (
                isinstance(node.word_limit, int) and node.word_limit > 0
            ):  # 是正整数就处理
                word_limit = node.word_limit
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

            insight_prompt = f"""
            你是一名专业的高级研究分析师，负责撰写一份深度调研报告的特定章节。

            当前任务不是写报告正文，而是进行结构化推理：  
            从给定资料中提炼“分析逻辑骨架”，用于后续正式写作。

            你的目标是形成：机制清晰、可比较、可验证、可反驳的分析结构。

            禁止：
            - 不要写正式段落
            - 不要写引用编号
            - 不要写Markdown格式
            - 不要写修辞语言
            - 不要生成完整正文

            ----------------------------------------------------------------------
            一、核心认知原则
            ----------------------------------------------------------------------

            1. 机制优先原则（必须建立因果链）

            禁止仅描述现象或结果。  
            每一个趋势、增长、挑战或优势，必须解释“为什么发生”。

            必须构建完整逻辑链条：

            驱动因素 → 作用机制 → 中间变量 → 结果 → 可能反馈效应

            不得出现孤立结论。

            2. 比较抽象原则（避免平行罗列）

            当存在两个及以上对象（公司/技术/路径/区域/模型）时：

            - 必须抽象出共同比较维度
            - 必须指出差异来源
            - 不允许逐个对象割裂分析

            比较维度示例（根据情境选择）：
            技术路径 / 成本结构 / 商业模式 / 融资能力 / 风险暴露 / 可持续性 / 政策依赖度

            3. 趋势拆解原则

            若涉及增长、扩张、改善、衰退等变化：

            - 必须区分内生因素、政策因素、周期因素
            - 必须判断持续性条件

            4. 风险与假设显性化

            每一节必须识别：

            - 隐含关键假设
            - 潜在风险
            - 数据局限来源

            禁止给出无条件断言。

            5. 情景化思维（如涉及预测）

            若涉及未来趋势：

            - 至少构建两种情景
            - 明确触发条件
            - 不得线性外推

            必须输出以下结构化内容：

            1. 本章节核心判断（1-2句话）

            2. 信息域枚举  
            （本章节理论上必须覆盖的维度）

            3. 关键机制链条  
            （可列出多条）

            4. 若存在多个对象：  
            统一比较维度 + 差异来源

            5. 章节级表格规划（仅在必要时，不要每一次都生成表格，也不要整个文章都没有表格）：
            - 是否需要表格：仅在本章节存在可横向比较的多个对象且有量化数据时
            - 预估数量：整个文章全文建议 1~4 个表格
            - 每个表格用途：展示核心对比维度，如经济资源对比、政治资源对比、教育/社保资源对比
            - 表头设计：仅列出必要维度
            - 核心数据字段：列出关键对比指标
            - 建议位置：标明在章节哪个部分最合理出现
            - 提示：若章节对象不可比较或数据不足，可跳过表格

            提示：如果本章节对象不可比较或数据不足，可跳过表格。表格规划应作为章节整体策略，而非每条机制链条都生成。
            
            6. 数据可得性与缺口

            7. 风险与关键假设

            8. 若涉及趋势：  
            情景划分 + 触发条件

            9. 篇幅估计：  
            字数要参考字数限制提供的信息，生成指导信息，不要让下游文档生成超出限制。
            ----------------------------------------------------------------------

            原始用户查询：
            { user_query }

            输入资料：
            {relevant_docs_text}

            完整大纲：
            { full_outline }

            已经生成内容：
            {completed_content}

            章节标题：
            {node.title}
                            
            字数限制:
            {word_limit}

            ----------------------------------------------------------------------

            输出要求：

            - 使用清晰分点结构
            - 每一部分必须单独列出
            - 不要生成正文
            - 不要加入多余解释
            """

            # temp_state = {
            #     "messages": [],
            #     "user_query": user_query,
            #     "full_outline": full_outline,
            #     "progress_context": progress_context,
            #     "completed_content": completed_content,
            #     # "reference_materials": relevant_docs_text,
            #     "draft_content": draft_content,
            #     "locale": locale,
            #     "word_limit": word_limit,  # 词数限制
            # }

            try:
                # step1 input
                # 先构建 insight 草稿
                insight_message = [HumanMessage(content=insight_prompt)]

                insight_response = llm.invoke(insight_message)
                insight_content = insight_response.content.strip()

                # 记录文档
                logger.info(
                    f"----=====SUPPORT_DOCS_START=====----{relevant_docs}----=====SUPPORT_DOCS_END=====----"
                )
                # 记录step1 input
                logger.info(
                    f"----=====STEP1_INPUT_START=====----{insight_message}----=====STEP1_INPUT_END=====----"
                )
                # 记录step1 output
                logger.info(
                    f"----=====STEP1_OUTPUT_START=====----{insight_content}----=====STEP1_OUTPUT_END=====----"
                )

                # step2 input
                # 再构建正文

                content_prompt = f"""你是一名专业的高级研究分析师，负责撰写一份深度调研报告的特定章节。你的目标不是简单整合资料，而是进行结构化推理，产出“机制清晰、可比较、可验证、可反驳”的高密度分析内容。

                本阶段任务：  
                基于“结构化分析骨架（insight）”与“原始资料（relevant_docs_text）”，生成当前章节的正式分析正文。

                你的输出必须严格围绕 insight 展开，不得偏离其逻辑结构。

                ----------------------------------------------------------------------
                一、输入材料
                ----------------------------------------------------------------------

                结构化分析骨架（必须作为逻辑主线）:
                {insight_content}

                证据资料（唯一事实来源）:
                {relevant_docs_text}

                原始用户查询:
                {user_query}

                完整大纲:
                {full_outline}

                当前章节标题:
                {node.title}

                ----------------------------------------------------------------------
                二、核心写作原则（高约束模式）
                ----------------------------------------------------------------------

                1. 骨架驱动原则（禁止自由发挥）

                - 正文结构必须严格对应 insight 的逻辑结构
                - 所有机制链条必须被展开
                - 所有比较维度必须被对齐展开
                - 所有风险、假设、情景必须被落实为分析段落
                - 不得新增 insight 未出现的重要判断

                insight 是“逻辑蓝图”，正文只是“展开表达”。

                ----------------------------------------------------------------------
                2. 引用强制规则（最重要）
                ----------------------------------------------------------------------

                - 每一个事实陈述必须附带引用编号
                - 每一段必须至少包含1个引用
                - 不允许出现无来源断言
                - 不允许编造引用
                - 不允许合并引用堆砌在段尾
                - 引用必须分散在句中或句末
                - 引用编号严格【1】【2】【3】形式
                - 情景与假设用段落文字描述，不放在【】中，不要输出这样的内容“【insight】”

                引用格式必须为：
                事实内容【1】

                禁止：
                - 在一段话末尾堆积【1】【2】【3】
                - 出现未在资料中出现的信息

                如果资料中未提供具体数据，必须写：
                “未提供相关具体数据”

                ----------------------------------------------------------------------
                3. 语言风格控制（高级研究报告风格）
                ----------------------------------------------------------------------

                - 直接陈述核心判断，不使用“本节将…”等过渡语言
                - 禁止口语化表达
                - 禁止空泛形容词（如“显著”、“巨大”、“领先”）
                - 必须使用机制语言：驱动、传导、约束、边际变化、成本结构、风险暴露等
                - 多用因果句式，少用并列罗列
                - 逻辑密度高于叙述密度

                ----------------------------------------------------------------------
                4. 表格使用规则（选择性，不强制）
                ----------------------------------------------------------------------

                当 insight 中存在“可横向对比的多个对象”，并且：

                - 维度一致
                - 数据可对齐
                - 数量≥2

                则使用 Markdown 表格呈现核心对比信息。

                若重点在机制链条或因果分析，则使用段落展开。

                禁止：
                - 为单一对象建立表格
                - 仅为装饰而建立表格

                表格必须包含清晰表头，语法规范：

                | 维度 | A | B |
                |------|---|---|
                | 指标1 | 数值 | 数值 |

                ----------------------------------------------------------------------
                5. 机制展开要求
                ----------------------------------------------------------------------

                每一个机制链条必须完整展开为：

                驱动因素 → 传导机制 → 中间变量变化 → 结果 → 潜在反馈

                不得只写结果。

                ----------------------------------------------------------------------
                6. 风险与假设显性化
                ----------------------------------------------------------------------

                正文必须包含：

                - 关键隐含假设
                - 潜在风险变量
                - 若条件变化，结论如何调整

                表达形式示例：

                当前判断依赖于X假设成立；若Y发生变化，则该机制可能弱化【3】

                ----------------------------------------------------------------------
                7. 情景分析（若 insight 中包含）
                ----------------------------------------------------------------------

                必须构建：

                - 基准情景
                - 风险情景（或替代路径）

                每个情景必须说明触发条件。

                禁止线性外推。

                ----------------------------------------------------------------------
                三、格式要求
                ----------------------------------------------------------------------

                - 使用规范 Markdown
                - 不添加标题（系统自动生成）
                - 可以使用表格
                - 不写结论性收尾
                - 每150-200字必须有结构分隔（表格或独立段落）

                ----------------------------------------------------------------------
                四、输出目标
                ----------------------------------------------------------------------

                生成“{node.title}”对应的高密度分析正文。

                必须：
                - 严格依赖 insight
                - 严格依赖资料
                - 严格执行引用规范
                - 仅生成当前章节新增内容"""
                content_message = [HumanMessage(content=content_prompt)]

                content_response = llm.invoke(content_message)
                content_content = content_response.content.strip()
                # messages = apply_prompt_template(
                #     "reporter_factstruct",
                #     temp_state,
                #     extra_context={
                #         "user_query": user_query,
                #         "full_outline": full_outline,
                #         "progress_context": progress_context,
                #         "completed_content": completed_content,
                #         # "reference_materials": relevant_docs_text,
                #         "draft_content": draft_content,
                #         "locale": locale,
                #         "word_limit": word_limit,  # 词数限制
                #     },
                # )

                # response = llm.invoke(messages)
                # content = response.content.strip()
                content = content_content  # naive版本来个直接拼的，看看效果如何。

                report_parts.append(f"{content}\n")
                logger.debug(f"  生成了 {len(content)} 个字符")
                # 记录step2 input
                logger.info(
                    f"----=====STEP2_INPUT_START=====----{content_message}----=====STEP2_INPUT_END=====----"
                )
                # 记录step2 output
                logger.info(
                    f"----=====STEP2_OUTPUT_START=====----{content_content}----=====STEP2_OUTPUT_END=====----"
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

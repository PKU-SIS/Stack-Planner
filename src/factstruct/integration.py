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
    # semantic_cls = CrossEncoder(
    #     "/data1/Yangzb/Model/StructBert/cross-encoder/nli-deberta-v3-small"
    # )
    semantic_cls =None

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
            你是一名具备“第一性原理”思维的高级研究分析师。
            你的任务是为调研报告的特定章节构建【逻辑骨架】，这是后续写作的灵魂。

            ### 1. 核心推理原则 (思维模型)
            - **机制还原**：严禁只描述结果。必须拆解为：驱动因素 → 传导路径 → 关键变量变化 → 最终效应。
            - **对立统一**：任何技术或商业选择都有 Trade-off（权衡）。识别出选择 A 意味着放弃了什么的深度代价。
            - **横向拉通**：禁止孤立分析对象。必须提炼出“通用比较维度”（如：成本结构、安全性上限、政策敏感度），并在同一维度下强制对标。
            - **结构化质疑**：识别资料中的非共识点、数据缺口或逻辑矛盾，并将其转化为待验证的假设。

            ### 2 任务完整性约束（强制执行）
            - **范围锁定**：所有分析必须严格围绕用户查询的限定条件（时间/地域/对象/问题范围）。
            - **子问题拆解**：若用户查询包含多个子问题，必须先拆解，并在逻辑骨架中逐条覆盖。
            - **覆盖自检**：在输出前，隐式检查是否遗漏任何关键影响因素（政策/市场/技术/竞争/风险）。
            - **禁止发散**：不得讨论与用户问题无直接关联的背景信息。

            ### 3. 必须输出的结构化推演
            请按以下顺序简洁输出，禁止任何修辞或废话：

            **零、任务拆解与覆盖映射**
            - 用户核心问题：
            - 子问题拆解：
            - 本章节负责回答的问题：
            - 本章节不负责但相关的边界问题（明确排除）：

            **一、 核心判断 (Insight)**
            用1-2句话点透本章最深层的逻辑逻辑脉络（例如：XX竞争的本质已从“参数竞赛”转为“工程落地效率”）。

            **二、 机制逻辑链 (Logic Chains)**
            - 链条1：A 导致 B 的具体传导路径，重点标注中间的“触发条件”。
            - 链条2：...

            **三、 跨对象对标矩阵 (Comparative Framework)**
            - **对标对象**：列出对象名
            - **统一比较维度**：维度A、维度B、维度C
            - **差异根源**：解释为什么它们在同一维度下表现不同（底层是技术差异、资源禀赋还是战略定位？）。

            **四、 关键权衡与风险 (Trade-offs & Risks)**
            - 隐含假设：若要结论成立，必须满足什么前提？
            - 潜在风险：哪些变量（外部冲击、技术瓶颈）会导致机制失效？

            **五、 章节级表格设计 (必要时)**
            - 若存在量化对比或多对象多维度对标，请设计表头。
            - 仅提供：表格用途 + 关键指标/表头。

            **六、 写作策略与字数配比**
            - 建议重点展开的逻辑点。
            - 篇幅限制参考: {word_limit}。
            
            **七、全面性覆盖自检（内部强制检查）**
            - 是否覆盖所有核心驱动因素？
            - 是否包含风险与反向情景？
            - 是否包含横向比较？
            - 是否标出关键变量与触发条件？
            - 是否存在逻辑跳跃？
            ----------------------------------------------------------------------
            原始输入
            用户查询：{user_query}
            章节标题：{node.title}
            输入资料：{relevant_docs_text}
            完整大纲上下文：{full_outline}
            ----------------------------------------------------------------------
            请直接开始你的逻辑推演：
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

                content_prompt = f"""
                你是一名高级研究员。现在请你根据【逻辑骨架】和【证据资料】，撰写报告章节的正文。

                ### 1. 核心写作标准
                - **逻辑显性化**：正文必须严丝合缝地对应【逻辑骨架】中的机制链条。禁止略过中间步骤直接写结论。
                - **高频对标**：禁止孤立描述。在分析对象 A 时，必须通过“相比之下”、“不同于 B”等话术，在同一段落内完成横向比较。
                - **证据颗粒度**：拒绝模糊词汇（如“领先”、“很多”）。必须使用资料中的具体数值、协议名称、团队背景或融资细节。
                - **专业语境**：自动识别本章所属领域，强制调用行业底层术语（如：具身智能的Sim-to-Real、金融的流动性螺旋、光谱的对称性破缺）。记得要解释术语，提高读者友好性
                - **机制深度（还原链条）**：拒绝跳跃式结论。必须按 [驱动因素→传导路径→中间变量→结果→反馈] 展开论述。
                - **语境去水（高级风格）**：禁止“显著、领先、巨大”等形容词；禁止“本节将、综上所述”等废话。强制使用：边际变化、风险暴露、成本结构、传导约束等专业词汇。最好使用引用文本提供的具体数据进行分析
                - **动态风险显性化**：段落中必须包含“逻辑成立前提”与“风险变量”。（例：若XX假设失效，则机制XX将向反向演化【1】）。
                - **非线性情景**：若涉及预测，必须区分[基准/风险]情景，并明确对应的“触发开关”。
                - **任务回应优先**：正文必须明确回应用户查询中的每一个核心问题。若存在多个子问题，必须在正文中显式对应，不得隐含回答。
                - **结构完整性**：正文必须按以下逻辑展开：① 问题界定 → ② 驱动因素 → ③ 机制推导 → ④ 横向对标 → ⑤ 权衡与风险 → ⑥ 情景演化。禁止跳跃或遗漏。因为你是生成的文档片段，所以如果这个地方做的不好，可以根据上下文逻辑进行调整。
                - **覆盖完整性要求**：若资料涉及政策、技术、市场、竞争、风险等多个层面，正文必须系统覆盖，不得只选择有利于结论的部分展开。




                ### 2. 引用与格式规则
                - **引用格式**：必须坚持“一话一引”，在事实后标注【编号】。禁止段末堆砌。
                - **缺失处理**：若资料无数据，严禁编造。请基于逻辑推导并注明“根据[某机制]推断”，或写“相关量化数据缺失”。
                - **引用执行**：坚持“一话一引”，标注在句中事实处。禁止段末堆砌。
                - **表格执行**：若骨架中建议了表格，且资料支持，请立即生成高密度的 Markdown 表格。

                ### 3. 输入材料
                - **章节标题**：{node.title}
                - **逻辑骨架 (主线)**：
                {insight_content}

                - **证据资料 (事实来源)**：
                {relevant_docs_text}

                - **背景参考**：{user_query} | {full_outline} (仅参考上下文逻辑)

                ---
                ### 4. 输出目标
                撰写“{node.title}”的正式分析正文（无需标题和结尾）。直接进入深度分析：
                输出必须采用规范 Markdown 格式；
                每个核心逻辑模块必须使用加粗小标题；
                每个机制链条必须独立分段；
                每个风险变量必须单独列点；
                若出现多对象对标，必须使用表格；
                禁止连续超过200字的纯文本段落。
                不添加标题（系统自动生成）；正文不得出现总结性收尾段落。全文必须保持高密度结构化排布，每 150–200 字必须通过表格、分隔块或独立小段落进行结构切分，禁止连续大段文本，但不得使用空泛过渡语。
                """
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

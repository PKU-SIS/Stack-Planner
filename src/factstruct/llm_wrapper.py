"""
LLM Wrapper: LLM 批量方法包装器

实现了 Stage 1 所需的批量 LLM 调用方法：
- batch_generate_queries: 批量生成查询
- batch_refine_outline: 批量修纲
- generate_initial_outline: 生成初始大纲
"""

import json
import re
from typing import List, Tuple, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.utils.logger import logger
from src.prompts import get_prompt_template
from .outline_node import OutlineNode
from .document import FactStructDocument
# from jinja2 import Template




class FactStructLLMWrapper:
    """
    FactStruct LLM 方法包装器

    封装了 Stage 1 所需的所有 LLM 调用方法。
    """

    def __init__(self, llm: BaseChatModel):
        """
        初始化 LLM 包装器

        参数:
            llm: LangChain BaseChatModel 实例
        """
        self.llm = llm

    def generate_initial_outline(
        self,
        query: str,
        docs: List[FactStructDocument],
        central_guidance=None,
        replan_result=None,
        instruction=None,
    ) -> OutlineNode:
        """
        生成初始大纲

        参数:
            query: 用户查询
            docs: 初始检索到的文档列表

        返回:
            根 OutlineNode
        """
    
    
        # 格式化文档信息
        docs_text = self._format_documents(docs)

        parts = []

        parts.append(
            "你是一个研究助手。你的任务是**生成一个初始研究大纲骨架**，"
            "用于后续逐步扩展，而不是一次性完成最终大纲。\n"
        )

        parts.append("## 用户查询\n")
        parts.append(query + "\n")

        if docs_text:
            parts.append("## 初始文档（仅供参考，不要求完全覆盖）\n")
            parts.append(docs_text + "\n")

        if central_guidance:
            parts.append("## 中枢智能体总体指导\n")
            parts.append("请在构建初始大纲时参考以下方向性建议：\n")
            parts.append(central_guidance + "\n")

        if replan_result:
            parts.append("## Replan Result\n")
            if isinstance(replan_result, dict):
                for k, v in replan_result.items():
                    parts.append(f"- {k}: {v}\n")
            else:
                parts.append(str(replan_result) + "\n")

        if instruction:
            parts.append("## 大纲智能体指导\n")
            parts.append(instruction + "\n")

        parts.append(
            """
        ## 生成要求（非常重要）

        1. **这是初始化阶段，只生成最小可用的大纲骨架**
        2. 一级节点数量建议 **3–5 个**
        3. 仅在必要时为一级节点添加二级节点，**避免展开过细**
        4. 不要求覆盖所有细节，允许后续通过 expandation 工具补充
        5. 每个节点只需提供清晰、概括性的标题
        6. 避免“穷举式”或“百科全书式”结构

        ## 输出格式要求
        - 仅输出 JSON
        - 不要输出任何解释性文字
        - JSON 结构如下：

        {
        "title": "根节点标题",
        "children": [
            {
            "title": "一级节点标题",
            "children": []
            }
        ]
        }
        """
        )


        prompt = "\n".join(parts)
        logger.info(f"大纲初始输入 prompt:{prompt}")
        try:
            messages = [HumanMessage(content=prompt)]
            logger.info(f"initial outline prompt:{messages}")
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # 尝试提取 JSON（可能被 markdown 代码块包裹）
            json_str = self._extract_json(content)
            outline_data = json.loads(json_str)

            # 构建 OutlineNode 树
            # root = self._build_outline_tree(outline_data, parent=None, node_counter=[0])
            root = self._build_outline_tree(outline_data, parent=None)

            # self._inherit_mab_state_for_existing_nodes(None, root)
            new_node_ids = []
            self._inherit_mab_state_for_existing_nodes(None, root, new_node_ids=new_node_ids)
            logger.info(
                f"Generated initial outline with {len(root.get_all_nodes())} nodes"
            )
            return root

        except Exception as e:
            import traceback

            logger.error(f"Failed to generate initial outline: {e}")
            logger.error(f"Detailed error:\n{traceback.format_exc()}")
            # 返回一个简单的默认大纲
            return OutlineNode(
                id="root_0",
                title=query,
                parent=None,
                children=[],
            )

    def batch_generate_queries(self, nodes: List[OutlineNode]) -> List[str]:
        """
        批量生成查询（单次 LLM 调用）

        参数:
            nodes: 需要生成查询的节点列表

        返回:
            查询字符串列表，与 nodes 一一对应
        """
        if not nodes:
            return []

        # 构建批量查询的 prompt
        nodes_info = []
        for i, node in enumerate(nodes, 1):
            parent_context = node.get_parent_context()
            context_str = f"（上下文：{parent_context}）" if parent_context else ""
            nodes_info.append(f"{i}. 节点: '{node.title}'{context_str}")

        prompt = f"""你是一个研究助手。请为以下 {len(nodes)} 个大纲节点分别生成一个精确的搜索查询。

        ## 节点列表
        {chr(10).join(nodes_info)}

        ## 要求
        1. 为每个节点生成一个精确、具体的搜索查询
        2. 查询应该能够帮助检索到与该节点主题相关的文档
        3. 如果节点有上下文信息，请在查询中体现
        4. 输出格式必须严格按照以下格式：
        查询 1: [查询内容]
        查询 2: [查询内容]
        ...
        查询 {len(nodes)}: [查询内容]

        请只输出查询，每行一个，严格按照上述格式。"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # 解析查询
            queries = self._parse_batch_queries(content, len(nodes))

            if len(queries) != len(nodes):
                logger.warning(
                    f"Expected {len(nodes)} queries, got {len(queries)}. "
                    "Using node titles as fallback."
                )
                # 使用节点标题作为备用查询
                queries = [node.title for node in nodes]

            return queries

        except Exception as e:
            import traceback

            logger.error(f"Failed to batch generate queries: {e}")
            logger.error(f"Detailed error:\n{traceback.format_exc()}")
            # 使用节点标题作为备用查询
            return [node.title for node in nodes]

    def batch_refine_outline(
        self,
        current_outline: OutlineNode,
        node_doc_pairs: List[Tuple[OutlineNode, List[FactStructDocument]]],
        memory: "Memory" = None,
    ) -> Tuple[
        OutlineNode,
        List[Tuple[OutlineNode, List[OutlineNode]]],
        Dict[str, List[FactStructDocument]],
    ]:
        """
        批量修纲（单次 LLM 调用）

        这是 Stage 1 最复杂的方法，要求 LLM 在单次调用中理解并执行 K 个独立的局部修改。

        参数:
            current_outline: 当前大纲根节点
            node_doc_pairs: (节点, 新文档列表) 的元组列表
            memory: Memory 实例，用于获取节点的累积文档映射

        返回:
            (修订后的大纲根节点, expanded_nodes_list, new_node_doc_mapping)
            expanded_nodes_list: [(父节点, [新子节点1, 新子节点2, ...]), ...] 的列表，
                                记录哪些节点被扩展了以及它们的新子节点
            new_node_doc_mapping: {新子节点ID: [匹配的文档列表]} 的字典，
                                记录每个新子节点与文档的匹配关系
        """
        if not node_doc_pairs:
            return current_outline, [], {}

        # 构建批量修纲的 prompt
        outline_text = current_outline.to_text_tree()

        optimization_tasks = []
        for i, (node, docs) in enumerate(node_doc_pairs, 1):
            docs_text = self._format_documents(docs, max_chars=500)  # 限制文档长度
            parent_context = node.get_parent_context()
            context_str = (
                f"（上下文：{parent_context} > {node.title}）"
                if parent_context
                else f"（节点：{node.title}）"
            )

            optimization_tasks.append(
            f"""
            优化任务 {i}:
            - 节点: '{node.title}'
            - 上下文: {context_str}
            - 新信息: {docs_text}
            - 要求: 在该节点下增加 2-4 个并列的子章节。**每个子章节的标题必须直接来源于或对应上述"新信息"中的某个具体内容点**，确保子章节与文档有明确的对应关系。"""
            )

        prompt = f"""你是一个研究助手。我们刚刚检索了 {len(node_doc_pairs)} 个节点，获得了新信息。你的任务是根据这些新信息，对大纲进行 {len(node_doc_pairs)} 次 *独立的局部优化*。

        ## 当前研究大纲
        {outline_text}

        {chr(10).join(optimization_tasks)}

        ## 要求
        1. 对每个优化任务，**独立地**进行局部修改
        2. **重要**：为每个目标叶子节点生成 2-4 个并列的子节点，避免只生成单个子节点导致大纲变成"斜树"
        3. 子节点之间应该是并列关系，覆盖该主题的不同方面，而不是层层嵌套
        4. 修改时只影响目标节点及其子节点，不要影响其他不相关的节点
        5. 修改后的大纲应该保持层次结构清晰、宽度均衡
        6. **关键**：每个新生成的子节点标题必须能够在对应的"新信息"文档中找到明确的内容支撑，不要生成与文档内容无关的空泛标题。子节点标题应该具体、有信息量，能够直接对应到某个文档的核心内容
        7. 输出格式必须是 JSON，结构如下：
        {{
            "title": "根节点标题",
            "children": [
                {{
                    "title": "子节点1标题",
                    "children": []
                }},
                {{
                    "title": "子节点2标题（被扩展的叶子节点）",
                    "children": [
                        {{
                            "title": "并列子节点A",
                            "children": []
                        }},
                        {{
                            "title": "并列子节点B",
                            "children": []
                        }},
                        {{
                            "title": "并列子节点C",
                            "children": []
                        }}
                    ]
                }}
            ]
        }}

        请只输出 JSON，不要包含其他解释性文字。输出完整的修订后大纲树。"""

        try:
            logger.info(f"batch_refine_outline prompt{prompt}")
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            logger.info(f"batch_refine_outline response{response}")
            content = response.content.strip()

            # 提取 JSON
            json_str = self._extract_json(content)
            outline_data = json.loads(json_str)

            # 构建新的 OutlineNode 树
            # new_root = self._build_outline_tree(outline_data, parent=None, node_counter=[0])
            new_root = self._build_outline_tree(outline_data, parent=None)
            # 验证：检查返回的大纲是否真的包含了对目标节点的修改
            # 这是一个启发式验证，检查目标节点是否仍然存在（可能被修改或移动）
            validation_passed = self._validate_batch_refine_result(
                current_outline, new_root, node_doc_pairs
            )

            if not validation_passed:
                logger.warning(
                    "Batch refine validation failed: returned outline may not contain expected changes. "
                    "Returning original outline."
                )
                return current_outline, [], {}

            # 重要：识别哪些节点被扩展了，并构建 expanded_nodes_list 和新子节点的文档映射
            expanded_nodes_list, new_node_doc_mapping = self._identify_expanded_nodes(
                current_outline, new_root, node_doc_pairs, memory
            )

            # 先继承所有现有节点的状态（通过路径匹配）
            # self._inherit_mab_state_for_existing_nodes(current_outline, new_root)
            new_node_ids = []
            self._inherit_mab_state_for_existing_nodes(current_outline, new_root, new_node_ids=new_node_ids)
            logger.info(
                f"Refined outline with {len(new_root.get_all_nodes())} nodes, "
                f"{len(expanded_nodes_list)} nodes expanded, "
                f"{len(new_node_doc_mapping)} new nodes mapped to documents"
            )
            return new_root, expanded_nodes_list, new_node_doc_mapping

        except Exception as e:
            import traceback

            logger.error(f"Failed to batch refine outline: {e}")
            logger.error(f"Detailed error:\n{traceback.format_exc()}")
            # 返回原大纲（不修改）
            return current_outline, [], {}

    def _format_documents(
        self,
        docs: List[FactStructDocument],
        max_chars: int = 1000,
    ) -> str:
        """格式化文档列表为文本"""
        if not docs:
            return "无文档"

        formatted = []
        total_chars = 0

        for i, doc in enumerate(docs[:5], 1):  # 最多显示 5 个文档
            doc_text = f"\n文档 {i}:\n"
            if doc.title:
                doc_text += f"标题: {doc.title}\n"
            if doc.url:
                doc_text += f"来源: {doc.url}\n"

            # 截断文档内容
            content = doc.text[:max_chars]
            if len(doc.text) > max_chars:
                content += "..."

            doc_text += f"内容: {content}\n"

            if total_chars + len(doc_text) > max_chars * 3:  # 总体限制
                formatted.append(f"\n... 还有 {len(docs) - i} 个文档")
                break

            formatted.append(doc_text)
            total_chars += len(doc_text)

        return "".join(formatted)

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON（可能被 markdown 代码块包裹）"""
        # 尝试提取代码块中的 JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # 尝试直接提取 JSON 对象
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # 如果都没找到，返回原文本
        return text

    def _parse_batch_queries(self, content: str, expected_count: int) -> List[str]:
        """解析批量查询输出"""
        queries = []

        # 尝试匹配 "查询 N: [内容]" 格式
        pattern = r"查询\s*(\d+)\s*:\s*(.+?)(?=查询\s*\d+\s*:|$)"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

        if matches:
            # 按编号排序
            matches.sort(key=lambda x: int(x[0]))
            queries = [match[1].strip() for match in matches]
        else:
            # 备用：按行分割
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            queries = lines[:expected_count]

        return queries

    # def _build_outline_tree(
    #     self,
    #     data: dict,
    #     parent: OutlineNode = None,
    #     node_counter: List[int] = None,
    # ) -> OutlineNode:
    #     """递归构建 OutlineNode 树"""
    #     if node_counter is None:
    #         node_counter = [0]

    #     node_counter[0] += 1
    #     node_id = f"node_{node_counter[0]}"

    #     node = OutlineNode(
    #         id=node_id,
    #         title=data.get("title", "未命名节点"),
    #         parent=parent,
    #         children=[],
    #     )

    #     # 递归构建子节点
    #     for child_data in data.get("children", []):
    #         child = self._build_outline_tree(
    #             child_data, parent=node, node_counter=node_counter
    #         )
    #         node.add_child(child)

    #     return node
    def _build_outline_tree(
        self,
        data: dict,
        parent: OutlineNode = None,
    ) -> OutlineNode:
        """递归构建 OutlineNode 树（只构结构，不分配 id）"""

        node = OutlineNode(
            id=None,  # 🔥 这里先不分配
            title=data.get("title", "未命名节点"),
            parent=parent,
            children=[],
        )

        for child_data in data.get("children", []):
            child = self._build_outline_tree(
                child_data, parent=node
            )
            node.add_child(child)

        return node



    def _identify_expanded_nodes(
        self,
        old_outline: OutlineNode,
        new_outline: OutlineNode,
        node_doc_pairs: List[Tuple[OutlineNode, List[FactStructDocument]]],
        memory: "Memory" = None,
    ) -> Tuple[
        List[Tuple[OutlineNode, List[OutlineNode]]], Dict[str, List[FactStructDocument]]
    ]:
        """
        识别哪些节点被扩展了（从叶子节点变成了有子节点的内部节点），
        并为每个新子节点匹配相关文档。

        参数:
            old_outline: 旧大纲根节点
            new_outline: 新大纲根节点
            node_doc_pairs: 目标节点-文档对列表（这些节点是我们希望被扩展的）
            memory: Memory 实例，用于获取节点的累积文档

        返回:
            (expanded_nodes_list, new_node_doc_mapping)
            expanded_nodes_list: [(父节点, [新子节点1, 新子节点2, ...]), ...] 的列表
                                使用列表而不是字典，因为 OutlineNode 不可哈希
            new_node_doc_mapping: {新子节点ID: [匹配的文档列表]} 的字典
        """

        def get_node_path(node: OutlineNode) -> str:
            """获取节点的完整路径（从根到当前节点）"""
            path_parts = []
            current = node
            while current is not None:
                path_parts.insert(0, current.title)
                current = current.parent
            return " > ".join(path_parts)

        def find_node_by_path(root: OutlineNode, target_path: str) -> OutlineNode:
            """根据路径查找节点"""
            for node in root.get_all_nodes():
                if get_node_path(node) == target_path:
                    return node
            return None

        def match_child_to_docs(
            child_node: OutlineNode, docs: List[FactStructDocument]
        ) -> List[FactStructDocument]:
            """
            为新子节点匹配最相关的文档。
            使用简单的文本匹配策略：检查子节点标题中的关键词是否出现在文档中。
            """
            if not docs:
                return []

            child_title_lower = child_node.title.lower()
            # 提取子节点标题中的关键词（去掉常见停用词）
            stopwords = {
                "的",
                "和",
                "与",
                "在",
                "是",
                "了",
                "有",
                "为",
                "the",
                "a",
                "an",
                "and",
                "or",
                "of",
                "in",
                "to",
                "for",
            }
            keywords = [
                w
                for w in child_title_lower.split()
                if w not in stopwords and len(w) > 1
            ]

            matched_docs = []
            for doc in docs:
                doc_text_lower = (doc.text + " " + (doc.title or "")).lower()
                # 计算匹配的关键词数量
                match_count = sum(1 for kw in keywords if kw in doc_text_lower)
                if match_count > 0:
                    matched_docs.append((match_count, doc))

            # 按匹配度排序，返回匹配的文档
            matched_docs.sort(key=lambda x: x[0], reverse=True)

            if matched_docs:
                return [doc for _, doc in matched_docs]
            else:
                # 如果没有明确匹配，返回所有文档（继承父节点的全部文档）
                return docs

        expanded_nodes_list = []
        new_node_doc_mapping: Dict[str, List[FactStructDocument]] = {}

        # 对于每个目标节点，检查它是否被扩展了
        for target_node, new_docs in node_doc_pairs:
            # 获取目标节点在旧大纲中的路径
            old_path = get_node_path(target_node)

            # 检查目标节点在旧大纲中是否是叶子节点
            if not target_node.is_leaf():
                # 如果目标节点在旧大纲中已经有子节点，跳过（我们不处理这种情况）
                continue

            # 在新大纲中查找对应的节点（通过路径匹配）
            new_matched_node = find_node_by_path(new_outline, old_path)

            # 如果路径匹配失败，尝试通过标题匹配（节点可能被重命名但路径变化不大）
            if new_matched_node is None:
                # 尝试查找标题相同且位置相似的节点
                new_matched_node = self._find_similar_node_by_title(
                    new_outline, target_node
                )

            if new_matched_node is None:
                logger.debug(
                    f"Could not find matched node for '{target_node.title}' "
                    f"(path: {old_path}) in new outline"
                )
                continue

            # 检查新节点是否有子节点（即是否被扩展了）
            if new_matched_node.children:
                # 节点被扩展了！记录新子节点
                expanded_nodes_list.append((target_node, new_matched_node.children))

                # 获取该节点的累积文档（包括之前迭代积累的 + 本轮新增的）
                # 优先使用 memory 中的累积文档，如果 memory 不可用则使用本轮新文档
                all_docs = new_docs  # 默认使用本轮新文档
                if memory is not None:
                    accumulated_docs = memory.get_docs_by_node(target_node.id)
                    if accumulated_docs:
                        all_docs = accumulated_docs
                        logger.debug(
                            f"Using {len(all_docs)} accumulated docs for node '{target_node.title}'"
                        )

                # 为每个新子节点匹配文档
                for child_node in new_matched_node.children:
                    matched_docs = match_child_to_docs(child_node, all_docs)
                    new_node_doc_mapping[child_node.id] = matched_docs
                    logger.debug(
                        f"Child node '{child_node.title}' matched {len(matched_docs)} documents"
                    )

                logger.info(
                    f"Node '{target_node.title}' expanded: "
                    f"{len(new_matched_node.children)} new children, "
                    f"all mapped to documents (from {len(all_docs)} available docs)"
                )

        return expanded_nodes_list, new_node_doc_mapping

    def _find_similar_node_by_title(
        self,
        root: OutlineNode,
        target_node: OutlineNode,
    ) -> OutlineNode:
        """
        通过标题和父节点上下文查找相似节点

        这个方法用于处理节点被重命名的情况。
        """
        # 获取目标节点的父节点上下文
        parent_context = target_node.get_parent_context()

        # 在新大纲中查找标题相同的节点
        for node in root.get_all_nodes():
            if node.title == target_node.title:
                # 检查父节点上下文是否匹配
                node_parent_context = node.get_parent_context()
                if node_parent_context == parent_context:
                    return node

        # 如果找不到完全匹配的，返回第一个标题相同的节点
        for node in root.get_all_nodes():
            if node.title == target_node.title:
                return node

        return None


    def _inherit_mab_state_for_existing_nodes(
        self,
        old_root: OutlineNode,
        new_root: OutlineNode,
        new_node_ids: List[str] = None,
    ):
        """
        为现有节点继承 MAB 状态（pull_count, reward_history）
        感觉就是因为有个 new_root有个 old_root所以做了个复制粘贴。
        通过节点路径（从根到当前节点的路径）匹配来找到对应的节点并继承状态。
        这个方法只处理旧大纲中已存在的节点，不包括新扩展的子节点。

        匹配策略：
        1. 优先使用完整路径匹配（根 > 父 > 当前）
        2. 如果路径匹配失败，回退到标题匹配
        3. 如果都失败，节点状态重置为初始值（pull_count=0, reward_history=[]）
        """

        if new_node_ids is None:
            new_node_ids = []

        if old_root is None:
            # 第一次初始化
            for node in new_root.get_all_nodes():
                node.id = OutlineNode.allocate_id()
                new_node_ids.append(node.id)  # 收集新节点
            return


        def get_node_path(node: OutlineNode) -> str:
            """获取节点的完整路径（从根到当前节点）"""
            path_parts = []
            current = node
            while current is not None:
                path_parts.insert(0, current.title)
                current = current.parent
            return " > ".join(path_parts)

        # 构建旧节点的路径映射（路径 -> 节点）
        old_nodes_by_path = {}
        old_nodes_by_title = {}
        for node in old_root.get_all_nodes():
            path = get_node_path(node)
            old_nodes_by_path[path] = node
            # 也保留标题映射作为备用（可能会有重复标题，但至少能找到第一个）
            if node.title not in old_nodes_by_title:
                old_nodes_by_title[node.title] = node



        def inherit_recursive(new_node: OutlineNode):
            """递归继承状态"""
            new_path = get_node_path(new_node)

            # 策略1：优先使用路径匹配（最准确）
            if new_path in old_nodes_by_path:
                old_node = old_nodes_by_path[new_path]
                new_node.pull_count = old_node.pull_count
                new_node.reward_history = old_node.reward_history.copy()
                new_node.id = old_node.id#增加node_id的继承
                logger.debug(
                    f"State inherited for node '{new_node.title}' via path match "
                    f"(pull_count={old_node.pull_count})"
                )
            # 策略2：回退到标题匹配（可能不够准确，但比丢失状态好）
            elif new_node.title in old_nodes_by_title:
                old_node = old_nodes_by_title[new_node.title]
                new_node.pull_count = old_node.pull_count
                new_node.reward_history = old_node.reward_history.copy()
                logger.debug(
                    f"State inherited for node '{new_node.title}' via title match "
                    f"(path may differ, pull_count={old_node.pull_count})"
                )
            # 策略3：无法匹配，保持默认状态（pull_count=0, reward_history=[]）
            else:
                new_node.id = OutlineNode.allocate_id()
                new_node_ids.append(new_node.id)  # 收集新节点
                logger.debug(
                    f"Node '{new_node.title}' (path: {new_path}) not found in old outline, "
                    "using default state"
                )

            # 递归处理子节点
            for child in new_node.children:
                inherit_recursive(child)

        inherit_recursive(new_root)

    def _validate_batch_refine_result(
        self,
        old_outline: OutlineNode,
        new_outline: OutlineNode,
        node_doc_pairs: List[Tuple[OutlineNode, List[FactStructDocument]]],
    ) -> bool:
        """
        验证批量修纲结果

        检查返回的新大纲是否真的包含了对目标节点的修改。
        这是一个启发式验证，主要检查：
        1. 新大纲是否比旧大纲有变化（节点数量或结构）
        2. 目标节点的标题是否在新大纲中存在（可能被修改或移动）

        参数:
            old_outline: 旧大纲根节点
            new_outline: 新大纲根节点
            node_doc_pairs: 目标节点-文档对列表

        返回:
            True 表示验证通过，False 表示验证失败
        """
        # 验证1：检查大纲是否有变化
        old_node_count = len(old_outline.get_all_nodes())
        new_node_count = len(new_outline.get_all_nodes())

        # 如果节点数量没有变化，且结构完全相同，可能 LLM 没有执行修改
        if old_node_count == new_node_count:
            old_titles = {node.title for node in old_outline.get_all_nodes()}
            new_titles = {node.title for node in new_outline.get_all_nodes()}
            if old_titles == new_titles:
                logger.warning(
                    "Batch refine returned outline with no changes in structure or titles"
                )
                return False

        # 验证2：检查目标节点的标题是否在新大纲中存在
        # （即使节点被修改或移动，标题应该仍然存在，或者有相似的新标题）
        target_titles = {node.title for node, _ in node_doc_pairs}
        new_titles = {node.title for node in new_outline.get_all_nodes()}

        # 至少应该有一些目标节点的标题在新大纲中出现
        # 或者新大纲中有明显的新节点（说明进行了扩展）
        matched_titles = target_titles & new_titles
        if not matched_titles and new_node_count <= old_node_count:
            logger.warning(
                f"Batch refine validation: None of target titles {target_titles} "
                f"found in new outline, and no expansion detected"
            )
            return False

        # 验证3：检查新大纲是否有明显的扩展（新增节点）
        if new_node_count > old_node_count:
            logger.debug(
                f"Batch refine validation passed: outline expanded "
                f"({old_node_count} -> {new_node_count} nodes)"
            )
            return True

        # 如果以上验证都通过，认为验证成功
        logger.debug(
            f"Batch refine validation passed: {len(matched_titles)}/{len(target_titles)} "
            f"target titles found in new outline"
        )
        return True



    def compress_under_parent(
        self,
        outline_root: "OutlineNode",
        parent_node: "OutlineNode",
        child_nodes: List["OutlineNode"],
        memory: "Memory" = None,
    ) -> Tuple[
        "OutlineNode",
        List[Tuple["OutlineNode", List["OutlineNode"]]],
        # Dict[str, List["FactStructDocument"]],
        Dict[str, List[str]],
        Dict[str, List[str]],
    ]:
        """
        在指定父节点下压缩多个相似子节点

        返回:
            new_root: 压缩后的大纲根节点
            compressed_nodes_list: [(父节点, [新生成子节点])]
            new_node_doc_mapping: {新节点ID: [文档列表]}
            merged_node_mapping: {新节点ID: [被压缩的旧节点ID列表]}
        """
        print("len(parent_node.children)",len(parent_node.children))
        print("child_nodes",child_nodes)
        if not parent_node.children:
            logger.info(f"父节点 '{parent_node.title}' 没有子节点，跳过压缩")
            return outline_root, [], {}, {}

        elif len(parent_node.children) <= 2:
            # 1 或 2 个子节点，进行结构折叠（不走 LLM）
            original_children = child_nodes

            # 1️⃣ 提升孙节点
            self.flatten_children(parent_node, original_children)

            # 2️⃣ 更新 memory
            merged_docs = []
            merged_child_ids = []

            if memory:
                merged_docs = []
                for child in original_children:
                    doc_ids = memory.node_to_docs.get(child.id, set())
                    if doc_ids:
                        merged_docs.extend(doc_ids)
                        del memory.node_to_docs[child.id]

                if merged_docs:
                    merged_doc_objs = [memory.documents[doc_id] for doc_id in merged_docs if doc_id in memory.documents]
                    memory.map_node_to_docs(parent_node.id, merged_doc_objs)

            logger.info(
                f"父节点 '{parent_node.title}' 子节点数量为 {len(original_children)}，已进行结构折叠"
            )

            return (
                outline_root,
                [(parent_node, original_children)],
                {parent_node.id: merged_docs},
                {parent_node.id: [child.id for child in original_children]},
            )


        # 多子节点压缩逻辑
        # 1️⃣ 构造当前大纲文本
        outline_text = outline_root.to_text_tree()

        # 2️⃣ 构造子节点描述（简要文献信息）
        children_desc = []
        merged_source_ids = []

        for node in child_nodes:
            merged_source_ids.append(node.id)
            docs = memory.node_to_docs.get(node.id, []) if memory else []

            if docs:
                doc_summaries = []
                for doc in docs:
                    if doc.title:
                        doc_summaries.append(doc.title().strip()) #doc.title.strip()
                    elif doc.text:
                        doc_summaries.append(doc.text[:50].strip() + "…")
                docs_brief = (
                    f"{len(docs)} 篇相关文献，主题包括：" + "；".join(doc_summaries[:5])
                )
                if len(doc_summaries) > 5:
                    docs_brief += f" 等（共 {len(doc_summaries)} 个主题锚点）"
            else:
                docs_brief = "无直接文献（由上层语义拆分而来）"

            children_desc.append(
                f"- 子节点标题: {node.title}\n- 文献信息摘要: {docs_brief}"
            )

        parent_context = parent_node.get_parent_context()
        context_str = f"{parent_context} > {parent_node.title}" if parent_context else parent_node.title



        # 3️⃣ 构造压缩 prompt
        prompt = f"""
        你是一个研究助手，正在对研究大纲进行**结构压缩优化**。
        你必须对父节点下的子节点至少执行一次合并或压缩操作，生成新的子节点。
        新生成的子节点标题必须具体且信息量充足，不能沿用原子节点标题原封不动。

        ## 当前大纲
        {outline_text}

        ## 压缩目标
        父节点：{context_str}

        该父节点下存在多个语义高度相似、信息量偏少的子节点，需要进行合并压缩。

        ## 待压缩子节点
        {chr(10).join(children_desc)}

        ## 要求
        1. 你的任务是对「某一父节点下的子节点」**独立地**进行局部修改语义层级压缩与结构重组。
        2. 当前父节点的子节点数量 ≥ 3，子节点均为同一语义层级，不涉及跨父节点调整。
        3. 在不丢失关键信息的前提下，对**部分或全部**子节点进行语义合并，生成新的子节点
        ## 子节点数量压缩规则（强约束，必须遵守）

        当前父节点下共有 N = {len(children_desc)} 个子节点。
        压缩后的子节点数量必须满足以下规则：
        1. 如果 N = 3 → 必须压缩为 2 个子节点
        2. 如果 N = 4 → 必须压缩为 2 或 3 个子节点
        3. 如果 N ≥ 5 → 必须压缩为 ⌊N/2⌋ 或 ⌈N/2⌉ 个子节点
        4. 无论任何情况：
        - 子节点数量必须 < N
        - 子节点数量必须 ≥ 2
        - 不允许保留原数量不变
        - 不允许压缩为 1 个

        如果你的输出不满足上述数量规则，则视为结构错误。
        你必须在语义合理的前提下，通过合并操作，使最终子节点数量严格符合规则。
        4. 压缩后的新的子节点之间应该是并列关系，覆盖该主题的不同方面，而不是层层嵌套，新生成的子节点需要在语义上完整覆盖其所合并的原子节点的核心信息
        5. 仅允许修改当前父节点的子节点，子节点必须要是叶子节点，非叶子节点不参与合并，父节点之外的任何节点结构、顺序、层级均不得修改。父节点的title 不能更改
        6. 合并后，父节点下仅保留，新生成的子节点，以及未参与合并的原始子节点，未参与合并的子节点，其标题与语义需保持不变
        7. **关键**：每个新生成的子节点标题必须能够在对应的"新信息"文档中找到明确的内容支撑，不要生成与文档内容无关的空泛标题。子节点标题应该具体、有信息量，能够直接对应到某个文档的核心内容
        8. 输出格式必须是 JSON，结构如下：
        {{
            "title": "根节点标题",
            "children": [
                {{
                    "title": "子节点1标题",
                    "children": []
                }},
                {{
                    "title": "子节点2标题（被扩展的叶子节点）",
                    "children": [
                        {{
                            "title": "并列子节点A",
                            "children": []
                        }},
                        {{
                            "title": "并列子节点B",
                            "children": []
                        }},
                        {{
                            "title": "并列子节点C",
                            "children": []
                        }}
                    ]
                }}
            ]
        }}

        请只输出 JSON，不要包含其他解释性文字。输出完整的修订后大纲树。"""


        try:
            logger.info(f"compress_under_parent prompt:\n{prompt}")
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            content = response.content.strip()
            logger.info(f"content{content}")
            # 4️⃣ 解析 JSON 并重建大纲树
            json_str = self._extract_json(content)
            outline_data = json.loads(json_str)
            # new_root = self._build_outline_tree(outline_data, parent=None, node_counter=[0])
            new_root = self._build_outline_tree(outline_data, parent=None)
            # 继承 MAB 状态
            new_node_ids = []
            self._inherit_mab_state_for_existing_nodes(outline_root, new_root, new_node_ids=new_node_ids)
            #原版本
            # self._inherit_mab_state_for_existing_nodes(outline_root, new_root)

            # 找到压缩后的父节点及其新子节点
            def get_node_path(node):
                path_parts = []
                current = node
                while current is not None:
                    path_parts.insert(0, current.title)
                    current = current.parent
                return " > ".join(path_parts)

            def find_node_by_path(root, target_path):
                for node in root.get_all_nodes():
                    if get_node_path(node) == target_path:
                        return node
                return None
            
            # 5️⃣ 找到压缩后的父节点及其新子节点
            target_path = get_node_path(parent_node)
            new_parent = find_node_by_path(new_root, target_path)
            # new_parent = new_root.find_node_by_path(parent_node.get_path_titles())
            if not new_parent:
                logger.warning("Parent node not found after compression")
                return outline_root, [], {}, {}

            # new_children = new_parent.children or []
            new_children = []
            # new_node_ids 是 id 列表
            for node_id in new_node_ids:
                node = new_root.find_node_by_id(node_id)
                if node:
                    new_children.append(node)
                else:
                    logger.warning(f"Node with id {node_id} not found in new_root")

            compressed_nodes_list = [(new_parent, new_children)]

            # 6️⃣ 构建 merged_node_mapping 和 new_node_doc_mapping
            merged_node_mapping = {}
            new_node_doc_mapping = {}
            for child in new_children:
                merged_node_mapping[child.id] = merged_source_ids
                merged_docs = []
                for old_id in merged_source_ids:
                    merged_docs.extend(memory.node_to_docs.get(old_id, []))
                if merged_docs:
                    new_node_doc_mapping[child.id] = merged_docs

        
            logger.info(f"Compression success: { len(parent_node.children)} -> {len(new_children)} nodes")
            return new_root, compressed_nodes_list, new_node_doc_mapping, merged_node_mapping

        except Exception as e:
            import traceback
            logger.error(f"Failed to compress under parent '{parent_node.title}': {e}")
            logger.error(traceback.format_exc())
            return outline_root, [], {}, {}


    def flatten_children(self,parent_node, child_nodes):
        """
        将 child_nodes 的子节点（孙节点）提升为 parent_node 的 children。
        如果所有 child 都没有 children，则 parent_node 变为叶子节点。
        """
        new_children = []

        for child in child_nodes:
            if child.children:
                new_children.extend(child.children)

        parent_node.children = new_children  # 可能是 []，这是合法的


    def update_under_parent(
        self,
        outline_root: "OutlineNode",
        parent_node: "OutlineNode",
        child_nodes: List["OutlineNode"],
        memory: "Memory" = None,
    ) -> Tuple[
        "OutlineNode",
        List[Tuple["OutlineNode", List["OutlineNode"]]],
        # Dict[str, List["FactStructDocument"]],
        Dict[str, List[str]],
        Dict[str, List[str]],
    ]:
        """
        Update 指定父节点（不同于 compression）
        
        规则：
        - 若子节点数 == 0 → 允许修改父节点标题
        - 若子节点数 > 0 → 不允许改变结构，只允许更新标题/语义
        """
        logger.info(f"Running update_under_parent on '{parent_node.title}'")
        logger.info(f"Children count: {len(parent_node.children)}")
        
        # ================================
        # 🟢 情况 1：没有子节点
        # ================================
        if len(parent_node.children) == 0:
        
            logger.info(f"父节点 '{parent_node.title}' 没有子节点，跳过标题修改")
            
            # 只更新文档映射,文档映射应该也不用，已经在外面的函数里做了
            # merged_docs = []
            # if memory:
            #     for doc in new_docs:
            #         merged_docs.append(doc)
            #     if merged_docs:
            #         memory.map_node_to_docs(parent_node.id, merged_docs)

            updated_nodes_list = [(parent_node, [])]

            # 更新后的节点映射
            updated_node_mapping = {parent_node.id: []}

            # 文档映射
            # new_node_doc_mapping = {parent_node.id: merged_docs}
            # merged_docs: List[FactStructDocument]

            # doc_ids = [doc.id for doc in merged_docs]
            doc_ids=[]
            new_node_doc_mapping = {parent_node.id: doc_ids}

            logger.info(f"Update success: '{parent_node.title}' (no children)")

            return (
                outline_root,
                updated_nodes_list,
                new_node_doc_mapping,
                updated_node_mapping,
            )

        # ================================
        # 🟢 情况 2：存在子节点
        # ================================

        outline_text = outline_root.to_text_tree()

        # 构造输入文档的文档摘要
        # doc_desc = []
        # for doc in new_docs:
        #     if doc.title and doc.text:
        #         doc_desc.append("标题"+doc.title.strip()+"内容"+doc.text[:100].strip() + "…")
        #     elif doc.text:
        #         doc_desc.append(doc.text[:100].strip() + "…")

        # new_docs_brief = "\n".join(f"- {d}" for d in doc_desc[:10])
        # if not new_docs:
        #     new_docs_brief = "无新增文档"

        parent_context = parent_node.get_parent_context()
        context_str = (
            f"{parent_context} > {parent_node.title}"
            if parent_context
            else parent_node.title
        )

        children_titles = [c.title for c in parent_node.children]
        # 2️⃣ 构造子节点描述（简要文献信息）
        children_desc = []
        merged_source_ids = []

        for node in child_nodes:
            merged_source_ids.append(node.id)
            docs = memory.node_to_docs.get(node.id, []) if memory else []

            if docs:
                doc_summaries = []
                for doc in docs:
                    if doc.title:
                        doc_summaries.append(doc.title().strip())   # ⚠ 修正这里
                    elif doc.text:
                        doc_summaries.append(doc.text[:50].strip() + "…")

                docs_brief = (
                    f"{len(docs)} 篇相关文献，主题包括：" + "；".join(doc_summaries[:5])
                )

                if len(doc_summaries) > 5:
                    docs_brief += f" 等（共 {len(doc_summaries)} 个主题锚点）"

            else:
                docs_brief = "无直接文献（由上层语义拆分而来）"

            children_desc.append(
                f"- 子节点标题: {node.title}\n  文献信息摘要: {docs_brief}"
            )


            # 3️⃣ 构造压缩 prompt
            prompt = f"""
            你是研究助手，需要根据新增文档对研究大纲进行【局部语义强化更新】。
            
            ⚠ 本任务不是压缩，也不是扩展，而是在保持结构完全不变的前提下，对现有子节点进行

            ## 当前完整大纲
            {outline_text}

            ## 目标父节点
            {context_str}
            该父节点下存在多个语义高度相似、信息量偏少的子节点，需要进行合并压缩。

            ## 当前子节点及其支撑文献
            {chr(10).join(children_desc)}


            ---
            ## 强约束规则（必须遵守）

            1. 不允许增加子节点数量
            2. 不允许减少子节点数量
            3. 不允许改变层级结构
            4. 不允许修改父节点标题
            5. 只允许修改当前父节点下的【叶子子节点标题】
            6. 至少必须修改 1 个子节点标题（不可全部保持不变）
            7. 修改必须基于新增文档内容
            8. 更新后的子节点标题必须更具体、更具信息量
            9. 不允许生成与文档无关的空泛概括标题
            10. 输出完整 JSON 树

            如果你的输出与原结构完全一致，则视为错误。
            {{
                "title": "根节点标题",
                "children": [
                    {{
                        "title": "子节点1标题",
                        "children": []
                    }},
                    {{
                        "title": "子节点2标题（被扩展的叶子节点）",
                        "children": [
                            {{
                                "title": "并列子节点A",
                                "children": []
                            }},
                            {{
                                "title": "并列子节点B",
                                "children": []
                            }},
                            {{
                                "title": "并列子节点C",
                                "children": []
                            }}
                        ]
                    }}
                ]
            }}

            请只输出 JSON，不要包含其他解释性文字。输出完整的修订后大纲树。"""



        try:
            logger.info(f"update_under_parent prompt:\n{prompt}")

            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            content = response.content.strip()
            logger.info(f"content{content}")
            json_str = self._extract_json(content)
            outline_data = json.loads(json_str)

            new_root = self._build_outline_tree(outline_data, parent=None)

            # 继承 MAB 状态
            new_node_ids = []
            self._inherit_mab_state_for_existing_nodes(
                outline_root, new_root, new_node_ids=new_node_ids
            )
            logger.info(f"new_node_ids{new_node_ids}")
            # ========= 路径工具函数 =========
            def get_node_path(node):
                path_parts = []
                current = node
                while current is not None:
                    path_parts.insert(0, current.title)
                    current = current.parent
                return " > ".join(path_parts)

            def find_node_by_path(root, target_path):
                for node in root.get_all_nodes():
                    if get_node_path(node) == target_path:
                        return node
                return None


            # ========= 找更新后的父节点 =========
            target_path = get_node_path(parent_node)
            new_parent = find_node_by_path(new_root, target_path)

            if not new_parent:
                logger.warning("Parent node not found after update")
                return outline_root, [], {}, {}

            # ========= 用 new_node_ids 找到更新后的子节点 =========
            new_children = []

            for node_id in new_node_ids:
                node = new_root.find_node_by_id(node_id)
                if node:
                    new_children.append(node)
                else:
                    logger.warning(f"Node with id {node_id} not found in new_root")

            updated_nodes_list = [(new_parent, new_children)]
            logger.info(f"new_children{new_children}")
            # ========= 构建 updated_node_mapping =========
            # update 是 1 对 1 语义增强
            # new_node_ids 和 old_child_ids 应该等长
            old_child_ids = [child.id for child in parent_node.children]

            updated_node_mapping = {}
            new_node_doc_mapping = {}
            # new_doc_ids = [doc.id for doc in new_docs] if new_docs else []
            for child in new_children:
                updated_node_mapping[child.id] = merged_source_ids
                merged_docs = []
                for old_id in merged_source_ids:
                    merged_docs.extend(memory.node_to_docs.get(old_id, []))
                
                # all_docs = merged_docs + new_doc_ids
                if merged_docs:#all_docs:
                    new_node_doc_mapping[child.id] = merged_docs
                    # new_node_doc_mapping[child.id] = all_docs


            logger.info(
                f"Update success: {len(old_child_ids)} -> {len(new_children)} nodes under '{parent_node.title}'"
            )

            return (
                new_root,
                updated_nodes_list,
                new_node_doc_mapping,
                updated_node_mapping,
            )



        except Exception as e:
            import traceback
            logger.error(
                f"Failed to update under parent '{parent_node.title}': {e}"
            )
            logger.error(traceback.format_exc())
            return outline_root, [], {}, {}





if __name__ == "__main__":
    """
    使用真实 Memory + 真实 Outline 测试 compress_under_parent
    测试场景：
    压缩 “中性粒细胞在脑缺血急性期的作用” 下的 4 个子节点
    """

    from src.factstruct.memory import Memory
    from src.factstruct.document import FactStructDocument

    print("========== START REAL DEBUG ==========")

    # -----------------------
    # 1️⃣ 构造真实 Memory
    # -----------------------
    memory = Memory()

    # 创建 3 个测试文档
    from datetime import datetime

    doc1 = FactStructDocument(
        id="doc_1",
        cite_id="CIT001",
        source_type="journal",
        title="中性粒细胞募集机制研究",
        text="急性期中性粒细胞通过趋化因子被募集到缺血区域。",
        embedding=None,
        timestamp=datetime.now()
    )

    doc2 = FactStructDocument(
        id="doc_2",
        cite_id="CIT002",
        source_type="journal",
        title="血脑屏障破坏机制",
        text="促炎因子释放导致血脑屏障通透性增加。",
        embedding=None,
        timestamp=datetime.now()
    )

    doc3 = FactStructDocument(
        id="doc_3",
        cite_id="CIT003",
        source_type="journal",
        title="炎症与神经损伤",
        text="炎症反应加剧脑水肿与神经损伤。",
        embedding=None,
        timestamp=datetime.now()
    )


    # -----------------------
    # 2️⃣ 构造真实 Outline
    # -----------------------
    root = OutlineNode(
        id="node_0",
        title="中性粒细胞在脑缺血中的作用",
        pull_count=2,
        reward_history=[0.8, 0.9],
        word_limit=500
    )

    acute = OutlineNode(
        id="node_1",
        title="中性粒细胞在脑缺血急性期的作用",
        pull_count=1,
        reward_history=[0.7],
        word_limit=300
    )

    n3 = OutlineNode(
        id="node_2",
        title="中性粒细胞的募集与激活机制",
        pull_count=0,
        reward_history=[],
        word_limit=100
    )
    n4 = OutlineNode(
        id="node_3",
        title="促炎因子释放与血-脑屏障破坏",
        pull_count=0,
        reward_history=[],
        word_limit=100
    )
    n5 = OutlineNode(
        id="node_4",
        title="炎症反应对脑水肿与神经损伤的影响",
        pull_count=0,
        reward_history=[],
        word_limit=100
    )
    n6 = OutlineNode(
        id="node_5",
        title="中性粒细胞在神经修复中的潜在作用",
        pull_count=0,
        reward_history=[],
        word_limit=100
    )


    acute.add_child(n3)
    acute.add_child(n4)
    acute.add_child(n5)
    acute.add_child(n6)

    root.add_child(acute)

    # -----------------------
    # 3️⃣ 建立真实节点-文档映射
    # -----------------------
    memory.map_node_to_docs("node_3", [doc1])
    memory.map_node_to_docs("node_4", [doc2])
    memory.map_node_to_docs("node_3", [doc3])

    print("\n--- BEFORE ---")
    # print(root.to_text_tree())
    print(root.to_text_tree(include_word_limit=True,include_mab_state=True))
    print("Memory node_to_docs:", memory.node_to_docs)

    # -----------------------
    # 4️⃣ 创建 wrapper
    # -----------------------
    # wrapper = FactStructLLMWrapper(llm=None)
    # === LLM ===
    from src.config.agents import AGENT_LLM_MAP
    from src.llms.llm import get_llm_by_type
    llm_type = AGENT_LLM_MAP.get("outline", "basic")
    llm = get_llm_by_type(llm_type)

    wrapper = FactStructLLMWrapper(llm)
    wrapper._inherit_mab_state_for_existing_nodes(old_root=None, new_root=root)
    # -----------------------
    # 5️⃣ 测试 2 子节点折叠（不走 LLM）
    # 只压缩 node_8 和 node_9
    # -----------------------
    # new_root, compressed_list, new_doc_map, merged_map = wrapper.compress_under_parent(
    #     outline_root=root,
    #     parent_node=acute,
    #     child_nodes=[n3, n4],
    #     memory=memory,
    # )

    # print("\n--- AFTER STRUCTURE ---")
    # print(new_root.to_text_tree(include_word_limit=True,include_mab_state=True))

    # print("\nCompressed list:")
    # for p, children in compressed_list:
    #     print("Parent:", p.title)
    #     print("Affected children:", [c.title for c in children])

    # print("\nNew node doc mapping:", new_doc_map)
    # print("Merged node mapping:", merged_map)

    # print("\nMemory after compression:")
    # print(memory.node_to_docs)

    # # -----------------------
    # # 6️⃣ 验证结构完整性
    # # -----------------------
    # def validate_tree(node):
    #     for child in node.children:
    #         assert child.parent == node, f"Parent pointer broken at {child.title}"
    #         validate_tree(child)

    # validate_tree(new_root)

    # print("\n✅ Tree structure valid.")
    # print("========== END DEBUG ==========")


    # -----------------------
    # 5️⃣ 测试 update_under_parent
    # -----------------------

    print("\n========== TEST UPDATE ==========")

    # 构造新增文档（模拟 retrieval 新结果）
    from datetime import datetime

    new_doc = FactStructDocument(
        id="doc_4",
        cite_id="CIT004",
        source_type="journal",
        title="急性期炎症级联反应研究",
        text="中性粒细胞释放NETs并激活炎症级联反应。",
        embedding=None,
        timestamp=datetime.now()
    )

    # 注意：update 版本不应该直接改 memory
    # 只传入 parent + memory + 由函数返回 new_doc_map

    new_root, updated_list, new_doc_map, updated_node_map = wrapper.update_under_parent(
        outline_root=root,
        parent_node=acute,      # 测试有子节点情况
        child_nodes=acute.children,
        memory=memory,
    )

    print("\n--- AFTER UPDATE STRUCTURE ---")
    print(new_root.to_text_tree(include_word_limit=True, include_mab_state=True))

    print("\nUpdated list:")
    for p, children in updated_list:
        print("Parent:", p.title)
        print("Children:", [c.title for c in children])

    print("\nNew node doc mapping:", new_doc_map)
    print("Updated node mapping:", updated_node_map)

    print("\n========== TEST UPDATE (NO CHILDREN) ==========")

    leaf_node = n6   # node_6 没有子节点

    new_root2, updated_list2, new_doc_map2, updated_node_map2 = wrapper.update_under_parent(
        outline_root=new_root,
        parent_node=leaf_node,
        child_nodes=[],
        memory=memory,
    )

    print("\n--- AFTER UPDATE (NO CHILDREN) ---")
    print(new_root2.to_text_tree(include_word_limit=True, include_mab_state=True))

    print("\nNew node doc mapping:", new_doc_map2)
    print("Updated node mapping:", updated_node_map2)

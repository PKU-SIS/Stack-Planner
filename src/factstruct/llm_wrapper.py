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

        prompt = f"""你是一个研究助手。请根据用户查询和提供的初始文档，生成一个结构化的研究大纲。

## 用户查询
{query}

## 初始文档
{docs_text}

## 要求
1. 生成一个层次化的研究大纲（建议 2-3 层）
2. 大纲应该覆盖查询的主要方面
3. 每个节点应该有一个清晰的标题
4. 输出格式必须是 JSON，结构如下：
{{
    "title": "根节点标题",
    "children": [
        {{
            "title": "子节点1标题",
            "children": []
        }},
        {{
            "title": "子节点2标题",
            "children": [
                {{
                    "title": "子节点2.1标题",
                    "children": []
                }}
            ]
        }}
    ]
}}

请只输出 JSON，不要包含其他解释性文字。"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # 尝试提取 JSON（可能被 markdown 代码块包裹）
            json_str = self._extract_json(content)
            outline_data = json.loads(json_str)

            # 构建 OutlineNode 树
            root = self._build_outline_tree(outline_data, parent=None, node_counter=[0])

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
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # 提取 JSON
            json_str = self._extract_json(content)
            outline_data = json.loads(json_str)

            # 构建新的 OutlineNode 树
            new_root = self._build_outline_tree(
                outline_data, parent=None, node_counter=[0]
            )

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
            self._inherit_mab_state_for_existing_nodes(current_outline, new_root)

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

    def _build_outline_tree(
        self,
        data: dict,
        parent: OutlineNode = None,
        node_counter: List[int] = None,
    ) -> OutlineNode:
        """递归构建 OutlineNode 树"""
        if node_counter is None:
            node_counter = [0]

        node_counter[0] += 1
        node_id = f"node_{node_counter[0]}"

        node = OutlineNode(
            id=node_id,
            title=data.get("title", "未命名节点"),
            parent=parent,
            children=[],
        )

        # 递归构建子节点
        for child_data in data.get("children", []):
            child = self._build_outline_tree(
                child_data, parent=node, node_counter=node_counter
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
    ):
        """
        为现有节点继承 MAB 状态（pull_count, reward_history）

        通过节点路径（从根到当前节点的路径）匹配来找到对应的节点并继承状态。
        这个方法只处理旧大纲中已存在的节点，不包括新扩展的子节点。

        匹配策略：
        1. 优先使用完整路径匹配（根 > 父 > 当前）
        2. 如果路径匹配失败，回退到标题匹配
        3. 如果都失败，节点状态重置为初始值（pull_count=0, reward_history=[]）
        """

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

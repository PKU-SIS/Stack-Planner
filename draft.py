


def run_compression(
    self,
    outline_root,
    memory,
    merge_candidates: List["OutlineNode"],
    max_merges: int,
    target_leaf_count: int,
    config: RunnableConfig,
):
    """
    执行大纲压缩（Batch-MAB 风格，每次选择一个父节点进行结构合并）
    """

    logger.info("Running outline compression")
    t = 0  # 压缩次数计数

    # 1️⃣ 按父节点分组（从叶子节点回溯）
    parent_to_children = {}
    for node in merge_candidates:
        if node.parent:
            parent_to_children.setdefault(node.parent, []).append(node)

    if not parent_to_children:
        logger.info("No parent groups found for compression")
        return outline_root, memory

    parents = list(parent_to_children.keys())
    logger.info(f"Found {len(parents)} candidate parent nodes")

    # 2️⃣ Batch-MAB 主循环（每轮只压一个 parent）
    while t < max_merges:

        ucb_scores = []
        for parent in parents:
            children = parent_to_children.get(parent, [])

            # 子节点相关性（是否适合压）
            cohesion = self.compute_children_cohesion(parent, children)

            # exploration / exploitation
            t_current = t + 1
            if parent.pull_count == 0:
                exploration = float("inf")
                avg_reward = 0.0
            else:
                avg_reward = parent.avg_reward()
                exploration = math.sqrt(2 * math.log(t_current) / parent.pull_count)

            # ✅ Compression 专用 UCB
            ucb_score = cohesion - (avg_reward + exploration)
            ucb_scores.append((ucb_score, parent))

        if not ucb_scores:
            logger.info("No parents scored, stopping compression")
            break

        # 3️⃣ 选择最“安全可压”的父节点
        ucb_scores.sort(key=lambda x: x[0], reverse=True)
        parent = ucb_scores[0][1]
        children = parent_to_children.get(parent, [])

        logger.info(
            f"Compressing under parent '{parent.title}' "
            f"(children={len(children)})"
        )

        try:
            # 4️⃣ 调用 LLM 做结构压缩
            outline_root, compressed_nodes_list, new_node_doc_mapping = (
                self.llm_wrapper.compress_under_parent(
                    outline_root=outline_root,
                    parent_node=parent,
                    child_nodes=children,
                    memory=memory,
                )
            )

            # 5️⃣ 状态继承（完全对齐 expansion）
            for parent_node, new_children in compressed_nodes_list:
                for child in new_children:
                    child.pull_count = parent_node.pull_count
                    child.reward_history = list(parent_node.reward_history)


            # --- 6️⃣ reward 更新和Memory 更新（Compression 专用逻辑）---
            # 暂时用 cohesion 作为 reward proxy（信息损失越小越好）
            # reward = cohesion
            # parent.reward_history.append(reward)
            parent.pull_count += 1

            logger.info(
                f"Compression success under '{parent.title}', "
                f"reward={reward:.4f}"
            )
            # merged_node_mapping: { new_node_id: [old_node_id1, old_node_id2, ...] }

            for new_node_id, old_node_ids in merged_node_mapping.items():
                merged_docs = []

                for old_id in old_node_ids:
                    # 收集旧节点的文档
                    docs = memory.node_to_docs.get(old_id, [])
                    merged_docs.extend(docs)

                if merged_docs:
                    memory.map_node_to_docs(new_node_id, merged_docs)
                    logger.debug(
                        f"Compressed node '{new_node_id}' mapped to {len(merged_docs)} documents"
                    )

            # --- 删除被压缩节点的文档映射（非常重要）---
            for old_node_ids in merged_node_mapping.values():
                for old_id in old_node_ids:
                    if old_id in memory.node_to_docs:
                        del memory.node_to_docs[old_id]
                        logger.debug(
                            f"Removed document mapping for compressed node '{old_id}'"
                        )


        except Exception as e:
            logger.error(f"Compression failed under parent '{parent.title}': {e}")
            parents.remove(parent)
            continue

        t += 1

        # 7️⃣ 检查终止条件
        current_leaf_count = len(outline_root.get_leaf_nodes())
        logger.info(f"Current leaf count: {current_leaf_count}")
        if current_leaf_count <= target_leaf_count:
            logger.info("Target leaf count reached, stopping compression")
            break

    logger.info(
        f"Compression finished: merges={t}, "
        f"total_nodes={len(outline_root.get_all_nodes())}"
    )
    return outline_root, memory


def compute_children_cohesion(
    self,
    parent: "OutlineNode",
    children: List["OutlineNode"],
) -> float:
    """
    计算子节点之间的语义内聚度（平均 pairwise cosine similarity）

    cohesion ∈ [-1, 1]（通常在 [0, 1]）
    越大表示这些子节点越应该被合并
    """

    n = len(children)
    if n < 2:
        return 0.0

    # 1️⃣ 构造子节点 embeddings（带父上下文，保证语义空间一致）
    child_embeddings = []
    parent_context = parent.title  # 只用 parent，不再引入 parent.parent

    for child in children:
        node_text = f"{parent_context} > {child.title}"
        emb = self.embedder.embed_text(node_text)
        child_embeddings.append(emb)

    if len(child_embeddings) < 2:
        return 0.0

    # 2️⃣ 计算 pairwise cosine similarity
    total_sim = 0.0
    pair_count = 0

    for i in range(len(child_embeddings)):
        for j in range(i + 1, len(child_embeddings)):
            sim = cosine_similarity(
                child_embeddings[i],
                child_embeddings[j],
            )
            total_sim += sim
            pair_count += 1

    if pair_count == 0:
        return 0.0

    cohesion = total_sim / pair_count
    return float(cohesion)


def compress_under_parent(
    self,
    outline_root: "OutlineNode",
    parent_node: "OutlineNode",
    child_nodes: List["OutlineNode"],
    memory: "Memory" = None,
) -> Tuple[
    "OutlineNode",
    List[Tuple["OutlineNode", List["OutlineNode"]]],
    Dict[str, List["FactStructDocument"]],
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

    if not child_nodes:
        logger.info(f"父节点 '{parent_node.title}' 没有子节点，跳过压缩")
        return outline_root, [], {}, {}

    elif len(child_nodes) == 1:
        # 只有一个子节点，直接提升它
        only_child = child_nodes[0]

        # 合并子节点标题到父节点（可选语义合并）
        # parent_node.title += f" / {only_child.title}"

        # 父节点变为叶子节点
        parent_node.children = []

        # 更新 memory
        # 这个地方应该写到外面那个函数上
        docs = memory.node_to_docs.get(only_child.id, []) if memory else []
        if memory and docs:
            memory.map_node_to_docs(parent_node.id, docs)
            # 删除旧节点映射
            del memory.node_to_docs[only_child.id]

        logger.info(f"父节点 '{parent_node.title}' 只有一个子节点，已合并提升为叶子")
        return (
            outline_root,
            [(parent_node, [only_child])],
            {parent_node.id: docs if docs else []},
            {parent_node.id: [only_child.id]},
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
                    doc_summaries.append(doc.title.strip())
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

    ## 当前大纲
    {outline_text}

    ## 压缩目标
    父节点：{context_str}

    该父节点下存在多个语义高度相似、信息量偏少的子节点，需要进行合并压缩。

    ## 待压缩子节点
    {chr(10).join(children_desc)}

    ## 压缩要求
    1. 对上述子节点进行**语义合并**
    2. 合并后生成 1-2 个新的子节点
    3. 新子节点需要：
      - 覆盖原子节点的核心信息
      - 标题具体、有信息量
      - 避免信息丢失或过度泛化
    4. 不要修改父节点之外的任何结构
    5. 输出完整的修订后大纲
    6. 输出格式必须为 JSON，结构与原大纲一致

    只输出 JSON，不要包含解释性文字。
    """

    try:
        logger.info(f"compress_under_parent prompt:\n{prompt}")
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        content = response.content.strip()

        # 4️⃣ 解析 JSON 并重建大纲树
        json_str = self._extract_json(content)
        outline_data = json.loads(json_str)
        new_root = self._build_outline_tree(outline_data, parent=None, node_counter=[0])

        # 5️⃣ 找到压缩后的父节点及其新子节点
        new_parent = new_root.find_node_by_path(parent_node.get_path_titles())
        if not new_parent:
            logger.warning("Parent node not found after compression")
            return outline_root, [], {}, {}

        new_children = new_parent.children or []
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

        # 7️⃣ 删除被压缩旧节点的文档映射
        if memory:
            for old_id in merged_source_ids:
                if old_id in memory.node_to_docs:
                    del memory.node_to_docs[old_id]

        # 8️⃣ 继承 MAB 状态
        self._inherit_mab_state_for_existing_nodes(outline_root, new_root)

        logger.info(f"Compression success: {len(child_nodes)} -> {len(new_children)} nodes")

        return new_root, compressed_nodes_list, new_node_doc_mapping, merged_node_mapping

    except Exception as e:
        import traceback
        logger.error(f"Failed to compress under parent '{parent_node.title}': {e}")
        logger.error(traceback.format_exc())
        return outline_root, [], {}, {}

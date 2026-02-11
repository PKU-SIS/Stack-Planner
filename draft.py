


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


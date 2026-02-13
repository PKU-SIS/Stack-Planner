def run_update(
    self,
    outline_root,
    memory,
    update_candidates: List["OutlineNode"],
    config: RunnableConfig,
):
    """
    执行大纲节点更新（单次更新，类似 Compression + Expansion，但不迭代）
    
    参数:
        outline_root: 当前大纲根节点
        memory: 当前 Memory 对象
        update_candidates: 待更新的节点列表
        config: RunnableConfig 配置对象
    
    返回:
        更新后的 outline_root 和 memory
    """
    import copy
    import traceback

    logger.info("Running outline update")

    # 1️⃣ 按父节点分组（从叶子节点回溯）
    parent_to_children = {}
    for node in update_candidates:
        if node.parent:
            parent = node.parent
            parent_to_children.setdefault(parent, []).append(node)

    logger.info(f"可更新父节点数量: {len(parent_to_children)}")
    logger.info("父节点列表:")
    for parent, children in parent_to_children.items():
        logger.info(
            f"- Parent: {parent.title} (id={parent.id}) "
            f"children_count={len(children)}"
        )

    parents = list(parent_to_children.keys())  # 现在是 OutlineNode 对象

    ucb_scores = []
    # 2️⃣ Batch-MAB 主逻辑
    for parent_iter in parents:
        # 用 id 在最新树中重新定位节点
        parent = outline_root.find_node_by_id(parent_iter.id)

        if parent is None:
            logger.warning(f"Parent {parent.id} not found in new tree, skipping")
            continue

        # 计算 exploration 和 exploitation
        t_current = 0 + 1  # 暂时这样处理
        if parent.pull_count == 0:
            exploration = float("inf")
            avg_reward = 0.0
        else:
            avg_reward = parent.avg_reward()
            exploration = math.sqrt(2 * math.log(t_current) / parent.pull_count)

        # ✅ update 专用 UCB
        ucb_score = avg_reward + exploration
        ucb_scores.append((ucb_score, parent))

    if not ucb_scores:
        logger.info("No parents scored, stopping update")
        return outline_root, memory

    # 3️⃣ 选择最“安全更新”的父节点
    ucb_scores.sort(key=lambda x: x[0], reverse=True)
    parent = ucb_scores[0][1]
    children = parent_to_children.get(parent, [])

    logger.info(f"Updating under parent '{parent.title}' (children={len(children)})")

    try:
        # 4️⃣ 调用 LLM 更新操作
        logger.info(f"update_under_parent 的 parent: {parent}")
        outline_root, updated_nodes_list, new_node_doc_mapping, updated_node_mapping = (
            self.llm_wrapper.update_under_parent(
                outline_root=outline_root,
                parent_node=parent,
                child_nodes=children,
                memory=memory,
            )
        )

        logger.info(f"updated_nodes_list: {updated_nodes_list}")
        logger.info(f"new_node_doc_mapping: {new_node_doc_mapping}")
        logger.info(f"updated_node_mapping: {updated_node_mapping}")

        # 5️⃣ 状态继承（完全对齐）
        for parent_node, new_children in updated_nodes_list:
            for child in new_children:
                child.pull_count = parent_node.pull_count
                child.reward_history = list(parent_node.reward_history)

        # --- 6️⃣ reward 更新和 Memory 更新 ---
        parent.pull_count += 1
        logger.info(
            f"Update success under '{parent.title}', reward={parent.reward_history}"
        )

        # 新建 memory，以免修改原 memory
        new_memory = copy.deepcopy(memory)

        # 7️⃣ 删除旧节点的文档映射
        for old_node_ids in updated_node_mapping.values():
            for old_id in old_node_ids:
                if old_id in new_memory.node_to_docs:
                    del new_memory.node_to_docs[old_id]
                    logger.debug(f"Removed document mapping for updated node '{old_id}'")

        # 8️⃣ 增加新节点的文档映射
        for new_node_id, old_node_ids in updated_node_mapping.items():
            updated_docs = []

            for old_id in old_node_ids:
                docs = memory.get_docs_by_node(old_id)
                updated_docs.extend(docs)

            # 更新 docs 列表，去除重复
            updated_docs = list({doc.id: doc for doc in updated_docs}.values())

            if updated_docs:
                new_memory.map_node_to_docs(new_node_id, updated_docs)
                logger.debug(f"New node '{new_node_id}' mapped to {len(updated_docs)} documents")

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Update failed under parent '{parent.title}': {e}\nTraceback:\n{tb_str}")
        return outline_root, memory

    logger.info(f"Update finished: total nodes={len(outline_root.get_all_nodes())}")

    return outline_root, new_memory



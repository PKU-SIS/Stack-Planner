#!/usr/bin/env python3


"""
FactStruct Stage 1 使用示例

演示如何使用 Batch-MAB 算法进行动态大纲生成与优化。
"""

from src.factstruct import run_factstruct_stage1, outline_node_to_text


def main():
    """主函数：演示 Stage 1 的使用"""

    # 用户查询
    query = "请分析人工智能在医疗领域的应用现状和发展趋势"

    print(f"=== FactStruct Stage 1 示例 ===\n")
    print(f"查询: {query}\n")
    print("开始运行 Batch-MAB 算法...\n")

    # 运行 Stage 1
    outline_root, memory = run_factstruct_stage1(
        query=query,
        max_iterations=20,  # 最大迭代次数
        batch_size=5,  # 批量大小
    )

    # 输出结果
    print("\n=== 生成的大纲 ===")
    outline_text = outline_node_to_text(outline_root)
    print(outline_text)

    print("\n=== 记忆库统计 ===")
    stats = memory.get_statistics()
    print(f"总文档数: {stats['total_documents']}")
    print(f"总节点数: {stats['total_nodes']}")
    print(f"索引大小: {stats['index_size']}")
    print(f"FAISS 可用: {stats['faiss_available']}")

    print("\n=== 节点-文档映射 ===")
    for node_id, doc_ids in memory.node_to_docs.items():
        node = outline_root.find_node_by_id(node_id)
        if node:
            print(f"节点 '{node.title}': {len(doc_ids)} 个文档")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from typing import Any, Dict, List
from collections import defaultdict

# 确保项目根目录在 sys.path 中，便于作为脚本直接运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.Reference.sp_reporter_utils import (  # type: ignore
    group_by_query_and_outline,
    naive_merge,
    extract_citations,
    parse_sections,
    node_id_to_name,
    parse_outline_bullets,
    citations_for_node_name,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference Test_SP: 仅测试文档拼接并写回 step3_input/step3_output"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="evaluation/Reference/datasets/parsed_dataset.json",
        help="输入 parsed_dataset.json 路径（会被原地修改）",
    )
    parser.add_argument(
        "--merge_algo",
        type=str,
        default="naive_merge",
        choices=["naive_merge"],
        help="拼接算法（当前仅支持 naive_merge）",
    )
    parser.add_argument(
        "--max_groups",
        type=int,
        default=0,
        help="最多处理多少个 (user_query, parent_node_id) 分组，0 表示全部",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    groups = group_by_query_and_outline(data)
    print(f"读取到 {len(data)} 条样本，分为 {len(groups)} 个 (user_query, outline) 分组。")

    processed_groups = 0

    for idx, ((user_query, outline), items) in enumerate(groups.items(), start=1):
        if args.max_groups and idx > args.max_groups:
            break

        print("\n" + "=" * 80)
        print(f"[MERGE GROUP {idx}] merge_algo={args.merge_algo}")
        print(f"user_query: {user_query}")
        print(f"outline: {outline}")

        # 只使用有 step1_output 的块
        valid_items = [it for it in items if it.get("step1_output")]
        if not valid_items:
            continue

        # 1️⃣ 文档拼接：naive_merge
        merged_report = naive_merge(
            user_query=user_query,
            items=valid_items,
            report_outline=outline,
        )

        print("\n[MERGED REPORT]")
        print(merged_report)

        # 拼接前每段文字引用
        node_to_outputs: Dict[str, str] = defaultdict(str)
        for it in valid_items:
            nid = it.get("node_id", "") or ""
            node_name = node_id_to_name(nid)
            text = it.get("step1_output") or ""
            if text.strip():
                node_to_outputs[node_name] = text

        outline_nodes = parse_outline_bullets(outline)
        print("\n" + "-" * 80)
        print("[ORIGINALCITATIONS BY OUTLINE STRUCTURE]")
        for level, title in outline_nodes:
            indent = "  " * level
            outputs = node_to_outputs.get(title, "")
            cits = extract_citations(outputs)
            print(f"{indent}- {title}")
            print(f"{indent}  citations={cits}")

        # 拼接后每段文字引用
        sections = parse_sections(merged_report)
        for heading, body in sections:
            body_cits = extract_citations(body)
            print(f"\n{heading}")
            print(f"  citations={body_cits}")

        # 2️⃣ 写回 step3_input / step3_output
        for item in items:
            # 每个章节都写回所在的整个全文
            item["step3_input"] = args.merge_algo
            item["step3_output"] = merged_report

        processed_groups += 1

    # 3️⃣ 原地写回 parsed_dataset.json
    with open(args.dataset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"\n全部完成，共处理 {processed_groups} 个分组。"
        f"\n已将拼接后的报告写回 step3_output，并将算法名写入 step3_input。"
    )


if __name__ == "__main__":
    main()


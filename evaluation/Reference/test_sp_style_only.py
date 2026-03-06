import argparse
import json
import os
import sys
from typing import Any, Dict, List

# 确保项目根目录在 sys.path 中，便于作为脚本直接运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.Reference.sp_reporter_utils import (  # type: ignore
    group_by_query_and_outline,
    group_by_query_and_outline_and_parent_node_id,
    outline_for_parent_node,
    naive_style_change,
    extract_citation_set,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference Test_SP: 仅测试风格转换并写回 step4_input/step4_output"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="evaluation/Reference/datasets/parsed_dataset.json",
        help="输入 parsed_dataset.json 路径（会被原地修改）",
    )
    parser.add_argument(
        "--style_algo",
        type=str,
        default="naive_style_change",
        choices=["naive_style_change"],
        help="风格切换算法（当前仅支持 naive_style_change）",
    )
    parser.add_argument(
        "--target_style",
        type=str,
        default="鲁迅",
        help="风格切换的目标风格（ROLE_CONSTRAINTS 中的 key 或自定义风格文本）",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="query_outline",
        choices=["query_outline", "query_outline_parent"],
        help="分组方式: query_outline=(user_query, outline); query_outline_parent=(user_query, outline, parent_node_id)",
    )
    parser.add_argument(
        "--max_groups",
        type=int,
        default=0,
        help="最多处理多少个分组，0 表示全部",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if args.group_by == "query_outline":
        groups = group_by_query_and_outline(data)
        print(f"读取到 {len(data)} 条样本，分为 {len(groups)} 个 (user_query, outline) 分组。")
    else:
        groups = group_by_query_and_outline_and_parent_node_id(data)
        print(f"读取到 {len(data)} 条样本，分为 {len(groups)} 个 (user_query, outline, parent_node_id) 分组。")

    processed_groups = 0
    total_with_refs = 0
    total_same_set = 0

    for idx, (key, items) in enumerate(groups.items(), start=1):
        if args.max_groups and idx > args.max_groups:
            break

        if args.group_by == "query_outline":
            user_query, outline = key
            report_outline = outline
        else:
            user_query, outline, parent_node_id = key
            report_outline = outline_for_parent_node(outline, parent_node_id)

        # 取这一组中任意一个非空的 step3_output 作为原始报告（同组内 step3_output 相同）
        base_item = next((it for it in items if it.get("step3_output")), None)
        if not base_item:
            continue

        merged_report = base_item["step3_output"]

        if not merged_report:
            continue

        print("\n" + "=" * 80)
        print(f"[STYLE GROUP {idx}] style_algo={args.style_algo}, group_by={args.group_by}")
        print(f"user_query: {user_query}")
        print(f"outline: {outline}")
        if args.group_by == "query_outline_parent":
            print(f"parent_node_id: {parent_node_id}")

        print("\n[ORIGINAL MERGED REPORT]")
        print(merged_report)

        before_set = extract_citation_set(merged_report)
        print(f"\n[STYLE EVAL] 风格切换前引用集合: {before_set}")
        print(f"[STYLE EVAL] before_set 大小: {len(before_set)}")

        # 1️⃣ 风格切换
        styled_report = naive_style_change(
            user_query=user_query,
            items=items,
            merged_report=merged_report,
            target_style=args.target_style,
            report_outline=report_outline,
        )

        print("\n[STYLED REPORT]")
        print(styled_report)

        after_set = extract_citation_set(styled_report)
        print(f"\n[STYLE EVAL] 风格切换后引用集合: {after_set}")
        print(f"[STYLE EVAL] after_set 大小: {len(after_set)}")

        if len(before_set) > 0:
            total_with_refs += 1
            if before_set == after_set:
                total_same_set += 1

        # 2️⃣ 写回 step4_input / step4_output
        for item in items:
            item["step4_input"] = args.target_style
            item["step4_output"] = styled_report

        processed_groups += 1

    # 3️⃣ 原地写回 parsed_dataset.json
    with open(args.dataset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"\n全部完成，共处理 {processed_groups} 个分组。"
        f"\n已将风格转换后的报告写回 step4_output，并将风格名写入 step4_input。"
    )

    print("\n" + "=" * 80)
    print("[GLOBAL STYLE STATS]")
    print(f"before_set 大小 > 0 的分组总数: {total_with_refs}")
    if total_with_refs > 0:
        print(f"其中 before_set 与 after_set 完全相同的分组数: {total_same_set}")
        print(f"占比: {total_same_set / total_with_refs:.2%}")


if __name__ == "__main__":
    main()


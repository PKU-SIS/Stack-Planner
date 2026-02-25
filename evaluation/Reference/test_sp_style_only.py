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

        # 取这一组中任意一个非空的 step3_output 作为原始报告
        base_item = next((it for it in items if it.get("step3_output")), None)
        if not base_item:
            continue

        merged_report = base_item["step3_output"]

        if not merged_report:
            continue

        print("\n" + "=" * 80)
        print(f"[STYLE GROUP {idx}] style_algo={args.style_algo}")
        print(f"user_query: {user_query}")
        print(f"outline: {outline}")

        print("\n[ORIGINAL MERGED REPORT]")
        print(merged_report)

        before_set = extract_citation_set(merged_report)
        print(f"\n[STYLE EVAL] 风格切换前引用集合: {before_set}")

        # 1️⃣ 风格切换
        styled_report = naive_style_change(
            user_query=user_query,
            items=items,
            merged_report=merged_report,
            target_style=args.target_style,
            report_outline=outline,
        )

        print("\n[STYLED REPORT]")
        print(styled_report)

        after_set = extract_citation_set(styled_report)
        print(f"\n[STYLE EVAL] 风格切换后引用集合: {after_set}")

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


if __name__ == "__main__":
    main()


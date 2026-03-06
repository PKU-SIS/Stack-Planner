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
    group_by_query_and_outline_and_parent_node_id,
    outline_for_parent_node,
    naive_merge,
    incremental_merge,
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
        choices=["naive_merge", "incremental_merge"],
        help="拼接算法：naive_merge=一次性合并；incremental_merge=分段逐章合并",
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
    # 全局统计：仅统计拼接前引用集合大小 > 0 的章节
    total_chapters_with_refs = 0
    total_decreased = 0
    total_increased = 0
    total_same_set = 0

    for idx, (key, items) in enumerate(groups.items(), start=1):
        if args.max_groups and idx > args.max_groups:
            break

        if args.group_by == "query_outline":
            user_query, outline = key
            parent_node_id = None
            report_outline = outline
        else:
            user_query, outline, parent_node_id = key
            report_outline = outline_for_parent_node(outline, parent_node_id)

        print("\n" + "=" * 80)
        print(f"[MERGE GROUP {idx}] merge_algo={args.merge_algo}, group_by={args.group_by}")
        print(f"user_query: {user_query}")
        print(f"outline: {outline}")
        if parent_node_id is not None:
            print(f"parent_node_id: {parent_node_id}")
            print(f"report_outline (本章节):\n{report_outline}")

        # 只使用有 step1_output 的块
        valid_items = [it for it in items if it.get("step1_output")]
        if not valid_items:
            continue

        # 1️⃣ 文档拼接
        if args.merge_algo == "incremental_merge":
            merged_report = incremental_merge(
                user_query=user_query,
                items=valid_items,
                report_outline=report_outline,
            )
        else:
            merged_report = naive_merge(
                user_query=user_query,
                items=valid_items,
                report_outline=report_outline,
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

        outline_nodes = parse_outline_bullets(report_outline)
        print("\n" + "-" * 80)
        print("[ORIGINAL CITATIONS BY OUTLINE STRUCTURE]")
        for level, title in outline_nodes:
            indent = "  " * level
            outputs = node_to_outputs.get(title, "")
            cits = extract_citations(outputs)
            print(f"{indent}- {title}")
            before_set = sorted({int(x) for x in cits}) if cits else []
            print(f"{indent}  citations={cits}")
            print(f"{indent}  citation_set={before_set}, unique_count={len(before_set)}")

        # 拼接后每段文字引用
        sections = parse_sections(merged_report)
        for heading, body in sections:
            body_cits = extract_citations(body)
            print(f"\n{heading}")
            body_set = sorted({int(x) for x in body_cits}) if body_cits else []
            print(f"  citations={body_cits}")
            print(f"  citation_set={body_set}, unique_count={len(body_set)}")

        # 按章节统计引用数目：拼接前 vs 拼接后
        print("\n" + "-" * 80)
        print("[OUTLINE CHAPTER CITATION COUNTS (BY UNIQUE CITATIONS SET)]")
        for level, title in outline_nodes:
            indent = "  " * level
            # 拼接前：该章节对应段落的引用集合大小
            before_outputs = node_to_outputs.get(title, "")
            before_cits = extract_citations(before_outputs)
            before_set = sorted({int(x) for x in before_cits}) if before_cits else []
            before_count = len(before_set)

            # 拼接后：在 merged_report 中标题包含该章节标题的 section 的引用集合大小
            after_cits_all: List[str] = []
            for heading, body in sections:
                if heading and title in heading:
                    after_cits_all.extend(extract_citations(body))
            after_set = sorted({int(x) for x in after_cits_all}) if after_cits_all else []
            after_count = len(after_set)

            print(f"{indent}- {title}: before={before_count}, after={after_count}")
            print(f"{indent}  before_set={before_set}")
            print(f"{indent}  after_set={after_set}")

            # 全局统计：只考虑拼接前集合大小 > 0 的章节
            if before_count > 0:
                total_chapters_with_refs += 1
                if after_count < before_count:
                    total_decreased += 1
                elif after_count > before_count:
                    total_increased += 1
                elif after_set == before_set:
                    total_same_set += 1

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

    # 全局引用迁移统计
    print("\n" + "=" * 80)
    print("[GLOBAL CITATION MIGRATION STATS]")
    print(f"original 引用集合大小 > 0 的章节总数: {total_chapters_with_refs}")
    if total_chapters_with_refs > 0:
        total_unchanged = total_chapters_with_refs - total_decreased - total_increased
        dec_ratio = total_decreased / total_chapters_with_refs
        inc_ratio = total_increased / total_chapters_with_refs
        unchg_ratio = total_unchanged / total_chapters_with_refs
        print(f"拼接后引用变少的章节数: {total_decreased} ({dec_ratio:.2%})")
        print(f"拼接后引用变多的章节数: {total_increased} ({inc_ratio:.2%})")
        print(f"拼接后引用不变的章节数: {total_unchanged} ({unchg_ratio:.2%})")
        same_ratio = total_same_set / total_chapters_with_refs
        print(f"拼接前后引用 set 完全相同的章节数: {total_same_set} ({same_ratio:.2%})")
    else:
        print("没有 original 引用数 > 0 的章节。")


if __name__ == "__main__":
    main()


import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from langchain_core.messages import HumanMessage

from src.prompts.template import apply_prompt_template
from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.agents.SubAgentManager import SubAgentManager


def extract_citations(text: str) -> List[str]:
    """按顺序提取正文中的所有【数字】引用（不去重）。"""
    return re.findall(r"【(\d+)】", text or "")


def extract_citation_set(text: str) -> List[int]:
    """提取正文中的引用编号集合（去重后按数字排序）。"""
    nums = re.findall(r"【(\d+)】", text or "")
    return sorted({int(n) for n in nums})


def group_by_query_and_parent(data: List[Dict[str, Any]]) -> Dict[Tuple[str, Any], List[Dict[str, Any]]]:
    """按 (user_query, parent_node_id) 分组 parsed_dataset."""
    groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for item in data:
        user_query = item.get("user_query", "")
        parent_node_id = item.get("parent_node_id")
        groups[(user_query, parent_node_id)].append(item)
    return groups


def build_reporter_messages(
    user_query: str,
    user_dst: str,
    report_outline: str,
    style_role: str,
    step1_outputs: List[str],
    original_report: str | None = None,
) -> List[HumanMessage]:
    """
    按照 SubAgentManager._generate_report_with_style 的逻辑，构造 reporter 的 prompt。

    - 使用相同的 reporter_xxqg 模板（apply_prompt_template）
    - 使用相同的 LLM 获取方式（get_llm_by_type + AGENT_LLM_MAP["reporter"]）
    - 将 step1_outputs 同时作为“各章节研究发现”和 observations 传入
    - original_report 存在时，附加引用保持要求和风格切换提示
    """
    # 1. 风格约束
    constraint = SubAgentManager.ROLE_CONSTRAINTS.get(style_role, "")
    if not constraint:
        constraint = style_role

    # 2. reporter 输入（与 SubAgentManager 中保持一致）
    reporter_input = {
        "messages": [
            HumanMessage(
                content=f"# Research Requirements\n\n## User Query\n\n{user_query}"
            )
        ],
        "locale": "zh-CN",
    }

    # 3. 将 step1_outputs 映射为“章节研究发现”
    chapter_results: Dict[str, List[str]] = {}
    for idx, obs in enumerate(step1_outputs, start=1):
        chapter_results[str(idx)] = [obs]

    context = {
        "user_query": user_query,
        "task_description": "生成最终报告",
        "chapter_results": chapter_results,
    }

    messages = apply_prompt_template(
        "reporter_xxqg", reporter_input, extra_context=context
    )

    reference_hint = ""
    style_change_hint = ""

    if original_report:
        citations = re.findall(r"【(\d+)】", original_report)
        if citations:
            unique_citations = sorted(set(citations), key=lambda x: int(x))
            reference_hint = (
                "\n\n## 引用保持要求\n\n原始报告使用了以下引用编号："
                + "、".join([f"【{c}】" for c in unique_citations])
                + "。请在新风格的报告中尽量保持使用相同的引用来源。"
            )

        style_change_hint = f"""

## 🔴 重要：风格切换要求

用户明确要求将报告风格切换为：**{style_role}**

原始报告已经生成，现在需要你**完全重写报告**，使用新的风格。

### 原始报告（仅供参考结构和引用）：
{original_report[:2000]}...

---

请基于原始报告的结构和引用信息，使用 **{style_role}** 风格完全重写报告。
"""

    style_emphasis = f"""
# 🎯 写作风格要求（最高优先级）

你正在撰写一篇 **{style_role}** 风格的报告。请严格遵守以下风格约束：

{constraint}

---
"""

    messages.append(
        HumanMessage(
            content=(
                f"{style_emphasis}"
                f"## 用户需求\n\n{user_query}\n\n"
                f"## 任务描述\n\n生成最终报告\n\n"
                f"## 用户补充要求\n\n{user_dst}\n\n"
                f"## 报告大纲\n\n{report_outline}"
                f"{style_change_hint}{reference_hint}"
            )
        )
    )

    # 4. 将 step1_outputs 作为 observations 追加
    for obs in step1_outputs:
        messages.append(
            HumanMessage(
                content=f"以下是检索智能体收集到的高质量信息: \n\n{obs}",
                name="search_agent",
            )
        )

    return messages


def call_reporter(
    user_query: str,
    step1_outputs: List[str],
    style_role: str,
    original_report: str | None = None,
    user_dst: str = "",
    report_outline: str = "用户未提供大纲",
) -> str:
    """调用与 SubAgentManager 中 reporter 相同的 LLM 接口生成报告。"""
    messages = build_reporter_messages(
        user_query=user_query,
        user_dst=user_dst,
        report_outline=report_outline,
        style_role=style_role,
        step1_outputs=step1_outputs,
        original_report=original_report,
    )

    llm_type = AGENT_LLM_MAP.get("reporter", "basic")
    llm = get_llm_by_type(llm_type)  # 与线上 reporter 使用完全相同的 API
    response = llm.invoke(messages)
    return response.content


def naive_merge(
    user_query: str,
    step1_outputs: List[str],
    merge_style: str,
) -> str:
    """
    naive_merge: 直接把同一 (user_query, parent_node_id) 下的所有 step1_output
    作为 reporter 的“章节研究发现”和 observations，生成一篇合并后的报告。
    """
    return call_reporter(
        user_query=user_query,
        step1_outputs=step1_outputs,
        style_role=merge_style,
        original_report=None,
    )


def naive_style_change(
    user_query: str,
    step1_outputs: List[str],
    merged_report: str,
    target_style: str,
) -> str:
    """
    naive_style_change: 在 naive_merge 生成的报告基础上进行风格切换。

    - original_report = merged_report
    - 仍然使用同一批 step1_outputs 作为研究发现 / observations
    - style_role 切换为 target_style
    """
    return call_reporter(
        user_query=user_query,
        step1_outputs=step1_outputs,
        style_role=target_style,
        original_report=merged_report,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reference Test_SP: 拼接 + 风格切换 引用迁移评测"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="evaluation/Reference/datasets/parsed_dataset.json",
        help="输入 parsed_dataset.json 路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/Reference/datasets/sp_merge_style_eval.json",
        help="评测结果输出路径（JSON）",
    )
    parser.add_argument(
        "--merge_algo",
        type=str,
        default="naive_merge",
        choices=["naive_merge"],
        help="拼接算法（当前仅支持 naive_merge）",
    )
    parser.add_argument(
        "--style_algo",
        type=str,
        default="naive_style_change",
        choices=["naive_style_change"],
        help="风格切换算法（当前仅支持 naive_style_change）",
    )
    parser.add_argument(
        "--merge_style",
        type=str,
        default="政策研究报告",
        help="拼接阶段使用的基础风格（ROLE_CONSTRAINTS 中的 key 或直接风格文本）",
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
        help="最多评测多少个 (user_query, parent_node_id) 分组，0 表示全部",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = group_by_query_and_parent(data)
    print(f"读取到 {len(data)} 条样本，分为 {len(groups)} 个 (user_query, parent_node_id) 分组。")

    results: List[Dict[str, Any]] = []

    for idx, ((user_query, parent_id), items) in enumerate(groups.items(), start=1):
        if args.max_groups and idx > args.max_groups:
            break

        # 只使用有 step1_output 的块
        valid_items = [it for it in items if it.get("step1_output")]
        if not valid_items:
            continue

        step1_outputs = [it["step1_output"] for it in valid_items]
        node_ids = [it.get("node_id") for it in valid_items]

        print("\n" + "=" * 80)
        print(f"[GROUP {idx}] merge_algo={args.merge_algo}, style_algo={args.style_algo}")
        print(f"user_query: {user_query}")
        print(f"parent_node_id: {parent_id}")
        print(f"node_ids: {node_ids}")

        # 1️⃣ 拼接阶段：naive_merge
        merged_report = naive_merge(
            user_query=user_query,
            step1_outputs=step1_outputs,
            merge_style=args.merge_style,
        )

        print("\n[MERGED REPORT]")
        print(merged_report)

        # 拼接前每段的引用（不去重）
        pre_segment_citations: List[List[str]] = []
        for i, seg in enumerate(step1_outputs, start=1):
            cits = extract_citations(seg)
            pre_segment_citations.append(cits)
            print(f"\n  [Segment {i}] node_id={node_ids[i-1]}")
            print(f"  step1_output 引用（原始顺序，不去重）: {cits}")

        # 拼接后报告的引用（不去重）
        merged_citations = extract_citations(merged_report)
        print(f"\n  [Merged] 报告整体引用（原始顺序，不去重）: {merged_citations}")

        # 2️⃣ 风格切换阶段：naive_style_change
        styled_report = naive_style_change(
            user_query=user_query,
            step1_outputs=step1_outputs,
            merged_report=merged_report,
            target_style=args.target_style,
        )

        print("\n[STYLED REPORT]")
        print(styled_report)

        # 风格切换前后全文引用集合（去重）
        before_set = extract_citation_set(merged_report)
        after_set = extract_citation_set(styled_report)
        print(f"\n[STYLE EVAL] 风格切换前引用集合: {before_set}")
        print(f"[STYLE EVAL] 风格切换后引用集合: {after_set}")

        result_item = {
            "user_query": user_query,
            "parent_node_id": parent_id,
            "node_ids": node_ids,
            "merge_algorithm": args.merge_algo,
            "merge_style": args.merge_style,
            "merged_report": merged_report,
            "segments": [
                {
                    "node_id": node_ids[i],
                    "step1_output": step1_outputs[i],
                    "citations_before": pre_segment_citations[i],
                }
                for i in range(len(step1_outputs))
            ],
            "merged_report_citations": merged_citations,
            "style_algorithm": args.style_algo,
            "target_style": args.target_style,
            "styled_report": styled_report,
            "style_before_citation_set": before_set,
            "style_after_citation_set": after_set,
        }
        results.append(result_item)

    # 写结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n全部完成，共评测 {len(results)} 个分组。结果已写入: {args.output_path}")


if __name__ == "__main__":
    main()


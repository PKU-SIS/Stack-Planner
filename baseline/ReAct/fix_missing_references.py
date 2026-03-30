#!/usr/bin/env python3
"""
亡羊补牢脚本: 为 ReAct_baseline.jsonl 中缺少参考文献章节的条目补充 ## 参考文献。

ReAct 输出的 research 字段格式与 IRCoT 不同：
  - IRCoT: research = {"1": {type, title, url, content}, "2": ...}
  - ReAct:  research = {"1": {type, title, url, content}, "2": ...}（经 research_list_to_rag_dict 转换后相同）

用法:
    python baseline/ReAct/fix_missing_references.py \
        --input_path evaluation/deep_research_bench/data/test_data/raw_data/ReAct_baseline.jsonl \
        --output_path evaluation/deep_research_bench/data/test_data/raw_data/ReAct_baseline.jsonl

若 --output_path 与 --input_path 相同则原地修改（先写临时文件再替换）。
"""
import json
import os
import re
import argparse
import shutil
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_references_section(research: dict) -> str:
    """根据 research 字典构建 ## 参考文献 章节字符串。
    兼容 ReAct 的 research_list_to_rag_dict 输出格式：
        {"1": {"type": ..., "title": ..., "url": ..., "content": ...}, ...}
    跳过 url 以 inline:// 开头的模型自生成伪来源。
    """
    if not research:
        return ""
    lines = ["\n\n## 参考文献\n"]
    for idx in sorted(research.keys(), key=lambda x: int(x)):
        doc = research[idx]
        url = doc.get("url", "")
        title = doc.get("title", "")
        # 跳过 inline:// 伪来源（ReAct 自生成的 observation）
        if url.startswith("inline://"):
            continue
        if url:
            lines.append(f"[{idx}] {url} - {title}")
        else:
            lines.append(f"[{idx}] {title}")
    if len(lines) == 1:
        # 只有标题行，没有任何有效条目
        return ""
    return "\n".join(lines)


def article_has_references(article: str) -> bool:
    """检查文章末尾是否已有参考文献章节。"""
    tail = article[-3000:] if len(article) > 3000 else article
    return bool(re.search(r"参考文献", tail))


def fix_jsonl(input_path: str, output_path: str):
    abs_input = os.path.join(_PROJECT_ROOT, input_path) if not os.path.isabs(input_path) else input_path
    abs_output = os.path.join(_PROJECT_ROOT, output_path) if not os.path.isabs(output_path) else output_path

    records = []
    with open(abs_input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    fixed_count = 0
    new_records = []
    for record in records:
        article = record.get("article", "")
        research = record.get("research", {})
        if not article_has_references(article) and research:
            ref_section = build_references_section(research)
            if ref_section:
                record["article"] = article + ref_section
                fixed_count += 1
        new_records.append(record)

    # 写到临时文件，再原子替换，避免原地写到一半时损坏文件
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", dir=os.path.dirname(abs_output))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        shutil.move(tmp_path, abs_output)
    except Exception:
        os.unlink(tmp_path)
        raise

    print(f"Done. Total records: {len(new_records)}, fixed (references appended): {fixed_count}")
    print(f"Output saved to: {abs_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="为 ReAct_baseline.jsonl 补充缺失的参考文献章节")
    parser.add_argument(
        "--input_path",
        type=str,
        default="evaluation/deep_research_bench/data/test_data/raw_data/ReAct_baseline2.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/deep_research_bench/data/test_data/raw_data/ReAct_baseline.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fix_jsonl(args.input_path, args.output_path)

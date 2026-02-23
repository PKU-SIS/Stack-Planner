import re
from collections import defaultdict
from datetime import datetime
from sentence_transformers import CrossEncoder
from typing import Union, List, Dict, Any


import json
import re
from tqdm import tqdm
from datetime import datetime

from sentence_transformers import CrossEncoder

from src.factstruct import (
    FactStructDocument,
    filter_content_by_relevant_docs,
    mark_content_with_support,
    repair_unknown_citations,
)


# =========================
# 解析 support_docs
# =========================
def parse_support_docs_to_objects(support_docs_str):

    docs = []

    chunks = support_docs_str.split("FactStructDocument(")

    for chunk in chunks[1:]:

        cite_match = re.search(r"cite_id=(\d+)", chunk)
        text_match = re.search(r"text='(.*?)',\s*source_type", chunk, re.S)
        url_match = re.search(r"url='(.*?)'", chunk)
        title_match = re.search(r"title='(.*?)'", chunk)
        observation_match = re.search(r"observation=(\{.*?\})", chunk, re.S)
        if not cite_match or not text_match:
            continue

        cite_id = int(cite_match.group(1))
        text = text_match.group(1)
        url = url_match.group(1) if url_match else ""
        title = title_match.group(1) if title_match else ""

        doc = FactStructDocument(
            id=f"doc_{cite_id}",
            cite_id=cite_id,
            text=text,
            source_type="page",
            timestamp=datetime.now(),
            embedding=None,
            url=url,
            title=title,
            observation=observation_match,
        )
        docs.append(doc)

    return docs


# =========================
# 主流程
# =========================
def main():

    input_path = "evaluation/Reference/datasets/parsed_dataset.json"
    output_path = "evaluation/Reference/datasets/nli_processed_dataset.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} samples...")

    # 加载 NLI 模型
    semantic_cls = CrossEncoder("/data1/Yangzb/Model/nli-deberta-v3-small")

    for item in tqdm(data):

        content = item.get("step1_output")
        support_docs_str = item.get("support_docs")

        if not content or not support_docs_str:
            continue

        # 1️⃣ 解析文档
        relevant_docs = parse_support_docs_to_objects(support_docs_str)

        if not relevant_docs:
            continue

        # 2️⃣ 判断支持关系
        supported = filter_content_by_relevant_docs(
            content=content, relevant_docs=relevant_docs, semantic_cls=semantic_cls
        )

        # 3️⃣ 标记支持情况
        new_content = mark_content_with_support(content=content, nli_results=supported)

        # 4️⃣ 修复 unknown 引用
        repair_content = repair_unknown_citations(
            content=new_content, relevant_docs=relevant_docs, semantic_cls=semantic_cls
        )

        # 保存结果
        item["nli_supported_raw"] = supported
        item["nli_marked_content"] = new_content
        item["nli_repaired_content"] = repair_content

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

import re
import json
import os


def extract_all_between(text, start_marker, end_marker):
    pattern = re.compile(
        re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
        re.DOTALL,
    )
    matches = pattern.findall(text)
    return [m.strip() for m in matches]


def safe_get(lst, idx):
    if idx < len(lst):
        return lst[idx]
    return ""


def parse_single_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    user_queries = extract_all_between(
        content,
        "----=====USER_QUERY_START=====----",
        "----=====USER_QUERY_END=====----",
    )

    outlines = extract_all_between(
        content,
        "----=====OUTLINE_START=====----",
        "----=====OUTLINE_END=====----",
    )

    node_ids = extract_all_between(
        content,
        "----=====NODE_ID_START=====----",
        "----=====NODE_ID_END=====----",
    )

    parent_node_ids = extract_all_between(
        content,
        "----=====PARENT_NODE_ID_START=====----",
        "----=====PARENT_NODE_ID_END=====----",
    )

    support_docs = extract_all_between(
        content,
        "----=====SUPPORT_DOCS_START=====----",
        "----=====SUPPORT_DOCS_END=====----",
    )

    step1_inputs = extract_all_between(
        content,
        "----=====STEP1_INPUT_START=====----",
        "----=====STEP1_INPUT_END=====----",
    )

    step1_outputs = extract_all_between(
        content,
        "----=====STEP1_OUTPUT_START=====----",
        "----=====STEP1_OUTPUT_END=====----",
    )

    step2_inputs = extract_all_between(
        content,
        "----=====STEP2_INPUT_START=====----",
        "----=====STEP2_INPUT_END=====----",
    )

    step2_outputs = extract_all_between(
        content,
        "----=====STEP2_OUTPUT_START=====----",
        "----=====STEP2_OUTPUT_END=====----",
    )

    # 以 step1_output 数量为主
    sample_count = len(step1_outputs)

    dataset = []
    base_name = os.path.splitext(os.path.basename(log_path))[0]

    for i in range(sample_count):
        sample_id = f"{base_name}_{str(i+1).zfill(4)}"

        dataset.append(
            {
                "id": sample_id,
                "user_query": safe_get(user_queries, i),
                "outline": safe_get(outlines, i),
                "node_id": safe_get(node_ids, i),
                "parent_node_id": safe_get(parent_node_ids, i),
                "support_docs": safe_get(support_docs, i),
                "step1_input": safe_get(step1_inputs, i),
                "step1_output": safe_get(step1_outputs, i),
                "step2_input": safe_get(step2_inputs, i),
                "step2_output": safe_get(step2_outputs, i),
                # 先不解析，统一填 None
                "step3_input": "None",
                "step3_output": "None",
                "step4_style": "None",
                "step4_output": "None",
            }
        )

    print(f"✅ {log_path} 提取 {len(dataset)} 条样本")
    return dataset


def parse_multiple_logs(log_paths):
    all_data = []

    for log_path in log_paths:
        if not os.path.exists(log_path):
            print(f"⚠ 文件不存在: {log_path}")
            continue

        file_data = parse_single_log(log_path)
        all_data.extend(file_data)

    print(f"\n🎯 总计提取 {len(all_data)} 条样本")
    return all_data


if __name__ == "__main__":

    log_files = [
        # "logs/20260222115328.log",
        "logs/20260223000906.log",
        "logs/20260223002342.log",
    ]

    dataset = parse_multiple_logs(log_files)

    output_path = "evaluation/Reference/datasets/parsed_dataset.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n🚀 已保存到 {output_path}")

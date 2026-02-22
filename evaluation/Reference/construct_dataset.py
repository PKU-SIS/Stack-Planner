import re
import json
import os


def extract_all_between(text, start_marker, end_marker):
    """
    提取所有 start_marker 和 end_marker 之间的内容
    """
    pattern = re.compile(
        re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
        re.DOTALL,
    )
    return [match.strip() for match in pattern.findall(text)]


def parse_single_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1️⃣ 草稿输入
    draft_inputs = extract_all_between(
        content,
        "草稿结果开始开始开始标志标志标志",
        "草稿结果结束结束结束标志标志标志",
    )

    # 2️⃣ 迁移输入（real_input）
    real_inputs = extract_all_between(
        content,
        "迁移输入开始开始开始标志标志标志",
        "迁移输入结束结束结束标志标志标志",
    )

    # 3️⃣ 迁移输出（model_output）
    model_outputs = extract_all_between(
        content,
        "迁移输出开始开始开始标志标志标志",
        "迁移输出结束结束结束标志标志标志",
    )

    # 4️⃣ 支撑文档
    docs_list = extract_all_between(
        content,
        "支撑文档开始开始开始标志标志标志",
        "支撑文档结束结束结束标志标志标志",
    )

    print(f"\n📄 {log_path}")
    print("draft:", len(draft_inputs))
    print("real_input:", len(real_inputs))
    print("output:", len(model_outputs))
    print("docs:", len(docs_list))

    sample_count = min(
        len(draft_inputs),
        len(real_inputs),
        len(model_outputs),
        len(docs_list),
    )

    # dataset = []
    # for i in range(sample_count):
    #     dataset.append({
    #         "source_log": os.path.basename(log_path),
    #         "input": draft_inputs[i],
    #         "real_input": real_inputs[i],
    #         "output": model_outputs[i],
    #         "doc": docs_list[i],
    #     })
    dataset = []
    base_name = os.path.splitext(os.path.basename(log_path))[0]

    for i in range(sample_count):
        sample_id = f"{base_name}_{str(i+1).zfill(4)}"

        dataset.append(
            {
                "id": sample_id,
                "source_log": os.path.basename(log_path),
                "index_in_file": i + 1,
                "input": draft_inputs[i],
                "real_input": real_inputs[i],
                "output": model_outputs[i],
                "doc": docs_list[i],
            }
        )
    print(f"✅ 本文件提取 {len(dataset)} 条")

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

    # 🔥 在这里填你要解析的 log 文件
    log_files = [
        "logs/20260222115328.log",
        # 你后面可以继续加
        # "logs/20260222103000.log",
        # "logs/20260222120000.log",
    ]

    dataset = parse_multiple_logs(log_files)

    output_path = "evaluation/Reference/datasets/parsed_dataset.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n🚀 已保存到 {output_path}")

# # 处理输入的
# import csv
# import json
# import os

# input_path = "evaluation/DeepConsult/datasets/DeepConsult/queries.csv"
# output_path = "evaluation/DeepConsult/datasets/DeepConsult/queries.jsonl"

# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# #不要了
# # with open(input_path, "r", encoding="utf-8") as csv_file, \
# #      open(output_path, "w", encoding="utf-8") as jsonl_file:
    
# #     reader = csv.reader(csv_file)
    
# #     for idx, row in enumerate(reader, start=0):
# #         if not row:
# #             continue
        
# #         prompt_text = row[0].strip()
        
# #         record = {
# #             "id": idx,
# #             "topic": None,
# #             "language": "en",
# #             "prompt": prompt_text
# #         }
        
# #         jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

# # print("Conversion completed successfully.")

# with open(input_path, "r", encoding="utf-8") as csv_file, \
#      open(output_path, "w", encoding="utf-8") as jsonl_file:

#     for idx, line in enumerate(csv_file):
#         prompt_text = line.strip()

#         if not prompt_text:
#             continue

#         record = {
#             "id": idx,
#             "topic": None,
#             "language": "en",
#             "prompt": prompt_text
#         }

#         jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

# 处理输出的
import json
import csv
import difflib
from collections import Counter

# ========= 路径 =========
SP_PATH = "evaluation/DeepConsult/datasets/DeepConsult/RAG.jsonl"
BASELINE_CSV_PATH = "evaluation/DeepConsult/datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv"
OUTPUT_CSV_PATH = "evaluation/DeepConsult/datasets/DeepConsult/responses_RAG_vs_Baseline.csv"

SIM_THRESHOLD = 0.99


def normalize(text):
    if not text:
        return ""
    return (
        text.strip()
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


# ===============================
# 1️⃣ 读取 baseline CSV
# ===============================
baseline_map = {}
baseline_questions = []
baseline_used = set()

with open(BASELINE_CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = normalize(row.get("question", ""))
        baseline_answer = row.get("baseline_answer", "")
        baseline_map[question] = baseline_answer
        baseline_questions.append(question)

print(f"Loaded baseline entries: {len(baseline_map)}")


# ===============================
# 2️⃣ 读取 SP.jsonl
# ===============================
sp_data = []

with open(SP_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            sp_data.append(json.loads(line))

print(f"Loaded SP entries: {len(sp_data)}")


# 🔎 检测 SP 是否有重复 question
sp_questions = [normalize(item.get("prompt", "")) for item in sp_data]
counter = Counter(sp_questions)

duplicates = [q for q, c in counter.items() if c > 1]

if duplicates:
    print("\n⚠️ SP 中存在重复问题:")
    print(f"重复数量: {len(duplicates)}")
else:
    print("\n✅ SP 中没有重复问题")


# ===============================
# 3️⃣ 构建新 CSV
# ===============================
matched = 0
fuzzy_matched = 0
unmatched = 0
duplicate_baseline_match = 0

with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["question", "baseline_answer", "candidate_answer"]
    )
    writer.writeheader()

    for item in sp_data:
        question = normalize(item.get("prompt", ""))
        candidate_answer = item.get("article", "")

        baseline_answer = baseline_map.get(question)
        matched_baseline_question = None

        # 1️⃣ exact match
        if baseline_answer is not None:
            matched_baseline_question = question
            matched += 1

        else:
            # 2️⃣ fuzzy match
            best_match = None
            best_score = 0

            for bq in baseline_questions:
                score = difflib.SequenceMatcher(None, question, bq).ratio()
                if score > best_score:
                    best_score = score
                    best_match = bq

            if best_score >= SIM_THRESHOLD:
                baseline_answer = baseline_map[best_match]
                matched_baseline_question = best_match
                fuzzy_matched += 1
                print(f"⚡ Fuzzy matched (score={best_score:.4f})")
            else:
                unmatched += 1
                print(f"❌ Unmatched (best_score={best_score:.4f})")
                baseline_answer = ""

        # 🚨 检查 baseline 是否已被使用
        if matched_baseline_question:
            if matched_baseline_question in baseline_used:
                duplicate_baseline_match += 1
                print("🚨 同一个 baseline 被多次匹配！")
            baseline_used.add(matched_baseline_question)

        writer.writerow({
            "question": question,
            "baseline_answer": baseline_answer,
            "candidate_answer": candidate_answer
        })


print("\nDone.")
print(f"Exact matched: {matched}")
print(f"Fuzzy matched: {fuzzy_matched}")
print(f"Unmatched: {unmatched}")
print(f"Baseline 被重复匹配次数: {duplicate_baseline_match}")
print(f"Output saved to: {OUTPUT_CSV_PATH}")


# 比较和原文问题的差异
# import json
# import pandas as pd
# import csv
# import difflib

# # ========= 路径 =========
# SP_PATH = "evaluation/DeepConsult/datasets/DeepConsult/SP.jsonl"
# BASELINE_CSV_PATH = "evaluation/DeepConsult/datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv"
# OUTPUT_CSV_PATH = "evaluation/DeepConsult/datasets/DeepConsult/responses_SP_vs_Baseline.csv"


# def normalize_text(text):
#     """统一做一次标准化，减少格式差异"""
#     if not text:
#         return ""
#     return (
#         text.strip()
#         .replace("\r\n", "\n")
#         .replace("\r", "\n")
#     )


# # ===============================
# # 1️⃣ 用 pandas 读取 baseline
# # ===============================
# print(f"Loading baseline from {BASELINE_CSV_PATH}")
# df = pd.read_csv(BASELINE_CSV_PATH)
# print(f"Loaded {len(df)} baseline examples")

# baseline_map = {}
# baseline_questions = []

# for _, row in df.iterrows():
#     question = normalize_text(str(row["question"]))
#     baseline_answer = row["baseline_answer"]
#     baseline_map[question] = baseline_answer
#     baseline_questions.append(question)


# # ===============================
# # 2️⃣ 读取 SP.jsonl
# # ===============================
# sp_data = []

# with open(SP_PATH, "r", encoding="utf-8") as f:
#     for line in f:
#         if line.strip():
#             sp_data.append(json.loads(line))

# print(f"Loaded {len(sp_data)} SP entries")


# # ===============================
# # 3️⃣ 构建新 CSV + Debug unmatched
# # ===============================
# matched = 0
# unmatched = 0

# with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
#     writer = csv.DictWriter(
#         f,
#         fieldnames=["question", "baseline_answer", "candidate_answer"]
#     )
#     writer.writeheader()

#     for item in sp_data:
#         raw_question = item.get("prompt", "")
#         question = normalize_text(raw_question)
#         candidate_answer = item.get("article", "")

#         baseline_answer = baseline_map.get(question)

#         if baseline_answer is None:
#             unmatched += 1

#             print("\n==============================")
#             print("❌ UNMATCHED QUESTION:")
#             print("----- 原始 -----")
#             print(raw_question)
#             print("----- repr -----")
#             print(repr(raw_question))
#             print("----- 长度 -----")
#             print("SP length:", len(question))

#             # 找最相似的 baseline
#             matches = difflib.get_close_matches(
#                 question, baseline_questions, n=3, cutoff=0.5
#             )

#             if matches:
#                 print("\n🔍 最相似 baseline:")
#                 for m in matches:
#                     print("-----")
#                     print(m)
#                     print("repr:", repr(m))
#                     print("长度:", len(m))
#                     print("相似度:",
#                           difflib.SequenceMatcher(None, question, m).ratio())
#             else:
#                 print("⚠️ 没有相似项")

#             baseline_answer = ""
#         else:
#             matched += 1

#         writer.writerow({
#             "question": question,
#             "baseline_answer": baseline_answer,
#             "candidate_answer": candidate_answer
#         })

# print("\nDone.")
# print(f"Matched: {matched}")
# print(f"Unmatched: {unmatched}")
# print(f"Output saved to: {OUTPUT_CSV_PATH}")
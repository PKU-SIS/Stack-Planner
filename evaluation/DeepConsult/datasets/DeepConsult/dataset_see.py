import pandas as pd

file_path = "evaluation/DeepConsult/datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv"

# 读取文件
df = pd.read_csv(file_path)

print("=" * 80)
print("📌 基本信息:")
print("=" * 80)
print(df.info())

print("\n" + "=" * 80)
print("📌 列名:")
print("=" * 80)
print(df.columns.tolist())

print("\n" + "=" * 80)
print("📌 前 5 个样本（字段内容超过 300 字符自动截断）:")
print("=" * 80)

def truncate_text(x, max_len=300):
    if isinstance(x, str) and len(x) > max_len:
        return x[:max_len] + " ... [TRUNCATED]"
    return x

preview_df = df.head(5).applymap(truncate_text)

print(preview_df.to_string(index=False))
import json
import os
import re
import ast
import time
from tqdm import tqdm
from openai import OpenAI
import ast
import yaml

import os
from dotenv import load_dotenv
from openai import OpenAI

# 自动加载 .env
load_dotenv()


def call_model(input):
    api_key = os.getenv("REF_API_KEY")
    base_url = os.getenv("REF_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model="deepseek-v3.2-20251201-160k-local",
        messages=input,
    )

    response = response.choices[0].message.content
    print(response)
    return response


prompt_extract = """你会看到一篇研究报告，研究报告正文中会有一些对参考文献的引用。
正文中的引用可能以如下形式出现：
1. 一段文字+空格+数字，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级 15"
2. 一段文字+[（一个或多个)数字]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15]"
3. 一段文字+[（一个或多个)数字†(一些行号等内容)]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15†L10][5L23][7†summary][9summary]"
4. [引用来源](引用链接)，例如："根据[ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s分类，中国社会可分为九个阶层"

请从正文中找出**所有**引用了参考文献的地方，提取出(fact, ref_idx, url)三元组，提取的时候，注意以下事项：
1. 由于后续需要检验这些facts是否正确，你可能需要在引用的前后寻找一些上下文，以确保fact是完整可理解的，而不是简单的词组或短语
2. 如果一个fact引用了多个文献，那么它应该对应多个三元组，例如如果引用了2个文献，则应该是(fact, ref_idx_1, url_1)和(fact, ref_idx_2, url_2)
3. 对于第三种形式的引用，ref_idx仅考虑第一个数字部分，不考虑其他指示具体位置的内容；对于第四种形式的引用（即引用来源和链接直接出现在正文中）的情况，ref_idx统一设置为0
4. 如果正文中没有标出引用的具体位置（比如仅在文章结尾列出了参考文献列表，而没有在正文中标出），请返回空列表

你应该返回json列表格式，列表中的每一项是一个三元组，例如：
[
    {{
        "fact": "原文中的文本片段，注意中文引号要用全角, 英文引号前加单个反斜杠转义",
        "ref_idx": "该段文字引用的参考文献在参考文献列表中的索引",
        "doc_text": None
    }}
]

下面是研究报告的正文：
{report_text}

下面开始提取，直接输出json列表，不要输出任何闲聊或解释。"""

prompt_validate = """你会看到一个参考资料和一些statement，请你判断对于参考资料来说statement是supported、unsupported、或者unknown，注意：
首先判断参考资料是否存在有效内容，如果参考资料中没有任何有效信息，如"page not found"页面，则认为所有statement的状态都是unknown。
除此之外，参考资料有效的情况下，对于一个statement来说，如果它包含的事实或数据在参考资料中可以全部或部分找到，就认为它是supported的（数据接受四舍五入）；如果statement中所有的事实和数据在参考资料中都找不到，认为它是unsupported的。

你应该返回json列表格式，列表中的每一项包含statement的序号和判断结果，例如：
[
    {{
        "idx": 1,
        "result": "supported"
    }}
]

下面是参考资料和statements：
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

下面开始判断，直接输出json列表，不要输出任何闲聊或解释。"""


# =========================
# 1️⃣ 解析 support_docs
# =========================
# def parse_support_docs(support_docs_str):
#     docs_map = {}

#     pattern = r"FactStructDocument\((.*?)\)"
#     matches = re.findall(pattern, support_docs_str, re.S)

#     for m in matches:
#         cite_match = re.search(r"cite_id=(\d+)", m)
#         text_match = re.search(r"text='(.*?)',\s*source_type", m, re.S)
#         print("cite_match",cite_match)
#         if cite_match and text_match:
#             cite_id = int(cite_match.group(1))
#             text = text_match.group(1)
#             docs_map[cite_id] = text

#     return docs_map
import re


def parse_support_docs(support_docs_str):

    docs_map = {}

    # 先按 FactStructDocument 分块
    chunks = support_docs_str.split("FactStructDocument(")

    for chunk in chunks[1:]:  # 第一个是空的，跳过

        # 找 cite_id
        cite_match = re.search(r"cite_id=(\d+)", chunk)
        if not cite_match:
            continue

        cite_id = int(cite_match.group(1))

        # 找 text='....', source_type
        text_match = re.search(r"text='(.*?)',\s*source_type", chunk, re.S)

        if not text_match:
            continue

        text = text_match.group(1)

        docs_map[cite_id] = text

    return docs_map


# =========================
# 2️⃣ 从 step1_output 抽 citation
# =========================
def extract_facts_with_llm(text):

    user_prompt = prompt_extract.format(report_text=text)

    messages = [{"role": "user", "content": user_prompt}]

    retries = 0
    while retries < 3:
        try:
            response = call_model(messages)

            if isinstance(response, dict):
                response = response["choices"][0]["message"]["content"]

            response = response.replace("```json", "").replace("```", "")

            return json.loads(response)

        except Exception as e:
            print("extract retrying...", e)
            time.sleep(2)
            retries += 1

    return []


# =========================
# 3️⃣ validate 核心逻辑
# =========================
def validate(reference_text, statement):

    user_prompt = prompt_validate.format(reference=reference_text, statements=statement)

    messages = [{"role": "user", "content": user_prompt}]

    retries = 0
    while retries < 3:
        try:
            # print("messages",messages)
            response = call_model(messages)

            if isinstance(response, dict):
                response = response["choices"][0]["message"]["content"]

            response = response.replace("```json", "").replace("```", "")

            return json.loads(response)

        except Exception as e:
            print("retrying...", e)
            time.sleep(2)
            retries += 1

    return [{"idx": 1, "result": "error"}]


# =========================
# 5️⃣ 统计函数
# =========================
def compute_statistics(data, output_path):

    total_samples = 0
    total_facts = 0
    total_citations = 0
    total_supported = 0

    for item in data:

        if "validation_results" not in item:
            continue

        results = item["validation_results"]

        if not results:
            continue

        total_samples += 1

        for r in results:

            total_facts += 1

            if r["result"] != "unknown":
                total_citations += 1

                if r["result"] == "supported":
                    total_supported += 1

    # 防止除零
    citation_per_sample = total_citations / total_samples if total_samples else 0
    supported_per_sample = total_supported / total_samples if total_samples else 0
    supported_rate = total_supported / total_citations if total_citations else 0

    print("\n===== Statistics =====")
    print("total_samples:", total_samples)
    print("total_facts:", total_facts)
    print("total_citations:", total_citations)
    print("total_supported:", total_supported)
    print("citation_per_sample:", citation_per_sample)
    print("supported_per_sample:", supported_per_sample)
    print("supported_rate:", supported_rate)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"total_samples: {total_samples}\n")
        f.write(f"total_facts: {total_facts}\n")
        f.write(f"total_citations: {total_citations}\n")
        f.write(f"total_supported: {total_supported}\n")
        f.write(f"citation_per_sample: {citation_per_sample}\n")
        f.write(f"supported_per_sample: {supported_per_sample}\n")
        f.write(f"supported_rate: {supported_rate}\n")


# =========================
# 4️⃣ 主流程
# =========================
if __name__ == "__main__":

    input_path = "evaluation/Reference/datasets/parsed_dataset.json"
    output_path = "evaluation/Reference/datasets/validated_dataset.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} samples...")
    count = 0
    for item in tqdm(data):
        # count=count+1
        # if count==3:
        #     break
        step1_output = item.get("step1_output")
        support_docs_str = item.get("support_docs")

        if not step1_output or not support_docs_str:
            continue

        docs_map = parse_support_docs(support_docs_str)
        facts = extract_facts_with_llm(step1_output)
        validation_results = []

        for fact_item in facts:
            # ref_idx = fact_item["ref_idx"]
            ref_idx = int(fact_item["ref_idx"])
            fact_text = fact_item["fact"]
            # print(ref_idx, type(ref_idx))
            # print(docs_map.keys())
            if ref_idx not in docs_map:
                print("ref_idx not in docs_map")
                validation_results.append(
                    {"fact": fact_text, "ref_idx": ref_idx, "result": "unknown"}
                )
                continue
            else:
                result = validate(docs_map[ref_idx], fact_text)
                # print("result",result)
                # exit()
                validation_results.append(
                    {
                        "fact": fact_text,
                        "ref_idx": ref_idx,
                        "result": result[0]["result"],
                    }
                )

        item["validation_results"] = validation_results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    stats_output_path = "evaluation/Reference/datasets/statistics.txt"
    compute_statistics(data, stats_output_path)
    print("Done.")

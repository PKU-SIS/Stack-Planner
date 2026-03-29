import json
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
from sentence_transformers import CrossEncoder
import math
import json
import requests
import yaml
import torch
# ========= 配置 =========
INPUT_PATH = "evaluation/deep_research_bench/results/fact/FactStruct/extracted.jsonl"
OUTPUT_PATH = (
    "evaluation/deep_research_bench/results/fact/FactStruct_fixed/extracted.jsonl"
)

ENTAILMENT_INDEX = 1  # ⚠️ 根据你的模型确认 entailment 对应 index
ENTAIL_THRESHOLD = 0.43

# 读取配置文件
with open("/data2/sp/Stack-Planner/conf.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 使用BASIC_MODEL配置
basic_model_config = config.get("BASIC_MODEL", {})
BASE_URL = basic_model_config.get("base_url")
MODEL_NAME = basic_model_config.get("model")
API_KEY = basic_model_config.get("api_key")
# API端点
API_ENDPOINT = f"{BASE_URL.rstrip('/')}/chat/completions"


def extract_numeric_keywords(text):
    """
    抽取：
    - 两位及以上整数
    - 小数
    - 百分比
    - 年份
    - 数字 + 中文单位
    - 数字 + 英文单位
    """

    pattern = r"""
        # 1️⃣ 两位及以上整数 / 小数 / 百分比
        (?<!\d)\d{2,}(?:\.\d+)?%?
        |
        (?<!\d)\d+\.\d+%?
        |
        (?<!\d)\d{2,}年
        |
        # 2️⃣ 数字 + 中文单位（允许后面跟 / 或 以上）
        (?<!\d)\d+(?:\.\d+)?[\u4e00-\u9fa5]{1,8}
        |
        # 3️⃣ 数字 + 英文单位
        (?<!\d)\d+(?:\.\d+)?\s*[a-zA-Z]{1,20}(?:\s+[a-zA-Z]{1,20})?
    """

    matches = re.findall(pattern, text, re.VERBOSE)

    filtered = []
    for m in matches:
        m = m.strip()

        # 过滤纯数字
        if re.fullmatch(r"\d+", m):
            continue

        filtered.append(m)

    return list(set(filtered))


def string_match_docs(fact, citeid2doc):

    keywords = extract_numeric_keywords(fact)
    matched_docs = set()

    for cid, doc in citeid2doc.items():
        text = doc["text"]

        for kw in keywords:
            if kw in text:
                # print("kw",kw)
                # print("text",text)
                matched_docs.add(cid)
                break

    return matched_docs


prompt_template = """你会看到一个参考资料和一些statement，请你判断对于参考资料来说statement是supported、unsupported、或者unknown，注意：
首先判断参考资料是否存在有效内容，如果参考资料中没有任何有效信息，如"page not found"页面，则认为所有statement的状态都是unknown。
除此之外，参考资料有效的情况下，对于一个statement来说，如果它包含的事实或数据在参考资料中可以全部或部分找到，就认为它是supported的（数据接受四舍五入）；如果statement中所有的事实和数据在参考资料中都找不到，认为它是unsupported的。

你应该返回json列表格式，列表中的每一项包含statement的序号和判断结果，例如：
[
    {{
        "idx": 1,
        "result": "supported"
    }},
    {{
        "idx": 2,
        "result": "unsupported"
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


def build_judge_prompt(reference_text, statements_list):
    """
    reference_text: str
    statements_list: list[str]
    """

    # 给 statement 编号
    formatted_statements = ""
    for i, stmt in enumerate(statements_list, 1):
        formatted_statements += f"{i}. {stmt}\n"

    final_prompt = prompt_template.format(
        reference=reference_text[:4000], statements=formatted_statements  # 防止过长
    )

    return final_prompt


def llm_judge_statements(reference_text, statements_list):

    judge_prompt = build_judge_prompt(reference_text, statements_list)
    # print("judge_prompt",judge_prompt)
    payload = {
        "model": MODEL_NAME,
        "temperature": 0.0,  # 判定任务建议 0
        "messages": [
            {"role": "system", "content": "你是一个严谨的事实核查专家。"},
            {"role": "user", "content": judge_prompt},
        ],
    }

    # extra_body 支持
    extra_body = basic_model_config.get("extra_body", {})
    if extra_body:
        payload.update(extra_body)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    response = requests.post(API_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()

    result_text = response.json()["choices"][0]["message"]["content"].strip()
    # print("result_text",result_text)
    # 尝试解析 JSON
    try:
        parsed = json.loads(result_text)
        return parsed
    except Exception as e:
        print("JSON 解析失败，原始输出：")
        print(result_text)
        return None


class BaseEntailmentModel:
    """
    统一接口：
    predict(premise, hypothesis) -> (label_id, score)
    label_id:
        0 = contradiction
        1 = entailment
        2 = neutral
    """

    def predict(self, premise: str, hypothesis: str):
        raise NotImplementedError


class StructBERTNLI(BaseEntailmentModel):

    def __init__(self, model_id, max_premise_chars=350, max_segments=3):
        

        # self.device_id = 0 if torch.cuda.is_available() else -1
        # print(f"当前使用设备ID：{self.device_id}（0=GPU，-1=CPU）")
        # exit()
        self.pipeline = pipeline(Tasks.nli, model_id, model_revision="master")

        self.label2id = {"矛盾": 0, "蕴涵": 1, "中立": 2}

        self.max_premise_chars = max_premise_chars
        self.max_segments = max_segments

    # =========================
    # 句子切块函数
    # =========================
    def split_text(self, text):

        if len(text) <= self.max_premise_chars:
            return [text]

        segments = []
        total_len = len(text)
        segment_len = math.ceil(total_len / self.max_segments)

        for i in range(self.max_segments):
            start = i * segment_len
            end = min((i + 1) * segment_len, total_len)
            segments.append(text[start:end])

        return segments

    # =========================
    # 单段预测
    # =========================
    def predict_single(self, premise, hypothesis):

        result = self.pipeline(input=(premise, hypothesis))
        # print("premise",premise)
        # print("hypothesis",hypothesis)
        # print("result",result)

        if "scores" in result and "labels" in result:

            scores = result["scores"]
            labels = result["labels"]

            max_idx = int(np.argmax(scores))
            label_name = labels[max_idx]
            score = float(scores[max_idx])

        elif "label" in result:

            label_name = result["label"]
            score = float(result.get("score", 1.0))

        else:
            raise ValueError(f"Unknown NLI output format: {result}")

        label_id = self.label2id.get(label_name, 2)

        return label_id, score

    # =========================
    # 支持自动分段 + OR 逻辑
    # =========================
    def predict(self, premise: str, hypothesis: str):

        segments = self.split_text(premise)

        best_score = 0.0
        best_label = 2  # 默认中立

        for seg in segments:

            label_id, score = self.predict_single(seg, hypothesis)

            # 记录最高分
            if score > best_score:
                best_score = score
                best_label = label_id

            # OR 逻辑：只要有一个是蕴涵就直接返回
            if label_id == 1:
                return 1, score

        return best_label, best_score


class CrossEncoderNLI(BaseEntailmentModel):

    def __init__(self, model_path):
        self.model = CrossEncoder(model_path)
        self.ENTAILMENT_INDEX = 1

    def predict(self, premise: str, hypothesis: str):

        scores = self.model.predict([(premise, hypothesis)])[0]
        pred_label = int(np.argmax(scores))
        score = float(scores[pred_label])
        # print("pred_label",pred_label)
        # print("scores",scores)
        return pred_label, score


def build_citeid2doc(research):
    """
    构建 cite_id -> doc 映射
    """
    citeid2doc = {}

    for cite_id, v in research.items():
        content = v.get("content", "").strip()
        url = v.get("url", "").strip()
        if not content:
            continue

        citeid2doc[int(cite_id)] = {
            "text": content,
            "url": url,
            "cite_id": int(cite_id),
            "id": v.get("id", None),
        }

    return citeid2doc


def validate_and_expand_citations(data, entail_model):
    ref_count = 0
    research = data.get("research", {})
    citations = data.get("citations", [])

    if not research or not citations:
        return data

    citeid2doc = build_citeid2doc(research)

    # ============================
    # 1️⃣ 先按 fact 聚合
    # ============================
    fact2refs = {}

    for c in citations:
        fact = c.get("fact", "").strip()
        if not fact:
            continue

        ref_idx = c.get("ref_idx")
        if ref_idx is None:
            continue

        fact2refs.setdefault(fact, set()).add(int(ref_idx))

    new_citations = []

    # ============================
    # 2️⃣ 每个 fact 统一处理
    # ============================
    llm_pass_cache = set()  # 已经被 LLM 判定 supported 的 (fact, cid)
    llm_fail_cache = set()  # 已经被 LLM 判定 unsupported 的 (fact, cid)
    for fact, original_refs in fact2refs.items():

        supported_refs = set()
        hypothesis = fact

        # --------------------------------
        # 1️⃣ 原引用 NLI 验证
        # --------------------------------
        for cite_id in original_refs:

            if cite_id not in citeid2doc:
                continue

            # premise = citeid2doc[cite_id]["text"]
            # label_id, score = entail_model.predict(premise, hypothesis)

            # if label_id == ENTAILMENT_INDEX and score > ENTAIL_THRESHOLD:
            #     supported_refs.add(cite_id)
            supported_refs.add(cite_id)

        # --------------------------------
        # 2️⃣ 全局 NLI 搜索
        # --------------------------------
        for cid, doc in citeid2doc.items():

            premise = doc["text"]
            label_id, score = entail_model.predict(premise, hypothesis)

            if label_id == ENTAILMENT_INDEX and score > ENTAIL_THRESHOLD:
                supported_refs.add(cid)
        # print("supported_refs",supported_refs)

        # --------------------------------
        # 3️⃣ 数字关键词匹配补充
        # --------------------------------
        string_matched = string_match_docs(fact, citeid2doc)
        # print("string_matched",string_matched)
        # union
        candidate_refs = supported_refs.union(string_matched)
        # print("candidate_refs",candidate_refs)
        # --------------------------------
        # 4️⃣ 送入 LLM 最终判断
        # --------------------------------
        final_supported = set()

        # for cid in candidate_refs:

        #     premise = citeid2doc[cid]["text"]

        #     result = llm_judge_statements(
        #         reference_text=premise,
        #         statements_list=[fact]
        #     )

        #     if result and result[0]["result"] == "supported":
        #         final_supported.add(cid)
        # print("llm_pass_cache",llm_pass_cache)
        # print("llm_fail_cache",llm_fail_cache)
        print("candidate_refs",candidate_refs)
        for cid in candidate_refs:

            cache_key = (fact, cid)

            # ✅ 已经通过的，直接加入
            if cache_key in llm_pass_cache:
                # 应该已经加了
                # final_supported.add(cid)
                print("pass cache_key", cache_key)
                print("pass llm_pass_cache", llm_pass_cache)
                continue

            # ❌ 已经失败的，跳过
            if cache_key in llm_fail_cache:
                print("fail cache_key", cache_key)
                print("fail llm_fail_cache", llm_fail_cache)
                continue

            premise = citeid2doc[cid]["text"]

            result = llm_judge_statements(
                reference_text=premise, statements_list=[fact]
            )
            print("result",result)
            if result and result[0]["result"] == "supported":
                final_supported.add(cid)
                llm_pass_cache.add(cache_key)
                ref_count = ref_count + 1
            else:
                llm_fail_cache.add(cache_key)
        # --------------------------------
        # 5️⃣ 生成结果
        # --------------------------------
        print("final_supported", final_supported)
        if final_supported:
            for ref in final_supported:
                new_citations.append(
                    {
                        "fact": fact,
                        "ref_idx": ref,
                        "url": citeid2doc[ref].get("url"),
                        "validation_status": "supported",
                    }
                )

    print("ref_count", ref_count)
    data["citations"] = new_citations
    return data



import os

def main():

    entail_model = StructBERTNLI("/data2/sp/Model/nlp_structbert_nli_chinese-tiny")

    # -----------------------------
    # 1️⃣ 读取已处理 id
    # -----------------------------
    processed_ids = set()

    if os.path.exists(OUTPUT_PATH):
        print("📂 发现已有输出文件，读取已处理数据...")
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "id" in item:
                        processed_ids.add(item["id"])
                except:
                    continue

        print(f"✅ 已处理样本数: {len(processed_ids)}")

    # -----------------------------
    # 2️⃣ 开始处理
    # -----------------------------
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:

        for line in tqdm(lines):

            data = json.loads(line)

            data_id = data.get("id")

            # ⛔ 已处理过，跳过
            if data_id in processed_ids:
                continue

            try:
                data = validate_and_expand_citations(data, entail_model)
            except Exception as e:
                print(f"Error on id {data_id}: {e}")
                continue

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()  # 防止中途崩溃丢数据

    print("✅ Citation validation and expansion complete.")


if __name__ == "__main__":
    main()

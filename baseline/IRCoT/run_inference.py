"""
IRCoT Baseline: 迭代检索 + 生成报告。
输入: evaluation/deep_research_bench/data/prompt_data/query.jsonl
输出: evaluation/deep_research_bench/data/test_data/raw_data/IRCoT.jsonl
"""
import json
import os
import re
import time
import yaml
import argparse
from openai import OpenAI
from tqdm import tqdm
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.tools.bocha_search.web_search_en import web_search


def load_api_config(conf_path=None):
    if conf_path is None:
        conf_path = os.path.join(_PROJECT_ROOT, "conf.yaml")
    with open(conf_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    return {
        "base_url": conf["BASIC_MODEL"]["base_url"],
        "api_key": conf["BASIC_MODEL"]["api_key"],
        "model": conf["BASIC_MODEL"]["model"],
    }


API_CONFIG = load_api_config()
SEARCH_TOP_K = 10


def load_file(filepath):
    path = os.path.join(_PROJECT_ROOT, filepath) if not os.path.isabs(filepath) else filepath
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_jsonl(filepath):
    data = []
    path = os.path.join(_PROJECT_ROOT, filepath) if not os.path.isabs(filepath) else filepath
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def do_retrieval(query_content: str, top_k: int = SEARCH_TOP_K) -> list:
    """执行检索，返回 list of {title, url, content}"""
    results = web_search(query_content, top_k=top_k)
    if not results:
        return []
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results
    ]


def merge_and_dedupe_results(all_results: list) -> dict:
    """合并多轮检索结果，去重，转为 research 格式"""
    seen_urls = set()
    research = {}
    idx = 1
    for batch in all_results:
        for item in batch:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                research[str(idx)] = {
                    "type": "page",
                    "title": item.get("title", ""),
                    "url": url,
                    "content": item.get("content", ""),
                }
                idx += 1
    return research


def build_report_context(research: dict) -> str:
    parts = []
    for idx, doc in research.items():
        content = doc.get("content", "") or ""
        if len(content) > 1500:
            content = content[:1500] + "..."
        parts.append(f"【{idx}】{content}")
    return "\n\n".join(parts)


def build_references_section(research: dict) -> str:
    """根据 research 字典构建 ## 参考文献 章节字符串。"""
    if not research:
        return ""
    lines = ["\n\n## 参考文献\n"]
    for idx in sorted(research.keys(), key=lambda x: int(x)):
        doc = research[idx]
        url = doc.get("url", "")
        title = doc.get("title", "")
        if url:
            lines.append(f"[{idx}] {url} - {title}")
        else:
            lines.append(f"[{idx}] {title}")
    return "\n".join(lines)


def summarize_first_round(batch: list) -> str:
    """将首轮检索结果摘要，供 LLM 分析"""
    parts = []
    for i, item in enumerate(batch[:5], 1):
        title = item.get("title", "")[:80]
        content = (item.get("content", "") or "")[:300]
        parts.append(f"[{i}] {title}\n{content}...")
    return "\n\n".join(parts)


def generate_follow_up_queries(client, query_content: str, first_round_summary: str, refinement_template: str, lang: str = "zh") -> list:
    """LLM 分析首轮结果，生成 1-2 个补充检索 query"""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    system_prompt = refinement_template.replace("{{ CURRENT_TIME }}", current_time)
    system_prompt = system_prompt.replace("{{ locale }}", lang)

    user_content = f"""Research Topic: {query_content}

First-round retrieval results (summary):
{first_round_summary}

Output 1-2 follow-up search queries:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    # print("messages",messages)
    response = client.chat.completions.create(
        model=API_CONFIG["model"],
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        extra_body={"enable_thinking": False},
    )
    content = response.choices[0].message.content
    print("content",content)
    if content is None:
        # Some endpoints return None for content when thinking mode is active;
        # fall back to reasoning_content if available, otherwise skip follow-ups.
        content = getattr(response.choices[0].message, "reasoning_content", None) or ""
    text = content.strip()
    raw = [q.strip() for q in text.split("\n") if q.strip()][:2]
    queries = []
    for q in raw:
        q = re.sub(r"^[\d]+[\.\-\、\)）]\s*", "", q).strip()
        if q and len(q) > 2:
            queries.append(q)
    return queries


def process_single_task(client, task, reporter_template, refinement_template):
    """IRCoT: 多轮检索 -> 合并 -> 生成报告"""
    try:
        query_content = task.get("prompt", "")
        task_id = task.get("id")
        lang = task.get("language", "zh")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # IRCoT: 第一轮检索
        first_batch = do_retrieval(query_content, top_k=SEARCH_TOP_K)
        all_results = [first_batch]

        # LLM 分析首轮结果，生成补充检索 query
        if first_batch:
            summary = summarize_first_round(first_batch)
            follow_up_queries = generate_follow_up_queries(
                client, query_content, summary, refinement_template, lang
            )
            for q in follow_up_queries:
                batch = do_retrieval(q, top_k=SEARCH_TOP_K)
                if batch:
                    all_results.append(batch)

        research = merge_and_dedupe_results(all_results)

        system_prompt = reporter_template.replace("{{ CURRENT_TIME }}", current_time)
        system_prompt = system_prompt.replace("{{locale}}", lang)

        if research:
            docs_context = build_report_context(research)
            user_content = f"""Research Topic: {query_content}

Retrieved documents:
{docs_context}

Please write a comprehensive report based on the above retrieved materials and the research topic. Use inline citations 【id】 to reference the sources."""
        else:
            user_content = f"""Research Topic: {query_content}

No retrieved documents were found. Please write a report based on your knowledge, acknowledging the lack of external sources. Do not use citation markers."""
            research = {}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = client.chat.completions.create(
            model=API_CONFIG["model"],
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            extra_body={"enable_thinking": False},
        )
        generated_article = response.choices[0].message.content

        # 自动在文章末尾追加参考文献章节
        if research and "## 参考文献" not in generated_article[-500:]:
            generated_article = generated_article + build_references_section(research)

        return {
            "id": task_id,
            "prompt": query_content,
            "article": generated_article,
            "research": research,
        }
    except Exception as e:
        print(f"Error processing task {task.get('id')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="IRCoT baseline inference")
    parser.add_argument(
        "--input_path",
        type=str,
        default="evaluation/deep_research_bench/data/prompt_data/query.jsonl",
        help="输入 jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/deep_research_bench/data/test_data/raw_data/IRCoT.jsonl",
        help="输出 jsonl",
    )
    parser.add_argument("--infer_num", type=int, default=None, help="推理样本数")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.join(_PROJECT_ROOT, args.input_path) if not os.path.isabs(args.input_path) else args.input_path
    output_path = os.path.join(_PROJECT_ROOT, args.output_path) if not os.path.isabs(args.output_path) else args.output_path

    client = OpenAI(base_url=API_CONFIG["base_url"], api_key=API_CONFIG["api_key"])
    reporter_template = load_file("baseline/RAG/reporter_prompt.md")
    refinement_template = load_file("baseline/IRCoT/query_refinement_prompt.md")
    queries = load_jsonl(input_path)
    if args.infer_num:
        queries = queries[: args.infer_num]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []
    print(f"Starting IRCoT inference with model {API_CONFIG['model']}...")
    print("Note: Ensure BOCHA_API_KEY is set for web search.")

    for task in tqdm(queries, desc="IRCoT Inference"):
        result = process_single_task(client, task, reporter_template, refinement_template)
        if result:
            results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Done! Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()

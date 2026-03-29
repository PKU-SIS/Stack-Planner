import json
import os
import re
import sys
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.tools.bocha_search.web_search_en import web_search

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
except ImportError:
    pass

API_CONFIG = {
    "base_url": "http://123.57.228.132:8286/api",
    "api_key": "sk-550b5def797e452184c320b368a10989",
    "model": "deepseek-v3.2-20251201-160k-local",
}

PATHS = {
    "prompt_file": "baseline/Webweaver/webweaver_prompt.md",
    "query_file": "evaluation/deep_research_bench/data/prompt_data/query.jsonl",
    "output_file": "evaluation/deep_research_bench/data/test_data/raw_data/Webweaver-DeepSeek.jsonl"
}


def load_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        return data
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate_text(text: str, max_len: int = 2200) -> str:
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + " ..."


def format_search_results(results):
    """
    这版有意保留更丰富的 source 信息，让模型更像 reporter 做综合，
    而不是只围着少数最稳句子写。
    """
    if not results:
        return "No external information found."

    formatted = []
    for i, res in enumerate(results, 1):
        title = clean_text(res.get("title", ""))
        content = truncate_text(res.get("content", ""), max_len=2200)
        url = clean_text(res.get("url", ""))
        source_type = clean_text(res.get("source", "page"))

        block = (
            f"【{i}】\n"
            f"Title: {title or 'Information not provided'}\n"
            f"Source Type: {source_type or 'page'}\n"
            f"URL: {url or 'Information not provided'}\n"
            f"Content: {content or 'Information not provided'}\n"
        )
        formatted.append(block)

    return "\n".join(formatted)


def normalize_article_citations(article: str) -> str:
    if not article:
        return article

    # [1] -> 【1】
    article = re.sub(r"\[(\d+)\]", r"【\1】", article)

    # 【1,2】 / 【1，2】 -> 【1】【2】
    def split_bracket_citations(match):
        nums = re.split(r"[，,、\s]+", match.group(1).strip())
        nums = [n for n in nums if n.isdigit()]
        return "".join([f"【{n}】" for n in nums]) if nums else match.group(0)

    article = re.sub(r"【([\d，,、\s]+)】", split_bracket_citations, article)
    article = re.sub(r"】\s+【", "】【", article)

    return article.strip()


def build_reference_section(research_dict):
    if not research_dict:
        return "\n\n---\n## References\n\nInformation not provided.\n"

    lines = ["\n\n---\n## References\n"]
    sort_keys = sorted(
        research_dict.keys(),
        key=lambda x: int(x) if str(x).isdigit() else str(x)
    )

    for idx in sort_keys:
        item = research_dict[idx]
        url = clean_text(item.get("url", ""))
        title = clean_text(item.get("title", ""))

        if url and title:
            lines.append(f"[{idx}] {url} - {title}")
        elif url:
            lines.append(f"[{idx}] {url}")
        elif title:
            lines.append(f"[{idx}] {title}")
        else:
            lines.append(f"[{idx}] Information not provided.")

    return "\n".join(lines) + "\n"


def process_single_task(client, task, prompt_template):
    try:
        query_content = task.get("prompt", "")
        lang = task.get("language", "zh")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 调回更高的 top_k，让材料更杂、更接近 reporter 综合场景
        search_results = web_search(query_content, top_k=8)
        formatted_results = format_search_results(search_results)

        research_dict = {}
        if search_results:
            for i, res in enumerate(search_results, 1):
                research_dict[str(i)] = {
                    "type": res.get("source", "page"),
                    "title": clean_text(res.get("title", "")),
                    "url": clean_text(res.get("url", "")),
                    "content": clean_text(res.get("content", ""))
                }

        system_prompt = prompt_template.replace("{{ CURRENT_TIME }}", current_time)
        system_prompt = system_prompt.replace("{{locale}}", lang)
        system_prompt = system_prompt.replace("{{ SEARCH_RESULTS }}", formatted_results)
        system_prompt = system_prompt.replace("{{ QUERY }}", query_content)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Research Topic: {query_content}\n\n"
                    "Please write a comprehensive reporter-style analytical report using the retrieved materials. "
                    "Synthesize, compare, and organize the sources into a coherent article rather than simply restating individual snippets. "
                    "Use inline citations in the exact format 【1】【2】 throughout the report for major claims, supporting statements, comparisons, summaries, and analytical passages that rely on the provided materials. "
                    "It is acceptable for a paragraph to include multiple citations when it integrates several sources. "
                    "Do not use [1] or (1). "
                    "Do not fabricate citation ids. "
                    "Do not manually add a references section. "
                    "Keep the report detailed, readable, and well-structured."
                ),
            },
        ]

        response = client.chat.completions.create(
            model=API_CONFIG["model"],
            messages=messages,
            temperature=0.95,
            max_tokens=4096
        )

        raw_article = response.choices[0].message.content or ""
        normalized_article = normalize_article_citations(raw_article)
        reference_section = build_reference_section(research_dict)
        article_with_refs = normalized_article + reference_section

        return {
            "id": task["id"],
            "prompt": query_content,
            "article": article_with_refs,
            "research": research_dict,
            "model": "Webweaver-DeepSeek"
        }

    except Exception as e:
        print(f"\nError processing task {task.get('id')}: {e}")
        return None


def main():
    client = OpenAI(
        base_url=API_CONFIG["base_url"],
        api_key=API_CONFIG["api_key"]
    )

    prompt_template = load_file(PATHS["prompt_file"])
    queries = load_jsonl(PATHS["query_file"])

    os.makedirs(os.path.dirname(PATHS["output_file"]), exist_ok=True)

    if os.path.exists(PATHS["output_file"]):
        os.remove(PATHS["output_file"])

    print("Starting Webweaver inference with DeepSeek-160k...")
    with open(PATHS["output_file"], "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_task = {
                executor.submit(process_single_task, client, task, prompt_template): task
                for task in queries
            }

            for future in tqdm(as_completed(future_to_task), total=len(queries), desc="Generating"):
                result = future.result()
                if result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

    print("Done! Webweaver Inference completed.")


if __name__ == "__main__":
    main()
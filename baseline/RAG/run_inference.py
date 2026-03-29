"""
RAG Baseline: 先进行 web 检索，再基于检索结果生成报告。
输入: evaluation/deep_research_bench/data/prompt_data/query.jsonl
输出: evaluation/deep_research_bench/data/test_data/raw_data/RAG.jsonl
"""
# import json
# import os
# import sys
# import time
# from openai import OpenAI
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor

import json
import os
import time
import yaml
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import sys

# 添加项目根目录以导入 src
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.tools.bocha_search.web_search_en import web_search

# ================= 配置区域 =================
# API_CONFIG = {
#     "base_url": "http://10.1.1.212:8000/v1",
#     "api_key": "sk-d47ad54165ee456093bc9ffd599e354e",
#     "model": "Qwen3-32B",
# }
def load_api_config(conf_path="conf.yaml"):
    with open(conf_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    return {
        "base_url": conf["BASIC_MODEL"]["base_url"],
        "api_key": conf["BASIC_MODEL"]["api_key"],
        "model": conf["BASIC_MODEL"]["model"]
    }
API_CONFIG = load_api_config()

# PATHS = {
#     "retrieval_prompt": os.path.join(os.path.dirname(__file__), "retrieval_prompt.md"),
#     "reporter_prompt": os.path.join(os.path.dirname(__file__), "reporter_prompt.md"),
#     "query_file": os.path.join(_PROJECT_ROOT, "evaluation/deep_research_bench/data/prompt_data/query.jsonl"),
#     "output_file": os.path.join(_PROJECT_ROOT, "evaluation/deep_research_bench/data/test_data/raw_data/Research.jsonl"),
# }

def parse_args():
    parser = argparse.ArgumentParser(description="Run agent with streaming API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8513/api/chat/sp_stream",
        help="API URL，例如 http://localhost:8513/api/chat/sp_stream",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="evaluation/deep_research_bench/data/prompt_data/query.jsonl",
        help="输入 jsonl 文件路径",
    )
    parser.add_argument("--reports_dir", type=str, default="reports", help="日志目录")
    parser.add_argument(
        "--infer_num", type=int, default=10, help="一共要 infer 多少个样本"
    )
    parser.add_argument(
        "--graph-format",
        type=str,
        default="sp_test",
        choices=["sp", "xxqg", "sp_xxqg", "sp_test", "base", "FactStruct"],
        help="Graph format to use (default: 'sp')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/deep_research_bench/data/test_data/raw_data/SP.jsonl",
        help="输出文件路径",
    )
    parser.add_argument(
        "--skip_exist", action="store_true", help="跳过已经生成过的样本"
    )

    return parser.parse_args()



# RAG 检索参数
SEARCH_TOP_K = 10
# ===========================================


def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def do_retrieval(query_content: str) -> dict:
    """
    执行 RAG 检索，返回 research 格式的 dict。
    research 格式: {"1": {"type": "page", "title": ..., "url": ..., "content": ...}, ...}
    """
    results = web_search(query_content, top_k=SEARCH_TOP_K)
    if not results:
        return {}

    research = {}
    for i, item in enumerate(results, start=1):
        research[str(i)] = {
            "type": "page",
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        }
    return research


def build_report_context(research: dict) -> str:
    """将 research 转为模型输入格式：【id】content"""
    parts = []
    for idx, doc in research.items():
        content = doc.get("content", "") or ""
        # 截断过长的 content，避免超出上下文
        if len(content) > 1500:
            content = content[:1500] + "..."
        parts.append(f"【{idx}】{content}")
    return "\n\n".join(parts)


def process_single_task(client, task, reporter_template):
    """处理单个任务：检索 -> 生成报告"""
    try:
        query_content = task.get("prompt", "")
        task_id = task.get("id")
        lang = task.get("language", "zh")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 1. RAG 检索
        research = do_retrieval(query_content)

        # 2. 准备 reporter prompt
        system_prompt = reporter_template.replace("{{ CURRENT_TIME }}", current_time)
        system_prompt = system_prompt.replace("{{locale}}", lang)

        # 3. 构建检索内容上下文
        if research:
            docs_context = build_report_context(research)
            user_content = f"""Research Topic: {query_content}

Retrieved documents:
{docs_context}

Please write a comprehensive report based on the above retrieved materials and the research topic. Use inline citations 【id】 to reference the sources."""
        else:
            user_content = f"""Research Topic: {query_content}

No retrieved documents were found. Please write a report based on your knowledge, acknowledging the lack of external sources. Do not use citation markers."""
            # 检索失败时 research 仍为空 dict
            research = {}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 4. 调用模型生成报告
        response = client.chat.completions.create(
            model=API_CONFIG["model"],
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            extra_body={"enable_thinking": False},
        )

        generated_article = response.choices[0].message.content

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


def main():
    # 1. 初始化客户端
    # client = OpenAI(
    #     base_url=API_CONFIG["base_url"],
    #     api_key=API_CONFIG["api_key"],
    # )
    # 初始化客户端
    args = parse_args()
    client = OpenAI(
        base_url=API_CONFIG["base_url"],
        api_key=API_CONFIG["api_key"]
    )
    # Prompt
    print(f"Loading Queries from {args.input_path}...")
    queries = load_jsonl(args.input_path)
    print("queries[0]",queries[0])
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    reporter_prompt = "baseline/RAG/reporter_prompt.md"

    # 2. 准备数据
    # print(f"Loading Reporter Prompt from {PATHS['reporter_prompt']}...")
    # reporter_template = load_file(PATHS["reporter_prompt"])
    print(f"Loading Reporter Prompt from {reporter_prompt}...")
    reporter_template = load_file(reporter_prompt)

    # print(f"Loading Queries from {PATHS['query_file']}...")
    # queries = load_jsonl(PATHS["query_file"])
    print(f"Loading Queries from {args.input_path}...")
    queries = load_jsonl(args.input_path)
    # os.makedirs(os.path.dirname(PATHS["output_file"]), exist_ok=True)

    # 3. 执行推理（串行执行，因检索有 API 限制）
    results = []
    print(f"Starting RAG inference with model {API_CONFIG['model']}...")
    print("Note: Ensure BOCHA_API_KEY is set for web search.")

    for task in tqdm(queries, desc="RAG Inference"):
        result = process_single_task(client, task, reporter_template)
        if result:
            results.append(result)

    # 4. 保存结果
    # print(f"Saving {len(results)} results to {PATHS['output_file']}...")
    # with open(PATHS["output_file"], 'w', encoding='utf-8') as f:
    print(f"Saving {len(results)} results to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done! RAG inference completed.")


if __name__ == "__main__":
    main()

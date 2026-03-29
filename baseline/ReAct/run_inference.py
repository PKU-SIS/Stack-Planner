import sys
import os
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 动态路径注入 Hack =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
eval_utils_path = os.path.join(project_root, "evaluation", "deep_research_bench")

if eval_utils_path not in sys.path:
    sys.path.append(eval_utils_path)
# 🌟 新增：把项目根目录也加进去，这样才能找到 src 文件夹
if project_root not in sys.path:
    sys.path.append(project_root)
# =====================================================

from utils.api import AIClient
# 🌟 听师兄的话：直接导入系统原生写好的工具
from src.tools.bocha_search.web_search_en import web_search


def load_prompt(prompt_path="baseline/ReAct/react_prompt.md"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, "react_prompt.md")
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def execute_search(query, collected_research):
    """
    使用系统原生的 web_search 工具，并提取 URL 满足 FACT 评测需求
    """
    print(f"  [Debug] 正在调用原生 web_search, 关键词: {query}")
    try:
        # 直接调用系统原生工具
        obs_text = web_search(query)

        if not obs_text:
            return "未检索到相关内容。"

        # 🌟 FACT 评测保命 Hack：从纯文本的观察结果中提取 URL 塞进 research 列表
        # 匹配文本中出现的所有 http/https 链接
        urls = re.findall(r'https?://[^\s<>"\']+|(?:www\.)[^\s<>"\']+', str(obs_text))
        # 去重
        unique_urls = list(set(urls))

        for url in unique_urls:
            # 去重检查，防止同一个 URL 存多次
            if not any(r.get("url") == url for r in collected_research):
                collected_research.append({
                    "url": url,
                    "title": "Web Search Result",  # 原生工具若无结构化标题，用默认值兜底
                    "content": str(obs_text)[:4000]  # 取一段文本作为内容供评测校验
                })

        print(f"  [Debug] ✅ 原生搜索成功！提取到 {len(unique_urls)} 个链接供 FACT 评测使用。")
        return str(obs_text)

    except Exception as e:
        print(f"  [Debug] ❌ web_search 调用失败: {e}")
        return "搜索失败，请尝试其他关键词。"


def parse_react_response(response_text, action):
    """根据动作类型安全地提取 Action Input"""
    action_input = None
    if action == "Search":
        match = re.search(r"\*?Action Input\*?:\s*([^\n]*)", response_text, re.IGNORECASE)
        if match:
            action_input = match.group(1).strip()
    elif action == "Finish":
        match = re.search(r"\*?Action Input\*?:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            action_input = match.group(1).strip()
    return action_input


def run_react_agent(task, client, system_prompt, max_steps=6):
    task_prompt = task.get("prompt")
    history = f"{system_prompt}\n\nTask: {task_prompt}\n"

    collected_research = []
    final_article = ""
    response = ""

    print(f"\n[开始处理任务 ID: {task.get('id', 'unknown')}]")

    for step in range(max_steps):
        response = client.generate(user_prompt=history, system_prompt="")

        action_match = re.search(r"\*?Action\*?:\s*\*?\[?(Search|Finish)\]?\*?", response, re.IGNORECASE)
        action = action_match.group(1).capitalize() if action_match else None

        # 核心防暴走 Hack：如果它是 Search，一刀砍掉后续幻觉
        if action == "Search":
            match = re.search(r"(\*?Action Input\*?:\s*[^\n]*)", response, re.IGNORECASE)
            if match:
                response = response[:match.end()].strip()

        history += f"\n{response}\n"

        print(f"\n--- 第 {step + 1} 轮模型输出 ---")
        print(response)

        action_input = parse_react_response(response, action)
        print(f"> 解析到的动作: {action}")

        if action == "Search":
            obs = execute_search(action_input, collected_research)
            history += f"\nObservation: {obs}\n"
            print(f"> 搜索完毕，将观察结果塞回历史")

        elif action == "Finish":
            print("> 模型决定结束任务，生成最终文章。")
            final_article = action_input
            break

        else:
            history += "\nObservation: 格式解析失败。请确保精确包含 'Action: Search' 或 'Action: Finish' 以及后续的 'Action Input:'\n"
            print("> 格式解析失败，强制纠正中...")

    if not final_article:
        final_article = response

    return final_article, collected_research


def process_single_task(task):
    client = AIClient()
    system_prompt = load_prompt()
    article, research = run_react_agent(task, client, system_prompt)

    return {
        "id": task.get("id"),
        "prompt": task.get("prompt"),
        "language": task.get("language"),
        "article": article,
        "research": research
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", type=str, default="evaluation/deep_research_bench/data/prompt_data/query.jsonl")
    parser.add_argument("--output_file", type=str,
                        default="evaluation/deep_research_bench/data/test_data/cleaned_data/ReAct_baseline.jsonl")
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = []
    if os.path.exists(args.query_file):
        with open(args.query_file, "r", encoding="utf-8") as f:
            for line in f:
                tasks.append(json.loads(line))
    else:
        print(f"Error: 找不到任务文件 {args.query_file}")
        return

    if args.limit:
        tasks = tasks[:args.limit]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Running ReAct Agent"):
            try:
                res = future.result()
                with open(args.output_file, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Task failed: {e}")


if __name__ == "__main__":
    main()
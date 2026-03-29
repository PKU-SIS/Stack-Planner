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
from baseline.Webweaver.planner import Planner
from baseline.Webweaver.writer import Writer
from baseline.Webweaver.memory_bank import MemoryBank

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
        "--output_path",
        type=str,
        default="evaluation/deep_research_bench/data/test_data/raw_data/WebWeaver.jsonl",
        help="输出文件路径",
    )
    parser.add_argument(
        "--skip_exist", action="store_true", help="跳过已经生成过的样本"
    )

    return parser.parse_args()



# RAG 检索参数
SEARCH_TOP_K = 3
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


from baseline.Webweaver.planner import Planner
from baseline.Webweaver.writer import Writer,parse_outline
from baseline.Webweaver.memory_bank import MemoryBank
from baseline.Webweaver.utils import llm
from baseline.Webweaver.prompts import INIT_OUTLINE_PROMPT, QUERY_PROMPT, REFINE_OUTLINE_PROMPT

# def process_single_task(task):
#     """处理单个任务：WebWeaver检索 -> 生成报告"""
#     try:
#         query_content = task.get("prompt", "")
#         task_id = task.get("id")
#         lang = task.get("language", "zh")
#         current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

#         # 1. 初始化 Planner 和 Memory
#         memory = MemoryBank()
#         planner = Planner(memory)
#         writer = Writer(memory)

#         # 2. 初始化大纲
#         outline = planner.init_outline(query_content)
#         print("outline",outline)
#         # 3. 生成查询并进行两次检索
#         queries = planner.generate_queries(query_content)
#         print("queries",queries)
#         print("\n===== STEP1: First Search =====\n")
#         evidence_ids,research_context = planner.search_and_store(queries)
#         print("Evidence IDs after first search:", evidence_ids)

#         # 4. 第一次更新大纲
#         refined_outline_1 = planner.refine_outline(query_content)
#         print("\n===== STEP2: First Outline Refinement =====\n", refined_outline_1)

#         # 5. 第二次检索
#         print("\n===== STEP3: Second Search =====\n")
#         evidence_ids,research_context = planner.search_and_store(queries)
#         print("Evidence IDs after second search:", evidence_ids)

#         # 6. 第二次更新大纲
#         refined_outline_2 = planner.refine_outline(query_content)
#         print("\n===== STEP4: Second Outline Refinement =====\n", refined_outline_2)

#         # 7. 使用 Writer 生成报告
#         # sections = refined_outline_2.split("\n")
#         # sections_with_eids = [(sec, evidence_ids) for sec in sections]  # 将所有 section 和证据配对
#         # print("sections_with_eids ",sections_with_eids )
#         # report = writer.write_report(sections_with_eids)
#         sections = parse_outline(refined_outline_2)
#         report = writer.write_report(sections)



#         # 8. 构建 research 字段
#         research = {}
#         for i, eid in enumerate(evidence_ids):
#             # 从 MemoryBank 获取证据的详细信息
#             evidence = memory.data.get(eid, {})
#             if evidence:
#                 research[i + 1] = {
#                     "type": "page",  # 假设所有证据都是网页类型
#                     "title": evidence.get('source', 'Unknown Title'),
#                     "url": evidence.get('source', 'Unknown URL'),  # 这里假设 source 字段包含了 URL
#                     "content": evidence.get('content', '')
#                 }

#         return {
#             "id": task_id,
#             "prompt": query_content,
#             "article": report,
#             "research": research
#         }

#     except Exception as e:
#         print(f"Error processing task {task.get('id')}: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

def process_single_task(task):
    """处理单个任务：WebWeaver检索 -> 生成报告"""
    try:
        query_content = task.get("prompt", "")
        task_id = task.get("id")
        lang = task.get("language", "zh")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 1. 初始化 Planner 和 Memory
        memory = MemoryBank()
        planner = Planner(memory)
        writer = Writer(memory)

        # 2. 初始化大纲
        outline = planner.init_outline(query_content)
        print("outline",outline)

        # 3. 生成查询并进行两次检索
        queries = planner.generate_queries(query_content)
        print("queries",queries)

        print("\n===== STEP1: First Search =====\n")
        evidence_ids_1 = planner.search_and_store(queries)
        evidence_ids_1=evidence_ids_1["evidence_ids"]
        print("Evidence IDs after first search:", evidence_ids_1)

        # 4. 第一次更新大纲
        refined_outline_1 = planner.refine_outline(query_content)
        print("\n===== STEP2: First Outline Refinement =====\n", refined_outline_1)

        # 5. 第二次检索
        print("\n===== STEP3: Second Search =====\n")
        evidence_ids_2= planner.search_and_store(queries)
        evidence_ids_2=evidence_ids_2["evidence_ids"]
        print("Evidence IDs after second search:", evidence_ids_2)

        # 6. 第二次更新大纲
        refined_outline_2 = planner.refine_outline(query_content)
        print("\n===== STEP4: Second Outline Refinement =====\n", refined_outline_2)

        # 7. 使用 Writer 生成报告
        sections = parse_outline(refined_outline_2)
        print("sections",sections)
        report = writer.write_report(sections)

        # 8. 合并第一轮和第二轮的 evidence_ids
        all_evidence_ids = evidence_ids_1 + evidence_ids_2  # 合并第一轮和第二轮的 evidence_ids
        print("all_evidence_ids",all_evidence_ids)
        # 9. 构建 research 字段
        research = {}
        for i, eid in enumerate(all_evidence_ids):  # 遍历合并后的 evidence_ids
            # 从 MemoryBank 获取证据的详细信息
            evidence = memory.research.get(eid, {})
            if evidence:
                research[i + 1] = {
                    "type": "page",  # 假设所有证据都是网页类型
                    "title": evidence.get('title', 'Unknown Title'),
                    "url": evidence.get('url', 'Unknown URL'),  # 这里假设 source 字段包含了 URL
                    "content": evidence.get('content', '')
                }

        return {
            "id": task_id,
            "prompt": query_content,
            "article": report,
            "research": research
        }

    except Exception as e:
        print(f"Error processing task {task.get('id')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # 1. 初始化客户端
    args = parse_args()

    # Prompt
    print(f"Loading Queries from {args.input_path}...")
    queries = load_jsonl(args.input_path)
    print(f"Loaded {len(queries)} queries from {args.input_path}")
    
    # 2. 创建输出目录（如果不存在的话）
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 3. 加载现有结果（如果跳过已生成样本）
    existing_results = set()
    if args.skip_exist and os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                existing_result = json.loads(line.strip())
                print("existing_result ",existing_result )
                existing_results.add(existing_result.get('prompt'))  # 假设 'task' 字段是唯一标识
    print("existing_results", existing_results)

    # 4. 控制推理数量
    total_inferences = 0
    with open(args.output_path, 'a', encoding='utf-8') as f:  # 'a' 模式追加
        for task in tqdm(queries, desc="WebWeaver Inference"):
            if task["prompt"] in existing_results:  # 如果当前任务已经推理过，跳过
                print("task[prompt]",task["prompt"])
                print("existing_results",existing_results)
                continue
            
            # 处理当前任务
            result = process_single_task(task)
            if result:
                # 写入结果到文件
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                total_inferences += 1
            
            if total_inferences >= args.infer_num:  # 如果达到推理数量，停止
                break

    print(f"Done! {total_inferences} RAG inferences completed.")

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()

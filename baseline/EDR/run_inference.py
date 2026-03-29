"""
EDR Baseline: 使用 enterprise-deep-research 进行深度研究。
输入: evaluation/deep_research_bench/data/prompt_data/query.jsonl
输出: evaluation/deep_research_bench/data/test_data/raw_data/EDR.jsonl
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# 路径设置
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_EDR_DIR = _SCRIPT_DIR / "enterprise-deep-research"
_BENCHMARKS_DIR = _EDR_DIR / "benchmarks"

sys.path.insert(0, str(_EDR_DIR))
sys.path.insert(0, str(_BENCHMARKS_DIR))
os.chdir(str(_EDR_DIR))

load_dotenv(dotenv_path=_EDR_DIR / ".env")
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")  # 项目根 .env（含 BOCHA_API_KEY 等）

from run_research import run_research_sync


def load_jsonl(filepath):
    data = []
    path = _PROJECT_ROOT / filepath if not os.path.isabs(filepath) else filepath
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="EDR batch inference")
    parser.add_argument(
        "--input_path",
        type=str,
        default="evaluation/deep_research_bench/data/prompt_data/query.jsonl",
        help="输入 jsonl 文件路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/deep_research_bench/data/test_data/raw_data/EDR.jsonl",
        help="输出文件路径",
    )
    parser.add_argument(
        "--infer_num",
        type=int,
        default=None,
        help="推理样本数量，默认全部",
    )
    parser.add_argument(
        "--max_loops",
        type=int,
        default=2,
        help="最大检索轮数",
    )
    return parser.parse_args()


async def process_single_task(task, max_loops):
    """调用 EDR run_research 处理单个任务"""
    query_content = task.get("prompt", "")
    task_id = task.get("id")

    result = await run_research_sync(
        query=query_content,
        max_web_search_loops=max_loops,
        visualization_disabled=True,
        extra_effort=False,
        minimum_effort=False,
        qa_mode=False,
        benchmark_mode=False,
        provider=None,
        model=None,
        output_file=None,
        file_path=None,
        steering_enabled=False,
        steering_messages=None,
    )

    if not result:
        return None

    return {
        "id": task_id,
        "prompt": query_content,
        "article": result.get("article", ""),
        "research": result.get("research", {}),
    }


async def main():
    args = parse_args()
    input_path = _PROJECT_ROOT / args.input_path if not os.path.isabs(args.input_path) else args.input_path
    output_path = _PROJECT_ROOT / args.output_path if not os.path.isabs(args.output_path) else args.output_path

    print(f"Loading queries from {input_path}...")
    queries = load_jsonl(str(input_path))
    if args.infer_num:
        queries = queries[: args.infer_num]
    print(f"Total tasks: {len(queries)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = []
    for task in tqdm(queries, desc="EDR Inference"):
        try:
            result = await process_single_task(task, args.max_loops)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing task {task.get('id')}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Saving {len(results)} results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done! EDR inference completed.")


if __name__ == "__main__":
    asyncio.run(main())

"""
STORM Baseline: 使用 STORM 算法进行深度研究。
输入: evaluation/deep_research_bench/data/prompt_data/query.jsonl
输出: evaluation/deep_research_bench/data/test_data/raw_data/STORM.jsonl

检索 API: src.tools.bocha_search.web_search_en (BOCHA_API_KEY)
推理 API: conf.yaml 的 BASIC_MODEL
"""
import json
import os
import re
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_STORM_DIR = _SCRIPT_DIR / "storm"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_STORM_DIR))
os.chdir(str(_PROJECT_ROOT))

import yaml
from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import BochaRM


def load_api_config():
    conf_path = _PROJECT_ROOT / "conf.yaml"
    with open(conf_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    return {
        "base_url": conf["BASIC_MODEL"]["base_url"],
        "api_key": conf["BASIC_MODEL"]["api_key"],
        "model": conf["BASIC_MODEL"]["model"],
    }


def load_jsonl(filepath):
    path = _PROJECT_ROOT / filepath if not str(filepath).startswith("/") else Path(filepath)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def sanitize_topic(topic: str) -> str:
    t = topic.replace(" ", "_").replace("/", "_")
    t = re.sub(r"[^\w\u4e00-\u9fff\-]", "", t)
    return t[:125] if t else "topic"


def url_to_info_to_research(ref_path: Path) -> dict:
    """将 STORM 的 url_to_info.json 转为 research 格式"""
    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    url_to_info = ref.get("url_to_info", ref)
    if not isinstance(url_to_info, dict):
        return {}
    research = {}
    for idx, (url, info) in enumerate(url_to_info.items(), 1):
        if isinstance(info, dict):
            snippets = info.get("snippets", [])
            content = " ".join(snippets) if snippets else info.get("description", "")
        else:
            content = ""
        research[str(idx)] = {
            "type": "page",
            "title": info.get("title", "") if isinstance(info, dict) else "",
            "url": url,
            "content": content[:2000] if content else "",
        }
    return research


def process_single_task(task, api_config, base_output_dir):
    task_id = task.get("id")
    prompt = task.get("prompt", "")

    lm_configs = STORMWikiLMConfigs()
    lm_kwargs = {
        "model": f"openai/{api_config['model']}",
        "api_key": api_config["api_key"],
        "api_base": api_config["base_url"],
        "temperature": 1.0,
        "top_p": 0.9,
    }

    conv_lm = LitellmModel(max_tokens=500, **lm_kwargs)
    qa_lm = LitellmModel(max_tokens=500, **lm_kwargs)
    outline_lm = LitellmModel(max_tokens=400, **lm_kwargs)
    article_lm = LitellmModel(max_tokens=700, **lm_kwargs)
    polish_lm = LitellmModel(max_tokens=4000, **lm_kwargs)

    lm_configs.set_conv_simulator_lm(conv_lm)
    lm_configs.set_question_asker_lm(qa_lm)
    lm_configs.set_outline_gen_lm(outline_lm)
    lm_configs.set_article_gen_lm(article_lm)
    lm_configs.set_article_polish_lm(polish_lm)

    task_output_dir = base_output_dir / str(task_id)
    engine_args = STORMWikiRunnerArguments(
        output_dir=str(task_output_dir),
        max_conv_turn=3,
        max_perspective=3,
        search_top_k=10,
        max_thread_num=3,
    )

    rm = BochaRM(k=10)
    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    try:
        runner.run(
            topic=prompt,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
            remove_duplicate=False,
        )
        runner.post_run()
    except Exception as e:
        print(f"STORM error task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

    article_dir_name = sanitize_topic(prompt.replace(" ", "_").replace("/", "_"))
    article_output_dir = task_output_dir / article_dir_name

    article_path = article_output_dir / "storm_gen_article_polished.txt"
    if not article_path.exists():
        article_path = article_output_dir / "storm_gen_article.txt"
    ref_path = article_output_dir / "url_to_info.json"

    if not article_path.exists():
        print(f"Task {task_id}: article not found")
        return None

    with open(article_path, "r", encoding="utf-8") as f:
        article = f.read()

    research = url_to_info_to_research(ref_path) if ref_path.exists() else {}

    return {
        "id": task_id,
        "prompt": prompt,
        "article": article,
        "research": research,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="STORM baseline inference")
    parser.add_argument(
        "--input_path",
        type=str,
        default="evaluation/deep_research_bench/data/prompt_data/query.jsonl",
        help="输入 jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/deep_research_bench/data/test_data/raw_data/STORM.jsonl",
        help="输出 jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="baseline/STORM/storm_output",
        help="STORM 中间输出目录",
    )
    parser.add_argument("--infer_num", type=int, default=None, help="推理样本数")
    return parser.parse_args()


def main():
    args = parse_args()
    api_config = load_api_config()
    input_path = _PROJECT_ROOT / args.input_path if not str(args.input_path).startswith("/") else Path(args.input_path)
    output_path = _PROJECT_ROOT / args.output_path if not str(args.output_path).startswith("/") else Path(args.output_path)
    base_output_dir = _PROJECT_ROOT / args.output_dir

    queries = load_jsonl(str(input_path))
    if args.infer_num:
        queries = queries[: args.infer_num]

    os.makedirs(output_path.parent, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)

    results = []
    print(f"Starting STORM inference with model {api_config['model']}...")
    print("Note: Ensure BOCHA_API_KEY is set for web search.")

    for task in tqdm(queries, desc="STORM Inference"):
        result = process_single_task(task, api_config, base_output_dir)
        if result:
            results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Done! Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# ReAct baseline：使用项目根目录 conf.yaml 中的 BASIC_MODEL 调用 OpenAI 兼容 API。
# 用法：在项目根目录执行 ./baseline/ReAct/run_react_inference.sh
# 或：bash baseline/ReAct/run_react_inference.sh --limit 2
# 断点续跑（跳过 output 里已有 id）：追加参数 --skip_existing

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONF="${CONF:-$ROOT/conf.yaml}"

exec python baseline/ReAct/run_inference.py \
  --conf "$CONF" \
  --query_file evaluation/deep_research_bench/data/prompt_data/query.jsonl \
  --output_file evaluation/deep_research_bench/data/test_data/raw_data/ReAct_baseline2.jsonl \
  --max_workers 1 \
  --max_steps 14 \
  --skip_existing \
  --start 80 \
  --limit 20 \
  "$@"

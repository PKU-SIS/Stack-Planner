#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/data/sp-yzb/miniconda3/envs/sp/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/data/sp-yzb/miniconda3/envs/evolvingintent-bird/bin/python}"
CASES="${1:?usage: run_sql_qwen32b.sh CASES_JSON [RUN_DIR]}"
RUN_DIR="${2:-${RUN_DIR:-${ROOT}/results/sp_sql_qwen32b_$(date -u +%Y%m%dT%H%M%SZ)}}"
PREDICTIONS="${RUN_DIR}/predictions.jsonl"
SUMMARY="${RUN_DIR}/summary.json"
LOG="${RUN_DIR}/run.log"

[[ -x "${PYTHON}" ]] || { echo "Python not found: ${PYTHON}" >&2; exit 2; }
[[ -x "${EVAL_PYTHON}" ]] || { echo "Evaluation Python not found: ${EVAL_PYTHON}" >&2; exit 2; }
[[ -f "${CASES}" ]] || { echo "Cases not found: ${CASES}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"

"${PYTHON}" - "${ROOT}/conf.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = str((config.get("BASIC_MODEL") or {}).get("model", ""))
if model.lower() != "qwen3-32b":
    raise SystemExit(f"Expected BASIC_MODEL=Qwen3-32B, found {model!r}")
PY

evaluate_partial() {
  if [[ -s "${PREDICTIONS}" ]]; then
    PYTHONPATH="${ROOT}" "${EVAL_PYTHON}" "${ROOT}/evaluation/multiturn/evaluate_sql.py" \
      --cases "${CASES}" \
      --predictions "${PREDICTIONS}" \
      --output "${SUMMARY}"
  fi
}

on_exit() {
  status=$?
  trap - EXIT
  evaluate_partial || true
  printf 'finished_at=%s\nexit_status=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" \
    >> "${RUN_DIR}/run_config.txt"
  exit "${status}"
}
trap on_exit EXIT

printf '%s\n' \
  "graph_format=sp_sql" \
  "task_family=sql" \
  "model=Qwen3-32B" \
  "execution_mode=each-turn" \
  "database_access=false" \
  "cases=${CASES}" \
  "predictions=${PREDICTIONS}" \
  "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${RUN_DIR}/run_config.txt"

runner_args=(
  --cases "${CASES}"
  --output "${PREDICTIONS}"
  --execution-mode each-turn
  --graph-format sp_sql
)
if [[ -n "${MAX_CASES:-}" ]]; then
  runner_args+=(--max-cases "${MAX_CASES}")
fi

{
  echo "Run directory: ${RUN_DIR}"
  echo "SQL agent database access: disabled"
  "${PYTHON}" "${ROOT}/evaluation/multiturn/run_evolving_intent.py" "${runner_args[@]}"
} 2>&1 | tee -a "${LOG}"

echo "SQL run completed."

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/data/sp-yzb/miniconda3/envs/sp/bin/python}"
CASES="${CASES:-${ROOT}/../evolving-intent-main/review_samples/math/gsm8k_operator_n20_seed20260821/operator_cases.json}"
RUN_DIR="${1:-${RUN_DIR:-${ROOT}/results/sp_math_profile_qwen32b_math20_$(date -u +%Y%m%dT%H%M%SZ)}}"
PREDICTIONS="${RUN_DIR}/predictions.jsonl"
SUMMARY="${RUN_DIR}/summary.json"
LOG="${RUN_DIR}/run.log"

mkdir -p "${RUN_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment does not exist: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CASES}" ]]; then
  echo "Math-20 cases do not exist: ${CASES}" >&2
  exit 2
fi

# Refuse to silently run a different configured model. No API key is printed.
"${PYTHON}" - "${ROOT}/conf.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = str((config.get("BASIC_MODEL") or {}).get("model", ""))
if model.lower() != "qwen3-32b":
    raise SystemExit(f"Expected BASIC_MODEL=Qwen3-32B, found {model!r}")
PY

printf '%s\n' \
  "graph_format=sp_math" \
  "task_family=math" \
  "model=Qwen3-32B" \
  "execution_mode=each-turn" \
  "cases=${CASES}" \
  "predictions=${PREDICTIONS}" \
  "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${RUN_DIR}/run_config.txt"

evaluate_partial() {
  if [[ -s "${PREDICTIONS}" ]]; then
    "${PYTHON}" "${ROOT}/evaluation/multiturn/evaluate_math.py" \
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

{
  echo "Run directory: ${RUN_DIR}"
  echo "Resume policy: completed task IDs are skipped; failed task IDs are retried."
  "${PYTHON}" "${ROOT}/evaluation/multiturn/run_evolving_intent.py" \
    --cases "${CASES}" \
    --output "${PREDICTIONS}" \
    --max-cases 20 \
    --execution-mode each-turn \
    --graph-format sp_math
} 2>&1 | tee -a "${LOG}"

echo "Math-20 completed."
echo "Predictions: ${PREDICTIONS}"
echo "Summary:     ${SUMMARY}"
echo "Log:         ${LOG}"

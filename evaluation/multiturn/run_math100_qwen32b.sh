#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/data/sp-yzb/miniconda3/envs/sp/bin/python}"
CASES="${CASES:-${ROOT}/../evolving-intent-main/review_samples/math/gsm8k_operator_n100_seed20260821_pro_final/operator_cases.json}"
MAX_CASES="${MAX_CASES:-100}"
MAX_PASSES="${MAX_PASSES:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-10}"
RUN_DIR="${1:-${RUN_DIR:-${ROOT}/results/sp_math_profile_qwen32b_math100_$(date -u +%Y%m%dT%H%M%SZ)}}"
PREDICTIONS="${RUN_DIR}/predictions.jsonl"
SUMMARY="${RUN_DIR}/summary.json"
LOG="${RUN_DIR}/run.log"

[[ -x "${PYTHON}" ]] || { echo "Python environment does not exist: ${PYTHON}" >&2; exit 2; }
[[ -f "${CASES}" ]] || { echo "Math cases do not exist: ${CASES}" >&2; exit 2; }
[[ "${MAX_CASES}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_CASES must be positive" >&2; exit 2; }
[[ "${MAX_PASSES}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_PASSES must be positive" >&2; exit 2; }

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG}") 2>&1

"${PYTHON}" - "${ROOT}/conf.yaml" "${CASES}" "${MAX_CASES}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = str((config.get("BASIC_MODEL") or {}).get("model", ""))
if model.lower() != "qwen3-32b":
    raise SystemExit(f"Expected BASIC_MODEL=Qwen3-32B, found {model!r}")
payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
cases = payload.get("cases") if isinstance(payload, dict) else payload
requested = int(sys.argv[3])
if not isinstance(cases, list) or len(cases) < requested:
    raise SystemExit(f"Requested {requested} cases, dataset contains {len(cases) if isinstance(cases, list) else 'invalid data'}")
PY

printf '%s\n' \
  "graph_format=sp_math" \
  "task_family=math" \
  "model=Qwen3-32B" \
  "execution_mode=each-turn" \
  "max_cases=${MAX_CASES}" \
  "max_passes=${MAX_PASSES}" \
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

progress_counts() {
  "${PYTHON}" - "${CASES}" "${PREDICTIONS}" "${MAX_CASES}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cases = payload.get("cases") if isinstance(payload, dict) else payload
selected = {str(case["task_id"]) for case in cases[:int(sys.argv[3])]}
latest = {}
path = Path(sys.argv[2])
if path.is_file():
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            task_id = str(row.get("task_id"))
            if task_id in selected:
                latest[task_id] = row
completed = sum(row.get("status") == "completed" for row in latest.values())
failed = sum(row.get("status") == "failed" for row in latest.values())
print(completed, failed, len(selected) - completed)
PY
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

completed=0
failed=0
remaining="${MAX_CASES}"
for ((pass=1; pass<=MAX_PASSES; pass++)); do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Math pass ${pass}/${MAX_PASSES}"
  "${PYTHON}" "${ROOT}/evaluation/multiturn/run_evolving_intent.py" \
    --cases "${CASES}" \
    --output "${PREDICTIONS}" \
    --max-cases "${MAX_CASES}" \
    --execution-mode each-turn \
    --graph-format sp_math

  read -r completed failed remaining < <(progress_counts)
  echo "Progress after pass ${pass}: completed=${completed} failed=${failed} remaining=${remaining}"
  if [[ "${completed}" -eq "${MAX_CASES}" ]]; then
    echo "Math-${MAX_CASES} completed."
    exit 0
  fi
  if [[ "${pass}" -lt "${MAX_PASSES}" ]]; then
    echo "Retrying incomplete cases after ${RETRY_DELAY_SECONDS}s."
    sleep "${RETRY_DELAY_SECONDS}"
  fi
done

echo "Math-${MAX_CASES} incomplete after ${MAX_PASSES} passes: completed=${completed}, failed=${failed}, remaining=${remaining}" >&2
exit 3

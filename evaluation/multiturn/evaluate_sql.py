#!/usr/bin/env python3
"""Evaluate every SQL turn with EvolvingIntent's execution evaluator."""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVOLVING_INTENT_ROOT = REPO_ROOT.parent / "evolving-intent-main"
if str(EVOLVING_INTENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EVOLVING_INTENT_ROOT))

from evaluation.common.sql_evaluator import evaluate_sql_response  # noqa: E402


def _read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("cases file must be a list or contain a 'cases' list")
    return cases


def _read_latest_predictions(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[str(row["task_id"])] = row
    return latest


def _resolve_db_path(case: dict[str, Any]) -> Path:
    metadata = case.get("metadata", {})
    configured = Path(str(metadata.get("db_path", "")))
    if configured.is_file():
        return configured
    db_id = str(metadata.get("db_id", ""))
    fallback = (
        EVOLVING_INTENT_ROOT
        / "intent_construction/intent_extraction/dataset_impl/bird_sql/data"
        / "dev_extracted/dev_20230613/dev_databases"
        / db_id
        / f"{db_id}.sqlite"
    )
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"database for {case.get('task_id')} not found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = {str(case["task_id"]): case for case in _read_cases(args.cases)}
    predictions = _read_latest_predictions(args.predictions)
    case_rows = []
    by_num_turns: dict[int, list[bool]] = defaultdict(list)
    total_turns = correct_turns = valid_turns = 0

    for task_id, prediction in predictions.items():
        if task_id not in source or prediction.get("status") != "completed":
            continue
        case = source[task_id]
        metadata = case.get("metadata", {})
        per_turn_gold = metadata.get("per_turn_gold", [])
        turn_results = prediction.get("turn_results", [])
        if len(turn_results) != len(per_turn_gold):
            raise ValueError(
                f"{task_id}: {len(turn_results)} predictions for "
                f"{len(per_turn_gold)} gold turns"
            )
        db_path = _resolve_db_path(case)
        evaluated_turns = []
        for turn_result, gold in zip(turn_results, per_turn_gold):
            result = evaluate_sql_response(
                model_response=str(turn_result.get("response", "")),
                gold_sql=str(gold["sql"]),
                db_path=db_path,
                expected_contract=gold.get("result_contract"),
                expected_semantic_slots=gold.get("semantic_slots"),
            )
            result_dict = dataclasses.asdict(result)
            evaluated_turns.append({
                "turn_index": int(turn_result["turn_index"]),
                **result_dict,
            })
            total_turns += 1
            correct_turns += int(result.execution_match)
            valid_turns += int(result.model_sql_valid)

        final_correct = evaluated_turns[-1]["execution_match"]
        num_turns = int(case["num_turns"])
        by_num_turns[num_turns].append(final_correct)
        case_rows.append({
            "task_id": task_id,
            "num_turns": num_turns,
            "final_correct": final_correct,
            "turns": evaluated_turns,
        })

    final_correct = sum(row["final_correct"] for row in case_rows)
    summary = {
        "evaluated_cases": len(case_rows),
        "final_correct": final_correct,
        "final_accuracy": final_correct / len(case_rows) if case_rows else None,
        "evaluated_turns": total_turns,
        "turn_correct": correct_turns,
        "turn_accuracy": correct_turns / total_turns if total_turns else None,
        "valid_sql_turns": valid_turns,
        "valid_sql_rate": valid_turns / total_turns if total_turns else None,
        "by_num_turns": {
            str(count): {
                "evaluated": len(values),
                "correct": sum(values),
                "accuracy": sum(values) / len(values),
            }
            for count, values in sorted(by_num_turns.items())
        },
        "rows": case_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

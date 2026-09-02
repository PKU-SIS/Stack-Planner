#!/usr/bin/env python3
"""Evaluate final StackPlanner responses against EvolvingIntent Math labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any

from run_evolving_intent import DEFAULT_CASES, DEFAULT_OUTPUT


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return boxed[-1].strip()
    gsm8k = re.findall(r"####\s*(-?[0-9][0-9,.]*)", text)
    if gsm8k:
        return gsm8k[-1]
    answer = re.findall(r"(?i)(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", text)
    if answer:
        return answer[-1].strip().rstrip(".")
    numbers = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def normalize_answer(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = re.sub(r"\\(?:text|mathrm)\{[^}]*\}", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "").strip()
    try:
        number = Decimal(cleaned)
        return format(number.normalize(), "f")
    except InvalidOperation:
        return cleaned.lower()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT.with_name("summary.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    source = {str(case["task_id"]): case for case in cases}
    latest = {str(row["task_id"]): row for row in _read_jsonl(args.predictions)}
    rows = []
    by_turn: dict[int, list[bool]] = defaultdict(list)
    for task_id, prediction in latest.items():
        if task_id not in source or prediction.get("status") != "completed":
            continue
        case = source[task_id]
        extracted = extract_answer(prediction.get("final_response"))
        correct = normalize_answer(extracted) == normalize_answer(str(case["label"]))
        by_turn[int(case["num_turns"])].append(correct)
        rows.append({
            "task_id": task_id,
            "num_turns": int(case["num_turns"]),
            "extracted_answer": extracted,
            "reference_answer": str(case["label"]),
            "correct": correct,
        })

    summary = {
        "evaluated": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": (sum(row["correct"] for row in rows) / len(rows) if rows else None),
        "by_num_turns": {
            str(turn_count): {
                "evaluated": len(values),
                "correct": sum(values),
                "accuracy": sum(values) / len(values),
            }
            for turn_count, values in sorted(by_turn.items())
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

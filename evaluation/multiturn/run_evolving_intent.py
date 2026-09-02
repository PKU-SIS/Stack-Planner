#!/usr/bin/env python3
"""Run StackPlanner on EvolvingIntent fixed multi-turn cases.

Only ``task_id`` and user ``turns`` enter StackPlanner. Reference answers and
operator metadata remain evaluation-side data and are never included in the
model prompt.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CASES = (
    REPO_ROOT.parent
    / "evolving-intent-main"
    / "review_samples"
    / "math"
    / "gsm8k_operator_n20_seed20260821"
    / "operator_cases.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "evolving_intent" / "predictions.jsonl"


def _read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("cases file must be a list or an object containing 'cases'")
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not case.get("task_id"):
            raise ValueError(f"case {index} has no task_id")
        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"case {case['task_id']} has no turns")
        for turn in turns:
            if set(turn) != {"role", "content"}:
                raise ValueError(
                    f"case {case['task_id']} turn contains fields other than role/content"
                )
            if turn["role"] != "user" or not isinstance(turn["content"], str):
                raise ValueError(f"case {case['task_id']} contains an invalid user turn")
    return cases


def build_transcript(turns: list[dict[str, str]]) -> str:
    """Project user turns into a deterministic, oracle-free agent prompt."""
    rendered = "\n\n".join(
        f"<turn index=\"{index}\" role=\"user\">\n{turn['content']}\n</turn>"
        for index, turn in enumerate(turns, start=1)
    )
    return (
        "The following is one evolving multi-turn user conversation. Interpret "
        "the requests in chronological order: later corrections replace the "
        "corresponding earlier facts, and explicitly withdrawn temporary "
        "requirements are no longer active. Complete the task that is active "
        "after the final turn, using the requested final output format.\n\n"
        f"<conversation>\n{rendered}\n</conversation>"
    )


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if row.get("status") in {"completed", "dry_run"}:
            completed.add(str(row["task_id"]))
    return completed


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


class StackPlannerRuntime:
    """Lazy StackPlanner graph wrapper so dry-runs require no agent imports."""

    def __init__(self, args: argparse.Namespace) -> None:
        from src.graph.builder import get_graph_by_format
        from src.graph.sp_nodes import init_agents

        self.init_agents = init_agents
        self.graph = get_graph_by_format(args.graph_format, with_memory=True)
        self.args = args

    def start_case(self) -> None:
        """Create one central agent whose internal stack persists across turns."""
        self.init_agents(self.args.graph_format)

    async def run(self, prompt: str, task_id: str, turn_index: int) -> str:
        initial_state = {
            # MemorySaver appends this new public user message to the same
            # thread. The harness never summarizes or resends a rewritten task.
            "messages": [{"role": "user", "content": prompt}],
            "auto_accepted_plan": True,
            "enable_background_investigation": self.args.background_investigation,
            "user_query": prompt,
            "final_report": "",
            "skip_perception": True,
        }
        config = {
            "configurable": {
                "thread_id": f"evolving-intent:{task_id}",
                "graph_format": self.args.graph_format,
                "max_plan_iterations": self.args.max_plan_iterations,
                "max_step_num": self.args.max_step_num,
                "mcp_settings": {"servers": {}},
            },
            "recursion_limit": self.args.recursion_limit,
        }
        final_state = await self.graph.ainvoke(initial_state, config=config)
        response = final_state.get("final_report") if isinstance(final_state, dict) else None
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("StackPlanner completed without a non-empty final_report")
        return response


async def run(args: argparse.Namespace) -> None:
    cases = _read_cases(args.cases)
    selected_ids = set(args.task_id or [])
    if selected_ids:
        cases = [case for case in cases if str(case["task_id"]) in selected_ids]
        missing = selected_ids - {str(case["task_id"]) for case in cases}
        if missing:
            raise ValueError(f"unknown task ids: {sorted(missing)}")
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    completed = _load_completed(args.output) if args.resume else set()
    runtime = None if args.dry_run else StackPlannerRuntime(args)
    dataset_sha256 = hashlib.sha256(args.cases.read_bytes()).hexdigest()

    for case_index, case in enumerate(cases, start=1):
        task_id = str(case["task_id"])
        if task_id in completed:
            print(f"[{case_index}/{len(cases)}] skip completed {task_id}", flush=True)
            continue
        turns = case["turns"]
        if runtime is not None:
            runtime.start_case()
        turn_results = []
        started = time.monotonic()
        status = "dry_run" if args.dry_run else "completed"
        error = None
        print(f"[{case_index}/{len(cases)}] {task_id} ({len(turns)} turns)", flush=True)
        try:
            if args.execution_mode == "each-turn":
                requests = [turn["content"] for turn in turns]
            else:
                requests = [build_transcript(turns)]
            for turn_index, prompt in enumerate(requests, start=1):
                response = None if runtime is None else await runtime.run(
                    prompt, task_id, turn_index
                )
                turn_results.append({
                    "turn_index": turn_index,
                    "prompt": prompt,
                    "response": response,
                })
        except Exception as exc:  # preserve failures for resumable long runs
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            print(f"  failed: {error}", flush=True)

        row = {
            "schema_version": 1,
            "task_id": task_id,
            "num_turns": len(turns),
            "status": status,
            "execution_mode": args.execution_mode,
            "graph_format": args.graph_format,
            "dataset_sha256": dataset_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "turn_results": turn_results,
            "final_response": (
                turn_results[-1]["response"] if turn_results else None
            ),
            "error": error,
        }
        _append_jsonl(args.output, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--execution-mode",
        choices=("final", "each-turn"),
        default="each-turn",
        help="Run once on the full transcript, or once for every transcript prefix.",
    )
    parser.add_argument(
        "--graph-format",
        default="sp_math",
        choices=("sp_math", "sp", "sp_xxqg"),
    )
    parser.add_argument("--max-plan-iterations", type=int, default=1)
    parser.add_argument("--max-step-num", type=int, default=3)
    parser.add_argument("--recursion-limit", type=int, default=100)
    parser.add_argument("--background-investigation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))

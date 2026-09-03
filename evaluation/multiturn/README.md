# EvolvingIntent multi-turn evaluation

This adapter runs StackPlanner on the fixed 20-case Math operator set in the
neighboring `evolving-intent-main` checkout. The runtime projection contains
only each case's `task_id` and `turns`; labels, source questions, operator plans,
and validation metadata never enter the agent prompt.

The default `sp_math` graph uses StackPlanner's central planner, an optional
coder for tool verification, and a lightweight conclusion agent. It does not
invoke web research or the long-form report agent. The default `each-turn` mode
runs the complete graph for every incoming user turn on one checkpointed
thread. Raw user/assistant messages and StackPlanner's own MemoryStack are
preserved; the harness performs no summarization. `final` runs the graph
on every conversation prefix and is much more expensive; only the last response
is scored.

```bash
cd /data/sp/yzb/AgentMultiTurn/Stack-Planner

# Validate data loading and inspect one rendered prompt without API calls.
/data/sp-yzb/miniconda3/envs/sp/bin/python \
  evaluation/multiturn/run_evolving_intent.py --dry-run --max-cases 1 \
  --output /tmp/stackplanner_dry_run.jsonl --no-resume

# One real smoke case.
/data/sp-yzb/miniconda3/envs/sp/bin/python \
  evaluation/multiturn/run_evolving_intent.py --max-cases 1

# Resume and finish all 20 cases (completed task IDs are skipped).
/data/sp-yzb/miniconda3/envs/sp/bin/python \
  evaluation/multiturn/run_evolving_intent.py

# Score completed predictions.
/data/sp-yzb/miniconda3/envs/sp/bin/python \
  evaluation/multiturn/evaluate_math.py
```

Predictions are appended to `results/evolving_intent/predictions.jsonl`; the
score report is written to `results/evolving_intent/summary.json`.

Run the complete online Math-20 experiment with the configured Qwen3-32B:

```bash
bash evaluation/multiturn/run_math20_qwen32b.sh
```

Pass an explicit run directory to resume the same JSONL after interruption:

```bash
bash evaluation/multiturn/run_math20_qwen32b.sh \
  results/sp_math_online_qwen32b_math20_<timestamp>
```

## SQL

`sp_sql` is a separate graph with only `central_agent`, `sql_agent`, and
`sql_conclusion`. Each public turn runs the complete graph using the same
checkpoint thread. The agent sees only the raw conversation, including schema
and sample rows supplied in the first turn. It does not receive database paths,
execution results, gold SQL, labels, or per-turn evaluator metadata.

Run a frozen SQL cases file with Qwen3-32B:

```bash
bash evaluation/multiturn/run_sql_qwen32b.sh /path/to/operator_cases.json
```

Set `MAX_CASES=3` for a smoke run. SQL evaluation uses EvolvingIntent's official
execution/semantic evaluator for every turn and for the final turn. A cases file
is intentionally required because no SQL set has yet been frozen under
`review_samples/sql/`; `.tmp` build artifacts are not silently selected.

The runner uses the `sp` environment for StackPlanner and the existing
`evolvingintent-bird` environment for SQL evaluation. Override these paths with
`PYTHON` and `EVAL_PYTHON` when needed.

## Math-100

Run the frozen 100-case Math set with Qwen3-32B:

```bash
bash evaluation/multiturn/run_math100_qwen32b.sh
```

The script runs every turn through `sp_math`, resumes completed task IDs, and
retries incomplete/API-failed cases for up to three passes. Override
`MAX_PASSES`, `RETRY_DELAY_SECONDS`, or `RUN_DIR` when needed.

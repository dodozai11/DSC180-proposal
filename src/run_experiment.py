"""Run every task x format x run, call the model, grade the code, and log one CSV row per call.

Usage:
    python src/run_experiment.py            # run (resumes automatically)
    python src/run_experiment.py --dry-run  # show what would run; no API calls, no key needed
"""

import argparse
import csv
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Make config.py (in the repo root) importable no matter where this is run from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from evaluate import extract_code, run_tests  # noqa: E402

FORMATS = ["structured", "informal"]
COLUMNS = ["task_id", "format", "run", "model", "input_tokens", "output_tokens",
           "tests_passed", "tests_total", "passed_all", "stop_reason", "timestamp"]

# Rough per-call token guesses, used only for the cost estimate printed before starting.
EST_INPUT_TOKENS, EST_OUTPUT_TOKENS = 250, 400
# USD per million tokens (input, output). Only used for the estimate.
PRICES = {"claude-haiku-4-5": (1.00, 5.00)}


def load_tasks():
    """Return {task_id: {"dir": Path, "structured": str, "informal": str}}."""
    tasks = {}
    for task_dir in sorted((ROOT / config.TASKS_DIR).glob("t*")):
        if task_dir.is_dir():
            tasks[task_dir.name] = {
                "dir": task_dir,
                "structured": (task_dir / "structured.txt").read_text(encoding="utf-8"),
                "informal": (task_dir / "informal.txt").read_text(encoding="utf-8"),
            }
    return tasks


def build_plan(task_ids):
    """Every (task_id, format, run) combination, shuffled in a reproducible order."""
    plan = [(task_id, fmt, run)
            for task_id in task_ids
            for fmt in FORMATS
            for run in range(1, config.N_RUNS + 1)]
    random.Random(config.SEED).shuffle(plan)
    return plan


def load_done(results_path):
    """Set of (task_id, format, run, model) already in the CSV, so we can resume."""
    if not results_path.exists():
        return set()
    with open(results_path, newline="", encoding="utf-8") as f:
        return {(r["task_id"], r["format"], int(r["run"]), r["model"]) for r in csv.DictReader(f)}


def append_row(results_path, row):
    """Append one row to the CSV right away (writing the header if the file is new)."""
    is_new = not results_path.exists()
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def call_model(client, prompt):
    """One API call with the fixed controls. Returns the response object."""
    return client.messages.create(
        model=config.MODEL,
        max_tokens=config.MAX_TOKENS,
        system=config.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        # SDK 1.x removed the `temperature` keyword, but the API still accepts it for
        # Haiku 4.5, so we send it in the request body directly.
        extra_body={"temperature": config.TEMPERATURE},
        # No `thinking` parameter -> no extended thinking.
        # No `cache_control` anywhere -> no prompt caching.
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="show the plan; make no API calls")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")  # puts ANTHROPIC_API_KEY from .env into the environment
    results_path = ROOT / config.RESULTS_PATH
    raw_dir = ROOT / config.RAW_DIR

    tasks = load_tasks()
    plan = build_plan(tasks)
    done = load_done(results_path)
    todo = [(t, f, r) for (t, f, r) in plan if (t, f, r, config.MODEL) not in done]

    in_price, out_price = PRICES.get(config.MODEL, (None, None))
    print(f"model={config.MODEL}  tasks={len(tasks)}  formats={len(FORMATS)}  runs={config.N_RUNS}")
    print(f"planned={len(plan)}  already done={len(plan) - len(todo)}  calls to make={len(todo)}")
    if in_price is not None:
        cost = len(todo) * (EST_INPUT_TOKENS * in_price + EST_OUTPUT_TOKENS * out_price) / 1_000_000
        print(f"rough cost estimate: ${cost:.2f}")

    if args.dry_run:
        print(f"\ntemperature={config.TEMPERATURE}  max_tokens={config.MAX_TOKENS}  seed={config.SEED}")
        print("first planned calls:")
        for task_id, fmt, run in todo[:10]:
            print(f"  {task_id:<20} {fmt:<11} run {run}")
        print("\n(dry run: no API calls made)")
        return

    if not todo:
        print("Nothing to do.")
        return

    import anthropic  # imported here so --dry-run works even without the SDK set up

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    results_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for i, (task_id, fmt, run) in enumerate(todo, start=1):
        task = tasks[task_id]
        try:
            response = call_model(client, task[fmt])
        except anthropic.RateLimitError as e:
            print(f"[{i}/{len(todo)}] {task_id} {fmt} run{run}: rate limited, skipping ({e})")
            continue
        except anthropic.APIStatusError as e:
            print(f"[{i}/{len(todo)}] {task_id} {fmt} run{run}: API error {e.status_code}, skipping ({e})")
            continue
        except anthropic.APIConnectionError as e:
            print(f"[{i}/{len(todo)}] {task_id} {fmt} run{run}: connection error, skipping ({e})")
            continue

        usage = response.usage
        if (usage.cache_read_input_tokens or 0) > 0 or (usage.cache_creation_input_tokens or 0) > 0:
            print("  WARNING: prompt caching was used on this call; input_tokens may be affected")

        # Save the raw reply so any result can be inspected later.
        text = "".join(block.text for block in response.content if block.type == "text")
        raw_name = f"{task_id}__{fmt}__run{run}__{config.MODEL}.txt"
        (raw_dir / raw_name).write_text(text, encoding="utf-8")

        passed, total = run_tests(extract_code(text), task["dir"], config.TEST_TIMEOUT)

        append_row(results_path, {
            "task_id": task_id,
            "format": fmt,
            "run": run,
            "model": config.MODEL,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "tests_passed": passed,
            "tests_total": total,
            "passed_all": passed == total,
            "stop_reason": response.stop_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"[{i}/{len(todo)}] {task_id} {fmt} run{run}: "
              f"in={usage.input_tokens} out={usage.output_tokens} tests={passed}/{total}")

    print(f"\nDone. Results in {config.RESULTS_PATH}")


if __name__ == "__main__":
    main()

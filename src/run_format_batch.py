"""Data-format v2: every format_tasks/ task x data format x run, sent to Claude as one Message Batch.

The batch API runs each request independently at half price; results come back within
24 hours (usually under one). Batch ids are kept in data/format_batch.json, so each step
can be run again safely.

Usage:
    python src/run_format_batch.py check              # offline: the tests accept/reject the right solutions
    python src/run_format_batch.py dry-run            # show the plan and a cost estimate; no API calls
    python src/run_format_batch.py count              # exact input tokens per prompt (free API call)
    python src/run_format_batch.py submit --limit 2   # a small test batch
    python src/run_format_batch.py submit             # everything not submitted yet
    python src/run_format_batch.py status
    python src/run_format_batch.py collect            # grade finished results into data/format_results.csv
    python src/run_format_batch.py submit --retry-failed   # resend requests that errored or expired
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from evaluate import count_tests, extract_code, run_tests  # noqa: E402

TASKS_DIR = ROOT / "format_tasks"
RESULTS_PATH = ROOT / "data" / "format_results.csv"
RAW_DIR = ROOT / "data" / "raw_format"
STATE_PATH = ROOT / "data" / "format_batch.json"

# Haiku 4.5: no thinking unless asked for, so output_tokens is exactly the visible reply,
# and it accepts temperature. TEMPERATURE, MAX_TOKENS and SYSTEM_PROMPT come from config.py.
MODEL = "claude-haiku-4-5"
N_RUNS = 40
FORMAT = "structured"  # instruction format; only one is used here
DATA_FORMATS = ["csv", "json", "xml", "markdown", "prose"]
COLUMNS = ["task_id", "format", "data_format", "run", "model", "input_tokens", "output_tokens",
           "tests_passed", "tests_total", "passed_all", "stop_reason", "timestamp",
           "response_chars", "code_chars", "cache_tokens"]

# Batch prices, USD per million tokens (input, output): half the standard $1 / $5.
BATCH_PRICES = (0.50, 2.50)
EST_OUTPUT_TOKENS = 1000  # earlier Haiku 4.5 parse replies averaged about 750-1,060


def load_tasks():
    """{task_id: {"dir": Path, "template": str, "data": {data_format: text}}}"""
    tasks = {}
    for task_dir in sorted(TASKS_DIR.glob("t*")):
        tasks[task_dir.name] = {
            "dir": task_dir,
            "template": (task_dir / f"{FORMAT}.txt").read_text(encoding="utf-8"),
            "data": {d: (task_dir / f"data_{d}.txt").read_text(encoding="utf-8").strip() for d in DATA_FORMATS},
        }
    if not tasks:
        sys.exit("No tasks in format_tasks/. Run:  python scripts/make_format_tasks.py")
    return tasks


def build_prompt(task, data_format):
    return task["template"].replace("{DATA}", task["data"][data_format])


def custom_id(task_id, data_format, run):
    return f"{task_id}__{data_format}__run{run:02d}"


def parse_custom_id(cid):
    task_id, data_format, run = cid.split("__")
    return task_id, data_format, int(run.removeprefix("run"))


def build_plan(tasks):
    """Every custom_id, in a fixed order (the batch API processes them in any order anyway)."""
    return [custom_id(t, d, r) for t in tasks for d in DATA_FORMATS for r in range(1, N_RUNS + 1)]


def request_params(tasks, cid):
    task_id, data_format, _ = parse_custom_id(cid)
    return {
        "model": MODEL,
        "max_tokens": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,  # the SDK passes it through in the request body
        "system": config.SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_prompt(tasks[task_id], data_format)}],
        # No `thinking` -> no extended thinking. No `cache_control` -> no prompt caching.
    }


# --- Saved state ---------------------------------------------------------------

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"batches": []}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def load_done():
    """custom_ids already graded into the results CSV."""
    if not RESULTS_PATH.exists():
        return set()
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        return {custom_id(r["task_id"], r["data_format"], int(r["run"])) for r in csv.DictReader(f)}


def append_row(row):
    is_new = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def make_client():
    """A client holding the key, with the key then removed from the environment.

    The tests run the model's code with our environment, and a reply could call the
    Claude API itself, so the key must not be visible to it (as in run_experiment_claude.py).
    """
    load_dotenv(ROOT / ".env")
    key = os.environ.pop("ANTHROPIC_API_KEY", None)
    if not key:
        sys.exit("No ANTHROPIC_API_KEY. Copy .env.example to .env and paste your key.")
    return anthropic.Anthropic(api_key=key)


# --- Commands --------------------------------------------------------------------

def cmd_check(tasks):
    """The tests accept a parsing solution and reject an empty or hard-coded one, in every format."""
    all_ok = True
    for task_id, task in tasks.items():
        total = count_tests(task["dir"] / "test_solution.py")
        reference = (task["dir"] / "reference.py").read_text(encoding="utf-8")
        # a working solution that also pastes the prompt's rows into the code
        rows = json.loads((task["dir"] / "sample.json").read_text(encoding="utf-8"))
        hard_coded = f"ROWS = {rows!r}\n\n{reference}"
        problems = []
        for d in DATA_FORMATS:
            for name, code, expected in (("reference", reference, total), ("empty", "", total - 1),
                                         ("hard-coded", hard_coded, total - 1)):
                passed, _ = run_tests(code, task["dir"], config.TEST_TIMEOUT, d)
                if passed != expected:
                    problems.append(f"{d}: {name} passes {passed}/{total} (expected {expected})")
        print(f"{task_id:<22} {'OK' if not problems else 'PROBLEMS'}")
        for p in problems:
            print(f"    - {p}")
        all_ok &= not problems
    sys.exit(0 if all_ok else 1)


def cmd_dry_run(tasks):
    plan = build_plan(tasks)
    chars = sum(len(build_prompt(tasks[t], d)) for t, d, _ in map(parse_custom_id, plan))
    est_in = chars / 3.5 + len(plan) * 20  # rough: ~3.5 characters per token, plus the system prompt
    est_out = len(plan) * EST_OUTPUT_TOKENS
    cost = (est_in * BATCH_PRICES[0] + est_out * BATCH_PRICES[1]) / 1_000_000
    print(f"model={MODEL}  tasks={len(tasks)}  data formats={DATA_FORMATS}  runs={N_RUNS}")
    print(f"temperature={config.TEMPERATURE}  max_tokens={config.MAX_TOKENS}  instruction format={FORMAT}")
    print(f"requests={len(plan)}  already graded={len(load_done() & set(plan))}")
    print(f"rough batch cost: ${cost:.2f} (~{est_in / len(plan):.0f} input and ~{EST_OUTPUT_TOKENS} output tokens per call)")
    print("first ids:", ", ".join(plan[:3]), "...", plan[-1])
    print(f"\n--- system ---\n{config.SYSTEM_PROMPT}\n--- user ({plan[0]}) ---\n{request_params(tasks, plan[0])['messages'][0]['content']}")


def cmd_count(tasks, client):
    """Exact input tokens for each task x format (the token-counting endpoint is free)."""
    print(f"{'task':<22}" + "".join(f"{d:>10}" for d in DATA_FORMATS))
    for task_id in tasks:
        counts = []
        for d in DATA_FORMATS:
            params = request_params(tasks, custom_id(task_id, d, 1))
            counts.append(client.messages.count_tokens(model=MODEL, system=params["system"],
                                                       messages=params["messages"]).input_tokens)
        print(f"{task_id:<22}" + "".join(f"{c:>10}" for c in counts))


def cmd_submit(tasks, client, limit, retry_failed):
    state = load_state()
    plan = build_plan(tasks)
    if retry_failed:
        open_batches = [b["id"] for b in state["batches"]
                        if client.messages.batches.retrieve(b["id"]).processing_status != "ended"]
        if open_batches:
            sys.exit(f"Batches still running: {open_batches}. Wait, then `collect` before retrying.")
        done = load_done()
        todo = [cid for cid in plan if cid not in done]
    else:
        submitted = {cid for b in state["batches"] for cid in b["custom_ids"]}
        todo = [cid for cid in plan if cid not in submitted]
    if limit:
        todo = todo[:limit]
    if not todo:
        print("Nothing to submit.")
        return

    batch = client.messages.batches.create(
        requests=[{"custom_id": cid, "params": request_params(tasks, cid)} for cid in todo])
    state["batches"].append({"id": batch.id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "custom_ids": todo})
    save_state(state)
    print(f"Submitted batch {batch.id} with {len(todo)} requests. Check it with:  python src/run_format_batch.py status")


def cmd_status(client):
    state = load_state()
    if not state["batches"]:
        print("No batches submitted yet.")
    for b in state["batches"]:
        batch = client.messages.batches.retrieve(b["id"])
        c = batch.request_counts
        print(f"{b['id']}  {batch.processing_status:<11} requests={len(b['custom_ids'])}  "
              f"processing={c.processing} succeeded={c.succeeded} errored={c.errored} "
              f"expired={c.expired} canceled={c.canceled}")


def cmd_collect(tasks, client):
    """Grade every finished result that isn't in the CSV yet, one at a time."""
    state = load_state()
    done = load_done()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    graded, failed = 0, []
    for b in state["batches"]:
        batch = client.messages.batches.retrieve(b["id"])
        if batch.processing_status != "ended":
            print(f"{b['id']}: still {batch.processing_status}, skipped")
            continue
        ended_at = batch.ended_at.isoformat(timespec="seconds")
        for result in client.messages.batches.results(b["id"]):  # streamed, one result at a time
            cid = result.custom_id
            if cid in done:
                continue
            if result.result.type != "succeeded":
                failed.append(f"{cid} ({result.result.type})")
                continue
            msg = result.result.message
            text = "".join(block.text for block in msg.content if block.type == "text")
            (RAW_DIR / f"{cid}.txt").write_text(text, encoding="utf-8")

            task_id, data_format, run = parse_custom_id(cid)
            code = extract_code(text)
            passed, total = run_tests(code, tasks[task_id]["dir"], config.TEST_TIMEOUT, data_format)
            usage = msg.usage
            cache = (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
            if cache:
                print(f"  WARNING: {cid} used prompt caching ({cache} tokens)")
            append_row({
                "task_id": task_id, "format": FORMAT, "data_format": data_format, "run": run, "model": MODEL,
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                "tests_passed": passed, "tests_total": total, "passed_all": passed == total,
                "stop_reason": msg.stop_reason, "timestamp": ended_at,
                "response_chars": len(text), "code_chars": len(code), "cache_tokens": cache,
            })
            done.add(cid)
            graded += 1
            print(f"{cid}: in={usage.input_tokens} out={usage.output_tokens} tests={passed}/{total}", flush=True)

    print(f"\nGraded {graded} new results. {len(done & set(build_plan(tasks)))} of {len(build_plan(tasks))} done.")
    if failed:
        print(f"{len(failed)} requests failed; resend them with  submit --retry-failed:")
        for f in failed[:20]:
            print(f"  {f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "dry-run", "count", "submit", "status", "collect"])
    parser.add_argument("--limit", type=int, help="submit: send only this many requests (a test batch)")
    parser.add_argument("--retry-failed", action="store_true", help="submit: resend everything not graded yet")
    args = parser.parse_args()

    tasks = load_tasks()
    if args.command == "check":
        return cmd_check(tasks)
    if args.command == "dry-run":
        return cmd_dry_run(tasks)

    client = make_client()
    if args.command == "count":
        cmd_count(tasks, client)
    elif args.command == "submit":
        cmd_submit(tasks, client, args.limit, args.retry_failed)
    elif args.command == "status":
        cmd_status(client)
    elif args.command == "collect":
        cmd_collect(tasks, client)


if __name__ == "__main__":
    main()

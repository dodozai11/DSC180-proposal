"""Run every task x format x run, call the model, grade the code, and log one CSV row per call.

Usage:
    python src/run_experiment.py            # run (resumes automatically)
    python src/run_experiment.py --dry-run  # show what would run; no model calls
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

FORMATS = ["structured", "informal"]           # instruction format
DATA_FORMATS = ["csv", "xml", "prose"]         # how the data is written in the prompt
COLUMNS = ["task_id", "format", "data_format", "run", "model", "input_tokens", "output_tokens",
           "tests_passed", "tests_total", "passed_all", "stop_reason", "timestamp"]

# Rough per-call token guesses (prompt with 20 rows of data; answer that copies the data),
# used only for the cost estimate of paid models.
EST_INPUT_TOKENS, EST_OUTPUT_TOKENS = 800, 1000
# USD per million tokens (input, output). Only used for the estimate.
PRICES = {"claude-haiku-4-5": (1.00, 5.00)}


def load_tasks():
    """Return {task_id: {"dir": Path, "structured": str, "informal": str, "data": {data_format: text}}}.

    Each task has data_csv.txt / data_xml.txt / data_prose.txt (the same rows in each
    format), and its prompt templates contain a {DATA} placeholder where the data goes.
    Copy tasks (t11-t16) and parse tasks (t21-t26) are handled the same way here.
    """
    tasks = {}
    for task_dir in sorted((ROOT / config.TASKS_DIR).glob("t*")):
        if not task_dir.is_dir():
            continue
        data = {}
        for data_format in DATA_FORMATS:
            path = task_dir / f"data_{data_format}.txt"
            if path.exists():
                data[data_format] = path.read_text(encoding="utf-8").strip()
        tasks[task_dir.name] = {
            "dir": task_dir,
            "structured": (task_dir / "structured.txt").read_text(encoding="utf-8"),
            "informal": (task_dir / "informal.txt").read_text(encoding="utf-8"),
            "data": data,
        }
    return tasks


def build_prompt(task, fmt, data_format):
    """The instruction template for `fmt`, with the data inserted in `data_format`."""
    return task[fmt].replace("{DATA}", task["data"][data_format])


def build_plan(tasks):
    """Every (task_id, format, data_format, run) combination, shuffled in a reproducible order."""
    plan = [(task_id, fmt, data_format, run)
            for task_id, task in tasks.items()
            for fmt in FORMATS
            for data_format in task["data"]
            for run in range(1, config.N_RUNS + 1)]
    random.Random(config.SEED).shuffle(plan)
    return plan


def load_done(results_path):
    """Set of (task_id, format, data_format, run, model) already in the CSV, so we can resume."""
    if not results_path.exists():
        return set()
    with open(results_path, newline="", encoding="utf-8") as f:
        return {(r["task_id"], r["format"], r["data_format"], int(r["run"]), r["model"])
                for r in csv.DictReader(f)}


def append_row(results_path, row):
    """Append one row to the CSV right away (writing the header if the file is new)."""
    is_new = not results_path.exists()
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# --- Model calls -------------------------------------------------------------
# Each provider has its own small function, and both return the same dict:
#   {"text": ..., "input_tokens": ..., "output_tokens": ..., "stop_reason": ...}

def call_ollama(prompt, run):
    """One call to a local Ollama model with the fixed controls."""
    import ollama

    response = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_TOKENS,  # Ollama's name for max_tokens
            "num_ctx": 16384,                  # room for prompt (incl. data) + longest allowed answer
            "seed": config.SEED + run,         # a different seed for each run number
        },
    )
    return {
        "text": response.message.content,
        "input_tokens": response.prompt_eval_count,   # tokens in system + user prompt
        "output_tokens": response.eval_count,         # tokens the model generated
        "stop_reason": response.done_reason,          # "stop" normally, "length" if cut off
    }


_anthropic_client = None


def call_anthropic(prompt, run):
    """One call to Claude via the paid API with the fixed controls."""
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    response = _anthropic_client.messages.create(
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
    usage = response.usage
    if (usage.cache_read_input_tokens or 0) > 0 or (usage.cache_creation_input_tokens or 0) > 0:
        print("  WARNING: prompt caching was used on this call; input_tokens may be affected")
    return {
        "text": "".join(block.text for block in response.content if block.type == "text"),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "stop_reason": response.stop_reason,  # "end_turn" normally, "max_tokens" if cut off
    }


def call_model(prompt, run):
    if config.PROVIDER == "ollama":
        return call_ollama(prompt, run)
    if config.PROVIDER == "anthropic":
        return call_anthropic(prompt, run)
    raise ValueError(f"Unknown PROVIDER in config.py: {config.PROVIDER!r}")


def call_errors():
    """The exceptions that mean "this call failed, skip it and retry on the next resume"."""
    if config.PROVIDER == "ollama":
        import ollama
        return (ollama.ResponseError, ConnectionError)
    import anthropic
    return (anthropic.APIStatusError, anthropic.APIConnectionError)


def check_ollama_ready():
    """Stop early with a clear message if Ollama isn't running or the model isn't downloaded."""
    import ollama

    try:
        ollama.show(config.MODEL)
    except ConnectionError:
        sys.exit("Can't reach Ollama. Start the Ollama app, then run this again.")
    except ollama.ResponseError:
        sys.exit(f"Model {config.MODEL!r} isn't downloaded. Run:  ollama pull {config.MODEL}")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="show the plan; make no model calls")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")  # puts ANTHROPIC_API_KEY from .env into the environment (if used)
    results_path = ROOT / config.RESULTS_PATH
    raw_dir = ROOT / config.RAW_DIR

    tasks = load_tasks()
    plan = build_plan(tasks)
    done = load_done(results_path)
    todo = [(t, f, d, r) for (t, f, d, r) in plan if (t, f, d, r, config.MODEL) not in done]

    print(f"provider={config.PROVIDER}  model={config.MODEL}  "
          f"tasks={len(tasks)}  formats={len(FORMATS)}  data formats={DATA_FORMATS}  runs={config.N_RUNS}")
    print(f"planned={len(plan)}  already done={len(plan) - len(todo)}  calls to make={len(todo)}")
    if config.PROVIDER == "ollama":
        print("cost: free (runs locally)")
    elif config.MODEL in PRICES:
        in_price, out_price = PRICES[config.MODEL]
        cost = len(todo) * (EST_INPUT_TOKENS * in_price + EST_OUTPUT_TOKENS * out_price) / 1_000_000
        print(f"rough cost estimate: ${cost:.2f}")

    if args.dry_run:
        print(f"\ntemperature={config.TEMPERATURE}  max_tokens={config.MAX_TOKENS}  seed={config.SEED}")
        print("first planned calls:")
        for task_id, fmt, data_format, run in todo[:10]:
            print(f"  {task_id:<20} {fmt:<11} {data_format:<6} run {run}")
        print("\n(dry run: no model calls made)")
        return

    if not todo:
        print("Nothing to do.")
        return

    if config.PROVIDER == "ollama":
        check_ollama_ready()

    errors = call_errors()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_model = config.MODEL.replace(":", "-").replace("/", "-")  # ":" isn't allowed in Windows filenames

    for i, (task_id, fmt, data_format, run) in enumerate(todo, start=1):
        task = tasks[task_id]
        label = f"[{i}/{len(todo)}] {task_id} {fmt} {data_format} run{run}"
        try:
            result = call_model(build_prompt(task, fmt, data_format), run)
        except errors as e:
            print(f"{label}: call failed, skipping ({e})")
            continue

        # Save the raw reply so any result can be inspected later.
        raw_name = f"{task_id}__{fmt}__{data_format}__run{run}__{safe_model}.txt"
        (raw_dir / raw_name).write_text(result["text"], encoding="utf-8")

        passed, total = run_tests(extract_code(result["text"]), task["dir"], config.TEST_TIMEOUT, data_format)

        append_row(results_path, {
            "task_id": task_id,
            "format": fmt,
            "data_format": data_format,
            "run": run,
            "model": config.MODEL,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "tests_passed": passed,
            "tests_total": total,
            "passed_all": passed == total,
            "stop_reason": result["stop_reason"],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"{label}: in={result['input_tokens']} out={result['output_tokens']} tests={passed}/{total}",
              flush=True)

    print(f"\nDone. Results in {config.RESULTS_PATH}")


if __name__ == "__main__":
    main()

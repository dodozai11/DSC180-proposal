"""Tool-result version: the data arrives as the result of a tool call, and the model answers directly.

Each call sends a conversation where the user asks three questions, the assistant has already
called a tool (scripted, not generated), and the tool returned the 20 rows in one data format.
Only the model's final answer is generated. Answers are graded against tool_tasks/*/answers.json.

Usage:
    python src/run_tool_experiment.py              # run (resumes automatically)
    python src/run_tool_experiment.py --dry-run    # show what would run; no model calls
    python src/run_tool_experiment.py --check      # grade the answer keys against themselves
"""

import argparse
import ast
import csv
import json
import math
import random
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from run_experiment import check_ollama_ready  # noqa: E402

TASKS_DIR = ROOT / "tool_tasks"
RESULTS_PATH = ROOT / "data" / "tool_results.csv"
RAW_DIR = ROOT / config.RAW_DIR / "tool"

# qwen2.5-coder:7b (the copy/parse model) got ~8% of these answers right, too few to compare
# formats, so this version uses a stronger general model. Its thinking mode is off: thinking
# adds thousands of hidden tokens per call and ran this laptop out of memory.
MODEL = "qwen3:8b"
THINK = False
NUM_CTX = 4096  # the longest prompt is ~1,700 tokens; a smaller window needs much less memory
config.MODEL = MODEL  # so check_ollama_ready() checks this model

DATA_FORMATS = ["csv", "markdown", "json", "json_pretty", "xml"]
QUESTIONS = ["q1", "q2", "q3"]
SYSTEM_PROMPT = "You are a data analysis assistant. Answer the user's questions using the tool results."
COLUMNS = (["task_id", "data_format", "run", "model", "input_tokens", "output_tokens"]
           + [f"{q}_answer" for q in QUESTIONS] + [f"{q}_correct" for q in QUESTIONS]
           + ["n_correct", "answered", "stop_reason", "seconds", "timestamp"])


def load_tasks():
    """{task_id: {"questions": str, "tool": dict, "arguments": dict, "answers": dict, "data": {format: text}}}"""
    tasks = {}
    for task_dir in sorted(TASKS_DIR.glob("t*")):
        tool = json.loads((task_dir / "tool.json").read_text(encoding="utf-8"))
        tasks[task_dir.name] = {
            "questions": (task_dir / "questions.txt").read_text(encoding="utf-8").strip(),
            "tool": tool["tool"],
            "arguments": tool["arguments"],
            "answers": json.loads((task_dir / "answers.json").read_text(encoding="utf-8")),
            "data": {d: (task_dir / f"data_{d}.txt").read_text(encoding="utf-8").strip() for d in DATA_FORMATS},
        }
    return tasks


def build_messages(task, data_format):
    """The conversation up to the model's answer: question, scripted tool call, tool result."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["questions"]},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": task["tool"]["name"], "arguments": task["arguments"]}}]},
        {"role": "tool", "content": task["data"][data_format], "tool_name": task["tool"]["name"]},
    ]


def build_plan(tasks):
    plan = [(task_id, d, run) for task_id in tasks for d in DATA_FORMATS for run in range(1, config.N_RUNS + 1)]
    random.Random(config.SEED).shuffle(plan)
    return plan


def load_done():
    if not RESULTS_PATH.exists():
        return set()
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        return {(r["task_id"], r["data_format"], int(r["run"]), r["model"]) for r in csv.DictReader(f)}


def append_row(row):
    is_new = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# --- Grading -----------------------------------------------------------------

ANSWERS_RE = re.compile(r"ANSWERS:\s*`*\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)
BARE_RE = re.compile(r"(\{[^{}]*[\"']q1[\"'][^{}]*\})", re.DOTALL)  # the same dict without "ANSWERS:"


def parse_answers(text):
    """The dict from the last `ANSWERS: {...}` line (or a bare {"q1": ...} dict), or None."""
    matches = ANSWERS_RE.findall(text) or BARE_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1]
    for load in (json.loads, ast.literal_eval):
        try:
            answers = load(raw)
            if isinstance(answers, dict):
                return {str(k).lower(): v for k, v in answers.items()}
        except (ValueError, SyntaxError):
            pass
    return None


def normalize(text):
    return " ".join(unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower().split())


def to_number(value):
    """A number from 0.6, "0.6", or "60%" (-> 0.6); None if it isn't one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    try:
        return float(text[:-1]) / 100 if text.endswith("%") else float(text)
    except ValueError:
        return None


def is_correct(given, key):
    """Numbers: within 0.01 or 1%, whichever is larger. Names: any tied name is accepted."""
    if key["type"] == "number":
        x = to_number(given)
        return x is not None and math.isclose(x, key["value"], rel_tol=0.01, abs_tol=0.01)
    accept = {normalize(n) for n in key["accept"]}
    names = given if isinstance(given, list) else [given]
    return bool(names) and all(normalize(n) in accept for n in names)


def grade(text, answer_key):
    """({q: given answer}, {q: correct?}, answered?)"""
    answers = parse_answers(text)
    if answers is None:
        return {q: "" for q in QUESTIONS}, {q: False for q in QUESTIONS}, False
    given = {q: answers.get(q, "") for q in QUESTIONS}
    return given, {q: is_correct(given[q], answer_key[q]) for q in QUESTIONS}, True


def self_check(tasks):
    """Each answer key must grade 3/3 against itself, and every row must appear in every format."""
    ok = True
    for task_id, task in tasks.items():
        key = task["answers"]
        fake = {q: (k["value"] if k["type"] == "number" else k["accept"][0]) for q, k in key.items()}
        _, correct, answered = grade("ANSWERS: " + json.dumps(fake), key)
        rows = json.loads(task["data"]["json"])
        names = {str(v) for r in rows for k, v in r.items() if k in ("team", "player", "champion")}
        missing = {d: [n for n in names if n not in text] for d, text in task["data"].items()}
        missing = {d: m for d, m in missing.items() if m}
        status = "ok" if answered and all(correct.values()) and not missing else "FAIL"
        ok &= status == "ok"
        print(f"{task_id:<20} key {sum(correct.values())}/3  missing names: {missing or 'none'}  {status}")
    return ok


# --- Model call --------------------------------------------------------------

def call_ollama(task, data_format, run):
    import ollama

    response = ollama.chat(
        model=config.MODEL,
        messages=build_messages(task, data_format),
        tools=[{"type": "function", "function": task["tool"]}],
        think=THINK,
        options={
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_TOKENS,
            "num_ctx": NUM_CTX,
            "seed": config.SEED + run,
        },
    )
    return {
        "text": response.message.content or "",
        "input_tokens": response.prompt_eval_count,
        "output_tokens": response.eval_count,
        "stop_reason": response.done_reason,
        "seconds": round(response.total_duration / 1e9, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="show the plan; make no model calls")
    parser.add_argument("--check", action="store_true", help="grade the answer keys against themselves")
    parser.add_argument("--limit", type=int, help="make at most this many calls (for a quick test)")
    args = parser.parse_args()

    tasks = load_tasks()
    if args.check:
        sys.exit(0 if self_check(tasks) else 1)

    if config.PROVIDER != "ollama":
        sys.exit("The tool-result version only supports PROVIDER = 'ollama'.")

    plan = build_plan(tasks)
    done = load_done()
    todo = [(t, d, r) for (t, d, r) in plan if (t, d, r, config.MODEL) not in done]
    print(f"model={config.MODEL}  tasks={len(tasks)}  data formats={DATA_FORMATS}  runs={config.N_RUNS}")
    print(f"planned={len(plan)}  already done={len(plan) - len(todo)}  calls to make={len(todo)}")
    if args.limit:
        todo = todo[:args.limit]
    if args.dry_run:
        for task_id, d, run in todo[:10]:
            print(f"  {task_id:<20} {d:<12} run {run}")
        print("\n(dry run: no model calls made)")
        return

    check_ollama_ready()
    import ollama
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = config.MODEL.replace(":", "-").replace("/", "-")

    for i, (task_id, data_format, run) in enumerate(todo, start=1):
        task = tasks[task_id]
        label = f"[{i}/{len(todo)}] {task_id} {data_format} run{run}"
        try:
            result = call_ollama(task, data_format, run)
        except (ollama.ResponseError, ConnectionError) as e:
            print(f"{label}: call failed, skipping ({e})")
            continue

        (RAW_DIR / f"{task_id}__{data_format}__run{run}__{safe_model}.txt").write_text(result["text"], encoding="utf-8")
        given, correct, answered = grade(result["text"], task["answers"])

        append_row({
            "task_id": task_id, "data_format": data_format, "run": run, "model": config.MODEL,
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
            **{f"{q}_answer": json.dumps(given[q], ensure_ascii=False) if given[q] != "" else "" for q in QUESTIONS},
            **{f"{q}_correct": correct[q] for q in QUESTIONS},
            "n_correct": sum(correct.values()), "answered": answered,
            "stop_reason": result["stop_reason"], "seconds": result["seconds"],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"{label}: in={result['input_tokens']} out={result['output_tokens']} "
              f"correct={sum(correct.values())}/3{'' if answered else ' (no ANSWERS line)'}", flush=True)

    print(f"\nDone. Results in {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

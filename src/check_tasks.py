"""Sanity-check every task before running the experiment.

Copy tasks (t11-t16) check that:
  1. reference.py passes every test       -> the tests are passable
  2. an empty solution passes no tests    -> the tests actually check something
  3. both prompt files exist and are non-empty
  4. a solution missing one row of data fails the "data embedded" check
  5. every row identifier appears in all three data formats, and CSV/XML have one entry per row
  6. both prompt templates contain the {DATA} placeholder

Parse tasks (t21-t26, with hidden_<format>.txt files) check, for every data format:
  1. reference.py passes both tests
  2. an empty solution fails the "runs on new data" check (it trivially "doesn't hard-code")
  3. a solution that copies the data (its copy-task twin's reference) fails "not hard-coded"
plus: both prompt files exist, the prompt and hidden samples are each consistent across
formats, the hidden sample shares no names with the prompt sample, and {DATA} is present.

Usage:
    python src/check_tasks.py
"""

import ast
import html
import json
import re
import sys
from pathlib import Path

# Make config.py (in the repo root) importable no matter where this is run from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from evaluate import count_tests, run_tests  # noqa: E402

DATA_FORMATS = ["csv", "xml", "prose"]


def read(path):
    return path.read_text(encoding="utf-8")


def list_in_tests(test_file, name):
    """Read a list constant (e.g. IDENTIFIERS) out of a test file without running it."""
    for node in ast.parse(read(test_file)).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    return []


def check_renderings(task_dir, prefix, sample_name, names):
    """The rows in `sample_name` must appear consistently in every <prefix>_<format>.txt file."""
    problems = []
    n_rows = len(json.loads(read(task_dir / sample_name)))
    for fmt in DATA_FORMATS:
        text = html.unescape(read(task_dir / f"{prefix}_{fmt}.txt"))  # XML writes & as &amp;
        missing = [n for n in names if n not in text]
        if missing:
            problems.append(f"{prefix}_{fmt}.txt is missing {missing[:3]}")
    csv_rows = len(read(task_dir / f"{prefix}_csv.txt").strip().splitlines()) - 1  # minus the header
    xml_rows = read(task_dir / f"{prefix}_xml.txt").count("<row>")
    if not csv_rows == xml_rows == n_rows:
        problems.append(f"{prefix} row counts differ: sample={n_rows} csv={csv_rows} xml={xml_rows}")
    return problems


def check_common(task_dir):
    problems = []
    for name in ("structured.txt", "informal.txt"):
        path = task_dir / name
        if not path.exists() or not read(path).strip():
            problems.append(f"{name} missing or empty")
        elif "{DATA}" not in read(path):
            problems.append(f"{name} has no {{DATA}} placeholder")
    return problems


def check_copy_task(task_dir, total):
    problems = []
    reference = read(task_dir / "reference.py")
    passed, _ = run_tests(reference, task_dir, config.TEST_TIMEOUT)
    if passed != total:
        problems.append(f"reference passes {passed}/{total}")
    passed, _ = run_tests("", task_dir, config.TEST_TIMEOUT)
    if passed != 0:
        problems.append(f"empty solution passes {passed}/{total}")

    # Removing one identifier from the reference must fail exactly the "data embedded" check.
    ids = list_in_tests(task_dir / "test_solution.py", "IDENTIFIERS")
    passed, _ = run_tests(reference.replace(ids[0], "XXXX"), task_dir, config.TEST_TIMEOUT)
    if passed != total - 1:
        problems.append(f"reference missing a row passes {passed}/{total} (expected {total - 1})")
    return problems + check_renderings(task_dir, "data", "sample.json", ids)


def check_parse_task(task_dir, total):
    problems = []
    reference = read(task_dir / "reference.py")
    twin = ROOT / config.TASKS_DIR / ("t1" + task_dir.name[2:].removesuffix("_parse"))
    copied_data = re.sub(r"def (q\d)\(\):", r"def \1(text):", read(twin / "reference.py"))

    for fmt in DATA_FORMATS:
        passed, _ = run_tests(reference, task_dir, config.TEST_TIMEOUT, fmt)
        if passed != total:
            problems.append(f"{fmt}: reference passes {passed}/{total}")
        passed, _ = run_tests("", task_dir, config.TEST_TIMEOUT, fmt)
        if passed != total - 1:
            problems.append(f"{fmt}: empty solution passes {passed}/{total} (expected {total - 1})")
        passed, _ = run_tests(copied_data, task_dir, config.TEST_TIMEOUT, fmt)
        if passed != total - 1:
            problems.append(f"{fmt}: copied-data solution passes {passed}/{total} (expected {total - 1})")

    prompt_names = list_in_tests(task_dir / "test_solution.py", "PROMPT_NAMES")
    hidden_rows = json.loads(read(task_dir / "hidden.json"))
    hidden_names = sorted({v for r in hidden_rows for k, v in r.items() if k in ("team", "player")})
    problems += check_renderings(task_dir, "data", "sample.json", prompt_names)
    problems += check_renderings(task_dir, "hidden", "hidden.json", hidden_names)
    shared = set(prompt_names) & set(hidden_names)
    if shared:
        problems.append(f"hidden sample shares names with the prompt: {sorted(shared)[:3]}")
    return problems


def check_task(task_dir):
    """Return (kind, number of tests, list of problems) for one task folder."""
    total = count_tests(task_dir / "test_solution.py")
    if (task_dir / "hidden_csv.txt").exists():
        return "parse", total, check_common(task_dir) + check_parse_task(task_dir, total)
    return "copy", total, check_common(task_dir) + check_copy_task(task_dir, total)


def main():
    task_dirs = sorted(p for p in (ROOT / config.TASKS_DIR).glob("t*") if p.is_dir())
    all_ok = True
    for task_dir in task_dirs:
        kind, total, problems = check_task(task_dir)
        status = "OK  " if not problems else "FAIL"
        print(f"{status} {task_dir.name:<26} {kind:<5}  tests={total}  {'; '.join(problems)}")
        all_ok = all_ok and not problems
    print(f"\n{len(task_dirs)} tasks checked: {'all OK' if all_ok else 'PROBLEMS FOUND'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

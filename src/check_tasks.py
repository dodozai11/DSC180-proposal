"""Sanity-check every task before running the experiment.

For every task folder this checks that:
  1. reference.py passes every test       -> the tests are passable
  2. an empty solution passes no tests    -> the tests actually check something
  3. both prompt files exist and are non-empty
  4. a solution missing one row of data fails the "data embedded" check
  5. every row identifier appears in all three data formats, and CSV/XML have one entry per row
  6. both prompt templates contain the {DATA} placeholder

Usage:
    python src/check_tasks.py
"""

import ast
import html
import json
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


def identifiers_in_tests(test_file):
    """Read the IDENTIFIERS list out of a data task's test file without running it."""
    for node in ast.parse(read(test_file)).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "IDENTIFIERS":
            return ast.literal_eval(node.value)
    return []


def check_data_task(task_dir, reference, total):
    """Checks specific to data tasks: missing-row detection, format consistency, templates."""
    problems = []
    ids = identifiers_in_tests(task_dir / "test_solution.py")
    n_rows = len(json.loads(read(task_dir / "sample.json")))

    # Removing one identifier from the reference must fail exactly the "data embedded" check.
    passed, _ = run_tests(reference.replace(ids[0], "XXXX"), task_dir, config.TEST_TIMEOUT)
    if passed != total - 1:
        problems.append(f"reference missing a row passes {passed}/{total} (expected {total - 1})")

    for fmt in DATA_FORMATS:
        text = html.unescape(read(task_dir / f"data_{fmt}.txt"))  # XML writes & as &amp;
        missing = [i for i in ids if i not in text]
        if missing:
            problems.append(f"data_{fmt}.txt is missing {missing[:3]}")
    csv_rows = len(read(task_dir / "data_csv.txt").strip().splitlines()) - 1  # minus the header
    xml_rows = read(task_dir / "data_xml.txt").count("<row>")
    if not csv_rows == xml_rows == n_rows:
        problems.append(f"row counts differ: sample={n_rows} csv={csv_rows} xml={xml_rows}")

    for name in ("structured.txt", "informal.txt"):
        if "{DATA}" not in read(task_dir / name):
            problems.append(f"{name} has no {{DATA}} placeholder")
    return problems


def check_task(task_dir):
    """Return (number of tests, list of problems) for one task folder (empty list = all good)."""
    problems = []
    total = count_tests(task_dir / "test_solution.py")

    reference = read(task_dir / "reference.py")
    passed, _ = run_tests(reference, task_dir, config.TEST_TIMEOUT)
    if passed != total:
        problems.append(f"reference passes {passed}/{total}")

    passed, _ = run_tests("", task_dir, config.TEST_TIMEOUT)
    if passed != 0:
        problems.append(f"empty solution passes {passed}/{total}")

    for name in ("structured.txt", "informal.txt"):
        path = task_dir / name
        if not path.exists() or not read(path).strip():
            problems.append(f"{name} missing or empty")

    problems += check_data_task(task_dir, reference, total)

    return total, problems


def main():
    task_dirs = sorted(p for p in (ROOT / config.TASKS_DIR).glob("t*") if p.is_dir())
    all_ok = True
    for task_dir in task_dirs:
        total, problems = check_task(task_dir)
        status = "OK  " if not problems else "FAIL"
        print(f"{status} {task_dir.name:<20} tests={total}  {'; '.join(problems)}")
        all_ok = all_ok and not problems
    print(f"\n{len(task_dirs)} tasks checked: {'all OK' if all_ok else 'PROBLEMS FOUND'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

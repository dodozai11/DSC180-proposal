"""Sanity-check every task before spending money on API calls.

For each task folder this checks that:
  1. reference.py passes every test       -> the tests are passable
  2. an empty solution passes no tests    -> the tests actually check something
  3. #tests == #requirement rows in checklist.md -> one test per requirement
  4. both prompt files exist and are non-empty

Usage:
    python src/check_tasks.py
"""

import re
import sys
from pathlib import Path

# Make config.py (in the repo root) importable no matter where this is run from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from evaluate import count_tests, run_tests  # noqa: E402


def check_task(task_dir):
    """Return a list of problems found in one task folder (empty list = all good)."""
    problems = []
    total = count_tests(task_dir / "test_solution.py")

    reference = (task_dir / "reference.py").read_text(encoding="utf-8")
    passed, _ = run_tests(reference, task_dir, config.TEST_TIMEOUT)
    if passed != total:
        problems.append(f"reference passes {passed}/{total}")

    passed, _ = run_tests("", task_dir, config.TEST_TIMEOUT)
    if passed != 0:
        problems.append(f"empty solution passes {passed}/{total}")

    checklist = (task_dir / "checklist.md").read_text(encoding="utf-8")
    n_reqs = len(re.findall(r"^\| R\d+ \|", checklist, re.MULTILINE))
    if n_reqs != total:
        problems.append(f"checklist has {n_reqs} requirements but {total} tests")

    for name in ("structured.txt", "informal.txt"):
        path = task_dir / name
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            problems.append(f"{name} missing or empty")

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

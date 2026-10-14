"""Re-score saved responses against the current tests, without calling the model again.

Useful after fixing a task's tests: the token counts stay as they were (they come from
the model), only tests_passed / tests_total / passed_all are recomputed from data/raw/.

Usage:
    python src/regrade.py                      # every row that has a saved raw response
    python src/regrade.py t11_side_winrate     # only these tasks
"""

import csv
import sys
from pathlib import Path

# Make config.py (in the repo root) importable no matter where this is run from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from evaluate import extract_code, run_tests  # noqa: E402


def main():
    only_tasks = set(sys.argv[1:])
    results_path = ROOT / config.RESULTS_PATH
    raw_dir = ROOT / config.RAW_DIR

    with open(results_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    changed = missing = 0
    for row in rows:
        if only_tasks and row["task_id"] not in only_tasks:
            continue
        safe_model = row["model"].replace(":", "-").replace("/", "-")
        raw = raw_dir / f"{row['task_id']}__{row['format']}__{row['data_format']}__run{row['run']}__{safe_model}.txt"
        if not raw.exists():
            missing += 1
            continue
        task_dir = ROOT / config.TASKS_DIR / row["task_id"]
        passed, total = run_tests(extract_code(raw.read_text(encoding="utf-8")), task_dir, config.TEST_TIMEOUT,
                                  row["data_format"])
        new = {"tests_passed": str(passed), "tests_total": str(total), "passed_all": str(passed == total)}
        if any(row[k] != v for k, v in new.items()):
            print(f"{row['task_id']} {row['format']} {row['data_format']} run{row['run']}: "
                  f"{row['tests_passed']}/{row['tests_total']} -> {passed}/{total}")
            row.update(new)
            changed += 1

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{changed} rows changed, {missing} rows skipped (no raw response saved)")


if __name__ == "__main__":
    main()

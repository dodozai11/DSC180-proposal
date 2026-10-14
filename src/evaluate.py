"""Grade a model response: pull out the Python code and run the task's hidden tests.

Usage (manual check of one file against one task):
    python src/evaluate.py tasks/t11_side_winrate path/to/response.txt
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Matches ```python ... ``` or plain ``` ... ``` blocks. DOTALL lets "." match newlines.
FENCE_RE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text):
    """Return the code inside markdown fences, or the whole text if there are none.

    If there are several fenced blocks we keep the longest one: models sometimes
    add a short usage-example block after the real solution.
    """
    blocks = FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def count_tests(test_file):
    """Count `def test_...` functions. Works even if the solution can't be imported."""
    text = Path(test_file).read_text(encoding="utf-8")
    return len(re.findall(r"^def test_", text, re.MULTILINE))


def run_tests(code, task_dir, timeout, data_format=None):
    """Run the task's tests against `code`. Returns (tests_passed, tests_total).

    Crashes, syntax errors, and timeouts all count as 0 passed.
    `data_format` (csv/xml/prose) is passed to the tests as the DATA_FORMAT environment
    variable; parse-task tests use it to pick their hidden data file.
    """
    test_file = Path(task_dir) / "test_solution.py"
    total = count_tests(test_file)

    # A fresh temporary folder per run, so solutions never mix with each other.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "solution.py").write_text(code, encoding="utf-8")
        shutil.copy(test_file, tmp / "test_solution.py")
        for hidden in Path(task_dir).glob("hidden_*.txt"):  # parse tasks only
            shutil.copy(hidden, tmp / hidden.name)
        env = dict(os.environ, DATA_FORMAT=data_format or "")

        try:
            subprocess.run(
                # sys.executable = the same Python (and venv) that is running this script
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 "--junitxml=report.xml", "test_solution.py"],
                cwd=tmp,
                timeout=timeout,
                capture_output=True,
                stdin=subprocess.DEVNULL,  # code that calls input() fails instead of hanging
                env=env,
            )
        except subprocess.TimeoutExpired:
            return 0, total

        report = tmp / "report.xml"
        if not report.exists():
            return 0, total

        # Each <testcase> that has no <failure>, <error>, or <skipped> child passed.
        passed = 0
        for case in ET.parse(report).getroot().iter("testcase"):
            if not any(child.tag in ("failure", "error", "skipped") for child in case):
                passed += 1

    # If collection failed, pytest may report a single error "testcase"; never exceed total.
    return min(passed, total), total


if __name__ == "__main__":
    task_dir, response_file = sys.argv[1], sys.argv[2]
    code = extract_code(Path(response_file).read_text(encoding="utf-8"))
    print(run_tests(code, task_dir, timeout=10))

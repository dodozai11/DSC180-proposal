"""Build notebooks/eda1.ipynb from the cell text in this file.

This script writes the notebook WITHOUT outputs; execute it afterwards:

    python scripts/make_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/eda1.ipynb

Editing the notebook by hand works too, but the next run of this script overwrites it.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "eda1.ipynb"

cells = []


def md(text):
    """Add a markdown cell."""
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    """Add a code cell."""
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Data format vs. token usage: exploratory analysis

**Question:** when a prompt contains a dataset, does the **format of that data** (CSV, XML, or prose) change how many tokens an LLM uses to write Python code for it?

- **Controls:** same model, temperature, and system prompt on every call.
- **Data:** 20-row samples from the [Oracle's Elixir](https://oracleselixir.com) 2025 LoL esports match data.
""")

code("""
import ast
import html
import json
import re
import sys
import textwrap
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import HTML, Markdown

sys.path.insert(0, "../src")
from evaluate import extract_code

SEED = 42
RESULTS_CSV = "../data/results.csv"
MODEL = "qwen2.5-coder:7b"

DATA_FORMATS = ["csv", "xml", "prose"]
DATA_COLORS = {"csv": "#008300", "xml": "#4a3aa7", "prose": "#e87ba4"}
DATA_MARKERS = {"csv": "o", "xml": "s", "prose": "D"}

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae",
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 10,
})

responses = pd.read_csv(RESULTS_CSV)
responses = responses[responses["model"] == MODEL].copy()
responses["total_tokens"] = responses["input_tokens"] + responses["output_tokens"]

is_parse = responses["task_id"].str.endswith("_parse")
copy_rows = responses[~is_parse].copy()    # sections 1-4: the code contains the data
parse_rows = responses[is_parse].copy()    # sections 5-7: the code receives the data as text
valid = copy_rows[copy_rows["passed_all"]].copy()   # code runs and contains all the data
print(f"{len(copy_rows)} copy rows ({len(valid)} valid), {len(parse_rows)} parse rows")
responses.head()
""")

code("""
TASKS_DIR = Path("../tasks")
TASK_IDS = sorted(p.name for p in TASKS_DIR.glob("t*") if p.is_dir() and not p.name.endswith("_parse"))

def read_task_file(task_id, name):
    return (TASKS_DIR / task_id / name).read_text(encoding="utf-8")

print("Tasks:", ", ".join(TASK_IDS))
""")

code("""
EXAMPLE_DATA_TASK = "t11_side_winrate"

def preview(text, data_format, n=4):
    \"\"\"The first n rows of a data block (prose rows are sentences, so cut by characters instead).\"\"\"
    if data_format == "prose":
        return textwrap.fill(text[:420], 60) + " ..."
    lines = text.splitlines()
    return "\\n".join(lines[:n + 1]) + "\\n..."   # +1 for the CSV header / opening <rows> tag

columns = "".join(
    f'<td style="vertical-align:top; padding:0 12px"><b>data_{d}.txt</b>'
    f'<pre style="white-space:pre-wrap">{html.escape(preview(read_task_file(EXAMPLE_DATA_TASK, f"data_{d}.txt"), d))}</pre></td>'
    for d in DATA_FORMATS
)
display(Markdown(f"#### `{EXAMPLE_DATA_TASK}`: first rows in each data format"))
display(HTML(f"<table><tr>{columns}</tr></table>"))
""")

code("""
data_rows = []
for task_id in TASK_IDS:
    row = {"task_id": task_id, "rows": len(json.loads(read_task_file(task_id, "sample.json")))}
    for d in DATA_FORMATS:
        row[f"{d} chars"] = len(read_task_file(task_id, f"data_{d}.txt"))
    data_rows.append(row)
data_table = pd.DataFrame(data_rows).set_index("task_id")

input_by_format = copy_rows.groupby(["task_id", "data_format"])["input_tokens"].mean().unstack()
for d in DATA_FORMATS:
    data_table[f"{d} input tokens"] = input_by_format[d].round(0)
data_table
""")

md("## 1. Tokens by data format")
code("""
valid.groupby("data_format")[["input_tokens", "output_tokens", "total_tokens"]].agg(["mean", "median"]).round(0).loc[DATA_FORMATS]
""")

md("## 2. Output tokens by data format")
code("""
per_task = valid.groupby(["task_id", "data_format"])["output_tokens"].mean().unstack()[DATA_FORMATS]
x = np.arange(len(DATA_FORMATS))

fig, ax = plt.subplots(figsize=(6.5, 4))
for task_id, row in per_task.iterrows():
    ax.plot(x, row.values, color="#d8d7d1", linewidth=1.2, zorder=1)
ax.plot(x, per_task.mean().values, color="#52514e", linewidth=2, zorder=2)
for i, d in enumerate(DATA_FORMATS):
    ax.scatter(i, per_task[d].mean(), s=90, color=DATA_COLORS[d], marker=DATA_MARKERS[d], zorder=3,
               edgecolor="white", linewidth=1.2)
ax.set_xticks(x, DATA_FORMATS)
ax.set_xlim(-0.3, len(DATA_FORMATS) - 0.7)
ax.set_xlabel("data format")
ax.set_ylabel("mean output tokens")
ax.set_title("Output tokens (gray: tasks, markers: mean)", loc="left")
ax.grid(axis="x", visible=False)
plt.show()
""")

md("## 3. Output vs input tokens")
code("""
valid["output_per_input"] = valid["output_tokens"] / valid["input_tokens"]

fig, ax = plt.subplots(figsize=(6, 4.2))
for d in DATA_FORMATS:
    sub_rows = valid[valid["data_format"] == d]
    ax.scatter(sub_rows["input_tokens"], sub_rows["output_tokens"], s=26, alpha=0.75, color=DATA_COLORS[d],
               marker=DATA_MARKERS[d], edgecolor="white", linewidth=0.5, label=d)
ax.set_xlabel("input tokens")
ax.set_ylabel("output tokens")
ax.set_title("Output vs input tokens, by data format", loc="left")
ax.legend(frameon=False, loc="lower right")
plt.show()

ratio = valid.groupby(["task_id", "data_format"])["output_per_input"].mean().unstack()[DATA_FORMATS]
ratio.mean().round(2).to_frame("mean output / input")
""")

md("""
## 4. How the model stored the data

Reads the saved responses in `data/raw/` (local only).
""")
code("""
def storage_style(code_text):
    \"\"\"How the model wrote the 20 rows into its code: dicts with a key per value, or bare tuples/lists.\"\"\"
    dict_rows = code_text.count('{"') + code_text.count("{'")          # {"team": ..., "side": ...}
    tuple_rows = sum(line.strip()[:2] in ('("', "('", '["', "['")      # ("DN Freecs", "blue", ...)
                     for line in code_text.splitlines())
    if dict_rows >= 15:
        return "dicts with keys"
    if tuple_rows >= 15:
        return "tuples/lists"
    return "other"

safe_model = MODEL.replace(":", "-").replace("/", "-")
style_rows = []
for path in Path("../data/raw").glob(f"*__{safe_model}.txt"):
    task_id, fmt, data_format, run, _ = path.stem.split("__")
    if data_format not in DATA_FORMATS:
        continue
    code_text = extract_code(path.read_text(encoding="utf-8"))
    style_rows.append({"task_id": task_id, "format": fmt, "data_format": data_format,
                       "run": int(run.removeprefix("run")), "style": storage_style(code_text)})
styles = pd.DataFrame(style_rows).merge(valid, on=["task_id", "format", "data_format", "run"])

display(pd.crosstab(styles["data_format"], styles["style"]).loc[DATA_FORMATS])
display(styles.groupby("style")["output_tokens"].agg(["count", "mean"]).round(0))
display(styles.groupby(["data_format", "style"])["output_tokens"].agg(["count", "mean"]).round(0).loc[DATA_FORMATS])
""")

md("""
## 5. Parse version: tokens by data format

The code receives the data as text (`q1(text)`) instead of containing it.
""")
code("""
def prompt_names(task_id):
    for node in ast.parse((TASKS_DIR / task_id / "test_solution.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "PROMPT_NAMES":
            return ast.literal_eval(node.value)

def normalize(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()

def saved_code(r):
    path = Path("../data/raw") / f"{r.task_id}__{r.format}__{r.data_format}__run{r.run}__{safe_model}.txt"
    return extract_code(path.read_text(encoding="utf-8"))

parse_rows["code_text"] = [saved_code(r) for r in parse_rows.itertuples()]
parse_rows["copied"] = [sum(normalize(n) in normalize(code_text) for n in prompt_names(task_id)) > 2
                        for task_id, code_text in zip(parse_rows["task_id"], parse_rows["code_text"])]
attempts = parse_rows[~parse_rows["copied"]].copy()   # responses that didn't copy the data
print(f"{len(attempts)} of {len(parse_rows)} parse responses didn't copy the data")
attempts.groupby("data_format")[["input_tokens", "output_tokens", "total_tokens"]].agg(["mean", "median"]).round(0).loc[DATA_FORMATS]
""")

md("## 6. Copy vs parse: output tokens")
code("""
copy_means = valid.groupby(["task_id", "data_format"])["output_tokens"].mean()
twin_ids = attempts["task_id"].str.removesuffix("_parse").str.replace("t2", "t1", n=1)
parse_means = attempts.assign(task_id=twin_ids).groupby(["task_id", "data_format"])["output_tokens"].mean()
copy_vs_parse = pd.DataFrame({"copy": copy_means, "parse": parse_means}).dropna().groupby("data_format").mean().loc[DATA_FORMATS].round(0)
copy_vs_parse["parse / copy"] = (copy_vs_parse["parse"] / copy_vs_parse["copy"]).round(2)

fig, ax = plt.subplots(figsize=(6.5, 3.2))
y = np.arange(len(DATA_FORMATS))
ax.hlines(y, copy_vs_parse["parse"], copy_vs_parse["copy"], color="#d8d7d1", linewidth=3, zorder=1)
ax.scatter(copy_vs_parse["copy"], y, s=70, color="#8a8984", zorder=2, label="copy")
for i, d in enumerate(DATA_FORMATS):
    ax.scatter(copy_vs_parse.loc[d, "parse"], i, s=70, color=DATA_COLORS[d], marker=DATA_MARKERS[d], zorder=3,
               edgecolor="white", linewidth=1)
ax.scatter([], [], s=70, color="#52514e", marker="D", label="parse (colored by format)")
ax.set_yticks(y, DATA_FORMATS)
ax.set_xlabel("mean output tokens")
ax.set_title("Output tokens: copy vs parse", loc="left")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
plt.show()
copy_vs_parse
""")

md("## 7. Parse version: did the code run on new data?")
code("""
parse_rows["code ran on new data"] = parse_rows["passed_all"] | ((parse_rows["tests_passed"] == 1) & parse_rows["copied"])
parse_rows["uses pandas"] = parse_rows["code_text"].str.contains("import pandas")
parse_rows["uses removed pandas call"] = parse_rows["code_text"].str.contains("compat.StringIO")
parse_rows.groupby("data_format")[["code ran on new data", "uses pandas", "uses removed pandas call"]].mean().round(2).loc[DATA_FORMATS]
""")

md("""
## Summary

- XML: about 2.3× the input tokens of CSV, about 50% more total tokens.
- Prose: more input than CSV, least output; CSV and prose about even in total.
- Output is driven by storage style: dicts with keys ≈ 1,120 output tokens, tuples ≈ 730.
- Parse version: output drops to 23–39% of the copy version; CSV needs the least parsing code (≈240 tokens).
- Parse version: XML code ran on new data 82% of the time, CSV and prose 52%; many CSV failures use a removed pandas call.
""")


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, NOTEBOOK_PATH)
print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)} with {len(cells)} cells (no outputs yet; execute it next)")

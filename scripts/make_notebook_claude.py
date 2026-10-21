"""Build notebooks/eda_claude.ipynb (Qwen vs Claude) from the cell text in this file.

Runs the same analyses as eda1.ipynb (sections 1-7), on both models side by side.
Reads data/results.csv (Qwen, read-only) and data/results_claude.csv (Claude).
Sections 4-7 read each model's raw replies (data/raw/, data/raw_claude/); a model whose
raw replies aren't on this machine shows its saved output from eda1.ipynb instead.

This script writes the notebook WITHOUT outputs; execute it afterwards:

    python scripts/make_notebook_claude.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/eda_claude.ipynb

Editing the notebook by hand works too, but the next run of this script overwrites it.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "eda_claude.ipynb"

cells = []


def md(text):
    """Add a markdown cell."""
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    """Add a code cell."""
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Data format vs. token usage: Qwen vs. Claude

**Question:** does the **format of the data** in a prompt (CSV, XML, or prose) change token usage the same way for different models?

- **Same experiment and analyses as eda1:** same tasks, prompts, temperature (0.7), system prompt, and tests; only the model differs.
- **Models:** `qwen2.5-coder:7b` (local, Ollama) and `claude-haiku-4-5` (Anthropic API, no extended thinking).
- **Data:** `data/results.csv` (Qwen) and `data/results_claude.csv` (Claude).
""")

code("""
import ast
import base64
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
from IPython.display import HTML, Image, Markdown

sys.path.insert(0, "../src")
from evaluate import extract_code

QWEN, CLAUDE = "qwen2.5-coder:7b", "claude-haiku-4-5"
MODELS = [QWEN, CLAUDE]
SHORT = {QWEN: "qwen", CLAUDE: "claude"}
RAW_DIRS = {QWEN: Path("../data/raw"), CLAUDE: Path("../data/raw_claude")}

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

responses = pd.concat([pd.read_csv("../data/results.csv"), pd.read_csv("../data/results_claude.csv")],
                      ignore_index=True)
responses = responses[responses["model"].isin(MODELS)].copy()
responses["total_tokens"] = responses["input_tokens"] + responses["output_tokens"]

is_parse = responses["task_id"].str.endswith("_parse")
copy_rows = responses[~is_parse].copy()    # sections 1-4: the code contains the data
parse_rows = responses[is_parse].copy()    # sections 5-7: the code receives the data as text
valid = copy_rows[copy_rows["passed_all"]].copy()   # code runs and contains all the data
for model in MODELS:
    m = responses["model"] == model
    print(f"{model}: {(~is_parse & m).sum()} copy rows ({(valid['model'] == model).sum()} valid), "
          f"{(is_parse & m).sum()} parse rows")
responses.head()
""")

code("""
def safe(model):
    return model.replace(":", "-").replace("/", "-")

def raw_path(r):
    return RAW_DIRS[r.model] / f"{r.task_id}__{r.format}__{r.data_format}__run{r.run}__{safe(r.model)}.txt"

HAS_RAW = {model: any(RAW_DIRS[model].glob(f"*__{safe(model)}.txt")) for model in MODELS}

EDA1 = json.loads(Path("eda1.ipynb").read_text(encoding="utf-8"))

def eda1_saved_output(heading):
    \"\"\"The saved outputs of eda1.ipynb's cells under the markdown heading starting with `heading`.\"\"\"
    outputs, inside = [], False
    for cell in EDA1["cells"]:
        if cell["cell_type"] == "markdown":
            inside = "".join(cell["source"]).startswith(heading)
            continue
        for out in cell.get("outputs", []) if inside else []:
            data = out.get("data", {})
            if "image/png" in data:
                outputs.append(Image(base64.b64decode(data["image/png"])))
            elif "text/html" in data:
                outputs.append(HTML("".join(data["text/html"])))
            elif "text" in out:
                outputs.append(Markdown(f"`{''.join(out['text']).strip()}`"))
    return outputs

def for_each_model(section, heading):
    \"\"\"Run `section(model)` per model, or show eda1's saved output if the raw replies are missing.\"\"\"
    for model in MODELS:
        display(Markdown(f"### {model}"))
        if HAS_RAW[model]:
            section(model)
        else:
            display(Markdown("*Raw replies aren't on this machine; saved output from eda1.ipynb:*"))
            for out in eda1_saved_output(heading):
                display(out)

print("raw replies available:", HAS_RAW)
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

for model in MODELS:
    input_by_format = (copy_rows[copy_rows["model"] == model]
                       .groupby(["task_id", "data_format"])["input_tokens"].mean().unstack())
    for d in DATA_FORMATS:
        data_table[f"{d} input tokens ({SHORT[model]})"] = input_by_format[d].round(0)
data_table
""")

md("## 1. Tokens by data format")
code("""
order = [(model, d) for model in MODELS for d in DATA_FORMATS]
(valid.groupby(["model", "data_format"])[["input_tokens", "output_tokens", "total_tokens"]]
 .agg(["mean", "median"]).round(0).loc[order])
""")

md("## 2. Output tokens by data format")
code("""
x = np.arange(len(DATA_FORMATS))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, model in zip(axes, MODELS):
    sub = valid[valid["model"] == model]
    per_task = sub.groupby(["task_id", "data_format"])["output_tokens"].mean().unstack()[DATA_FORMATS]
    for task_id, row in per_task.iterrows():
        ax.plot(x, row.values, color="#d8d7d1", linewidth=1.2, zorder=1)
    ax.plot(x, per_task.mean().values, color="#52514e", linewidth=2, zorder=2)
    for i, d in enumerate(DATA_FORMATS):
        ax.scatter(i, per_task[d].mean(), s=90, color=DATA_COLORS[d], marker=DATA_MARKERS[d], zorder=3,
                   edgecolor="white", linewidth=1.2)
    ax.set_xticks(x, DATA_FORMATS)
    ax.set_xlim(-0.3, len(DATA_FORMATS) - 0.7)
    ax.set_xlabel("data format")
    ax.set_title(model, loc="left")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("mean output tokens")
fig.suptitle("Output tokens (gray: tasks, markers: mean)", x=0.07, ha="left")
fig.tight_layout()
plt.show()
""")

md("## 3. Output vs input tokens")
code("""
valid["output_per_input"] = valid["output_tokens"] / valid["input_tokens"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
for ax, model in zip(axes, MODELS):
    for d in DATA_FORMATS:
        sub_rows = valid[(valid["model"] == model) & (valid["data_format"] == d)]
        ax.scatter(sub_rows["input_tokens"], sub_rows["output_tokens"], s=26, alpha=0.75, color=DATA_COLORS[d],
                   marker=DATA_MARKERS[d], edgecolor="white", linewidth=0.5, label=d)
    ax.set_xlabel("input tokens")
    ax.set_title(model, loc="left")
axes[0].set_ylabel("output tokens")
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle("Output vs input tokens, by data format", x=0.07, ha="left")
fig.tight_layout()
plt.show()

ratio = valid.groupby(["model", "task_id", "data_format"])["output_per_input"].mean()
ratio.groupby(["data_format", "model"]).mean().unstack("model").loc[DATA_FORMATS, MODELS].round(2)
""")

md("""
## 4. How the model stored the data

Reads the saved responses in `data/raw/` (Qwen, local only) and `data/raw_claude/` (Claude).
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

def section4(model):
    style_rows = []
    for path in RAW_DIRS[model].glob(f"*__{safe(model)}.txt"):
        task_id, fmt, data_format, run, _ = path.stem.split("__")
        if data_format not in DATA_FORMATS:
            continue
        code_text = extract_code(path.read_text(encoding="utf-8"))
        style_rows.append({"task_id": task_id, "format": fmt, "data_format": data_format,
                           "run": int(run.removeprefix("run")), "style": storage_style(code_text)})
    styles = pd.DataFrame(style_rows).merge(valid[valid["model"] == model],
                                            on=["task_id", "format", "data_format", "run"])
    display(pd.crosstab(styles["data_format"], styles["style"]).loc[DATA_FORMATS])
    display(styles.groupby("style")["output_tokens"].agg(["count", "mean"]).round(0))
    display(styles.groupby(["data_format", "style"])["output_tokens"].agg(["count", "mean"]).round(0).loc[DATA_FORMATS])

for_each_model(section4, "## 4.")
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
    return extract_code(raw_path(r).read_text(encoding="utf-8"))

parse_by_model, attempts_by_model = {}, {}
for model in MODELS:
    if not HAS_RAW[model]:
        continue
    rows = parse_rows[parse_rows["model"] == model].copy()
    rows["code_text"] = [saved_code(r) for r in rows.itertuples()]
    rows["copied"] = [sum(normalize(n) in normalize(code_text) for n in prompt_names(task_id)) > 2
                      for task_id, code_text in zip(rows["task_id"], rows["code_text"])]
    parse_by_model[model] = rows
    attempts_by_model[model] = rows[~rows["copied"]].copy()   # responses that didn't copy the data

def section5(model):
    rows, attempts = parse_by_model[model], attempts_by_model[model]
    print(f"{len(attempts)} of {len(rows)} parse responses didn't copy the data")
    display(attempts.groupby("data_format")[["input_tokens", "output_tokens", "total_tokens"]]
            .agg(["mean", "median"]).round(0).loc[DATA_FORMATS])

for_each_model(section5, "## 5.")
""")

md("## 6. Copy vs parse: output tokens")
code("""
def section6(model):
    attempts = attempts_by_model[model]
    copy_means = valid[valid["model"] == model].groupby(["task_id", "data_format"])["output_tokens"].mean()
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
    if model == CLAUDE:
        # Start where Qwen's chart (eda1's saved image) starts, so the two are easier to compare:
        # its data spans 240-1105 tokens, and matplotlib pads that by 5% on each side.
        ax.set_xlim(left=240 - 0.05 * (1105 - 240))
    ax.set_xlabel("mean output tokens")
    ax.set_title("Output tokens: copy vs parse", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.show()
    display(copy_vs_parse)

for_each_model(section6, "## 6.")
""")

md("## 7. Parse version: did the code run on new data?")
code("""
def section7(model):
    rows = parse_by_model[model]
    rows["code ran on new data"] = rows["passed_all"] | ((rows["tests_passed"] == 1) & rows["copied"])
    rows["uses pandas"] = rows["code_text"].str.contains("import pandas")
    rows["uses removed pandas call"] = rows["code_text"].str.contains("compat.StringIO")
    display(rows.groupby("data_format")[["code ran on new data", "uses pandas", "uses removed pandas call"]]
            .mean().round(2).loc[DATA_FORMATS])

for_each_model(section7, "## 7.")
""")

md("""
## 8. Claude replies that call an LLM

Not in eda1. On informal prompts Claude often wrote code that asks the Claude API for the answer instead of
computing it. The tests run without an API key, so this code always fails.
""")
code("""
claude_rows = responses[responses["model"] == CLAUDE].copy()
claude_rows["kind"] = np.where(claude_rows["task_id"].str.endswith("_parse"), "parse", "copy")
claude_rows["calls LLM"] = [bool(re.search(r"anthropic|openai", raw_path(r).read_text(encoding="utf-8"), re.I))
                            for r in claude_rows.itertuples()]
print(f"{(claude_rows['calls LLM'] & ~claude_rows['passed_all']).sum()} of "
      f"{(~claude_rows['passed_all']).sum()} failed Claude replies call an LLM")
claude_rows.groupby(["format", "kind"]).agg(
    replies=("calls LLM", "size"),
    call_llm=("calls LLM", "sum"),
    passed_all=("passed_all", "mean"),
).round(2)
""")

md("""
## Summary

- Input tokens: nearly the same for both models (same prompts); Claude's tokenizer counts 1–9% more, most on prose. XML ≈ 2.3× the input of CSV for both.
- Output (copy version): Claude writes more than Qwen in every format. Format effect differs: Qwen's prose is its *shortest* output (836 vs 1,043 CSV), Claude's prose is among its *longest* (1,341 vs 1,069 CSV); XML is longest for both.
- Total (copy version): CSV is cheapest for Claude; for Qwen, CSV and prose are about even.
- Storage style drives output for both: dicts with keys ≈ 1,310 tokens for Claude (Qwen ≈ 1,120), tuples ≈ 800 (Qwen ≈ 730). It explains the prose difference: Qwen stored prose data mostly as tuples (29 of 48), Claude mostly as dicts (39 of 48).
- Parse version: Claude's output drops to 36–46% of the copy version (Qwen 23–39%), but stays about 2× Qwen's (≈ 500–580 vs ≈ 240–330 tokens).
- Parse version: Claude's code ran on new data 58–67% of the time (Qwen: XML 82%, CSV and prose 52%). Claude never used pandas; nearly all its failures are informal-prompt replies that call the Claude API (section 8). These replies also paste the prompt's rows into their API prompt, which is why eda1's copy check flags 83 Claude parse replies as copied (Qwen: 4) and sections 5–6 leave them out.
- Qwen's sections 4–7 are eda1's saved outputs, since its raw replies aren't in this repo.
""")


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, NOTEBOOK_PATH)
print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)} with {len(cells)} cells (no outputs yet; execute it next)")

"""Build notebooks/test4.ipynb from the cell text in this file.

This script writes the notebook WITHOUT outputs; execute it afterwards:

    python scripts/make_notebook_test4.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/test4.ipynb

It reads data/format_results.csv, data/raw_format/ (the saved replies) and format_tasks/.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "test4.ipynb"

cells = []


def md(text):
    """Add a markdown cell."""
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    """Add a code cell."""
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Data format vs. output tokens: test 4 (Claude Haiku 4.5)

## 1. Introduction

**Question:** when the data in a prompt is written as CSV, JSON, XML, a Markdown table or prose, does the model spend a different number of **output tokens** writing code for it?

**Hypothesis:** formats that are harder to parse (prose, XML) make the model write longer code than formats with a standard parser (CSV, JSON).

**Design**

| | |
|---|---|
| Task | the parse task: write `q1(text)`, `q2(text)`, `q3(text)`, functions that receive the data as a string and parse it. This is how people usually ask for code; the earlier "copy the data into the code" version is dropped |
| Data formats | `csv`, `json`, `markdown`, `xml`, `prose`: the same 20 rows, written five ways |
| Tasks | 12, each with its own 20-row sample: `t41`–`t46` are the earlier parse tasks (`t21`–`t26`) again, `t47`–`t52` are new (new leagues, columns and questions) |
| Instructions | the structured prompt only (labeled sections, numbered requirements) |
| Model | `claude-haiku-4-5`, temperature 0.7, max 4,096 output tokens, no extended thinking, so output tokens are exactly the visible reply |
| Runs | 40 per task × format, so 12 × 5 × 40 = 2,400 calls, sent as Message Batches (half price; every request is independent) |
| Validity | the code runs on a hidden 20-row sample from another league, and doesn't hard-code the prompt's data. Answers aren't graded |
""")

code("""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from scipy import stats

sys.path.insert(0, "../scripts")
from make_format_tasks import ALL_TASKS as TASKS  # task definitions: leagues, columns
from make_format_tasks import NEW_TASKS, format_id, render_all

SEED = 42
N_PERM = 10_000
RESULTS_CSV = Path("../data/format_results.csv")
RAW_DIR = Path("../data/raw_format")
TASKS_DIR = Path("../format_tasks")
MODEL = "claude-haiku-4-5"
N_ROWS = 20
LIST_PRICES = (1.00, 5.00)   # USD per million tokens (input, output)
BATCH_PRICES = (0.50, 2.50)  # what this run paid

# Roughly compact -> wordy. Colors match eda1-eda3; the order keeps neighbors distinguishable.
DATA_FORMATS = ["csv", "json", "markdown", "xml", "prose"]
DATA_COLORS = {"csv": "#008300", "json": "#1d6fb8", "markdown": "#d4762c", "xml": "#4a3aa7", "prose": "#e87ba4"}
DATA_MARKERS = {"csv": "o", "json": "v", "markdown": "^", "xml": "s", "prose": "D"}

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
responses["task"] = responses["task_id"].str[4:]  # t41_side_winrate -> side_winrate
NEW_IDS = {format_id(t) for t in NEW_TASKS}
responses["task set"] = np.where(responses["task_id"].isin(NEW_IDS), "new (t47–t52)", "original (t41–t46)")
responses["data_format"] = pd.Categorical(responses["data_format"], DATA_FORMATS, ordered=True)
responses["custom_id"] = (responses["task_id"] + "__" + responses["data_format"].astype(str)
                          + "__run" + responses["run"].map("{:02d}".format))
responses["total_tokens"] = responses["input_tokens"] + responses["output_tokens"]
responses["reply"] = [(RAW_DIR / f"{cid}.txt").read_text(encoding="utf-8") if (RAW_DIR / f"{cid}.txt").exists() else ""
                      for cid in responses["custom_id"]]
TASK_NAMES = sorted(responses["task"].unique())

spent = (responses["input_tokens"].sum() * BATCH_PRICES[0] + responses["output_tokens"].sum() * BATCH_PRICES[1]) / 1e6
print(f"{len(responses)} responses, {responses['task'].nunique()} tasks, "
      f"{responses['run'].max()} runs per task × format")
print(f"tokens: {responses['input_tokens'].sum():,} input, {responses['output_tokens'].sum():,} output "
      f"-> about ${spent:.2f} at batch prices")
""")

md("""
## 2. Data source

The rows come from [Oracle's Elixir](https://oracleselixir.com)'s 2025 League of Legends esports match data. `scripts/make_format_tasks.py` builds both sets of tasks:

- **Original 6** (`t41`–`t46`): the same samples as the earlier parse version (`t21`–`t26`), read from each parse task's `sample.json` and `hidden.json`. Their CSV, XML and prose files are byte-identical to the earlier ones.
- **New 6** (`t47`–`t52`): sampled the same way (fixed seed, games with complete data only) from leagues, columns and questions the original tasks don't use. They test whether the format effect holds beyond the original 6 questions.

For every task:

- **Prompt sample:** 20 rows from one league, shown to the model.
- **Hidden sample:** 20 rows from a different league with no shared team or player names, used only by the validity check.
- Team tasks have one row per team per game (10 games × 2 teams); player tasks have one row per player (20 players, one position).
""")

md("""
## 3. The 12 tasks
""")
code("""
def questions(task_dir):
    text = (task_dir / "structured.txt").read_text(encoding="utf-8")
    return [re.sub(r"^\\d+\\. Define a function ", "", line) for line in text.splitlines() if "Define a function" in line]

task_table = pd.DataFrame([
    {"task": format_id(t)[4:], "set": "new" if t in NEW_TASKS else "original",
     "rows": "team per game" if t["kind"] == "team" else f"player ({t['position']})",
     "prompt league": t["league"], "hidden league": t["hidden_league"], "columns": ", ".join(t["columns"]),
     **{f"q{i}": q for i, q in enumerate(questions(TASKS_DIR / format_id(t)), start=1)}}
    for t in TASKS
]).set_index("task")
with pd.option_context("display.max_colwidth", None):
    display(task_table)
""")

md("""
## 4. The 5 data formats

The first 2 rows of the `side_winrate` task, as each format writes them. The prompt contains all 20 rows in exactly one format.
""")
code("""
example_task = TASKS[0]
example_rows = json.loads((TASKS_DIR / format_id(example_task) / "sample.json").read_text(encoding="utf-8"))[:2]
for fmt, text in render_all(example_rows, example_task).items():
    print(f"--- {fmt} ---\\n{text}\\n")
""")
code("""
# One full prompt, exactly as the model received it (after the system prompt
# "You are a Python programmer. Reply with only Python code.")
template = (TASKS_DIR / format_id(example_task) / "structured.txt").read_text(encoding="utf-8")
data = (TASKS_DIR / format_id(example_task) / "data_json.txt").read_text(encoding="utf-8").strip()
print(template.replace("{DATA}", data))
""")

md("""
## 5. Data quality

What came back, and whether anything needs cleaning. The input data is fixed and needs none; the checks are on the **responses**:

- every planned request should be here exactly once;
- `stop_reason` should be `end_turn`. `max_tokens` would mean a reply was cut off, which would shorten its output count;
- no prompt caching (it would change how input tokens are reported);
- validity per format.

**Rule:** invalid responses stay in the token analysis, because they still cost tokens. Section 9 repeats the main comparisons on valid responses only.
""")
code("""
expected = {f"{format_id(t)}__{d}__run{r:02d}" for t in TASKS for d in DATA_FORMATS for r in range(1, 41)}
got = set(responses["custom_id"])
print(f"expected {len(expected)}, received {len(got)}, missing {len(expected - got)}, "
      f"duplicates {responses['custom_id'].duplicated().sum()}")
print("stop reasons:", responses["stop_reason"].value_counts().to_dict())
print("responses using prompt caching:", int((responses["cache_tokens"] > 0).sum()))
print("saved replies found:", int((responses["reply"] != "").sum()))

def wilson(k, n, z=1.96):
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half

validity = responses.groupby("data_format", observed=True).agg(
    n=("passed_all", "size"), valid=("passed_all", "sum"), tokens=("total_tokens", "sum"))
validity["rate"] = validity["valid"] / validity["n"]
validity["low"], validity["high"] = wilson(validity["valid"], validity["n"])
validity["tests passed: 0 / 1 / 2"] = [
    " / ".join(str(((responses["data_format"] == d) & (responses["tests_passed"] == k)).sum()) for k in (0, 1, 2))
    for d in validity.index]
chi_p = stats.chi2_contingency(pd.crosstab(responses["data_format"], responses["passed_all"]))[1]
print(f"\\nvalidity depends on format? chi-square p = {chi_p:.4f}")
validity[["n", "valid", "rate", "low", "high", "tests passed: 0 / 1 / 2"]].round(3)
""")
code("""
fig, ax = plt.subplots(figsize=(6.5, 2.8))
for i, d in enumerate(DATA_FORMATS):
    r = validity.loc[d]
    ax.hlines(i, r["low"], r["high"], color="#d8d7d1", linewidth=3, zorder=1)
    ax.scatter(r["rate"], i, s=80, color=DATA_COLORS[d], marker=DATA_MARKERS[d], zorder=2, edgecolor="white", linewidth=1)
    ax.annotate(f"{r['rate']:.0%}", (r["rate"], i), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color="#52514e")
ax.set_yticks(range(len(DATA_FORMATS)), DATA_FORMATS)
ax.set_ylim(-0.6, len(DATA_FORMATS) - 0.3)
ax.set_xlim(0, 1.05)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("share of valid responses (95% Wilson CI)")
plt.tight_layout()
plt.show()

# by task: a format that fails on one task only shows up here
responses.pivot_table(index="task", columns="data_format", values="passed_all", aggfunc="mean", observed=True).round(2)
""")

md("""
## 6. Input tokens

The same prompt always splits into the same tokens, so every repeat of a prompt reports the same input count: there's no noise, and differences between formats are exact (the method of `eda2` section 1). Each prompt holds the same 20 rows, so the extra cost of a format can be stated per data row.
""")
code("""
cells_ = responses.groupby(["task", "data_format"], observed=True)["input_tokens"]
print("prompts whose runs all had the same input count:", (cells_.nunique() == 1).sum(), "of", cells_.ngroups)

inputs = responses.drop_duplicates(["task", "data_format"])
per_prompt = inputs.pivot_table(index="task", columns="data_format", values="input_tokens", observed=True)[DATA_FORMATS]
input_table = pd.DataFrame({
    "mean input tokens": per_prompt.mean(),
    "× csv": (per_prompt.div(per_prompt["csv"], axis=0)).mean(),
    "extra tokens per row vs csv": (per_prompt.sub(per_prompt["csv"], axis=0) / N_ROWS).mean(),
})
display(per_prompt)
input_table.round(2)
""")
code("""
fig, ax = plt.subplots(figsize=(6.5, 2.6))
y = np.arange(len(DATA_FORMATS))
ax.barh(y, input_table["mean input tokens"], color=[DATA_COLORS[d] for d in DATA_FORMATS], height=0.6)
for i, d in enumerate(DATA_FORMATS):
    ax.annotate(f"{input_table.loc[d, 'mean input tokens']:.0f}  ({input_table.loc[d, '× csv']:.2f}× csv)",
                (input_table.loc[d, "mean input tokens"], i), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color="#52514e")
ax.set_yticks(y, DATA_FORMATS)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("input tokens per prompt (mean over 12 tasks)")
ax.set_xlim(0, input_table["mean input tokens"].max() * 1.3)
plt.tight_layout()
plt.show()
""")

md("""
## 7. Output tokens

Unlike input, output varies from run to run: each reply is a new sample. This section describes the 2,400 output counts. Section 9 tests whether the differences are bigger than chance.
""")
code("""
rng = np.random.default_rng(SEED)

def boot_ci(values, n=5000):
    values = np.asarray(values)
    means = rng.choice(values, (n, len(values))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])

summary = responses.groupby("data_format", observed=True)["output_tokens"].agg(["mean", "median", "std", "min", "max"])
summary[["95% CI low", "95% CI high"]] = [boot_ci(responses.loc[responses["data_format"] == d, "output_tokens"])
                                         for d in DATA_FORMATS]
summary["× csv (mean)"] = summary["mean"] / summary.loc["csv", "mean"]
summary.round(1)
""")
code("""
fig, ax = plt.subplots(figsize=(7.5, 3.6))
jitter = np.random.default_rng(SEED)
for i, d in enumerate(DATA_FORMATS):
    vals = responses.loc[responses["data_format"] == d, "output_tokens"]
    ax.scatter(i + jitter.uniform(-0.18, 0.18, len(vals)), vals, s=10, alpha=0.35, color=DATA_COLORS[d],
               marker=DATA_MARKERS[d], linewidth=0)
    lo, hi = summary.loc[d, ["95% CI low", "95% CI high"]]
    ax.hlines(summary.loc[d, "mean"], i - 0.3, i + 0.3, color="#52514e", linewidth=2)
    ax.annotate(f"{summary.loc[d, 'mean']:.0f}", (i + 0.3, summary.loc[d, "mean"]), xytext=(3, 0),
                textcoords="offset points", va="center", fontsize=8, color="#52514e")
ax.set_xticks(range(len(DATA_FORMATS)), DATA_FORMATS)
ax.set_ylabel("output tokens per reply")
ax.grid(axis="x", visible=False)
ax.set_title(f"Each dot is one reply ({len(responses) // len(DATA_FORMATS)} per format); bar = mean", loc="left", fontsize=9, color="#52514e")
plt.tight_layout()
plt.show()
""")

md("""
**By task.** Mean output tokens per task × format, then each task relative to its own CSV mean: a gray dot per task, a large marker for the average across tasks. The formats are categories, so the dots aren't connected. If a format's dots all sit on the same side of the CSV line, the effect isn't driven by one task.
""")
code("""
task_means = responses.pivot_table(index="task", columns="data_format", values="output_tokens",
                                   aggfunc="mean", observed=True)[DATA_FORMATS]
relative = task_means.div(task_means["csv"], axis=0)
display(task_means.round(0))

fig, ax = plt.subplots(figsize=(6.5, 3.6))
others = DATA_FORMATS[1:]  # CSV is the reference line at 1
jitter = np.random.default_rng(SEED)
for i, d in enumerate(others):
    ax.scatter(i + jitter.uniform(-0.12, 0.12, len(relative)), relative[d], s=28, color="#b5b4ae",
               edgecolor="white", linewidth=0.6, zorder=2)
    ax.scatter(i, relative[d].mean(), s=110, color=DATA_COLORS[d], marker=DATA_MARKERS[d], zorder=3,
               edgecolor="white", linewidth=1.2)
    ax.annotate(f"{relative[d].mean():.2f}×", (i, relative[d].mean()), xytext=(14, 0), textcoords="offset points",
                va="center", fontsize=8, color="#52514e")
ax.axhline(1, color="#8a8984", linewidth=1, linestyle=":")
ax.annotate("csv = 1", (len(others) - 0.5, 1), xytext=(0, 3), textcoords="offset points", ha="right",
            fontsize=8, color="#8a8984")
ax.set_xticks(range(len(others)), others)
ax.set_xlim(-0.5, len(others) - 0.5)
ax.set_ylabel("output tokens relative to CSV")
ax.set_title("Gray dot = one task (values in the table below); large marker = average", loc="left",
             fontsize=9, color="#52514e")
ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.show()
relative.round(2)
""")

md("""
**Run-to-run variability.** For each prompt (task × format), the coefficient of variation (standard deviation / mean) over its 40 runs. Higher means the same prompt gives less predictable token counts.
""")
code("""
cv = responses.groupby(["task", "data_format"], observed=True)["output_tokens"].agg(lambda s: s.std() / s.mean()).unstack()
display(cv[DATA_FORMATS].round(3))
print("median CV:", cv[DATA_FORMATS].median().round(3).to_dict())
print("Friedman test (same tasks, 5 formats): p =", round(stats.friedmanchisquare(*[cv[d] for d in DATA_FORMATS]).pvalue, 4))
""")

md("""
**How many runs are enough?** The mean output of each format using only runs 1..N, averaged over tasks. Earlier experiments used 5 runs; where the lines flatten, more runs stop changing the answer.
""")
code("""
fig, ax = plt.subplots(figsize=(7, 3.4))
ns = np.arange(1, responses["run"].max() + 1)
for d in DATA_FORMATS:
    sub = responses[responses["data_format"] == d]
    curve = [sub[sub["run"] <= n].groupby("task")["output_tokens"].mean().mean() for n in ns]
    ax.plot(ns, curve, color=DATA_COLORS[d], linewidth=2)
    ax.annotate(d, (ns[-1], curve[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color="#52514e")
ax.axvline(5, color="#8a8984", linewidth=1, linestyle=":")
ax.annotate("5 runs (earlier tests)", (5, ax.get_ylim()[0]), xytext=(4, 4), textcoords="offset points",
            fontsize=8, color="#8a8984")
ax.set_xlabel("runs used (1..N)")
ax.set_ylabel("mean output tokens")
ax.set_xlim(1, ns[-1] + 6)
plt.tight_layout()
plt.show()
""")

md("""
## 8. What the output tokens go to

Output tokens are spent on code, comments, docstrings and any text outside the code block. These measures come from the saved replies (`data/raw_format/`) and are counted in lines or characters, since only the total token count is reported per reply.

**Parse approach:** which tool the code uses to read the data, detected from the code (a reply can use several): the `csv` module, `json`, an XML parser, regular expressions, or plain string splitting (`.split(...)`).
""")
code("""
TRIPLE = '"' * 3

def anatomy(reply):
    fence = re.findall(r"```[ \\t]*(?:python|py)?[ \\t]*\\n(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    code_ = max(fence, key=len) if fence else reply
    lines = code_.splitlines()
    doc_lines, in_doc = 0, False
    for line in lines:
        n = line.count(TRIPLE)
        if in_doc or n:
            doc_lines += 1
        if n % 2 == 1:
            in_doc = not in_doc
    return {
        "code lines": sum(1 for l in lines if l.strip()),
        "comment lines": sum(1 for l in lines if l.strip().startswith("#")),
        "docstring lines": doc_lines,
        "functions defined": len(re.findall(r"^\\s*def ", code_, re.MULTILINE)),
        "text outside code (chars)": len(reply.strip()) - len(code_.strip()) if fence else 0,
        "uses csv module": bool(re.search(r"\\bcsv\\.|import csv", code_)),
        "uses json": bool(re.search(r"\\bjson\\.loads|import json", code_)),
        "uses XML parser": bool(re.search(r"ElementTree|xml\\.dom|minidom|lxml", code_)),
        "uses regex": bool(re.search(r"\\bre\\.(?:search|findall|match|compile|finditer|sub|split)", code_)),
        "uses .split(": ".split(" in code_,
    }

anat = pd.concat([responses[["task", "data_format", "output_tokens", "passed_all"]].reset_index(drop=True),
                  pd.DataFrame([anatomy(r) for r in responses["reply"]])], axis=1)
anat_table = anat.groupby("data_format", observed=True).mean(numeric_only=True).drop(columns=["passed_all"])
anat_table.T.round(2)
""")
code("""
# How much of the output-token gap do code lines explain? Tokens per non-blank code line, by format.
per_line = anat.groupby("data_format", observed=True).apply(
    lambda g: g["output_tokens"].sum() / g["code lines"].sum(), include_groups=False).rename("output tokens per code line")
print(per_line.round(1).to_string())
print("\\ncorrelation of output tokens with code lines:", round(anat["output_tokens"].corr(anat["code lines"]), 3))
""")
code("""
fig, ax = plt.subplots(figsize=(7, 3))
approaches = ["uses csv module", "uses json", "uses XML parser", "uses regex", "uses .split("]
width = 0.16
for i, d in enumerate(DATA_FORMATS):
    ax.bar(np.arange(len(approaches)) + (i - 2) * width, anat_table.loc[d, approaches], width * 0.9,
           color=DATA_COLORS[d], label=d)
ax.set_xticks(np.arange(len(approaches)), [a.removeprefix("uses ") for a in approaches])
ax.set_ylabel("share of replies")
ax.set_ylim(0, 1.05)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
plt.tight_layout()
plt.show()
""")

md("""
**One example reply per format** (`side_winrate` task): the reply whose output count is closest to that format's median for the task, so it's typical rather than extreme.
""")
code("""
example = responses[responses["task"] == "side_winrate"]
for d in DATA_FORMATS:
    sub = example[example["data_format"] == d]
    if sub.empty:
        continue
    row = sub.iloc[(sub["output_tokens"] - sub["output_tokens"].median()).abs().argmin()]
    print(f"=== {d}: run {row['run']}, {row['output_tokens']} output tokens, valid={row['passed_all']} ===")
    print(row["reply"].strip(), "\\n")
""")

md("""
## 9. Statistical follow-up

**Permutation test (eda2's method).** For a pair of formats, take the gap in mean log output tokens within each task (both formats share the same questions and data), and average it over the 12 tasks. Shuffling the format labels within each task 10,000 times shows how large a gap chance alone produces. The gap is reported as a ratio: 1.20 means 20% more output tokens than the second format.

**Holm correction.** 10 pairs are tested at once, so some would look significant by luck. Holm's method adjusts the p-values for that; `p (Holm)` is the one to read.

**Task CI.** A 95% bootstrap interval for the ratio, from resampling the 12 tasks. It shows whether the effect holds across tasks or comes from a few.

**Overall test.** A Friedman test on the 12 task means asks whether the five formats differ at all, treating each task as a block.

`p = 0.0001` is the smallest possible value with 10,000 shuffles (none of them produced a gap that large).
""")
code("""
def blocks_for(rows, a, b, column="output_tokens"):
    out = []
    for task, g in rows[rows["data_format"].isin([a, b])].groupby("task"):
        is_a = (g["data_format"] == a).to_numpy()
        if is_a.any() and (~is_a).any():
            out.append((np.log(g[column].to_numpy()), is_a, task))
    return out

def mean_gap(blocks, labels=None):
    labels = labels or [b[1] for b in blocks]
    return np.mean([y[m].mean() - y[~m].mean() for (y, _, _), m in zip(blocks, labels)])

def compare(rows, a, b):
    blocks = blocks_for(rows, a, b)
    observed = mean_gap(blocks)
    perm = np.array([mean_gap(blocks, [rng.permutation(m) for _, m, _ in blocks]) for _ in range(N_PERM)])
    p = (np.sum(np.abs(perm) >= abs(observed)) + 1) / (N_PERM + 1)
    tasks = [t for _, _, t in blocks]
    boot = [mean_gap([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) for _ in range(2000)]
    lo, hi = np.exp(np.percentile(boot, [2.5, 97.5]))
    higher = sum(y[m].mean() > y[~m].mean() for y, m, _ in blocks)
    return {"ratio": np.exp(observed), "95% CI low": lo, "95% CI high": hi, "p": p,
            "tasks where higher": f"{higher} of {len(tasks)}"}

def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out

PAIRS = [(a, b) for i, b in enumerate(DATA_FORMATS) for a in DATA_FORMATS[i + 1:]]

def effects_table(rows):
    table = pd.DataFrame([{"comparison": f"{a} / {b}", **compare(rows, a, b)} for a, b in PAIRS]).set_index("comparison")
    table["p (Holm)"] = holm(table["p"])
    table["significant"] = np.where(table["p (Holm)"] < 0.05, "yes", "no")
    return table

friedman = stats.friedmanchisquare(*[task_means[d] for d in DATA_FORMATS])
print(f"Friedman test on task means, all responses: p = {friedman.pvalue:.4f}")
effects = effects_table(responses)
effects.round(4)
""")

md("""
**Robustness: valid responses only.** The same tests using only replies that passed both validity checks. If the ratios barely move, the format effect isn't caused by broken replies.
""")
code("""
valid_only = responses[responses["passed_all"]]
effects_valid = effects_table(valid_only)
pd.DataFrame({
    "ratio (all)": effects["ratio"],
    "ratio (valid only)": effects_valid["ratio"],
    "95% CI low": effects_valid["95% CI low"],
    "95% CI high": effects_valid["95% CI high"],
    "p (Holm, valid only)": effects_valid["p (Holm)"],
    "significant (valid only)": effects_valid["significant"],
}).round(4)
""")

md("""
**Do the new tasks agree with the original ones?** The ratios vs. CSV computed separately on the original 6 tasks and on the 6 new ones. If both halves point the same way, the effect isn't tied to the original questions.
""")
code("""
halves = {}
for name, rows in responses.groupby("task set"):
    halves[name] = pd.Series({f"{a} / csv": compare(rows, a, "csv")["ratio"] for a in DATA_FORMATS[1:]})
halves = pd.DataFrame(halves)
halves["same direction"] = np.where(np.sign(np.log(halves.iloc[:, 0])) == np.sign(np.log(halves.iloc[:, 1])), "yes", "no")
display(halves.round(3))
responses.pivot_table(index="task set", columns="data_format", values="passed_all", aggfunc="mean", observed=True).round(2)
""")

md("""
## 10. Total cost per format

What a user pays per call: input + output tokens, in dollars at list price and at the batch price this run used. **Cost per valid reply** divides everything spent by the number of valid replies, the cost of retrying until the code works.
""")
code("""
cost = responses.groupby("data_format", observed=True).agg(
    input=("input_tokens", "mean"), output=("output_tokens", "mean"), total=("total_tokens", "mean"),
    valid_rate=("passed_all", "mean"))
cost["output share"] = cost["output"] / cost["total"]
cost["$ per 1,000 calls (list)"] = (cost["input"] * LIST_PRICES[0] + cost["output"] * LIST_PRICES[1]) / 1000
cost["$ per 1,000 calls (batch)"] = cost["$ per 1,000 calls (list)"] / 2
cost["$ per 1,000 valid replies (list)"] = cost["$ per 1,000 calls (list)"] / cost["valid_rate"]
cost["× csv (list)"] = cost["$ per 1,000 calls (list)"] / cost.loc["csv", "$ per 1,000 calls (list)"]
cost.round(3)
""")
code("""
fig, ax = plt.subplots(figsize=(6.5, 2.8))
y = np.arange(len(DATA_FORMATS))
inp = cost["input"] * LIST_PRICES[0] / 1000
out = cost["output"] * LIST_PRICES[1] / 1000
ax.barh(y, inp, color="#b5b4ae", height=0.6, label="input")
ax.barh(y, out, left=inp + inp.max() * 0.01, color=[DATA_COLORS[d] for d in DATA_FORMATS], height=0.6, label="output")
for i, d in enumerate(DATA_FORMATS):
    ax.annotate(f"${cost.loc[d, '$ per 1,000 calls (list)']:.2f}", (inp[d] + out[d], i), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8, color="#52514e")
ax.set_yticks(y, DATA_FORMATS)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("$ per 1,000 calls at list price (gray = input, color = output)")
ax.set_xlim(0, (inp + out).max() * 1.2)
plt.tight_layout()
plt.show()
""")

md("""
## 11. Findings

The numbers below are computed from the data, so they stay correct if the notebook is re-run.
""")
code("""
lines = []
lines.append(f"- **Input** (exact): " + ", ".join(
    f"{d} {input_table.loc[d, 'mean input tokens']:.0f} ({input_table.loc[d, '× csv']:.2f}× csv)" for d in DATA_FORMATS) + ".")
lines.append(f"- **Output** (mean per reply): " + ", ".join(
    f"{d} {summary.loc[d, 'mean']:.0f}" for d in DATA_FORMATS) + f". Friedman test across formats: p = {friedman.pvalue:.4f}.")
sig = effects[effects["significant"] == "yes"]
if len(sig):
    lines.append("- **Significant pairs** (Holm p < 0.05): " + "; ".join(
        f"{c} = {r['ratio']:.2f}× (CI {r['95% CI low']:.2f}–{r['95% CI high']:.2f}, higher in {r['tasks where higher']} tasks)"
        for c, r in sig.iterrows()) + ".")
else:
    lines.append("- **No pair of formats differs significantly** after the Holm correction.")
held = (effects["significant"] == effects_valid["significant"]).sum()
lines.append(f"- **Valid replies only:** {held} of {len(effects)} pairs keep the same verdict.")
lines.append(f"- **Original vs. new tasks:** {(halves['same direction'] == 'yes').sum()} of {len(halves)} "
             "ratios vs. CSV point the same way in both halves.")
lines.append("- **Validity:** " + ", ".join(f"{d} {validity.loc[d, 'rate']:.0%}" for d in DATA_FORMATS)
             + f" (chi-square p = {chi_p:.4f}).")
cheapest, dearest = cost["$ per 1,000 calls (list)"].idxmin(), cost["$ per 1,000 calls (list)"].idxmax()
lines.append(f"- **Cost:** cheapest {cheapest} (${cost.loc[cheapest, '$ per 1,000 calls (list)']:.2f} per 1,000 calls), "
             f"most expensive {dearest} (${cost.loc[dearest, '$ per 1,000 calls (list)']:.2f}); "
             f"output is {cost['output share'].min():.0%}–{cost['output share'].max():.0%} of tokens.")
display(Markdown("\\n".join(lines)))
""")

md("""
**Compared with the earlier Haiku 4.5 parse run.** CSV, XML and prose were also run in the earlier parse version (`t21`–`t26`, 5 runs, both instruction styles). Only its structured-prompt rows are compared here, with this run's original 6 tasks (the same prompts).
""")
code("""
old_path = Path("../data/results_claude.csv")
if old_path.exists():
    old = pd.read_csv(old_path)
    old = old[old["task_id"].str.startswith("t2") & (old["format"] == "structured") & (old["model"] == MODEL)]
    old_means = old.groupby("data_format")["output_tokens"].agg(["mean", "size"])
    same_prompts = responses[~responses["task_id"].isin(NEW_IDS)]
    new_means = same_prompts.groupby("data_format", observed=True)["output_tokens"].agg(["mean", "size"])
    display(pd.DataFrame({"earlier run (5 runs)": old_means["mean"], "n earlier": old_means["size"],
                          "this run (40 runs)": new_means["mean"], "n this run": new_means["size"]})
            .reindex(DATA_FORMATS).round(0))
else:
    print("data/results_claude.csv not found; skipped")
""")


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, NOTEBOOK_PATH)
print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)} with {len(cells)} cells (no outputs yet; execute it next)")

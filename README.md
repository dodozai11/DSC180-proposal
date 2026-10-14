# DSC180-proposal: Prompt format vs. token usage

An exploratory study of how properties of a prompt relate to the tokens an LLM uses when it writes Python code. There are two experiments, and they share one pipeline and one results table:

- **Experiment 1:** does **instruction format** (structured vs. informal) change token usage on 10 small coding tasks? Correctness is checked with hidden unit tests.
- **Experiment 2:** does **data format** (CSV vs. XML vs. prose) change token usage when the prompt contains a dataset the model must put into its code? It's crossed with instruction format, on 6 tasks using real League of Legends esports data. This experiment measures **token usage only**: answers aren't graded, and a validity check flags responses that skipped the data.

## Design: Experiment 1

| | |
|---|---|
| **Independent variable** | prompt format: `structured` (labeled sections: Goal, Input, Requirements as bullets, Output) or `informal` (one casual paragraph with the requirements scattered through it) |
| **Dependent variables** | `input_tokens` and `output_tokens`, the exact counts the model reports with every response |
| **Secondary check** | how many hidden unit tests the returned code passes (one test per requirement), so token differences can be compared against correctness |
| **Controls** | same model, temperature, max output tokens, and system prompt on every call; no extended thinking |
| **Model** | `qwen2.5-coder:7b`, an open-source coding model run locally with [Ollama](https://ollama.com). It's free, and anyone can rerun the study with exactly the same model weights |
| **Tasks** | 10 small Python tasks (4–8 requirements each, 58 requirements total). Each has both prompt formats with exactly the same requirements, and both ask for only Python code |
| **Repetitions** | every task × format is run N times (default 5) in a randomized but reproducible order. That's 100 calls by default |

## Design: Experiment 2

| | |
|---|---|
| **Independent variables** | `data_format`: the same 20 rows written as `csv`, `xml`, or `prose` (one templated sentence per row). **Crossed with** the instruction `format` (`structured` / `informal`) |
| **Dependent variables** | `input_tokens` and `output_tokens` |
| **Task** | "here's my data, write me a script": the model must write Python that **contains the data** and defines `q1()`, `q2()`, `q3()`, each answering a question about it |
| **Validity check** (not grading) | 2 tests: `q1`–`q3` run without crashing, and every row's identifier (game id, team or player name) appears in the code. `tests_passed`/`tests_total` hold these 2 checks for Experiment 2 rows, so `passed_all` means "valid response" |
| **Tasks** | 6 tasks (`t11`–`t16`), each with its own 20-row sample from a different league. 3 use team rows (side win rate, gold lead at 15, objectives); 3 use player rows (mid-lane KDA, jungle picks, support vision) |
| **Repetitions** | 6 tasks × 3 data formats × 2 instruction formats × 5 runs = 180 calls |
| **Data source** | [Oracle's Elixir](https://oracleselixir.com) 2025 LoL esports match data. Only the 6 × 20-row samples are committed, not the full file |

## Repo layout

```
config.py                 every experiment setting (provider, model, N, temperature, seed, ...)
tasks/tNN_<name>/
    structured.txt        prompt, structured format (Experiment 2: a template with a {DATA} slot)
    informal.txt          prompt, informal format (Experiment 2: a template with a {DATA} slot)
    test_solution.py      hidden tests (Experiment 2: the 2 validity checks)
    reference.py          a known-correct solution, used only to check the tests
    checklist.md          where each requirement appears in both prompts
    sample.json           Experiment 2 only: the 20 canonical rows
    data_csv.txt          Experiment 2 only: those rows as CSV
    data_xml.txt          Experiment 2 only: ... as XML
    data_prose.txt        Experiment 2 only: ... as prose
src/
    run_experiment.py     calls the model and appends one row per call to data/results.csv
    evaluate.py           extracts code from a response and runs the hidden tests
    check_tasks.py        sanity-checks every task
data/
    results.csv           the dataset (created by run_experiment.py)
    raw/                  the raw text of every response (git-ignored)
notebooks/eda.ipynb       analysis: tables, plots, paired statistics
scripts/make_notebook.py  builds notebooks/eda.ipynb (the notebook's text and code live here)
scripts/make_data_tasks.py builds the Experiment 2 tasks from the Oracle's Elixir file
```

## Setup

1. **Python 3.10 or newer**, then install the packages:
   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Ollama:** install it from https://ollama.com and start the app. Then download the model, about 4.7 GB:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```
   A GPU makes it much faster, but it also runs on a CPU.

No API key or account is needed.

## Running the experiment

Run all commands from the repo root. Everything is free.

```bash
# 1. Check the tasks: each reference solution must pass all its tests, and an empty solution none.
python src/check_tasks.py

# 2. Preview what will run.
python src/run_experiment.py --dry-run

# 3. Run. Prints one line per call: input tokens, output tokens, tests passed.
python src/run_experiment.py

# 4. Analyze.
jupyter notebook notebooks/eda.ipynb
```

The notebook is generated by `scripts/make_notebook.py`. To change its text or code, edit that script, then rebuild and execute the notebook:

```bash
python scripts/make_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

You can also edit `eda.ipynb` directly, but the next run of the script overwrites those edits.

### Resuming

Each row is written to `data/results.csv` as soon as its call finishes. If the run stops (crash, Ctrl+C, Ollama closed), run the same command again. Rows already in the CSV, matched on `(task_id, format, data_format, run, model)`, are skipped, so nothing is duplicated. A call that fails is skipped without writing a row, so the next run retries it. The same mechanism lets you raise `N_RUNS` later and only run the new repetitions.

### Output columns (`data/results.csv`)

| column | meaning |
|---|---|
| `task_id` | task folder name, e.g. `t04_password` |
| `format` | instruction format: `structured` or `informal` |
| `data_format` | `none` (Experiment 1), or `csv` / `xml` / `prose` (Experiment 2) |
| `run` | repetition number, 1..N |
| `model` | model used |
| `input_tokens` | tokens in the system prompt + user prompt (Ollama's `prompt_eval_count`) |
| `output_tokens` | tokens the model generated (Ollama's `eval_count`) |
| `tests_passed`, `tests_total` | Experiment 1: hidden tests passed / total. Experiment 2: validity checks passed / 2 |
| `passed_all` | Experiment 1: every test passed. Experiment 2: the response is valid |
| `stop_reason` | `stop` normally; `length` would mean the answer was cut off |
| `timestamp` | UTC time of the call |

## How grading works

`src/evaluate.py` takes the code from the response (the contents of the markdown code fence if there is one, otherwise the whole reply). It saves it as `solution.py` in a fresh temporary folder next to that task's `test_solution.py`, and runs pytest in a separate process with a timeout (`TEST_TIMEOUT` in `config.py`). Code that fails to import, crashes, or times out counts as 0 tests passed.

**Safety note:** this runs code written by the model on your computer. The tasks are small pure functions, and the subprocess has a timeout, but don't point this at untrusted prompts.

## Reproducibility notes

- **Model:** `qwen2.5-coder:7b` through Ollama. This model has no hidden "thinking" step, so `output_tokens` is exactly the visible answer.
- **Temperature:** 0.7, Qwen's recommended default. It's above 0, so the 5 runs of a prompt differ and show natural variability.
- **Seeds:** call order is shuffled with `SEED = 42`, and each call gets the seed `SEED + run`. Rerunning gets close to the same answers, but not always identical ones: the GPU and Ollama's internal cache can change results slightly. In testing, the first call after the model loaded sometimes differed from later calls with the same seed.
- **Prompt caching:** Ollama reuses work for repeated prompt prefixes internally, but we checked that its reported `input_tokens` always counts the full prompt, whatever the call order.
- **Versions:** the Python packages are pinned in `requirements.txt`. Record your Ollama version (`ollama --version`) in your report. This study was built with Ollama 0.40.3.

## Optional: run the same experiment on Claude

The code also supports Claude through Anthropic's paid API, for checking whether results carry over to a commercial model. The default 100 calls on Claude Haiku 4.5 cost roughly $0.25.

1. Copy `.env.example` to `.env` and add your API key. `.env` is git-ignored, and the code never contains the key.
2. In `config.py`, set `PROVIDER = "anthropic"` and `MODEL = "claude-haiku-4-5"`. Leave the other settings as they are, so both models run under the same conditions.
3. Run as above. Claude rows go into the same CSV under their own model name, and set `MODEL` at the top of the notebook to analyze them.

Keep `claude-haiku-4-5`. Newer Claude models always use extended thinking and reject custom temperatures, so they'd break the controls.

## Regenerating the Experiment 2 tasks

The data tasks are built from the full Oracle's Elixir file, which isn't in the repo. Download the 2025 match data from https://oracleselixir.com, place it next to the repo folder (or pass `--source`), then run:

```bash
python scripts/make_data_tasks.py
python src/check_tasks.py
```

Sampling uses a fixed seed, so the same source file always produces the same 20-row samples. `check_tasks.py` confirms that every row appears in all three data formats.

## Adding or changing a task

Create `tasks/t11_<name>/` with the same five files. The runner picks up every `tasks/t*/` folder automatically. Run `python src/check_tasks.py` to confirm the new task is consistent. If you change any prompt, update its `checklist.md` so the two formats still list the same requirements, and delete that task's old rows from `data/results.csv`.

## Cleaning up afterwards

- Remove the model, about 4.7 GB: `ollama rm qwen2.5-coder:7b`
- Uninstall Ollama: Windows Settings → Apps → Ollama. Then delete `%USERPROFILE%\.ollama` if it's still there.
- Delete the `.venv` folder, about 1 GB. Recreate it later with `pip install -r requirements.txt`.

Your results in `data/results.csv` and the notebook aren't affected.

## Limitations

- One small open-source model at one temperature. Larger commercial models may respond to prompt format differently (see the optional Claude run above).
- One prompt pair per task, so the particular wording of each informal paragraph is mixed in with the effect of "informality" itself.
- 10 tasks give limited statistical power. The notebook reports bootstrap confidence intervals alongside the Wilcoxon test.
- The two formats' prompts aren't exactly the same length. The notebook measures this gap as a possible confound.

# DSC180-proposal: Prompt format vs. token usage

**Research question:** Does prompt format (structured vs. informal) affect token usage when an LLM writes Python code for the same requirements?

## Design

| | |
|---|---|
| **Independent variable** | prompt format: `structured` (labeled sections: Goal, Input, Requirements as bullets, Output) or `informal` (one casual paragraph with the requirements scattered through it) |
| **Dependent variables** | `input_tokens` and `output_tokens`, taken from the API response's `usage` field |
| **Secondary check** | how many hidden unit tests the returned code passes (one test per requirement), so token differences can be compared against correctness |
| **Controls** | same model, temperature, `max_tokens`, and system prompt on every call; no prompt caching; no extended thinking |
| **Tasks** | 10 small Python tasks (4–8 requirements each, 58 requirements total). Each has both prompt formats with exactly the same requirements, and both ask for only Python code |
| **Repetitions** | every task × format is run N times (default 5) in a randomized but reproducible order. That's 100 calls by default |

## Repo layout

```
config.py                 every experiment setting (model, N, temperature, max_tokens, seed, ...)
tasks/tNN_<name>/
    structured.txt        prompt, structured format
    informal.txt          prompt, informal format
    test_solution.py      hidden tests, one per requirement (never shown to the model)
    reference.py          a known-correct solution, used only to check the tests
    checklist.md          where each requirement appears in both prompts, and its test
src/
    run_experiment.py     calls the API and appends one row per call to data/results.csv
    evaluate.py           extracts code from a response and runs the hidden tests
    check_tasks.py        sanity-checks every task (run this before spending money)
data/
    results.csv           the dataset (created by run_experiment.py)
    raw/                  the raw text of every response (git-ignored)
notebooks/eda.ipynb       analysis: tables, plots, paired statistics
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Then copy `.env.example` to `.env` and paste in your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is git-ignored, and the code never contains the key. The SDK reads it from the environment.

## Running the experiment

Run all commands from the repo root.

```bash
# 1. Check the tasks: each reference solution must pass all its tests, and an empty solution none.
python src/check_tasks.py

# 2. Preview what will run. Makes no API calls and needs no key.
python src/run_experiment.py --dry-run

# 3. (Recommended) Smoke test: set N_RUNS = 1 in config.py, run, then look at data/results.csv.
python src/run_experiment.py

# 4. Full run: set N_RUNS back to 5 and run again. Already-finished rows are skipped.
python src/run_experiment.py

# 5. Analyze.
jupyter notebook notebooks/eda.ipynb
```

Before each run, the runner prints how many calls are planned, already done, and still to make, plus a rough cost estimate. The default 100 calls on Claude Haiku 4.5 cost roughly $0.25.

### Resuming

Each row is written to `data/results.csv` as soon as its call finishes. If the run stops (crash, Ctrl+C, network problem), run the same command again. Rows already in the CSV, matched on `(task_id, format, run, model)`, are skipped, so nothing is duplicated. A call that fails with an API error is skipped without writing a row, so the next run retries it.

### Output columns (`data/results.csv`)

| column | meaning |
|---|---|
| `task_id` | task folder name, e.g. `t04_password` |
| `format` | `structured` or `informal` |
| `run` | repetition number, 1..N |
| `model` | model ID used |
| `input_tokens`, `output_tokens` | from the API's `usage` field |
| `tests_passed`, `tests_total` | hidden tests passed / total for that task |
| `passed_all` | `True` if every test passed |
| `stop_reason` | `end_turn` normally; `max_tokens` would mean the answer was cut off |
| `timestamp` | UTC time of the call |

## How grading works

`src/evaluate.py` takes the code from the response (the contents of the markdown code fence if there is one, otherwise the whole reply). It saves it as `solution.py` in a fresh temporary folder next to that task's `test_solution.py`, and runs pytest in a separate process with a timeout (`TEST_TIMEOUT` in `config.py`). Code that fails to import, crashes, or times out counts as 0 tests passed.

**Safety note:** this runs code written by the model on your computer. The tasks are small pure functions, and the subprocess has a timeout, but don't point this at untrusted prompts.

## Reproducibility notes

- **Model:** `claude-haiku-4-5`. When the `thinking` parameter is left out, it runs with no extended thinking, and it accepts `temperature`, so both controls hold. Newer models (Claude Sonnet 5.5, Opus 5.5) always use thinking and reject custom temperatures, so switching to them would break the controls.
- **Temperature:** 1.0, the API default. At 0 the repeated runs would be nearly identical. Because the Anthropic Python SDK 1.x no longer has a `temperature` argument, it's sent in the request body (`extra_body`). The API still accepts it for Haiku 4.5.
- **Order:** calls are shuffled with `SEED = 42`, so the same config always produces the same order.
- **Caching:** no `cache_control` is sent. The runner prints a warning if the API ever reports cached tokens.
- **Versions:** pinned in `requirements.txt`.
- Model outputs are not deterministic even at a fixed temperature, so re-running produces new samples rather than identical rows. The analysis uses per-task means to account for this.

## Adding or changing a task

Create `tasks/t11_<name>/` with the same five files. The runner picks up every `tasks/t*/` folder automatically. Run `python src/check_tasks.py` to confirm the new task is consistent. If you change any prompt, update its `checklist.md` so the two formats still list the same requirements.

## Limitations

- One model at one temperature.
- One prompt pair per task, so the particular wording of each informal paragraph is mixed in with the effect of "informality" itself.
- 10 tasks give limited statistical power. The notebook reports bootstrap confidence intervals alongside the Wilcoxon test.
- The two formats' prompts aren't exactly the same length. The notebook measures this gap as a possible confound.

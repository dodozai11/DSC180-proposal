"""All experiment settings live here. Change a value, re-run, done."""

# Model used for every call. claude-haiku-4-5 runs with no extended thinking
# when the `thinking` parameter is omitted, and it accepts `temperature`, so
# both controls hold. (Newer models always think and reject temperature.)
MODEL = "claude-haiku-4-5"

N_RUNS = 5          # repetitions per task x format
TEMPERATURE = 1.0   # API default; keeps natural run-to-run variability
MAX_TOKENS = 4096   # generous so code is never cut off (stop_reason is logged to check)
SEED = 42           # makes the shuffled call order reproducible
TEST_TIMEOUT = 10   # seconds allowed for the hidden tests on one response

# Identical for every call.
SYSTEM_PROMPT = "You are a Python programmer. Reply with only Python code."

# Paths are relative to the repo root.
TASKS_DIR = "tasks"
RESULTS_PATH = "data/results.csv"
RAW_DIR = "data/raw"

"""All experiment settings live here. Change a value, re-run, done."""

# Where the model runs:
#   "ollama"    - free, runs locally (install Ollama, then `ollama pull <MODEL>`)
#   "anthropic" - Claude via the paid API (needs ANTHROPIC_API_KEY in .env)
PROVIDER = "ollama"

# Model used for every call.
#   ollama:    "qwen2.5-coder:7b" (a coding model; no hidden "thinking" step)
#   anthropic: "claude-haiku-4-5" (no extended thinking when `thinking` is omitted,
#              and it accepts `temperature`; newer Claude models don't allow either)
MODEL = "qwen2.5-coder:7b"

N_RUNS = 5          # repetitions per task x format
TEMPERATURE = 0.7   # Qwen's recommended default; >0 so repeated runs differ
MAX_TOKENS = 4096   # generous so code is never cut off (stop_reason is logged to check)
SEED = 42           # makes the shuffled call order reproducible (Ollama also gets SEED + run)
TEST_TIMEOUT = 10   # seconds allowed for the hidden tests on one response

# Identical for every call.
SYSTEM_PROMPT = "You are a Python programmer. Reply with only Python code."

# Paths are relative to the repo root.
TASKS_DIR = "tasks"
RESULTS_PATH = "data/results.csv"
RAW_DIR = "data/raw"

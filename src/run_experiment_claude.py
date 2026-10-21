"""Run the same experiment on Claude, kept separate from the Qwen run.

Reuses src/run_experiment.py unchanged; only the settings below differ from config.py,
and results go to their own files, so data/results.csv is never touched.

Usage:
    python src/run_experiment_claude.py            # run (resumes automatically)
    python src/run_experiment_claude.py --dry-run  # show what would run; no model calls
"""

import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402

# Everything else (prompts, TEMPERATURE, MAX_TOKENS, SEED, SYSTEM_PROMPT, tests) comes
# from config.py, so the only difference from the Qwen run is the model.
config.PROVIDER = "anthropic"
config.MODEL = "claude-haiku-4-5"  # accepts temperature; no thinking unless asked for
config.N_RUNS = 5
config.RESULTS_PATH = "data/results_claude.csv"
config.RAW_DIR = "data/raw_claude"

import run_experiment  # noqa: E402

# The hidden tests run the model's code with our environment. Some replies call the
# Claude API themselves, so the key must not be visible to them: give it to our client
# only, then remove it from the environment.
load_dotenv(ROOT / ".env")
if "--dry-run" not in sys.argv:
    run_experiment._anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
os.environ.pop("ANTHROPIC_API_KEY", None)
run_experiment.load_dotenv = lambda *args, **kwargs: None  # don't let main() load it back

if __name__ == "__main__":
    run_experiment.main()

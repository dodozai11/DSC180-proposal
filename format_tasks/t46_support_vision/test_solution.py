"""Validity checks for this parse task. Answers are NOT graded (the parse tasks measure tokens).

test_runs_on_new_data: q1, q2, q3 run without crashing on a hidden 20-row sample from another
                       league, in the same data format as the prompt (DATA_FORMAT env variable)
test_not_hard_coded:   at most 2 of the prompt sample's names appear in the code
                       (a name in an example comment is fine; copying the data is not)
"""
import os
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
SOLUTION = HERE / "solution.py"
PROMPT_NAMES = ['J3rkie',
 'JACTROLL',
 'Peto',
 'Rekkles',
 'Prime',
 'Shredder',
 'Nsurr',
 'Tockimo',
 'Piero',
 'Centu',
 'Mahonix',
 'Visdom',
 'Brizz',
 'Vlaren',
 'Speedy1',
 'Max1',
 'Rutzou',
 'Zoelys',
 'Sparz',
 'Farfetch']


def normalize(text):
    """Lowercase and strip accents, so 'Shōnen' and 'Shonen' both count."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()


def test_runs_on_new_data():
    import solution  # imported here, so a crash only fails this test

    text = (HERE / f"hidden_{os.environ['DATA_FORMAT']}.txt").read_text(encoding="utf-8").strip()
    for name in ("q1", "q2", "q3"):
        getattr(solution, name)(text)


def test_not_hard_coded():
    source = normalize(SOLUTION.read_text(encoding="utf-8"))
    found = [n for n in PROMPT_NAMES if normalize(n) in source]
    assert len(found) <= 2, f"the code contains the prompt's data: {found[:5]}"

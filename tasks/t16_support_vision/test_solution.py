"""Validity checks for this data task. Answers are NOT graded (the experiment measures tokens).

test_runs:          q1, q2, q3 exist and run without crashing
test_data_embedded: every game id / name from the data appears in the code,
                    so a response that skipped copying the data is flagged as invalid

The two checks are independent: the data check reads solution.py as text without
running it, so code that crashes still gets credit for containing the data.
"""
import unicodedata
from pathlib import Path

SOLUTION = Path(__file__).parent / "solution.py"
IDENTIFIERS = ['J3rkie',
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


def test_runs():
    import solution  # imported here, so a crash only fails this test

    for name in ("q1", "q2", "q3"):
        getattr(solution, name)()


def test_data_embedded():
    source = normalize(SOLUTION.read_text(encoding="utf-8"))
    missing = [i for i in IDENTIFIERS if normalize(i) not in source]
    assert not missing, f"missing from the code: {missing}"

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
IDENTIFIERS = ['G01',
 'GiantX',
 'Movistar KOI',
 'G02',
 'G03',
 'Karmine Corp',
 'Fnatic',
 'G04',
 'G05',
 'SK Gaming',
 'Team Vitality',
 'G06',
 'Team Heretics',
 'G07',
 'G08',
 'G09',
 'G10',
 'G2 Esports']


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

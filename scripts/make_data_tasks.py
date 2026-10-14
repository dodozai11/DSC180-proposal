"""Create the data tasks from the Oracle's Elixir 2025 match data.

Copy tasks (t11-t16): the model must write the data into its code.
Parse tasks (t21-t26): the same data and questions, but each function receives the data
as a text string in the shown format and must parse it.

Each task gets 20 rows of real League of Legends esports data, written out three ways
(CSV, XML, prose) from the same canonical rows, plus two instruction templates
(structured / informal) with a {DATA} placeholder where the data goes.
Parse tasks also get a hidden 20-row sample from another league, used by their tests.

Usage (needs the Oracle's Elixir CSV, which is not committed):
    python scripts/make_data_tasks.py
    python scripts/make_data_tasks.py --source path/to/2025_LoL_esports_match_data_from_OraclesElixir.csv

Data source: Oracle's Elixir (https://oracleselixir.com), 2025 LoL esports match data.
"""

import argparse
import csv
import io
import json
import pprint
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT.parent / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
SEED = 42
N_TEAM_GAMES = 10   # team tasks: 10 games x 2 teams = 20 rows
N_PLAYERS = 20      # player tasks: 20 different players

EMBED_RULE_STRUCTURED = "Put the data directly in the code as Python data structures. Do not read it from a file."
EMBED_RULE_INFORMAL = "Please put the data right in the code as Python data structures instead of reading it from a file."

PARSE_RULE_STRUCTURED = ("Each function takes one argument, `text`: a string containing data in exactly the format "
                         "shown above, possibly with different rows. Parse it inside the function. "
                         "Do not hard-code the data.")
PARSE_RULE_INFORMAL = ("Each function should take the data as one string called text, in exactly the format above "
                       "but possibly with different rows, and parse it itself instead of hard-coding the data.")
DATA_FORMATS = ["csv", "xml", "prose"]

# Words used in the prose version for values that read badly as-is.
PROSE_RESULT = {"win": "won", "loss": "lost"}
PROSE_FIRST_DRAGON = {"yes": "took the first dragon", "no": "did not take the first dragon"}


# --- Task definitions ----------------------------------------------------------
# kind:      "team" rows (one per team per game) or "player" rows (one per player)
# columns:   the columns shown to the model, in order
# ids:       columns whose values must all appear in the model's code (validity check)
# prose:     builds one sentence from one row
# questions: (structured requirement text, function body for the reference solution)
# informal:  the informal request paragraph (same questions, different order and wording)

TASKS = [
    {
        "id": "t11_side_winrate", "hidden_league": "LFL", "kind": "team", "league": "LCK",
        "columns": ["game", "team", "side", "result", "game_minutes"],
        "ids": ["game", "team"],
        "prose": lambda r: (f"In game {r['game']}, {r['team']} played on the {r['side']} side and "
                            f"{PROSE_RESULT[r['result']]}; the game lasted {r['game_minutes']} minutes."),
        "questions": [
            ("`q1()` returns the win rate of blue-side teams, as a number between 0 and 1.",
             "blue = [r for r in ROWS if r['side'] == 'blue']\n    return sum(r['result'] == 'win' for r in blue) / len(blue)"),
            ("`q2()` returns the average length in minutes of the games won by the red-side team.",
             "wins = [r['game_minutes'] for r in ROWS if r['side'] == 'red' and r['result'] == 'win']\n    return sum(wins) / len(wins) if wins else 0.0"),
            ("`q3()` returns the name of the team with the most wins.",
             "wins = {}\n    for r in ROWS:\n        wins[r['team']] = wins.get(r['team'], 0) + (r['result'] == 'win')\n    return max(wins, key=wins.get)"),
        ],
        "informal": ("can you write me some Python that answers a few things about it? I'd like a function q2() "
                     "that gives the average length in minutes of the games the red-side team won, and q1() should "
                     "return the win rate of blue-side teams as a number between 0 and 1. Oh, and q3() should return "
                     "the name of the team with the most wins."),
    },
    {
        "id": "t12_gold_lead", "hidden_league": "LCK", "kind": "team", "league": "LEC",
        "columns": ["game", "team", "side", "result", "gold_diff_15"],
        "ids": ["game", "team"],
        "prose": lambda r: (f"In game {r['game']}, {r['team']} played on the {r['side']} side and "
                            f"{PROSE_RESULT[r['result']]}, with a gold difference of {r['gold_diff_15']} at 15 minutes."),
        "questions": [
            ("`q1()` returns the average gold difference at 15 minutes of the winning teams.",
             "wins = [r['gold_diff_15'] for r in ROWS if r['result'] == 'win']\n    return sum(wins) / len(wins)"),
            ("`q2()` returns how many teams won despite a negative gold difference at 15 minutes.",
             "return sum(r['result'] == 'win' and r['gold_diff_15'] < 0 for r in ROWS)"),
            ("`q3()` returns the win rate of teams that were ahead in gold at 15 minutes, as a number between 0 and 1.",
             "ahead = [r for r in ROWS if r['gold_diff_15'] > 0]\n    return sum(r['result'] == 'win' for r in ahead) / len(ahead)"),
        ],
        "informal": ("could you write some Python for it? I'm curious how many teams won even though they were behind "
                     "in gold at 15 minutes, so make that q2(). Then q3() should give the win rate of teams that were "
                     "ahead in gold at 15 minutes, as a number between 0 and 1, and q1() should return the average "
                     "gold difference at 15 minutes of the winning teams."),
    },
    {
        "id": "t13_objectives", "hidden_league": "LEC", "kind": "team", "league": "LAS",
        "columns": ["game", "team", "result", "first_dragon", "dragons", "barons", "towers"],
        "ids": ["game", "team"],
        "prose": lambda r: (f"In game {r['game']}, {r['team']} {PROSE_RESULT[r['result']]}; they "
                            f"{PROSE_FIRST_DRAGON[r['first_dragon']]} and finished with {r['dragons']} dragons, "
                            f"{r['barons']} barons and {r['towers']} towers."),
        "questions": [
            ("`q1()` returns the win rate of teams that took the first dragon, as a number between 0 and 1.",
             "first = [r for r in ROWS if r['first_dragon'] == 'yes']\n    return sum(r['result'] == 'win' for r in first) / len(first)"),
            ("`q2()` returns the average number of barons taken by winning teams.",
             "wins = [r['barons'] for r in ROWS if r['result'] == 'win']\n    return sum(wins) / len(wins)"),
            ("`q3()` returns the total number of towers destroyed across all rows.",
             "return sum(r['towers'] for r in ROWS)"),
        ],
        "informal": ("I want some Python to dig into it. First, q3() should just add up the towers destroyed across "
                     "all of it. q1() should return the win rate of teams that took the first dragon, as a number "
                     "between 0 and 1, and q2() should give the average number of barons the winning teams took."),
    },
    {
        "id": "t14_mid_kda", "hidden_league": "LJL", "kind": "player", "position": "mid", "league": "NLC",
        "columns": ["player", "team", "champion", "kills", "deaths", "assists", "result"],
        "ids": ["player"],
        "prose": lambda r: (f"{r['player']} of {r['team']} played {r['champion']} in mid and "
                            f"{PROSE_RESULT[r['result']]} with {r['kills']} kills, {r['deaths']} deaths "
                            f"and {r['assists']} assists."),
        "questions": [
            ("`q1()` returns the name of the player with the highest KDA, where KDA = (kills + assists) / max(deaths, 1).",
             "return max(ROWS, key=lambda r: (r['kills'] + r['assists']) / max(r['deaths'], 1))['player']"),
            ("`q2()` returns the average number of kills in games that were won.",
             "wins = [r['kills'] for r in ROWS if r['result'] == 'win']\n    return sum(wins) / len(wins)"),
            ("`q3()` returns the name of the most-picked champion.",
             "picks = {}\n    for r in ROWS:\n        picks[r['champion']] = picks.get(r['champion'], 0) + 1\n    return max(picks, key=picks.get)"),
        ],
        "informal": ("can you write some Python to look at it? I want q3() to tell me which champion got picked the "
                     "most, and q1() to return the name of the player with the highest KDA, which is (kills + assists) "
                     "divided by max(deaths, 1). And q2() should be the average number of kills in the games that "
                     "were won."),
    },
    {
        "id": "t15_jungle_picks", "hidden_league": "EM", "kind": "player", "position": "jng", "league": "LJL",
        "columns": ["player", "champion", "result", "deaths"],
        "ids": ["player"],
        "prose": lambda r: (f"{r['player']} played {r['champion']} in the jungle and "
                            f"{PROSE_RESULT[r['result']]}, dying {r['deaths']} times."),
        "questions": [
            ("`q1()` returns the number of different champions played.",
             "return len({r['champion'] for r in ROWS})"),
            ("`q2()` returns the name of the champion with the best win rate among champions played at least twice "
             "(return None if no champion was played twice).",
             "games = {}\n    for r in ROWS:\n        games.setdefault(r['champion'], []).append(r['result'] == 'win')\n"
             "    rates = {c: sum(g) / len(g) for c, g in games.items() if len(g) >= 2}\n"
             "    return max(rates, key=rates.get) if rates else None"),
            ("`q3()` returns the average number of deaths in games that were lost.",
             "losses = [r['deaths'] for r in ROWS if r['result'] == 'loss']\n    return sum(losses) / len(losses)"),
        ],
        "informal": ("could you write some Python for it? q3() should give the average number of deaths in the games "
                     "that were lost, and q1() should count how many different champions were played. For q2(), "
                     "among champions played at least twice, return the name of the one with the best win rate "
                     "(or None if no champion was played twice)."),
    },
    {
        "id": "t16_support_vision", "hidden_league": "LJL", "kind": "player", "position": "sup", "league": "EM",
        "columns": ["player", "team", "result", "vision_score", "wards_placed", "control_wards"],
        "ids": ["player"],
        "prose": lambda r: (f"{r['player']} of {r['team']} {PROSE_RESULT[r['result']]} with a vision score of "
                            f"{r['vision_score']}, {r['wards_placed']} wards placed and {r['control_wards']} "
                            f"control wards bought."),
        "questions": [
            ("`q1()` returns the average vision score of players who won minus the average vision score of players who lost.",
             "wins = [r['vision_score'] for r in ROWS if r['result'] == 'win']\n"
             "    losses = [r['vision_score'] for r in ROWS if r['result'] == 'loss']\n"
             "    return sum(wins) / len(wins) - sum(losses) / len(losses)"),
            ("`q2()` returns the name of the player who placed the most wards.",
             "return max(ROWS, key=lambda r: r['wards_placed'])['player']"),
            ("`q3()` returns how many players bought at least 15 control wards.",
             "return sum(r['control_wards'] >= 15 for r in ROWS)"),
        ],
        "informal": ("can you write me some Python? I'd like q2() to return the name of the player who placed the most "
                     "wards, and q3() to count how many players bought at least 15 control wards. Also q1() should "
                     "return the average vision score of the players who won minus the average for the players "
                     "who lost."),
    },
]

INTRO = {
    "team": "Here is some League of Legends esports match data, one row per team per game:",
    "player": "Here is some League of Legends esports match data, one row per player:",
}
INTRO_PARSE = {
    "team": "Here is a sample of some League of Legends esports match data, one row per team per game:",
    "player": "Here is a sample of some League of Legends esports match data, one row per player:",
}
INTRO_INFORMAL_PARSE = {
    "team": "hey, here's a sample of some League of Legends esports match data, one row per team per game:",
    "player": "hey, here's a sample of some League of Legends esports match data, one row per player:",
}
INTRO_INFORMAL = {
    "team": "hey, I've got some League of Legends esports match data here, one row per team per game:",
    "player": "hey, I've got some League of Legends esports match data here, one row per player:",
}


# --- Sampling --------------------------------------------------------------------

def load_source(path):
    columns = ["gameid", "datacompleteness", "league", "date", "side", "position", "playername", "teamname",
               "champion", "result", "gamelength", "kills", "deaths", "assists", "firstdragon", "dragons",
               "barons", "towers", "golddiffat15", "visionscore", "wardsplaced", "controlwardsbought"]
    df = pd.read_csv(path, usecols=columns, low_memory=False, encoding="utf-8")
    return df[df["datacompleteness"] == "complete"]


def team_rows(df, league, exclude_names=()):
    """10 random games from the league, both teams' rows, in date order (skipping games with excluded teams)."""
    teams = df[(df["position"] == "team") & (df["league"] == league)]
    if exclude_names:
        bad_games = teams.loc[teams["teamname"].isin(exclude_names), "gameid"]
        teams = teams[~teams["gameid"].isin(bad_games)]
    game_ids = pd.Series(sorted(teams["gameid"].unique())).sample(N_TEAM_GAMES, random_state=SEED)
    picked = teams[teams["gameid"].isin(game_ids)].sort_values(["date", "side"])
    game_number = {gid: f"G{i:02d}" for i, gid in enumerate(picked["gameid"].unique(), start=1)}
    rows = []
    for _, r in picked.iterrows():
        rows.append({
            "game": game_number[r["gameid"]],
            "team": r["teamname"],
            "side": r["side"].lower(),
            "result": "win" if r["result"] == 1 else "loss",
            "game_minutes": round(r["gamelength"] / 60, 1),
            "gold_diff_15": int(r["golddiffat15"]),
            "first_dragon": "yes" if r["firstdragon"] == 1 else "no",
            "dragons": int(r["dragons"]),
            "barons": int(r["barons"]),
            "towers": int(r["towers"]),
        })
    return rows


def player_rows(df, league, position, exclude_names=()):
    """20 different players from the league at one position (one game each), in date order."""
    players = df[(df["position"] == position) & (df["league"] == league)].drop_duplicates("playername")
    if exclude_names:
        players = players[~players["playername"].isin(exclude_names) & ~players["teamname"].isin(exclude_names)]
    picked = players.sample(N_PLAYERS, random_state=SEED).sort_values("date")
    rows = []
    for _, r in picked.iterrows():
        rows.append({
            "player": r["playername"],
            "team": r["teamname"],
            "champion": r["champion"],
            "result": "win" if r["result"] == 1 else "loss",
            "kills": int(r["kills"]),
            "deaths": int(r["deaths"]),
            "assists": int(r["assists"]),
            "vision_score": int(r["visionscore"]),
            "wards_placed": int(r["wardsplaced"]),
            "control_wards": int(r["controlwardsbought"]),
        })
    return rows


# --- Rendering the same rows three ways ------------------------------------------

def render_csv(rows, columns):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for r in rows:
        writer.writerow([r[c] for c in columns])
    return out.getvalue().strip()


def render_xml(rows, columns):
    # Generic <rows>/<row> tags: a column can't clash with the element name (e.g. a "game" column).
    root, item = "rows", "row"
    lines = [f"<{root}>"]
    for r in rows:
        fields = "".join(f"<{c}>{escape(str(r[c]))}</{c}>" for c in columns)
        lines.append(f"  <{item}>{fields}</{item}>")
    lines.append(f"</{root}>")
    return "\n".join(lines)


def render_all(rows, task):
    """The same rows in every data format: {format: text}."""
    return {
        "csv": render_csv(rows, task["columns"]),
        "xml": render_xml(rows, task["columns"]),
        "prose": render_prose(rows, task["prose"]),
    }


def render_prose(rows, sentence):
    return " ".join(sentence(r) for r in rows)


def check_prose_row(row, sentence_text, columns):
    """Every value of the row must appear in its prose sentence (with result/first_dragon as words)."""
    for c in columns:
        value = row[c]
        if c == "result":
            value = PROSE_RESULT[value]
        elif c == "first_dragon":
            value = PROSE_FIRST_DRAGON[value]
        assert str(value) in sentence_text, f"{c}={value!r} missing from: {sentence_text}"


# --- Prompts, tests, reference, checklist ---------------------------------------------

def structured_template(task):
    reqs = [EMBED_RULE_STRUCTURED] + [
        "Define a function " + q.replace("()` returns", "()` that returns", 1) for q, _ in task["questions"]
    ]
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(reqs, start=1))
    return (f"## Goal\nWrite Python code that answers three questions about the data below.\n\n"
            f"## Data\n{INTRO[task['kind']]}\n\n{{DATA}}\n\n"
            f"## Requirements\n{numbered}\n\n"
            f"## Output\nRespond with only Python code.\n")


def informal_template(task):
    return (f"{INTRO_INFORMAL[task['kind']]}\n\n{{DATA}}\n\n"
            f"{task['informal']} {EMBED_RULE_INFORMAL} Just send back the Python code, nothing else.\n")


def identifiers(rows, task):
    found = []
    for r in rows:
        for c in task["ids"]:
            if r[c] not in found:
                found.append(r[c])
    return found


def test_file(ids):
    return f'''"""Validity checks for this data task. Answers are NOT graded (the experiment measures tokens).

test_runs:          q1, q2, q3 exist and run without crashing
test_data_embedded: every game id / name from the data appears in the code,
                    so a response that skipped copying the data is flagged as invalid

The two checks are independent: the data check reads solution.py as text without
running it, so code that crashes still gets credit for containing the data.
"""
import unicodedata
from pathlib import Path

SOLUTION = Path(__file__).parent / "solution.py"
IDENTIFIERS = {pprint.pformat(ids, width=100)}


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
    assert not missing, f"missing from the code: {{missing}}"
'''


def reference_file(rows, task):
    shown = [{c: r[c] for c in task["columns"]} for r in rows]
    functions = "\n\n\n".join(f"def q{i}():\n    {body}" for i, (_, body) in enumerate(task["questions"], start=1))
    return f"ROWS = {pprint.pformat(shown, width=110, sort_dicts=False)}\n\n\n{functions}\n"


def checklist_file(task):
    lines = [f"# {task['id']} checklist", "",
             "Answers are not graded; the tests only check that the code runs and contains all the data.", "",
             "| Item | Requirement | Structured | Informal |", "|---|---|---|---|",
             f"| E | data embedded in the code, not read from a file | Requirements #1 | \"{EMBED_RULE_INFORMAL}\" |"]
    for i, (text, _) in enumerate(task["questions"], start=1):
        lines.append(f"| Q{i} | {text} | Requirements #{i + 1} | describes q{i}() in the paragraph |")
    lines += ["", "Shared by both: the same intro line, the same {DATA} block in the same position, "
              "\"only Python code\"."]
    return "\n".join(lines) + "\n"


def write_task(task, rows):
    folder = ROOT / "tasks" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    shown = [{c: r[c] for c in task["columns"]} for r in rows]

    for r in shown:
        check_prose_row(r, task["prose"](r), task["columns"])

    files = {
        "sample.json": json.dumps(shown, indent=2, ensure_ascii=False) + "\n",
        **{f"data_{fmt}.txt": text + "\n" for fmt, text in render_all(shown, task).items()},
        "structured.txt": structured_template(task),
        "informal.txt": informal_template(task),
        "test_solution.py": test_file(identifiers(shown, task)),
        "reference.py": reference_file(rows, task),
        "checklist.md": checklist_file(task),
    }
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    print(f"wrote {folder.relative_to(ROOT)} ({len(shown)} rows)")


# --- Experiment 3: parse tasks ------------------------------------------------------------------

def parse_id(task):
    """t11_side_winrate -> t21_side_winrate_parse"""
    return "t2" + task["id"][2:] + "_parse"


def as_parse_question(text):
    """`q1()` returns ... -> `q1(text)` returns ..."""
    return text.replace("()`", "(text)`", 1)


def structured_parse_template(task):
    reqs = [PARSE_RULE_STRUCTURED] + [
        "Define a function " + as_parse_question(q).replace("(text)` returns", "(text)` that returns", 1)
        for q, _ in task["questions"]
    ]
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(reqs, start=1))
    return (f"## Goal\nWrite Python functions that answer three questions about data in the format below.\n\n"
            f"## Data\n{INTRO_PARSE[task['kind']]}\n\n{{DATA}}\n\n"
            f"## Requirements\n{numbered}\n\n"
            f"## Output\nRespond with only Python code.\n")


def informal_parse_template(task):
    paragraph = task["informal"]
    for i in (1, 2, 3):
        paragraph = paragraph.replace(f"q{i}()", f"q{i}(text)")
    return (f"{INTRO_INFORMAL_PARSE[task['kind']]}\n\n{{DATA}}\n\n"
            f"{paragraph} {PARSE_RULE_INFORMAL} Just send back the Python code, nothing else.\n")


def names(rows, task):
    """Team/player names in the rows (game ids like G01 are excluded: every sample has them)."""
    found = []
    for r in rows:
        for c in task["ids"]:
            if c != "game" and r[c] not in found:
                found.append(r[c])
    return found


def parse_test_file(prompt_names):
    return f'''"""Validity checks for this parse task. Answers are NOT graded (the parse tasks measure tokens).

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
PROMPT_NAMES = {pprint.pformat(prompt_names, width=100)}


def normalize(text):
    """Lowercase and strip accents, so 'Shōnen' and 'Shonen' both count."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()


def test_runs_on_new_data():
    import solution  # imported here, so a crash only fails this test

    text = (HERE / f"hidden_{{os.environ['DATA_FORMAT']}}.txt").read_text(encoding="utf-8").strip()
    for name in ("q1", "q2", "q3"):
        getattr(solution, name)(text)


def test_not_hard_coded():
    source = normalize(SOLUTION.read_text(encoding="utf-8"))
    found = [n for n in PROMPT_NAMES if normalize(n) in source]
    assert len(found) <= 2, f"the code contains the prompt's data: {{found[:5]}}"
'''


PARSE_REFERENCE = """def q1(text):
    # Minimal non-hard-coded solution, used only to check that the tests work.
    return len(text.strip().splitlines())


def q2(text):
    return len(text)


def q3(text):
    return text.split()[0] if text.split() else None
"""


def parse_checklist_file(task):
    lines = [f"# {parse_id(task)} checklist", "",
             f"Parse-mode twin of {task['id']}: same data and questions, but each function receives the data as text.",
             "Answers are not graded; the tests only check that the code runs on new data and doesn't copy the data.", "",
             "| Item | Requirement | Structured | Informal |", "|---|---|---|---|",
             f"| P | functions take `text` and parse it; don't hard-code the data | Requirements #1 | \"{PARSE_RULE_INFORMAL}\" |"]
    for i, (text, _) in enumerate(task["questions"], start=1):
        lines.append(f"| Q{i} | {as_parse_question(text)} | Requirements #{i + 1} | describes q{i}(text) in the paragraph |")
    lines += ["", "Shared by both: the same intro line, the same {DATA} block in the same position, "
              "\"only Python code\"."]
    return "\n".join(lines) + "\n"


def write_parse_task(task, rows, hidden_rows):
    folder = ROOT / "tasks" / parse_id(task)
    folder.mkdir(parents=True, exist_ok=True)
    shown = [{c: r[c] for c in task["columns"]} for r in rows]
    hidden = [{c: r[c] for c in task["columns"]} for r in hidden_rows]
    shared = set(names(shown, task)) & set(names(hidden, task))
    assert not shared, f"hidden sample shares names with the prompt sample: {shared}"

    files = {
        "sample.json": json.dumps(shown, indent=2, ensure_ascii=False) + "\n",
        "hidden.json": json.dumps(hidden, indent=2, ensure_ascii=False) + "\n",
        **{f"data_{fmt}.txt": text + "\n" for fmt, text in render_all(shown, task).items()},
        **{f"hidden_{fmt}.txt": text + "\n" for fmt, text in render_all(hidden, task).items()},
        "structured.txt": structured_parse_template(task),
        "informal.txt": informal_parse_template(task),
        "test_solution.py": parse_test_file(names(shown, task)),
        "reference.py": PARSE_REFERENCE,
        "checklist.md": parse_checklist_file(task),
    }
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    print(f"wrote {folder.relative_to(ROOT)} ({len(shown)} rows + {len(hidden)} hidden rows from {task['hidden_league']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Oracle's Elixir 2025 CSV")
    args = parser.parse_args()

    df = load_source(args.source)
    for task in TASKS:
        if task["kind"] == "team":
            rows = team_rows(df, task["league"])
        else:
            rows = player_rows(df, task["league"], task["position"])
        write_task(task, rows)

        shown = [{c: r[c] for c in task["columns"]} for r in rows]
        exclude = names(shown, task)
        if task["kind"] == "team":
            hidden_rows = team_rows(df, task["hidden_league"], exclude)
        else:
            hidden_rows = player_rows(df, task["hidden_league"], task["position"], exclude)
        write_parse_task(task, rows, hidden_rows)


if __name__ == "__main__":
    main()

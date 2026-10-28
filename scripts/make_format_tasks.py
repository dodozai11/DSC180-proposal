"""Build the data-format v2 tasks (t41-t52) in five data formats: CSV, JSON, XML, Markdown and prose.

t41-t46: the parse tasks t21-t26 again: the same 20-row samples, hidden samples, questions
         and structured prompt (read from tasks/t2X_*_parse/).
t47-t52: six new tasks with new leagues, columns and questions, sampled the same way from
         the Oracle's Elixir file (fixed seed; the hidden sample comes from another league
         and shares no names with the prompt sample).

    python scripts/make_format_tasks.py
    python scripts/make_format_tasks.py --source path/to/2025_LoL_esports_match_data_from_OraclesElixir.csv

For each task this writes format_tasks/t4X_<name>/ (or t5X_<name>/):
    data_<format>.txt     the prompt's 20 rows in each format
    hidden_<format>.txt   the hidden 20 rows (another league) in each format, for the tests
    structured.txt        the prompt template, with a {DATA} slot
    sample.json, hidden.json, test_solution.py, reference.py

The tasks live outside tasks/ so src/run_experiment.py doesn't pick them up.
"""
import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from make_data_tasks import (DEFAULT_SOURCE, N_PLAYERS, N_TEAM_GAMES, PARSE_REFERENCE, PROSE_RESULT,  # noqa: E402
                             SEED, TASKS, parse_id, parse_test_file, render_csv, render_prose, render_xml,
                             structured_parse_template)
from make_data_tasks import names as id_names  # noqa: E402

SOURCE_DIR = ROOT / "tasks"
OUT_DIR = ROOT / "format_tasks"
DATA_FORMATS = ["csv", "json", "xml", "markdown", "prose"]
COPIED = ["sample.json", "hidden.json", "structured.txt", "test_solution.py", "reference.py"]
YES_NO = {1: "yes", 0: "no"}

# --- New tasks (t47-t52) ---------------------------------------------------------------
# Same fields as make_data_tasks.TASKS, plus `fields`: output column -> (source column, conversion).
# Questions only need their text here: parse tasks have no worked reference answers.

NEW_TASKS = [
    {
        "id": "t47_first_tower", "kind": "team", "league": "LCKC", "hidden_league": "TCL",
        "columns": ["game", "team", "result", "first_tower", "void_grubs", "heralds"],
        "ids": ["game", "team"],
        "fields": {"first_tower": ("firsttower", lambda v: YES_NO[int(v)]), "void_grubs": ("void_grubs", int),
                   "heralds": ("heralds", int)},
        "prose": lambda r: (f"In game {r['game']}, {r['team']} {PROSE_RESULT[r['result']]}; they "
                            f"{'took' if r['first_tower'] == 'yes' else 'did not take'} the first tower and "
                            f"finished with {r['void_grubs']} void grubs and {r['heralds']} heralds."),
        "questions": [
            ("`q1()` returns the win rate of teams that took the first tower, as a number between 0 and 1.", ""),
            ("`q2()` returns the average number of void grubs taken by winning teams.", ""),
            ("`q3()` returns how many teams took at least one herald and still lost.", ""),
        ],
    },
    {
        "id": "t48_early_kills", "kind": "team", "league": "PCS", "hidden_league": "TCL",
        "columns": ["game", "team", "side", "result", "kills_at_15", "xp_diff_15"],
        "ids": ["game", "team"],
        "fields": {"kills_at_15": ("killsat15", int), "xp_diff_15": ("xpdiffat15", int)},
        "prose": lambda r: (f"In game {r['game']}, {r['team']} played on the {r['side']} side and "
                            f"{PROSE_RESULT[r['result']]}, with {r['kills_at_15']} kills and an experience "
                            f"difference of {r['xp_diff_15']} at 15 minutes."),
        "questions": [
            ("`q1()` returns the average number of kills at 15 minutes of red-side teams.", ""),
            ("`q2()` returns the name of the team with the largest experience difference at 15 minutes.", ""),
            ("`q3()` returns how many teams won despite a negative experience difference at 15 minutes.", ""),
        ],
    },
    {
        "id": "t49_inhibitors", "kind": "team", "league": "LTA N", "hidden_league": "LVP SL",
        "columns": ["game", "team", "result", "game_minutes", "inhibitors", "tower_damage"],
        "ids": ["game", "team"],
        "fields": {"game_minutes": ("gamelength", lambda v: round(v / 60, 1)), "inhibitors": ("inhibitors", int),
                   "tower_damage": ("damagetotowers", int)},
        "prose": lambda r: (f"In game {r['game']}, {r['team']} {PROSE_RESULT[r['result']]} after "
                            f"{r['game_minutes']} minutes, destroying {r['inhibitors']} inhibitors and dealing "
                            f"{r['tower_damage']} damage to towers."),
        "questions": [
            ("`q1()` returns the average number of inhibitors destroyed by winning teams.", ""),
            ("`q2()` returns the average damage to towers of losing teams.", ""),
            ("`q3()` returns the game label (for example G01) of the longest game.", ""),
        ],
    },
    {
        "id": "t50_bot_farm", "kind": "player", "position": "bot", "league": "NACL", "hidden_league": "HC",
        "columns": ["player", "team", "champion", "result", "cs_per_min", "gold_share"],
        "ids": ["player"],
        "fields": {"cs_per_min": ("cspm", lambda v: round(v, 2)), "gold_share": ("earnedgoldshare", lambda v: round(v, 3))},
        "prose": lambda r: (f"{r['player']} of {r['team']} played {r['champion']} in the bot lane and "
                            f"{PROSE_RESULT[r['result']]}, with {r['cs_per_min']} CS per minute and a gold share "
                            f"of {r['gold_share']}."),
        "questions": [
            ("`q1()` returns the name of the player with the highest CS per minute.", ""),
            ("`q2()` returns the average gold share of players who won.", ""),
            ("`q3()` returns how many players had more than 9 CS per minute.", ""),
        ],
    },
    {
        "id": "t51_top_damage", "kind": "player", "position": "top", "league": "LPLOL", "hidden_league": "HLL",
        "columns": ["player", "team", "champion", "result", "damage_per_min", "damage_taken_per_min"],
        "ids": ["player"],
        "fields": {"damage_per_min": ("dpm", lambda v: round(v, 1)),
                   "damage_taken_per_min": ("damagetakenperminute", lambda v: round(v, 1))},
        "prose": lambda r: (f"{r['player']} of {r['team']} played {r['champion']} in the top lane and "
                            f"{PROSE_RESULT[r['result']]}, dealing {r['damage_per_min']} damage per minute and "
                            f"taking {r['damage_taken_per_min']} damage per minute."),
        "questions": [
            ("`q1()` returns the average damage per minute of players who lost.", ""),
            ("`q2()` returns the name of the player who took the most damage per minute.", ""),
            ("`q3()` returns the number of different teams.", ""),
        ],
    },
    {
        "id": "t52_jungle_early", "kind": "player", "position": "jng", "league": "CD", "hidden_league": "PCS",
        "columns": ["player", "champion", "result", "cs_diff_10", "kills_at_10", "assists_at_10"],
        "ids": ["player"],
        "fields": {"cs_diff_10": ("csdiffat10", int), "kills_at_10": ("killsat10", int),
                   "assists_at_10": ("assistsat10", int)},
        "prose": lambda r: (f"{r['player']} played {r['champion']} in the jungle and {PROSE_RESULT[r['result']]}; "
                            f"at 10 minutes they had a CS difference of {r['cs_diff_10']}, {r['kills_at_10']} "
                            f"kills and {r['assists_at_10']} assists."),
        "questions": [
            ("`q1()` returns the win rate of players with a positive CS difference at 10 minutes, as a number "
             "between 0 and 1.", ""),
            ("`q2()` returns the total number of kills at 10 minutes across all rows.", ""),
            ("`q3()` returns the name of the champion with the most assists at 10 minutes in total.", ""),
        ],
    },
]


def format_id(task):
    """t11_side_winrate -> t41_side_winrate; new tasks already have their final id."""
    return task["id"] if task in NEW_TASKS else "t4" + task["id"][2:]


ALL_TASKS = TASKS + NEW_TASKS


def render_markdown(rows, columns):
    # The same pipe table as scripts/make_tool_tasks.py:to_markdown.
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def render_all(rows, task):
    """The same rows in every data format: {format: text}."""
    columns = task["columns"]
    return {
        "csv": render_csv(rows, columns),
        "json": json.dumps(rows, indent=2, ensure_ascii=False),
        "xml": render_xml(rows, columns),
        "markdown": render_markdown(rows, columns),
        "prose": render_prose(rows, task["prose"]),
    }


def names(rows):
    return sorted({str(r[c]) for r in rows for c in ("team", "player") if c in r})


# --- Sampling the new tasks (same method as make_data_tasks.team_rows / player_rows) ----------

def load_source(path):
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    return df[df["datacompleteness"] == "complete"]


def convert(r, task):
    row = {}
    for c in task["columns"]:
        if c == "game":
            continue
        if c in task["fields"]:
            source, fn = task["fields"][c]
            row[c] = fn(r[source])
        elif c == "result":
            row[c] = "win" if r["result"] == 1 else "loss"
        elif c == "side":
            row[c] = r["side"].lower()
        else:
            row[c] = {"team": r["teamname"], "player": r["playername"], "champion": r["champion"]}[c]
    return row


def sample_team(df, task, league, exclude_names=()):
    """10 random games from the league, both teams' rows, in date order."""
    needed = [s for s, _ in task["fields"].values()]
    teams = df[(df["position"] == "team") & (df["league"] == league)]
    bad = teams.loc[teams[needed].isna().any(axis=1) | teams["teamname"].isin(exclude_names), "gameid"]
    teams = teams[~teams["gameid"].isin(bad)]
    game_ids = pd.Series(sorted(teams["gameid"].unique())).sample(N_TEAM_GAMES, random_state=SEED)
    picked = teams[teams["gameid"].isin(game_ids)].sort_values(["date", "side"])
    game_number = {gid: f"G{i:02d}" for i, gid in enumerate(picked["gameid"].unique(), start=1)}
    return [{"game": game_number[r["gameid"]], **convert(r, task)} for _, r in picked.iterrows()]


def sample_players(df, task, league, exclude_names=()):
    """20 different players from the league at one position (one game each), in date order."""
    needed = [s for s, _ in task["fields"].values()]
    players = df[(df["position"] == task["position"]) & (df["league"] == league)].dropna(subset=needed + ["playername"])
    players = players.drop_duplicates("playername")
    players = players[~players["playername"].isin(exclude_names)]
    picked = players.sample(N_PLAYERS, random_state=SEED).sort_values("date")
    return [convert(r, task) for _, r in picked.iterrows()]


# --- Writing ---------------------------------------------------------------------------

def write_formats(folder, task, prefix, rows, old_dir=None):
    for fmt, text in render_all(rows, task).items():
        # For t41-t46, CSV, XML and prose must match the parse task's files exactly.
        old = old_dir / f"{prefix}_{fmt}.txt" if old_dir else None
        if old and old.exists():
            assert text == old.read_text(encoding="utf-8").strip(), f"{old} differs from the re-rendering"
        missing = [n for n in names(rows) if n not in html.unescape(text)]
        assert not missing, f"{folder.name} {prefix}_{fmt}: missing {missing[:3]}"
        (folder / f"{prefix}_{fmt}.txt").write_text(text + "\n", encoding="utf-8")


def write_parse_twin_task(task):
    source = SOURCE_DIR / parse_id(task)
    folder = OUT_DIR / format_id(task)
    folder.mkdir(parents=True)
    for name in COPIED:
        shutil.copy(source / name, folder / name)
    for prefix, sample in (("data", "sample.json"), ("hidden", "hidden.json")):
        write_formats(folder, task, prefix, json.loads((source / sample).read_text(encoding="utf-8")), source)
    print(f"wrote {folder.relative_to(ROOT)} (from {source.name})")


def write_new_task(task, df):
    sample = sample_team if task["kind"] == "team" else sample_players
    shown = sample(df, task, task["league"])
    hidden = sample(df, task, task["hidden_league"], exclude_names=id_names(shown, task))
    shared = set(names(shown)) & set(names(hidden))
    assert not shared, f"{task['id']}: hidden sample shares names with the prompt sample: {shared}"

    folder = OUT_DIR / format_id(task)
    folder.mkdir(parents=True)
    files = {
        "sample.json": json.dumps(shown, indent=2, ensure_ascii=False) + "\n",
        "hidden.json": json.dumps(hidden, indent=2, ensure_ascii=False) + "\n",
        "structured.txt": structured_parse_template(task),
        "test_solution.py": parse_test_file(id_names(shown, task)),
        "reference.py": PARSE_REFERENCE,
    }
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    write_formats(folder, task, "data", shown)
    write_formats(folder, task, "hidden", hidden)
    print(f"wrote {folder.relative_to(ROOT)} ({task['league']}, hidden: {task['hidden_league']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Oracle's Elixir 2025 CSV (for t47-t52)")
    args = parser.parse_args()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for task in TASKS:
        write_parse_twin_task(task)
    df = load_source(args.source)
    for task in NEW_TASKS:
        write_new_task(task, df)

"""Build the tool-result tasks (t31-t36) from the copy tasks' 20-row samples.

The model doesn't write code here: the data arrives as the result of a (scripted) tool
call, in one of several formats, and the model answers three questions directly.

    python scripts/make_tool_tasks.py

For each task this writes tool_tasks/t3X_<name>/:
    data_<format>.txt   the same 20 rows as the tool would return them (5 formats)
    questions.txt       the user's message: three questions + the ANSWERS line to end with
    tool.json           the tool's schema and the arguments of the scripted call
    answers.json        the answer key

It only reads tasks/t1X_<name>/sample.json, so it doesn't need the Oracle's Elixir file.
"""
import json
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "tasks"
OUT_DIR = ROOT / "tool_tasks"

DATA_FORMATS = ["csv", "markdown", "json", "json_pretty", "xml"]

TEAM_TOOL = {
    "name": "get_team_games",
    "description": "Get recent League of Legends esports games for a league, one row per team per game.",
    "parameters": {
        "type": "object",
        "properties": {
            "league": {"type": "string", "description": "League code, e.g. LCK"},
            "limit": {"type": "integer", "description": "Number of games"},
        },
        "required": ["league"],
    },
}
PLAYER_TOOL = {
    "name": "get_player_stats",
    "description": "Get one recent game per player for a league and position, one row per player.",
    "parameters": {
        "type": "object",
        "properties": {
            "league": {"type": "string", "description": "League code, e.g. LCK"},
            "position": {"type": "string", "enum": ["top", "jng", "mid", "bot", "sup"]},
            "limit": {"type": "integer", "description": "Number of players"},
        },
        "required": ["league", "position"],
    },
}


# --- Answer helpers -------------------------------------------------------------
# Questions are lookups and small counts, not averages: a 7-8B model answering in one pass
# got ~0-8% of average/rate questions right in every format, too few to compare formats.
# q1 = one number from one row, q2 = one name from one row, q3 = count rows matching a value.

def number(x):
    return {"type": "number", "value": round(float(x), 6)}


def name(x):
    return {"type": "name", "accept": [x]}


def lookup(rows, field, **match):
    """The `field` of the only row whose columns equal `match`."""
    found = [r[field] for r in rows if all(r[k] == v for k, v in match.items())]
    assert len(found) == 1, (field, match, found)
    return found[0]


def count(rows, **match):
    return sum(all(r[k] == v for k, v in match.items()) for r in rows)


TASKS = [
    {
        "id": "t31_side_winrate", "source": "t11_side_winrate", "tool": TEAM_TOOL,
        "args": {"league": "LCK", "limit": 10},
        "intro": "Can you pull the last 10 LCK games and answer these?",
        "questions": [
            "How long was game G07, in minutes?",
            "Which team played on the red side in G06?",
            "How many games did Hanwha Life Esports win?",
        ],
        "answers": lambda rows: [
            number(lookup(rows, "game_minutes", game="G07", side="blue")),
            name(lookup(rows, "team", game="G06", side="red")),
            number(count(rows, team="Hanwha Life Esports", result="win")),
        ],
    },
    {
        "id": "t32_gold_lead", "source": "t12_gold_lead", "tool": TEAM_TOOL,
        "args": {"league": "LEC", "limit": 10},
        "intro": "Can you pull the last 10 LEC games and answer these?",
        "questions": [
            "What was Team Vitality's gold difference at 15 minutes in G09?",
            "Which team won G10?",
            "How many games did GiantX win?",
        ],
        "answers": lambda rows: [
            number(lookup(rows, "gold_diff_15", game="G09", team="Team Vitality")),
            name(lookup(rows, "team", game="G10", result="win")),
            number(count(rows, team="GiantX", result="win")),
        ],
    },
    {
        "id": "t33_objectives", "source": "t13_objectives", "tool": TEAM_TOOL,
        "args": {"league": "LAS", "limit": 10},
        "intro": "Can you pull the last 10 LAS games and answer these?",
        "questions": [
            "How many towers did KT Rolster Academy destroy in G04?",
            "Which team took the first dragon in G06?",
            "How many games did DRX Academy play?",
        ],
        "answers": lambda rows: [
            number(lookup(rows, "towers", game="G04", team="KT Rolster Academy")),
            name(lookup(rows, "team", game="G06", first_dragon="yes")),
            number(count(rows, team="DRX Academy")),
        ],
    },
    {
        "id": "t34_mid_kda", "source": "t14_mid_kda", "tool": PLAYER_TOOL,
        "args": {"league": "NLC", "position": "mid", "limit": 20},
        "intro": "Can you pull stats for 20 NLC mid laners and answer these?",
        "questions": [
            "How many assists did Air have?",
            "Which champion did Sencux play?",
            "How many players played Orianna?",
        ],
        "answers": lambda rows: [
            number(lookup(rows, "assists", player="Air")),
            name(lookup(rows, "champion", player="Sencux")),
            number(count(rows, champion="Orianna")),
        ],
    },
    {
        "id": "t35_jungle_picks", "source": "t15_jungle_picks", "tool": PLAYER_TOOL,
        "args": {"league": "LJL", "position": "jng", "limit": 20},
        "intro": "Can you pull stats for 20 LJL junglers and answer these?",
        "questions": [
            "How many deaths did Racon have?",
            "Which champion did DG hatinyang play?",
            "How many players played Viego?",
        ],
        "answers": lambda rows: [
            number(lookup(rows, "deaths", player="Racon")),
            name(lookup(rows, "champion", player="DG hatinyang")),
            number(count(rows, champion="Viego")),
        ],
    },
    {
        "id": "t36_support_vision", "source": "t16_support_vision", "tool": PLAYER_TOOL,
        "args": {"league": "EM", "position": "sup", "limit": 20},
        "intro": "Can you pull stats for 20 EMEA Masters supports and answer these?",
        "questions": [
            "What was Brizz's vision score?",
            "Which team does Rekkles play for?",
            "How many players are on Karmine Corp Blue?",
        ],
        "answers": lambda rows: [
            number(lookup(rows, "vision_score", player="Brizz")),
            name(lookup(rows, "team", player="Rekkles")),
            number(count(rows, team="Karmine Corp Blue")),
        ],
    },
]

ANSWER_LINE = """End your reply with one line in exactly this form:
ANSWERS: {"q1": ..., "q2": ..., "q3": ...}
Use numbers for numeric answers and strings for names."""


# --- Data formats --------------------------------------------------------------

def to_markdown(rows):
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def to_xml(rows):
    lines = ["<rows>"]
    for r in rows:
        lines.append("  <row>" + "".join(f"<{k}>{escape(str(v))}</{k}>" for k, v in r.items()) + "</row>")
    lines.append("</rows>")
    return "\n".join(lines)


def write_task(task):
    source = SOURCE_DIR / task["source"]
    rows = json.loads((source / "sample.json").read_text(encoding="utf-8"))
    folder = OUT_DIR / task["id"]
    folder.mkdir(parents=True, exist_ok=True)

    data = {
        "csv": (source / "data_csv.txt").read_text(encoding="utf-8").strip(),  # the same CSV as the copy task
        "markdown": to_markdown(rows),
        "json": json.dumps(rows, ensure_ascii=False),
        "json_pretty": json.dumps(rows, ensure_ascii=False, indent=2),
        "xml": to_xml(rows),
    }
    for d, text in data.items():
        (folder / f"data_{d}.txt").write_text(text + "\n", encoding="utf-8")

    questions = "\n".join(f"{i}. {q}" for i, q in enumerate(task["questions"], start=1))
    (folder / "questions.txt").write_text(f"{task['intro']}\n\n{questions}\n\n{ANSWER_LINE}\n", encoding="utf-8")
    (folder / "tool.json").write_text(json.dumps({"tool": task["tool"], "arguments": task["args"]}, indent=2) + "\n",
                                      encoding="utf-8")
    answers = dict(zip(["q1", "q2", "q3"], task["answers"](rows)))
    (folder / "answers.json").write_text(json.dumps(answers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {folder.relative_to(ROOT)}: " + ", ".join(f"{q}={a.get('value', a.get('accept'))}"
                                                           for q, a in answers.items()))


if __name__ == "__main__":
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for task in TASKS:
        write_task(task)

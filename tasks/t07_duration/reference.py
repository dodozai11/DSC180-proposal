import re

PATTERN = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")


def parse_duration(s):
    s = s.replace(" ", "").lower()
    if not s:
        raise ValueError("empty duration")
    match = PATTERN.fullmatch(s)
    if not match:
        raise ValueError(f"invalid duration: {s!r}")
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds

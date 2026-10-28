def q1(text):
    # Minimal non-hard-coded solution, used only to check that the tests work.
    return len(text.strip().splitlines())


def q2(text):
    return len(text)


def q3(text):
    return text.split()[0] if text.split() else None

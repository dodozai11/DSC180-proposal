def normalize_whitespace(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(text.split())

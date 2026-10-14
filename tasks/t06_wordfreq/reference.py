def word_frequencies(text, top_k=None):
    for ch in ".,!?;:":
        text = text.replace(ch, "")
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    result = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        result = result[:top_k]
    return result

def chunk_list(items, size):
    if size < 1:
        raise ValueError("size must be at least 1")
    return [list(items[i:i + size]) for i in range(0, len(items), size)]

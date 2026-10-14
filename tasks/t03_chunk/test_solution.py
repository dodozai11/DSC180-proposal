import pytest

from solution import chunk_list


def test_r1_consecutive_chunks_in_order():
    assert chunk_list([1, 2, 3, 4, 5, 6], 2) == [[1, 2], [3, 4], [5, 6]]
    assert chunk_list([1, 2, 3, 4, 5, 6], 3) == [[1, 2, 3], [4, 5, 6]]


def test_r2_short_last_chunk():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk_list([1, 2, 3], 5) == [[1, 2, 3]]


def test_r3_size_below_one_raises_value_error():
    with pytest.raises(ValueError):
        chunk_list([1, 2, 3], 0)
    with pytest.raises(ValueError):
        chunk_list([1, 2, 3], -1)


def test_r4_empty_items():
    assert chunk_list([], 3) == []


def test_r5_input_not_modified():
    items = [1, 2, 3, 4, 5]
    original = list(items)
    chunk_list(items, 2)
    assert items == original

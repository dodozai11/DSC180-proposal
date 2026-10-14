import copy

import pytest

from solution import merge_intervals


def merged(intervals):
    """Call the solution and turn each interval into a list, so tuples vs lists doesn't matter."""
    return [list(item) for item in merge_intervals(intervals)]


def test_r1_overlapping_merge():
    assert merged([[1, 4], [2, 6]]) == [[1, 6]]
    assert merged([[1, 10], [2, 3]]) == [[1, 10]]


def test_r2_touching_merge():
    assert merged([[1, 3], [3, 5]]) == [[1, 5]]


def test_r3_unsorted_input_sorted_output():
    assert merged([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]
    assert merged([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]


def test_r4_input_not_modified():
    intervals = [[8, 10], [1, 3], [2, 6]]
    original = copy.deepcopy(intervals)
    merge_intervals(intervals)
    assert intervals == original


def test_r5_empty_input():
    assert merged([]) == []


def test_r6_start_after_end_raises_value_error():
    with pytest.raises(ValueError):
        merge_intervals([[5, 1]])
    with pytest.raises(ValueError):
        merge_intervals([[1, 2], [4, 3]])

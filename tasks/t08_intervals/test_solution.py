import copy

import pytest

from solution import merge_intervals


def test_r1_overlapping_merge():
    assert merge_intervals([[1, 4], [2, 6]]) == [[1, 6]]
    assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]


def test_r2_touching_merge():
    assert merge_intervals([[1, 3], [3, 5]]) == [[1, 5]]


def test_r3_unsorted_input_sorted_output():
    assert merge_intervals([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]
    assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]


def test_r4_input_not_modified():
    intervals = [[8, 10], [1, 3], [2, 6]]
    original = copy.deepcopy(intervals)
    merge_intervals(intervals)
    assert intervals == original


def test_r5_empty_input():
    assert merge_intervals([]) == []


def test_r6_start_after_end_raises_value_error():
    with pytest.raises(ValueError):
        merge_intervals([[5, 1]])
    with pytest.raises(ValueError):
        merge_intervals([[1, 2], [4, 3]])

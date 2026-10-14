import pytest

from solution import normalize_whitespace


def test_r1_collapse_spaces():
    assert normalize_whitespace("a   b c") == "a b c"
    assert normalize_whitespace("one  two") == "one two"


def test_r2_tabs_and_newlines():
    assert normalize_whitespace("a\tb\nc") == "a b c"
    assert normalize_whitespace("a \t\n b") == "a b"


def test_r3_strip_ends():
    assert normalize_whitespace("  a b  ") == "a b"
    assert normalize_whitespace(" x") == "x"


def test_r4_non_string_raises_type_error():
    with pytest.raises(TypeError):
        normalize_whitespace(5)
    with pytest.raises(TypeError):
        normalize_whitespace(None)

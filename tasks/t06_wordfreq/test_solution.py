from solution import word_frequencies


def test_r1_case_insensitive_lowercase():
    assert word_frequencies("The the THE") == [("the", 3)]


def test_r2_punctuation_removed():
    assert word_frequencies("hi! hi. hi,") == [("hi", 3)]
    assert word_frequencies("yes? yes; yes:") == [("yes", 3)]


def test_r3_list_of_tuples():
    result = word_frequencies("a b")
    assert isinstance(result, list)
    assert all(isinstance(item, tuple) for item in result)
    assert result == [("a", 1), ("b", 1)]


def test_r4_sorted_by_count_then_alphabetically():
    assert word_frequencies("b a b c a") == [("a", 2), ("b", 2), ("c", 1)]
    assert word_frequencies("z y y") == [("y", 2), ("z", 1)]


def test_r5_top_k():
    assert word_frequencies("b a b c a", top_k=2) == [("a", 2), ("b", 2)]
    assert word_frequencies("b a b c a", top_k=None) == [("a", 2), ("b", 2), ("c", 1)]


def test_r6_empty_text():
    assert word_frequencies("") == []
    assert word_frequencies("   ") == []

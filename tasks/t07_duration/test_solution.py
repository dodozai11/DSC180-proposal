import pytest

from solution import parse_duration


def test_r1_full_duration():
    assert parse_duration("1h30m15s") == 5415
    assert parse_duration("2h0m3s") == 7203


def test_r2_units_optional():
    assert parse_duration("45m") == 2700
    assert parse_duration("2h5s") == 7205
    assert parse_duration("10s") == 10


def test_r3_case_insensitive():
    assert parse_duration("1H2M") == 3720
    assert parse_duration("3S") == 3


def test_r4_spaces_ignored():
    assert parse_duration("1h 30m") == 5400
    assert parse_duration(" 2 h 5 s ") == 7205


def test_r5_invalid_format_raises_value_error():
    for bad in ["abc", "5x", "5", "m5", "30m1h"]:
        with pytest.raises(ValueError):
            parse_duration(bad)


def test_r6_empty_raises_value_error():
    with pytest.raises(ValueError):
        parse_duration("")
    with pytest.raises(ValueError):
        parse_duration("   ")

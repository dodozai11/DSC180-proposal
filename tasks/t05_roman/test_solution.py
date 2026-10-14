import pytest

from solution import roman_to_int


def test_r1_basic_symbols_add():
    assert roman_to_int("III") == 3
    assert roman_to_int("LVIII") == 58
    assert roman_to_int("MDCLXVI") == 1666


def test_r2_subtractive_pairs():
    assert roman_to_int("IV") == 4
    assert roman_to_int("IX") == 9
    assert roman_to_int("XL") == 40
    assert roman_to_int("XC") == 90
    assert roman_to_int("CD") == 400
    assert roman_to_int("CM") == 900
    assert roman_to_int("MCMXCIV") == 1994


def test_r3_lowercase_accepted():
    assert roman_to_int("xiv") == 14
    assert roman_to_int("mcmxciv") == 1994


def test_r4_invalid_character_raises_value_error():
    with pytest.raises(ValueError):
        roman_to_int("XIZ")
    with pytest.raises(ValueError):
        roman_to_int("12")


def test_r5_empty_raises_value_error():
    with pytest.raises(ValueError):
        roman_to_int("")

from solution import validate_password

# Meets every requirement. Each test changes only the one thing it checks.
# Results are checked for truthiness (not `is True`), so returning e.g. a regex match
# object instead of a bool doesn't fail every test at once; each test checks one rule.
VALID = "Abcdef1!"


def test_r1_min_length():
    assert validate_password(VALID)
    assert not validate_password("Abcd1!x")  # 7 characters


def test_r2_uppercase():
    assert not validate_password("abcdef1!")


def test_r3_lowercase():
    assert not validate_password("ABCDEF1!")


def test_r4_digit():
    assert not validate_password("Abcdefg!")


def test_r5_special_character():
    assert not validate_password("Abcdefg1")
    assert not validate_password("Abcdef1&")  # & is not in the allowed set


def test_r6_no_spaces():
    assert not validate_password("Abc def1!")

from solution import validate_password

# Meets every requirement. Each test changes only the one thing it checks.
VALID = "Abcdef1!"


def test_r1_min_length():
    assert validate_password(VALID) is True
    assert validate_password("Abcd1!x") is False  # 7 characters


def test_r2_uppercase():
    assert validate_password("abcdef1!") is False


def test_r3_lowercase():
    assert validate_password("ABCDEF1!") is False


def test_r4_digit():
    assert validate_password("Abcdefg!") is False


def test_r5_special_character():
    assert validate_password("Abcdefg1") is False
    assert validate_password("Abcdef1&") is False  # & is not in the allowed set


def test_r6_no_spaces():
    assert validate_password("Abc def1!") is False

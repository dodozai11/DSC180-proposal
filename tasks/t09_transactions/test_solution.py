from solution import summarize_transactions


def row(date, category, amount):
    return {"date": date, "category": category, "amount": amount}


def test_r1_totals_per_category():
    rows = [row("2024-01-01", "food", 10), row("2024-01-02", "rent", 500), row("2024-01-03", "food", 5)]
    assert summarize_transactions(rows) == {"food": 15, "rent": 500}


def test_r2_categories_case_insensitive():
    rows = [row("2024-01-01", "Food", 10), row("2024-01-02", "food", 5), row("2024-01-03", "FOOD", 1)]
    assert summarize_transactions(rows) == {"food": 16}


def test_r3_negative_amounts_subtract():
    rows = [row("2024-01-01", "food", 10), row("2024-01-02", "food", -3)]
    assert summarize_transactions(rows) == {"food": 7}


def test_r4_rounded_to_two_decimals():
    rows = [row("2024-01-01", "food", 0.1), row("2024-01-02", "food", 0.2)]
    assert summarize_transactions(rows) == {"food": 0.3}
    rows = [row("2024-01-01", "fun", 1.005), row("2024-01-02", "fun", 2.111)]
    assert summarize_transactions(rows)["fun"] == round(1.005 + 2.111, 2)


def test_r5_rows_without_amount_skipped():
    rows = [row("2024-01-01", "food", 10), {"date": "2024-01-02", "category": "food"}]
    assert summarize_transactions(rows) == {"food": 10}


def test_r6_inclusive_date_filter():
    rows = [
        row("2024-01-01", "food", 1),
        row("2024-01-05", "food", 10),
        row("2024-01-10", "food", 100),
        row("2024-01-15", "food", 1000),
    ]
    assert summarize_transactions(rows, "2024-01-05", "2024-01-10") == {"food": 110}
    assert summarize_transactions(rows, start_date="2024-01-10") == {"food": 1100}
    assert summarize_transactions(rows, end_date="2024-01-05") == {"food": 11}


def test_r7_no_rows_empty_dict():
    assert summarize_transactions([]) == {}

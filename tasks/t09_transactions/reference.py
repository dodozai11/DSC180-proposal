def summarize_transactions(rows, start_date=None, end_date=None):
    totals = {}
    for row in rows:
        if "amount" not in row:
            continue
        if start_date is not None and row["date"] < start_date:
            continue
        if end_date is not None and row["date"] > end_date:
            continue
        category = row["category"].lower()
        totals[category] = totals.get(category, 0) + row["amount"]
    return {category: round(total, 2) for category, total in totals.items()}

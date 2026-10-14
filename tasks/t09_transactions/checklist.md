# t09_transactions checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | dict category -> sum of amounts | Requirements #1 | "a dict that maps each category to the sum of its amounts" | `test_r1_totals_per_category` |
| R2 | categories case-insensitive, lowercase keys | Requirements #2 | "categories should be case-insensitive, so use the lowercase category name as the key and \"Food\" and \"food\" get combined" | `test_r2_categories_case_insensitive` |
| R3 | negative amounts reduce total | Requirements #3 | "Refunds show up as negative amounts and they should reduce the category total" | `test_r3_negative_amounts_subtract` |
| R4 | round totals to 2 dp | Requirements #4 | "with each total rounded to 2 decimal places" | `test_r4_rounded_to_two_decimals` |
| R5 | skip rows with no "amount" key | Requirements #5 | "Some rows don't have an \"amount\" key at all, so just skip those" | `test_r5_rows_without_amount_skipped` |
| R6 | inclusive start/end date filter, either may be None | Requirements #6 | "only count rows whose date is between them, inclusive, and either one can be None, meaning no limit on that side" | `test_r6_inclusive_date_filter` |
| R7 | nothing to include -> {} | Requirements #7 | "If there are no rows to include, just return an empty dict" | `test_r7_no_rows_empty_dict` |

Shared by both prompts: name and signature `summarize_transactions(rows, start_date=None, end_date=None)`, row keys "date"/"category"/"amount", date format "YYYY-MM-DD", "only Python code".

# t05_roman checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | symbols I..M add up | Requirements #1 | "The symbols are I=1, V=5, ... M=1000, and normally you add their values up, like \"LVIII\" = 58" | `test_r1_basic_symbols_add` |
| R2 | subtractive pairs | Requirements #2 | "handle the subtractive pairs IV=4, IX=9, XL=40, XC=90, CD=400, CM=900, so \"MCMXCIV\" = 1994" | `test_r2_subtractive_pairs` |
| R3 | lowercase accepted | Requirements #3 | "Lowercase should work just like uppercase, so \"xiv\" gives 14" | `test_r3_lowercase_accepted` |
| R4 | invalid char -> ValueError | Requirements #4 | "raise a ValueError if it has any character that isn't a Roman numeral symbol" | `test_r4_invalid_character_raises_value_error` |
| R5 | empty -> ValueError | Requirements #5 | "If the string is empty, raise a ValueError" | `test_r5_empty_raises_value_error` |

Shared by both prompts: name `roman_to_int(s)`, returns an integer, "only Python code".

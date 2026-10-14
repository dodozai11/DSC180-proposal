# t07_duration checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | numbers + h/m/s units -> total int seconds | Requirements #1 | "made of whole numbers that are each followed by a unit, h for hours, m for minutes, s for seconds" + "total number of seconds as an integer ... 5415" | `test_r1_full_duration` |
| R2 | units optional, order h, m, s | Requirements #2 | "Any of the units can be left out, but whichever ones are there have to come in the order h, m, s" | `test_r2_units_optional` |
| R3 | case-insensitive units | Requirements #3 | "the units are case-insensitive, so \"1H2M\" = 3720" | `test_r3_case_insensitive` |
| R4 | spaces ignored anywhere | Requirements #4 | "Spaces can show up anywhere and should just be ignored" | `test_r4_spaces_ignored` |
| R5 | bad format -> ValueError (5x, 5, m5, 30m1h) | Requirements #5 | "Anything else that doesn't follow that format should raise a ValueError too ..." | `test_r5_invalid_format_raises_value_error` |
| R6 | empty/only spaces -> ValueError | Requirements #6 | "If the string is empty or only spaces, raise a ValueError" | `test_r6_empty_raises_value_error` |

Shared by both prompts: name `parse_duration(s)`, returns int seconds, same worked examples, "only Python code".

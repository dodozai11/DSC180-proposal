# t04_password checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | length >= 8 | Requirements #1 | "it has to be at least 8 characters long" | `test_r1_min_length` |
| R2 | >= 1 uppercase | Requirements #2 | "at least one uppercase letter" | `test_r2_uppercase` |
| R3 | >= 1 lowercase | Requirements #3 | "at least one lowercase letter" | `test_r3_lowercase` |
| R4 | >= 1 digit | Requirements #4 | "it needs at least one digit in there somewhere" | `test_r4_digit` |
| R5 | >= 1 of ! @ # $ % | Requirements #5 | "at least one of the special characters ! @ # $ or %" | `test_r5_special_character` |
| R6 | no spaces | Requirements #6 | "Spaces aren't allowed anywhere in it" | `test_r6_no_spaces` |

Shared by both prompts: name `validate_password(pw)`, returns True/False, "only Python code".

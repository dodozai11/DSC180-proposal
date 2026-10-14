# t02_whitespace checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | runs of 2+ spaces -> one space | Requirements #1 | "wherever there are two or more spaces in a row, I just want one space" | `test_r1_collapse_spaces` |
| R2 | tabs/newlines count as whitespace | Requirements #2 | "tabs and newlines count as whitespace too, so any mix ... should end up as a single space" | `test_r2_tabs_and_newlines` |
| R3 | strip start and end | Requirements #3 | "There shouldn't be any whitespace left hanging at the start or the end" | `test_r3_strip_ends` |
| R4 | non-string -> TypeError | Requirements #4 | "if someone passes in something that isn't a string, it should raise a TypeError" | `test_r4_non_string_raises_type_error` |

Shared by both prompts: name `normalize_whitespace(text)`, returns a string, "only Python code".

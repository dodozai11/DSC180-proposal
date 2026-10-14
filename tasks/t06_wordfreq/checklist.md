# t06_wordfreq checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | case-insensitive, lowercase words | Requirements #1 | "treat words case-insensitively, reporting each one in lowercase, so \"The\" and \"the\" are the same word" | `test_r1_case_insensitive_lowercase` |
| R2 | remove . , ! ? ; : | Requirements #2 | "strip out these punctuation characters: . , ! ? ; :" | `test_r2_punctuation_removed` |
| R3 | list of (word, count) tuples | Requirements #3 | "Give me the results back as a list of (word, count) tuples" | `test_r3_list_of_tuples` |
| R4 | count desc, ties alphabetical | Requirements #4 | "sorted by count from highest to lowest, and when two words tie, put them in alphabetical order" | `test_r4_sorted_by_count_then_alphabetically` |
| R5 | top_k limits; None = all | Requirements #5 | "If I pass a top_k, only give me the first top_k items ... if it's None, give me all of them" | `test_r5_top_k` |
| R6 | empty/whitespace text -> [] | Requirements #6 | "If the text is empty or only whitespace, I just want an empty list" | `test_r6_empty_text` |

Shared by both prompts: name `word_frequencies(text, top_k=None)`, words separated by whitespace, "only Python code".

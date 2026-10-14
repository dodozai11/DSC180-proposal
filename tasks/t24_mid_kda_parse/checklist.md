# t24_mid_kda_parse checklist

Parse-mode twin of t14_mid_kda: same data and questions, but each function receives the data as text.
Answers are not graded; the tests only check that the code runs on new data and doesn't copy the data.

| Item | Requirement | Structured | Informal |
|---|---|---|---|
| P | functions take `text` and parse it; don't hard-code the data | Requirements #1 | "Each function should take the data as one string called text, in exactly the format above but possibly with different rows, and parse it itself instead of hard-coding the data." |
| Q1 | `q1(text)` returns the name of the player with the highest KDA, where KDA = (kills + assists) / max(deaths, 1). | Requirements #2 | describes q1(text) in the paragraph |
| Q2 | `q2(text)` returns the average number of kills in games that were won. | Requirements #3 | describes q2(text) in the paragraph |
| Q3 | `q3(text)` returns the name of the most-picked champion. | Requirements #4 | describes q3(text) in the paragraph |

Shared by both: the same intro line, the same {DATA} block in the same position, "only Python code".

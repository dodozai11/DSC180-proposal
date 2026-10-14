# t01_fizzbuzz checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | list of str for 1..n in order; other numbers as strings | Requirements #1 | "a list of strings, one for each number from 1 up to n in order" + "Every other number just shows up as itself written as a string" | `test_r1_numbers_as_strings` |
| R2 | multiple of 3 (not 5) -> "Fizz" | Requirements #2 | "a multiple of 3 (but not 5) should say \"Fizz\"" | `test_r2_fizz` |
| R3 | multiple of 5 (not 3) -> "Buzz" | Requirements #3 | "a multiple of 5 (but not 3) should say \"Buzz\"" | `test_r3_buzz` |
| R4 | multiple of 3 and 5 -> "FizzBuzz" | Requirements #4 | "a multiple of both 3 and 5 should say \"FizzBuzz\"" | `test_r4_fizzbuzz` |
| R5 | n <= 0 -> [] | Requirements #5 | "If n is 0 or negative, just give me an empty list" | `test_r5_non_positive_n` |

Shared by both prompts: name `fizzbuzz_list(n)`, returns a list of strings, "only Python code".

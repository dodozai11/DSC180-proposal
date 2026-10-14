# t08_intervals checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | overlapping intervals merge | Requirements #1 | "merging intervals that overlap into one, so [[1, 4], [2, 6]] becomes [[1, 6]]" | `test_r1_overlapping_merge` |
| R2 | touching intervals merge | Requirements #2 | "intervals that only touch should merge too, so [[1, 3], [3, 5]] becomes [[1, 5]]" | `test_r2_touching_merge` |
| R3 | any input order; output sorted by start | Requirements #3 | "The input might come in any order, but the result needs to be sorted by start value" | `test_r3_unsorted_input_sorted_output` |
| R4 | input list not modified | Requirements #4 | "Please don't modify the input list or the intervals inside it" | `test_r4_input_not_modified` |
| R5 | empty -> [] | Requirements #5 | "an empty input should just return an empty list" | `test_r5_empty_input` |
| R6 | start > end -> ValueError | Requirements #6 | "If any interval has a start greater than its end, raise a ValueError" | `test_r6_start_after_end_raises_value_error` |

Shared by both prompts: name `merge_intervals(intervals)`, input/output are lists of [start, end] lists of ints, "only Python code".

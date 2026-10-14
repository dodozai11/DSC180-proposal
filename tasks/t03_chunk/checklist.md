# t03_chunk checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | consecutive chunks of `size`, in order, each a list | Requirements #1 | "split items into consecutive chunks of length size, keeping everything in the original order ... each chunk is its own list" | `test_r1_consecutive_chunks_in_order` |
| R2 | last chunk holds leftovers | Requirements #2 | "the last chunk just holds whatever's left over, so it's shorter" | `test_r2_short_last_chunk` |
| R3 | size < 1 -> ValueError | Requirements #3 | "If size is less than 1 it should raise a ValueError" | `test_r3_size_below_one_raises_value_error` |
| R4 | empty items -> [] | Requirements #4 | "an empty items list should give back an empty list" | `test_r4_empty_items` |
| R5 | input list not modified | Requirements #5 | "make sure it doesn't change the items list I pass in" | `test_r5_input_not_modified` |

Shared by both prompts: name `chunk_list(items, size)`, returns a list of lists, "only Python code".

# t10_lru_cache checklist

| Req | Requirement | Structured | Informal (quoted phrase) | Test |
|---|---|---|---|---|
| R1 | get missing -> -1 | Requirements #1 | "get(key) should return -1 if the key isn't in the cache" | `test_r1_get_missing_returns_minus_one` |
| R2 | put stores; get returns it | Requirements #2 | "With put(key, value) the value gets stored and a later get(key) returns it" | `test_r2_put_then_get` |
| R3 | evict LRU when over capacity | Requirements #3 | "When a put adds a new key and the cache would go over capacity, remove the least recently used key first" | `test_r3_evicts_least_recently_used` |
| R4 | get refreshes recency | Requirements #4 | "a successful get(key) does count as using it, so it becomes the most recently used" | `test_r4_get_refreshes_recency` |
| R5 | put existing updates value + recency, no eviction | Requirements #5 | "If you put a key that's already there, just update its value and count that as using it, and it shouldn't evict anything" | `test_r5_put_existing_updates_value_and_recency` |
| R6 | capacity < 1 -> ValueError | Requirements #6 | "If capacity is less than 1, raise a ValueError when the cache is created" | `test_r6_capacity_below_one_raises_value_error` |
| R7 | len(cache) = item count | Requirements #7 | "len(cache) should return how many items are currently stored" | `test_r7_len` |
| R8 | `in` returns bool, no recency change | Requirements #8 | "Doing key in cache should give True or False but it doesn't count as using the key" | `test_r8_contains_does_not_refresh_recency` |

Shared by both prompts: class name `LRUCache(capacity)`, methods `get`/`put`, `len()` and `in` support, "only Python code".

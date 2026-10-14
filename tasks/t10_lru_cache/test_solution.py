import pytest

from solution import LRUCache


def test_r1_get_missing_returns_minus_one():
    cache = LRUCache(2)
    assert cache.get("missing") == -1


def test_r2_put_then_get():
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.get(1) == "a"
    assert cache.get(2) == "b"


def test_r3_evicts_least_recently_used():
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")
    assert cache.get(1) == -1
    assert cache.get(2) == "b"
    assert cache.get(3) == "c"


def test_r4_get_refreshes_recency():
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.get(1)          # 1 is now the most recently used, so 2 is evicted next
    cache.put(3, "c")
    assert cache.get(2) == -1
    assert cache.get(1) == "a"


def test_r5_put_existing_updates_value_and_recency():
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(1, "z")     # update, no eviction; 1 is now most recently used
    cache.put(3, "c")
    assert cache.get(2) == -1
    assert cache.get(1) == "z"
    single = LRUCache(1)
    single.put(1, "a")
    single.put(1, "b")
    assert single.get(1) == "b"


def test_r6_capacity_below_one_raises_value_error():
    with pytest.raises(ValueError):
        LRUCache(0)
    with pytest.raises(ValueError):
        LRUCache(-5)


def test_r7_len():
    cache = LRUCache(3)
    assert len(cache) == 0
    cache.put(1, "a")
    cache.put(2, "b")
    assert len(cache) == 2
    cache.put(1, "c")
    assert len(cache) == 2


def test_r8_contains_does_not_refresh_recency():
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    assert (1 in cache) is True
    assert (3 in cache) is False
    cache.put(3, "c")     # `in` did not refresh 1, so 1 is still the least recently used
    assert (1 in cache) is False
    assert (2 in cache) is True

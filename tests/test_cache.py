"""Tests for the hand-written LRU cache."""

import pytest

from leafcheck.cache import LRUCache


def test_get_after_put_returns_value() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    assert cache.get("a") == 1


def test_get_missing_key_returns_none() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    assert cache.get("nope") is None


def test_evicts_least_recently_used_when_full() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # "a" is oldest -> evicted
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_get_refreshes_recency() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # now "b" is the least recently used
    cache.put("c", 3)
    assert "a" in cache
    assert "b" not in cache
    assert cache.keys_newest_first() == ["c", "a"]


def test_size_never_exceeds_capacity() -> None:
    cache: LRUCache[int, int] = LRUCache(3)
    for i in range(100):
        cache.put(i, i)
        assert len(cache) <= 3
    assert cache.keys_newest_first() == [99, 98, 97]


def test_updating_existing_key_changes_value_and_keeps_size() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)  # update, also makes "a" most recent
    assert len(cache) == 2
    assert cache.get("a") == 10
    cache.put("c", 3)  # evicts "b", not "a"
    assert "b" not in cache
    assert cache.get("a") == 10


def test_capacity_one() -> None:
    cache: LRUCache[str, int] = LRUCache(1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") is None
    assert cache.get("b") == 2


def test_invalid_capacity_raises() -> None:
    with pytest.raises(ValueError):
        LRUCache(0)

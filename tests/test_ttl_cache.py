"""Tests for the TTL cache (uses a fake clock, never sleeps)."""

import threading

import pytest

from leafcheck.cache import LRUCache
from leafcheck.ttl_cache import TTLCache


class FakeClock:
    """A clock the test can move forward by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_get_before_expiry_returns_value() -> None:
    clock = FakeClock()
    cache: TTLCache[str, int] = TTLCache(4, ttl_seconds=10, clock=clock)
    cache.put("a", 1)
    clock.now += 9.9
    assert cache.get("a") == 1


def test_get_after_expiry_returns_none_and_drops_entry() -> None:
    clock = FakeClock()
    cache: TTLCache[str, int] = TTLCache(4, ttl_seconds=10, clock=clock)
    cache.put("a", 1)
    clock.now += 10
    assert cache.get("a") is None
    assert len(cache) == 0


def test_missing_key_returns_none() -> None:
    cache: TTLCache[str, int] = TTLCache(4, ttl_seconds=10, clock=FakeClock())
    assert cache.get("nope") is None


def test_put_again_resets_expiry() -> None:
    clock = FakeClock()
    cache: TTLCache[str, int] = TTLCache(4, ttl_seconds=10, clock=clock)
    cache.put("a", 1)
    clock.now += 8
    cache.put("a", 2)
    clock.now += 8
    assert cache.get("a") == 2


def test_lru_eviction_still_applies() -> None:
    cache: TTLCache[str, int] = TTLCache(2, ttl_seconds=100, clock=FakeClock())
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert len(cache) == 2


def test_clear_empties_cache() -> None:
    cache: TTLCache[str, int] = TTLCache(2, ttl_seconds=100, clock=FakeClock())
    cache.put("a", 1)
    cache.clear()
    assert len(cache) == 0 and cache.get("a") is None


def test_rejects_non_positive_ttl() -> None:
    with pytest.raises(ValueError):
        TTLCache(2, ttl_seconds=0)


def test_concurrent_puts_keep_size_bounded() -> None:
    cache: TTLCache[int, int] = TTLCache(50, ttl_seconds=100)

    def worker(start: int) -> None:
        for i in range(start, start + 200):
            cache.put(i, i)
            cache.get(i - 1)

    threads = [threading.Thread(target=worker, args=(k * 1000,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(cache) == 50


def test_lru_remove() -> None:
    lru: LRUCache[str, int] = LRUCache(2)
    lru.put("a", 1)
    assert lru.remove("a") is True
    assert lru.remove("a") is False
    assert len(lru) == 0 and lru.keys_newest_first() == []

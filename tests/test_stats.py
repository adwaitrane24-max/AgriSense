"""Tests for hit/miss counters and timing averages."""

from leafcheck.stats import Stats


def test_empty_stats() -> None:
    snap = Stats().snapshot()
    assert snap["requests"] == 0
    assert snap["hit_rate"] == 0.0
    assert snap["avg_hit_ms"] is None
    assert snap["avg_miss_ms"] is None


def test_counts_and_averages() -> None:
    stats = Stats()
    stats.record(hit=False, elapsed_ms=300.0)
    stats.record(hit=False, elapsed_ms=500.0)
    stats.record(hit=True, elapsed_ms=2.0)
    snap = stats.snapshot()
    assert (snap["hits"], snap["misses"], snap["requests"]) == (1, 2, 3)
    assert snap["hit_rate"] == round(1 / 3, 4)
    assert snap["avg_miss_ms"] == 400.0
    assert snap["avg_hit_ms"] == 2.0

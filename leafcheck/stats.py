"""Counters for the /stats page: cache hits vs misses and response times.

We keep running totals (count + total milliseconds) instead of a list of every
timing, so memory stays constant no matter how many requests arrive.

Complexity: record O(1) time, snapshot O(1) time, O(1) space.
"""

import threading
from typing import Any


class Stats:
    """Thread-safe request statistics (Flask may serve requests in threads)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self._hit_ms_total = 0.0
        self._miss_ms_total = 0.0

    def record(self, hit: bool, elapsed_ms: float) -> None:
        """Add one served prediction. O(1)."""
        with self._lock:
            if hit:
                self.hits += 1
                self._hit_ms_total += elapsed_ms
            else:
                self.misses += 1
                self._miss_ms_total += elapsed_ms

    def snapshot(self) -> dict[str, Any]:
        """Current totals, hit rate and average times. O(1)."""
        with self._lock:
            total = self.hits + self.misses
            return {
                "requests": total,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "avg_hit_ms": round(self._hit_ms_total / self.hits, 2) if self.hits else None,
                "avg_miss_ms": round(self._miss_ms_total / self.misses, 2) if self.misses else None,
            }

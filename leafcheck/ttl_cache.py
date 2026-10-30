"""Time-to-live (TTL) cache built on top of the hand-written LRU cache.

Why: weather and soil data come from slow web APIs. Weather changes during the
day, so a stored answer must expire; soil changes almost never. A TTL cache
remembers an answer for `ttl_seconds` and then forgets it.

Design: reuse `LRUCache` and store `(expires_at, value)` as the value.
    - size is still bounded by LRU eviction,
    - expiry is checked lazily on `get` (an expired entry is dropped then).

The clock is injectable so tests can move time forward without sleeping.
A `threading.Lock` makes get/put safe when Flask serves several requests at once.

Complexity (n = capacity):
    get     O(1) average time
    put     O(1) average time
    space   O(n)
"""

import threading
import time
from typing import Callable, Generic, Hashable, Optional, TypeVar

from leafcheck.cache import LRUCache

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """LRU cache whose entries also expire `ttl_seconds` after being stored."""

    def __init__(
        self,
        capacity: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty cache. O(1).

        Args:
            capacity: maximum number of entries (LRU eviction beyond this).
            ttl_seconds: how long an entry stays valid; must be > 0.
            clock: function returning the current time in seconds.
        """
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._lru: LRUCache[K, tuple[float, V]] = LRUCache(capacity)
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()

    def get(self, key: K) -> Optional[V]:
        """Return the value for `key`, or None if missing or expired. O(1) average.

        An expired entry is removed so it stops taking up space.
        """
        with self._lock:
            entry = self._lru.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                self._lru.remove(key)
                return None
            return value

    def put(self, key: K, value: V) -> None:
        """Store `value` under `key`, valid for `ttl_seconds` from now. O(1) average."""
        with self._lock:
            self._lru.put(key, (self._clock() + self._ttl, value))

    def clear(self) -> None:
        """Forget every entry. O(1) (old nodes are garbage collected)."""
        with self._lock:
            self._lru = LRUCache(self._lru.capacity)

    def __len__(self) -> int:
        """Number of stored entries, including expired ones not yet dropped. O(1)."""
        return len(self._lru)

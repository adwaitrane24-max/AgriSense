"""Hand-written LRU (Least Recently Used) cache.

Why: running the CNN costs hundreds of milliseconds; looking up a stored result
costs microseconds. When the same image is uploaded again we return the saved
answer instead of predicting again.

Design: a hash map (Python dict) + a doubly linked list.

    dict:  key -> node                 (find any node in O(1))
    list:  head <-> newest ... oldest <-> tail
           (move / remove any node in O(1), because each node knows
            both its neighbours)

`head` and `tail` are sentinel (dummy) nodes, so we never have to check for
"is the list empty?" or "is this the first node?" when linking.

Complexity (n = capacity):
    get     O(1) average time (dict lookup + 2 pointer moves)
    put     O(1) average time (dict insert + pointer moves + maybe 1 eviction)
    space   O(n) for n nodes plus the dict entries
"""

from typing import Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class _Node(Generic[K, V]):
    """One entry in the doubly linked list."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Optional[K] = None, value: Optional[V] = None) -> None:
        self.key = key
        self.value = value
        self.prev: Optional["_Node[K, V]"] = None
        self.next: Optional["_Node[K, V]"] = None


class LRUCache(Generic[K, V]):
    """Fixed-size cache that evicts the least recently used entry when full."""

    def __init__(self, capacity: int) -> None:
        """Create an empty cache. O(1).

        Args:
            capacity: maximum number of entries; must be at least 1.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._map: dict[K, _Node[K, V]] = {}
        # Sentinels: head.next is the most recent, tail.prev the least recent.
        self._head: _Node[K, V] = _Node()
        self._tail: _Node[K, V] = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    # ----- private list helpers (each O(1)) -----

    def _unlink(self, node: _Node[K, V]) -> None:
        """Remove `node` from the list by joining its two neighbours."""
        node.prev.next = node.next  # type: ignore[union-attr]
        node.next.prev = node.prev  # type: ignore[union-attr]

    def _push_front(self, node: _Node[K, V]) -> None:
        """Insert `node` right after head, marking it most recently used."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node  # type: ignore[union-attr]
        self._head.next = node

    # ----- public API -----

    def get(self, key: K) -> Optional[V]:
        """Return the value for `key`, or None if absent. O(1) average.

        A hit also marks the entry as most recently used.
        """
        node = self._map.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key: K, value: V) -> None:
        """Insert or update `key`. O(1) average.

        If the cache is full and `key` is new, the least recently used entry
        (the node just before tail) is evicted first.
        """
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return

        if len(self._map) >= self._capacity:
            lru = self._tail.prev
            self._unlink(lru)  # type: ignore[arg-type]
            del self._map[lru.key]  # type: ignore[union-attr, index]

        node = _Node(key, value)
        self._map[key] = node
        self._push_front(node)

    def keys_newest_first(self) -> list[K]:
        """Keys ordered from most to least recently used. O(n); for tests/debug."""
        keys: list[K] = []
        node = self._head.next
        while node is not self._tail:
            keys.append(node.key)  # type: ignore[union-attr, arg-type]
            node = node.next  # type: ignore[union-attr]
        return keys

    def __contains__(self, key: object) -> bool:
        """Membership check without changing recency. O(1) average."""
        return key in self._map

    def __len__(self) -> int:
        """Number of stored entries. O(1)."""
        return len(self._map)

    @property
    def capacity(self) -> int:
        """Maximum number of entries."""
        return self._capacity

"""Dedup store: abstraction for "have we seen this event id recently?".

The ingestor calls `add_if_new(event_id)` once per event. True = first
time seen (process it); False = already seen (skip it).

Why a store and not a set on the ingestor:
- Swapping in a durable store (SQLite, Redis) is a config change.
- Testing different bounded/unbounded behaviors is trivial.
- The ingestor stays single-purpose: pull -> handle -> commit.

Two implementations ship in-repo:
- `InMemoryDedupStore` - bounded LRU (default; same as before)
- `SQLiteDedupStore`  - durable on local disk; TTL-pruned
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Protocol, runtime_checkable


@runtime_checkable
class DedupStore(Protocol):
    def add_if_new(self, event_id: str) -> bool:  # pragma: no cover - protocol
        """Return True if `event_id` is new (caller should process);
        False if it was already recorded.
        """
        ...

    def close(self) -> None:  # pragma: no cover - protocol
        ...


class InMemoryDedupStore:
    """Bounded LRU. Fast, no IO, but state is lost on restart.

    Bounded by `max_size`. On overflow, the oldest entry is evicted.
    Eviction means the id may be treated as "new" if it arrives again
    after the eviction window - which is safe because the ingestion
    handler is idempotent.
    """

    def __init__(self, *, max_size: int = 1024) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._seen: OrderedDict[str, None] = OrderedDict()

    def add_if_new(self, event_id: str) -> bool:
        if event_id in self._seen:
            # Refresh recency on hit (LRU semantics).
            self._seen.move_to_end(event_id)
            return False
        self._seen[event_id] = None
        while len(self._seen) > self._max_size:
            self._seen.popitem(last=False)
        return True

    def close(self) -> None:
        self._seen.clear()

    # ---- introspection (tests) ----

    @property
    def size(self) -> int:
        return len(self._seen)

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._seen

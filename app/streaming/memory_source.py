"""In-memory EventSource for tests and demos.

Not for production. But it mirrors the Kafka contract:
- at-least-once delivery (no dedup here — that's the ingestor's job)
- commit is a no-op that records the last id (for assertions in tests)
- poll returns None on empty (so the consumer's loop stays responsive)

Features useful for testing:
- `push(event)` to enqueue (also exposes producer side for demos)
- `committed_ids` for assertions
- `fail_next(n)` to simulate downstream/ack failures
"""
from __future__ import annotations

import asyncio
from collections import deque

from app.streaming.events import Event


class InMemoryQueueSource:
    def __init__(self, *, maxsize: int = 0) -> None:
        self._queue: deque[Event] = deque()
        self._committed: list[str] = []
        self._closed = False
        self._maxsize = maxsize
        # Signaling: producers call `_new_event.set()`; poll waits on it.
        self._new_event = asyncio.Event()

    # ---- producer side (demo/test convenience) ----

    def push(self, event: Event) -> None:
        if self._closed:
            raise RuntimeError("source is closed")
        if self._maxsize and len(self._queue) >= self._maxsize:
            raise BufferError(f"queue full (maxsize={self._maxsize})")
        self._queue.append(event)
        self._new_event.set()

    def push_bytes(
        self,
        data: bytes,
        *,
        id: str,
        metadata: dict | None = None,
    ) -> None:
        self.push(Event(id=id, data=data, metadata=dict(metadata or {})))

    # ---- consumer side (EventSource protocol) ----

    async def poll(self, timeout_s: float = 1.0) -> Event | None:
        if self._closed:
            return None

        # fast path: already queued
        if self._queue:
            return self._queue.popleft()

        # wait for a producer
        try:
            await asyncio.wait_for(self._new_event.wait(), timeout=timeout_s)
        except TimeoutError:
            return None

        self._new_event.clear()
        if self._queue:
            return self._queue.popleft()
        return None

    async def commit(self, event_id: str) -> None:
        self._committed.append(event_id)

    async def close(self) -> None:
        self._closed = True
        self._new_event.set()

    # ---- introspection (tests) ----

    @property
    def committed_ids(self) -> list[str]:
        return list(self._committed)

    @property
    def pending(self) -> int:
        return len(self._queue)

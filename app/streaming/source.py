"""EventSource protocol.

Pull-based, async: the consumer decides when to pull the next event,
which is how backpressure is expressed. Two methods:

- `poll(timeout_s)`: return the next event, or None on timeout/empty.
- `commit(event_id)`: mark an event as processed (offset commit / ack).

A source is expected to be idempotent-friendly:
- The same event id may be delivered more than once (at-least-once).
- `commit(event_id)` is safe to call more than once.

Whether to implement `commit` as no-op (in-memory) or a real offset
commit (Kafka) is up to the adapter.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.streaming.events import Event


@runtime_checkable
class EventSource(Protocol):
    async def poll(self, timeout_s: float = 1.0) -> Event | None:  # pragma: no cover
        ...

    async def commit(self, event_id: str) -> None:  # pragma: no cover
        ...

    async def close(self) -> None:  # pragma: no cover
        ...

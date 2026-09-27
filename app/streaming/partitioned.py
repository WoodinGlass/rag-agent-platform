"""PartitionedSource: a source that fans out into per-partition sources.

Why: some brokers (Kafka) require one consumer per partition for correct
offset tracking and rebalancing. Some systems (S3) can be sharded by a
key prefix. This abstraction lets the parallel ingestor treat both the
same way: enumerate partitions, get an EventSource for each.

The offline implementation (`InMemoryPartitionedSource`) is used in
tests and demos. Kafka and S3 would provide their own.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.streaming.events import Event
from app.streaming.memory_source import InMemoryQueueSource
from app.streaming.source import EventSource


@runtime_checkable
class PartitionedSource(Protocol):
    def partitions(self) -> list[str]:  # pragma: no cover - protocol
        """Return the current set of partition ids (may change over time)."""
        ...

    def source_for(self, partition: str) -> EventSource:  # pragma: no cover
        """Return an EventSource scoped to this partition."""
        ...


class InMemoryPartitionedSource:
    """Offline implementation. Each partition is an InMemoryQueueSource.

    Push helpers accept a partition id. Partitions are created lazily on
    first push so tests can populate them in any order.
    """

    def __init__(self) -> None:
        self._sources: dict[str, InMemoryQueueSource] = {}

    # ---- producer side (demo/test) ----

    def partition(self, name: str) -> InMemoryQueueSource:
        """Return (creating if needed) the source for a partition."""
        if name not in self._sources:
            self._sources[name] = InMemoryQueueSource()
        return self._sources[name]

    def push(self, partition: str, event: Event) -> None:
        self.partition(partition).push(event)

    def push_bytes(
        self,
        partition: str,
        data: bytes,
        *,
        id: str,
        metadata: dict | None = None,
    ) -> None:
        self.partition(partition).push_bytes(data, id=id, metadata=metadata)

    # ---- consumer side (PartitionedSource protocol) ----

    def partitions(self) -> list[str]:
        return sorted(self._sources)

    def source_for(self, partition: str) -> EventSource:
        return self.partition(partition)

    async def close(self) -> None:
        for src in self._sources.values():
            await src.close()

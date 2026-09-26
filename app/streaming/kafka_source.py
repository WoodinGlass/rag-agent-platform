"""Kafka adapter for EventSource.

Lazy import so `kafka-python` is only required when streaming_backend
is set to "kafka". Missing package -> ImportError with a clear message.

Concurrency note: `kafka-python`'s consumer is not thread-safe but is
safe to use from a single asyncio task if we offload blocking calls to
a thread. We do that via `asyncio.to_thread` so the event loop never
blocks on network I/O.

Semantics (mirror InMemoryQueueSource):
- at-least-once: `commit(event_id)` commits the offset of that message.
  We track the last seen offset per event id, so callers can commit by
  id. This is required by our EventSource protocol.
- `poll(timeout_s)` returns None on timeout, not on partition EOF, so
  the ingestor loop stays responsive.
- `close()` closes the underlying consumer.

Not production-hardened yet:
- No rebalance listener / partition assignment tracking.
- No transaction support (exactly-once) -- deliberately, since our
  contract is at-least-once plus ingestor-side dedup.
- `auto_offset_reset` defaults to "earliest" for demo friendliness.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.streaming.events import Event

log = get_logger(__name__)


@dataclass
class KafkaConfig:
    bootstrap_servers: str
    topic: str
    group_id: str = "rag-agent-platform"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    # extra kwargs passed straight to KafkaConsumer
    consumer_kwargs: dict = field(default_factory=dict)


class KafkaEventSource:
    """EventSource backed by a Kafka topic (via kafka-python)."""

    def __init__(self, config: KafkaConfig) -> None:
        try:
            from kafka import KafkaConsumer
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "kafka-python not installed; run: "
                "pip install 'rag-agent-platform[kafka]'"
            ) from e

        self._config = config
        self._consumer = KafkaConsumer(
            config.topic,
            bootstrap_servers=config.bootstrap_servers.split(","),
            group_id=config.group_id,
            auto_offset_reset=config.auto_offset_reset,
            enable_auto_commit=config.enable_auto_commit,
            **config.consumer_kwargs,
        )
        # Map event id -> (TopicPartition, offset) for id-based commit.
        self._offsets: dict[str, tuple[Any, int]] = {}
        # Buffered records from a poll() batch (FIFO).
        self._pending: deque[tuple[Any, Any]] = deque()
        self._closed = False

    async def poll(self, timeout_s: float = 1.0) -> Event | None:
        if self._closed:
            return None

        # 1) drain local buffer first — no network needed
        if self._pending:
            tp, record = self._pending.popleft()
            return self._to_event(tp, record)

        # 2) ask the consumer for more; blocking call offloaded to a thread
        timeout_ms = max(1, int(timeout_s * 1000))
        records = await asyncio.to_thread(
            self._consumer.poll, timeout_ms=timeout_ms
        )
        if not records:
            return None

        # poll returns dict[TopicPartition, list[ConsumerRecord]]
        for tp, batch in records.items():
            if not batch:
                continue
            for rec in batch:
                self._pending.append((tp, rec))

        if not self._pending:
            return None
        tp, record = self._pending.popleft()
        return self._to_event(tp, record)

    def _to_event(self, tp: Any, record: Any) -> Event:
        offset = int(record.offset)
        event_id = f"{tp.topic}:{tp.partition}:{offset}"
        self._offsets[event_id] = (tp, offset)
        raw = record.value if isinstance(record.value, bytes) else bytes(record.value)
        key = record.key.decode("utf-8", errors="replace") if record.key else None
        return Event(
            id=event_id,
            data=raw,
            metadata={
                "topic": tp.topic,
                "partition": tp.partition,
                "offset": offset,
                "key": key,
            },
        )

    async def commit(self, event_id: str) -> None:
        """Commit offset for this event id (idempotent)."""
        target = self._offsets.get(event_id)
        if target is None:
            log.debug("kafka.commit_unknown_id", extra={"event_id": event_id})
            return
        tp, offset = target
        # Commit the *next* offset (Kafka convention: offset+1).
        try:
            await asyncio.to_thread(
                self._consumer.commit,
                {tp: offset + 1},
            )
        except Exception:
            log.warning(
                "kafka.commit_failed", extra={"event_id": event_id}, exc_info=True
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.to_thread(self._consumer.close)
        except Exception:
            log.debug("kafka.close_failed", exc_info=True)


def get_source(config: KafkaConfig) -> KafkaEventSource:
    """Factory mirroring `get_store` / `get_reranker` patterns."""
    return KafkaEventSource(config)

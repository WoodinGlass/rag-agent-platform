"""StreamingIngestor: pull events, process them, ack on success.

Guarantees:
- At-least-once: an event is committed only after the handler returns
  successfully. On failure, the event is retried up to `max_retries`
  and NOT committed, so the source can redeliver (broker semantics).
- Dedup within a bounded window: the same event id seen twice inside
  `dedup_window` is skipped, even if the source redelivers. This makes
  the pipeline robust to the common at-least-once + retry pattern.
- Bounded memory: the dedup window is an LRU (OrderedDict), not a set
  that grows forever.

Backpressure:
- One event is in flight at a time. This is deliberate: it gives
  predictable memory usage and makes the loop testable. For throughput,
  a prefetch queue can be added later behind a config knob.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.core.tracing import span
from app.streaming.events import Event
from app.streaming.prefetch import Prefetcher
from app.streaming.source import EventSource

log = get_logger(__name__)

# Handler: async function that takes an Event and returns None on success.
EventHandler = Callable[[Event], Awaitable[None]]


@dataclass
class IngestorStats:
    pulled: int = 0
    committed: int = 0
    deduped: int = 0
    failed: int = 0
    retries: int = 0
    handler_errors: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict:
        return {
            "pulled": self.pulled,
            "committed": self.committed,
            "deduped": self.deduped,
            "failed": self.failed,
            "retries": self.retries,
            "handler_errors": self.handler_errors,
            "uptime_s": round(time.monotonic() - self.started_at, 3),
        }


class StreamingIngestor:
    def __init__(
        self,
        source: EventSource,
        handler: EventHandler,
        *,
        poll_timeout_s: float = 1.0,
        max_retries: int = 3,
        dedup_window: int = 1024,
        idle_sleep_s: float = 0.05,
        retry_backoff_base_s: float = 0.1,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if dedup_window < 1:
            raise ValueError("dedup_window must be >= 1")

        self.source = source
        self.handler = handler
        self.poll_timeout_s = poll_timeout_s
        self.max_retries = max_retries
        self.dedup_window = dedup_window
        self.idle_sleep_s = idle_sleep_s
        self.retry_backoff_base_s = retry_backoff_base_s

        self.stats = IngestorStats()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._stop = asyncio.Event()

    # ---------- dedup ----------

    def _is_duplicate(self, event_id: str) -> bool:
        if event_id in self._seen:
            return True
        self._seen[event_id] = None
        while len(self._seen) > self.dedup_window:
            self._seen.popitem(last=False)
        return False

    # ---------- handler with retry ----------

    async def _handle_with_retry(self, event: Event) -> bool:
        """Return True if handled, False if gave up after retries."""
        attempt = 0
        while True:
            try:
                with span("streaming.handle", event_id=event.id, attempt=attempt):
                    await self.handler(event)
                return True
            except Exception as e:  # noqa: BLE001 -- handler boundary
                self.stats.handler_errors += 1
                if attempt >= self.max_retries:
                    log.error(
                        "streaming.handler_gaveup",
                        extra={"event_id": event.id, "attempt": attempt, "error": str(e)[:200]},
                    )
                    return False
                attempt += 1
                self.stats.retries += 1
                backoff = self.retry_backoff_base_s * (2 ** (attempt - 1))
                log.warning(
                    "streaming.handler_retry",
                    extra={"event_id": event.id, "attempt": attempt, "backoff_s": backoff},
                )
                await asyncio.sleep(backoff)

    # ---------- main loop ----------

    async def run_once(self) -> bool:
        """Pull and process one event. Return False if loop should stop.

        Returns:
          True  -> keep running
          False -> stop requested (or source closed)
        """
        if self._stop.is_set():
            return False

        event = await self.source.poll(timeout_s=self.poll_timeout_s)
        if event is None:
            await asyncio.sleep(self.idle_sleep_s)
            return True

        self.stats.pulled += 1

        if self._is_duplicate(event.id):
            self.stats.deduped += 1
            log.info("streaming.dedup", extra={"event_id": event.id})
            # Safe to commit: we already processed it earlier.
            await self.source.commit(event.id)
            return True

        ok = await self._handle_with_retry(event)
        if ok:
            await self.source.commit(event.id)
            self.stats.committed += 1
        else:
            self.stats.failed += 1
            # Do NOT commit -> source will redeliver.
        return True

    async def run(self) -> None:
        """Run until `stop()` is called (or source is closed)."""
        log.info("streaming.start", extra={"dedup_window": self.dedup_window, "max_retries": self.max_retries})
        try:
            while True:
                keep_going = await self.run_once()
                if not keep_going:
                    break
        finally:
            log.info("streaming.stop", extra=self.stats.as_dict())


    async def run_with_prefetch(
        self,
        *,
        prefetch_n: int = 8,
        poll_timeout_s: float | None = None,
    ) -> None:
        """Convenience: wrap source in a Prefetcher and run.

        `prefetch_n` is the prefetcher buffer size. Passing 1 reduces to
        the non-prefetch path (buffer of one). The prefetcher hides
        source-side latency; the ingestor logic is unchanged.
        """
        prefetcher = Prefetcher(
            self.source,
            buffer_size=max(1, prefetch_n),
            poll_timeout_s=poll_timeout_s or self.poll_timeout_s,
        )
        await prefetcher.start()
        original_source = self.source
        self.source = prefetcher
        try:
            await self.run()
        finally:
            await prefetcher.close()
            self.source = original_source

    def stop(self) -> None:
        self._stop.set()

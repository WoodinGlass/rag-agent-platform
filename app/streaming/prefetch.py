"""Prefetcher: bounded buffer between source and ingestor.

Why: `StreamingIngestor` polls one event at a time. When the source's
`poll()` has non-trivial latency (network, Kafka broker round-trip),
the loop spends most of its time waiting. A prefetcher runs `poll()`
in a background task and hands events to the ingestor from a local
buffer, hiding that latency.

Properties:
- Bounded queue: buffer_size caps memory. When full, the background
  task waits -- natural backpressure, same as not having a prefetcher.
- Commit delegate: `commit(event_id)` forwards to the source. The
  prefetcher is not an authority on offsets; it's a transport.
- Graceful shutdown: `close()` cancels the background task and closes
  the source.
- Failure isolation: if the source's `poll()` raises, the background
  task logs and retries with a small delay; the ingestor loop stays up.

Not implemented (deliberate): prefetch acks/commits ahead of processing.
We only ever commit what the ingestor finished, so at-least-once is
preserved without extra bookkeeping.
"""
from __future__ import annotations

import asyncio
from typing import Final

from app.core.logging import get_logger
from app.streaming.events import Event
from app.streaming.source import EventSource

log = get_logger(__name__)

DEFAULT_BUFFER_SIZE: Final[int] = 8
DEFAULT_POLL_TIMEOUT_S: Final[float] = 1.0
# Small delay before re-polling after a source error. Keeps the loop
# from busy-spinning when the source is down.
ERROR_BACKOFF_S: Final[float] = 0.5


class Prefetcher:
    """Wrap an EventSource with a bounded in-process buffer."""

    def __init__(
        self,
        source: EventSource,
        *,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        poll_timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        self._source = source
        self._buffer_size = buffer_size
        self._poll_timeout_s = poll_timeout_s

        # `_queue` holds events ready for the ingestor. `None` is the
        # sentinel the background task pushes to signal "source closed".
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=buffer_size)
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    # ---- lifecycle ----

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._fill_loop(), name="prefetcher.fill")

    async def _fill_loop(self) -> None:
        while not self._closed:
            try:
                event = await self._source.poll(timeout_s=self._poll_timeout_s)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("prefetch.source_error", exc_info=True)
                await asyncio.sleep(ERROR_BACKOFF_S)
                continue

            if event is None:
                # No event within timeout; loop again. Yield control so
                # the ingestor can drain what's already buffered.
                await asyncio.sleep(0)
                continue

            # Blocks when the buffer is full -> backpressure to source.
            # CancelledError propagates naturally; no handler needed.
            await self._queue.put(event)

    # ---- EventSource protocol ----

    async def poll(self, timeout_s: float = 1.0) -> Event | None:
        if self._closed and self._queue.empty():
            return None
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        except TimeoutError:
            return None
        if item is None:
            return None
        return item

    async def commit(self, event_id: str) -> None:
        await self._source.commit(event_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # expected: we just cancelled it
            except Exception:
                log.debug("prefetch.fill_task_error", exc_info=True)
            self._task = None
        try:
            await self._source.close()
        except Exception:
            log.debug("prefetch.source_close_error", exc_info=True)

"""Prefetcher: buffering, backpressure, error isolation, close."""
import asyncio

import pytest

from app.streaming.events import Event
from app.streaming.ingestor import StreamingIngestor
from app.streaming.memory_source import InMemoryQueueSource
from app.streaming.prefetch import Prefetcher
from app.streaming.source import EventSource


def _event(i: int) -> Event:
    return Event(id=f"e{i}", data=f"doc {i}".encode())


# ---------- protocol conformance ----------

def test_prefetcher_satisfies_protocol():
    src = InMemoryQueueSource()
    pf = Prefetcher(src)
    assert isinstance(pf, EventSource)


# ---------- basic pass-through ----------

@pytest.mark.asyncio
async def test_prefetcher_passes_events_through():
    src = InMemoryQueueSource()
    for i in range(3):
        src.push(_event(i))

    pf = Prefetcher(src, buffer_size=8, poll_timeout_s=0.02)
    await pf.start()
    try:
        got = []
        for _ in range(3):
            e = await pf.poll(timeout_s=0.5)
            assert e is not None
            got.append(e.id)
        assert got == ["e0", "e1", "e2"]
    finally:
        await pf.close()


@pytest.mark.asyncio
async def test_commit_delegates_to_source():
    src = InMemoryQueueSource()
    pf = Prefetcher(src, poll_timeout_s=0.02)
    await pf.start()
    try:
        await pf.commit("abc")
        assert src.committed_ids == ["abc"]
    finally:
        await pf.close()


@pytest.mark.asyncio
async def test_poll_timeout_when_empty():
    src = InMemoryQueueSource()
    pf = Prefetcher(src, poll_timeout_s=0.02)
    await pf.start()
    try:
        assert await pf.poll(timeout_s=0.05) is None
    finally:
        await pf.close()


# ---------- validation ----------

def test_buffer_size_validated():
    src = InMemoryQueueSource()
    with pytest.raises(ValueError, match="buffer_size"):
        Prefetcher(src, buffer_size=0)


# ---------- close ----------

@pytest.mark.asyncio
async def test_close_is_idempotent():
    src = InMemoryQueueSource()
    pf = Prefetcher(src, poll_timeout_s=0.02)
    await pf.start()
    await pf.close()
    await pf.close()  # no error
    assert await pf.poll(timeout_s=0.05) is None


@pytest.mark.asyncio
async def test_close_closes_underlying_source():
    src = InMemoryQueueSource()
    pf = Prefetcher(src, poll_timeout_s=0.02)
    await pf.start()
    await pf.close()
    # source is closed; push now raises
    with pytest.raises(RuntimeError, match="closed"):
        src.push(_event(0))


# ---------- backpressure (bounded buffer) ----------

@pytest.mark.asyncio
async def test_buffer_bounded_backpressure():
    """With buffer_size=2 and no consumer, fill loop must not overrun."""
    src = InMemoryQueueSource()
    for i in range(10):
        src.push(_event(i))

    pf = Prefetcher(src, buffer_size=2, poll_timeout_s=0.01)
    await pf.start()
    try:
        # Give the fill loop time to overfill if it were broken.
        await asyncio.sleep(0.15)
        # Drain a few slowly; queue should never grow beyond 2.
        assert pf._queue.qsize() <= 2
    finally:
        await pf.close()


# ---------- error isolation ----------

@pytest.mark.asyncio
async def test_source_error_does_not_kill_prefetcher():
    class FlakySource:
        def __init__(self):
            self.calls = 0

        async def poll(self, timeout_s: float = 1.0):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            if self.calls == 2:
                return _event(0)
            await asyncio.sleep(0.05)
            return None

        async def commit(self, event_id: str) -> None:
            return None

        async def close(self) -> None:
            return None

    src = FlakySource()
    pf = Prefetcher(src, buffer_size=4, poll_timeout_s=0.02)
    await pf.start()
    try:
        e = await pf.poll(timeout_s=1.0)
        assert e is not None
        assert e.id == "e0"
    finally:
        await pf.close()


# ---------- integration with StreamingIngestor ----------

@pytest.mark.asyncio
async def test_ingestor_run_with_prefetch_processes_events():
    src = InMemoryQueueSource()
    for i in range(5):
        src.push(_event(i))

    seen: list[str] = []

    async def handler(e: Event) -> None:
        seen.append(e.id)

    ing = StreamingIngestor(
        src,
        handler,
        poll_timeout_s=0.02,
        idle_sleep_s=0.001,
        retry_backoff_base_s=0.0,
    )

    # simulate a small workload: run_with_prefetch loops until stop()
    async def stopper():
        await asyncio.sleep(0.4)
        ing.stop()

    asyncio.create_task(stopper())
    await ing.run_with_prefetch(prefetch_n=4)

    assert seen[:5] == ["e0", "e1", "e2", "e3", "e4"]
    assert ing.stats.committed >= 5

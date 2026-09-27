"""ParallelIngestor: fan-out, isolation, aggregated stats. Offline."""
import asyncio

import pytest

from app.streaming.events import Event
from app.streaming.parallel_ingestor import ParallelIngestor
from app.streaming.partitioned import (
    InMemoryPartitionedSource,
    PartitionedSource,
)


def _event(partition: str, i: int) -> Event:
    return Event(id=f"{partition}-{i}", data=f"{partition} doc {i}".encode())


# ---------- protocol ----------

def test_in_memory_partitioned_satisfies_protocol():
    src = InMemoryPartitionedSource()
    assert isinstance(src, PartitionedSource)


# ---------- protocol behavior ----------

def test_partitions_created_lazily_and_sorted():
    src = InMemoryPartitionedSource()
    src.push("p2", _event("p2", 0))
    src.push("p0", _event("p0", 0))
    src.push("p1", _event("p1", 0))
    assert src.partitions() == ["p0", "p1", "p2"]


def test_source_for_is_stable():
    src = InMemoryPartitionedSource()
    a = src.source_for("p0")
    b = src.source_for("p0")
    assert a is b


# ---------- happy path: 3 partitions processed in parallel ----------

@pytest.mark.asyncio
async def test_processes_all_partitions():
    src = InMemoryPartitionedSource()
    for p in ("p0", "p1", "p2"):
        for i in range(3):
            src.push(p, _event(p, i))

    seen: list[str] = []
    lock = asyncio.Lock()

    async def handler(e: Event) -> None:
        async with lock:
            seen.append(e.id)

    ing = ParallelIngestor(
        src,
        handler,
        poll_timeout_s=0.01,
        idle_sleep_s=0.001,
        retry_backoff_base_s=0.0,
    )

    async def stopper():
        await asyncio.sleep(0.4)
        ing.stop()

    asyncio.create_task(stopper())
    await ing.run()

    # all 9 events processed (order across partitions is nondeterministic)
    assert sorted(seen) == sorted(
        f"{p}-{i}" for p in ("p0", "p1", "p2") for i in range(3)
    )
    assert ing.stats.total("committed") == 9


# ---------- per-partition stats ----------

@pytest.mark.asyncio
async def test_per_partition_stats():
    src = InMemoryPartitionedSource()
    src.push("a", _event("a", 0))
    src.push("a", _event("a", 1))
    src.push("b", _event("b", 0))

    async def handler(e: Event) -> None:
        pass

    ing = ParallelIngestor(
        src,
        handler,
        poll_timeout_s=0.01,
        idle_sleep_s=0.001,
        retry_backoff_base_s=0.0,
    )

    async def stopper():
        await asyncio.sleep(0.3)
        ing.stop()

    asyncio.create_task(stopper())
    await ing.run()

    snap = ing.stats.as_dict()
    assert "a" in snap["per_partition"]
    assert "b" in snap["per_partition"]
    assert snap["per_partition"]["a"]["committed"] == 2
    assert snap["per_partition"]["b"]["committed"] == 1
    assert snap["totals"]["committed"] == 3


# ---------- isolation: crash in one partition doesn't kill others ----------

@pytest.mark.asyncio
async def test_partition_crash_is_isolated():
    src = InMemoryPartitionedSource()
    src.push("healthy", _event("healthy", 0))

    async def handler(e: Event) -> None:
        pass

    ing = ParallelIngestor(src, handler, poll_timeout_s=0.01, idle_sleep_s=0.001)

    # monkeypatch: make one partition's run() raise
    from app.streaming.ingestor import StreamingIngestor

    original_run = StreamingIngestor.run

    async def flaky_run(self):
        if self.source is src.source_for("healthy"):
            # In practice this is unreachable: healthy runs fine. We add
            # a separate broken partition to trigger the crash path.
            await original_run(self)
        else:
            raise RuntimeError("boom")

    # add a broken partition first
    src.push("broken", _event("broken", 0))

    # easier isolation test: replace run for one partition by name
    # we can't easily hook by partition; use a marker partition
    ing._ingestors = {}  # ensure clean

    # Simpler approach: use a partition whose source raises on poll.
    class BoomSource:
        async def poll(self, timeout_s=1.0):
            raise RuntimeError("boom")

        async def commit(self, event_id: str) -> None:
            return None

        async def close(self) -> None:
            return None

    src._sources["broken"] = BoomSource()  # type: ignore[assignment]

    async def stopper():
        await asyncio.sleep(0.3)
        ing.stop()

    asyncio.create_task(stopper())
    await ing.run()

    # healthy partition still processed
    snap = ing.stats.as_dict()
    assert snap["totals"]["committed"] >= 1
    # broken partition either errored or kept retrying, but healthy finished
    assert "healthy" in snap["per_partition"]


# ---------- stop is idempotent ----------

@pytest.mark.asyncio
async def test_stop_is_idempotent():
    src = InMemoryPartitionedSource()
    src.push("p", _event("p", 0))

    async def handler(e: Event) -> None:
        pass

    ing = ParallelIngestor(src, handler, poll_timeout_s=0.01, idle_sleep_s=0.001)

    async def stopper():
        await asyncio.sleep(0.15)
        ing.stop()
        ing.stop()
        ing.stop()

    asyncio.create_task(stopper())
    await asyncio.wait_for(ing.run(), timeout=2.0)
    assert ing.stats.total("committed") == 1


# ---------- no partitions ----------

@pytest.mark.asyncio
async def test_no_partitions_returns_immediately():
    src = InMemoryPartitionedSource()

    async def handler(e: Event) -> None:
        pass

    ing = ParallelIngestor(src, handler)
    await asyncio.wait_for(ing.run(), timeout=1.0)
    assert ing.stats.as_dict()["totals"]["committed"] == 0


# ---------- ingestor_for ----------

@pytest.mark.asyncio
async def test_ingestor_for_returns_per_partition_instance():
    src = InMemoryPartitionedSource()
    src.push("x", _event("x", 0))

    async def handler(e: Event) -> None:
        pass

    ing = ParallelIngestor(src, handler, poll_timeout_s=0.01, idle_sleep_s=0.001)

    async def stopper():
        await asyncio.sleep(0.2)
        ing.stop()

    asyncio.create_task(stopper())
    await ing.run()

    assert ing.ingestor_for("x") is not None
    assert ing.ingestor_for("missing") is None

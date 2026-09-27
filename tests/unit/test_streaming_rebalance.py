"""ParallelIngestor rebalance: assign/revoke lifecycle. Offline."""
import asyncio

import pytest

from app.streaming.events import Event
from app.streaming.parallel_ingestor import ParallelIngestor
from app.streaming.partitioned import InMemoryPartitionedSource


def _event(partition: str, i: int) -> Event:
    return Event(id=f"{partition}-{i}", data=f"{partition} doc {i}".encode())


async def _noop(e: Event) -> None:
    pass


# ---------- assign ----------

@pytest.mark.asyncio
async def test_assign_starts_ingestors():
    src = InMemoryPartitionedSource()
    ing = ParallelIngestor(src, _noop, poll_timeout_s=0.01, idle_sleep_s=0.001)

    await ing.on_partitions_assigned(["p0", "p1"])
    try:
        assert ing.running_partitions() == ["p0", "p1"]
        assert ing.ingestor_for("p0") is not None
    finally:
        await ing.on_partitions_revoked(["p0", "p1"])


@pytest.mark.asyncio
async def test_assign_is_idempotent():
    src = InMemoryPartitionedSource()
    ing = ParallelIngestor(src, _noop, poll_timeout_s=0.01, idle_sleep_s=0.001)

    await ing.on_partitions_assigned(["p0"])
    first = ing.ingestor_for("p0")
    await ing.on_partitions_assigned(["p0"])  # same partition again
    second = ing.ingestor_for("p0")

    try:
        assert first is second  # same instance, not replaced
    finally:
        await ing.on_partitions_revoked(["p0"])


# ---------- revoke ----------

@pytest.mark.asyncio
async def test_revoke_stops_ingestors():
    src = InMemoryPartitionedSource()
    src.push("p0", _event("p0", 0))

    ing = ParallelIngestor(src, _noop, poll_timeout_s=0.01, idle_sleep_s=0.001)

    await ing.on_partitions_assigned(["p0", "p1"])
    await ing.on_partitions_revoked(["p0"])

    try:
        assert ing.running_partitions() == ["p1"]
        assert ing.ingestor_for("p0") is None
        # stats snapshot for the revoked partition is preserved
        assert "p0" in ing.stats.per_partition
    finally:
        await ing.on_partitions_revoked(["p1"])


@pytest.mark.asyncio
async def test_revoke_unknown_partition_is_noop():
    src = InMemoryPartitionedSource()
    ing = ParallelIngestor(src, _noop, poll_timeout_s=0.01, idle_sleep_s=0.001)

    await ing.on_partitions_assigned(["p0"])
    await ing.on_partitions_revoked(["does-not-exist"])  # no error
    try:
        assert ing.running_partitions() == ["p0"]
    finally:
        await ing.on_partitions_revoked(["p0"])


# ---------- grace period ----------

@pytest.mark.asyncio
async def test_revoke_waits_for_in_flight_event():
    """An event being handled when revoke arrives should finish."""
    src = InMemoryPartitionedSource()
    src.push("p0", _event("p0", 0))

    started = asyncio.Event()
    finish = asyncio.Event()

    async def slow_handler(e: Event) -> None:
        started.set()
        await finish.wait()

    ing = ParallelIngestor(
        src, slow_handler,
        poll_timeout_s=0.01, idle_sleep_s=0.001,
        rebalance_grace_s=2.0,
    )
    await ing.on_partitions_assigned(["p0"])

    # wait for the handler to start
    await asyncio.wait_for(started.wait(), timeout=1.0)

    # fire revoke in the background; it must block on the in-flight event
    revoke_task = asyncio.create_task(ing.on_partitions_revoked(["p0"]))
    await asyncio.sleep(0.05)
    assert not revoke_task.done(), "revoke should wait for in-flight event"

    finish.set()
    await asyncio.wait_for(revoke_task, timeout=1.0)

    assert ing.running_partitions() == []
    # the event was processed and committed
    assert ing.stats.per_partition["p0"]["committed"] == 1


@pytest.mark.asyncio
async def test_revoke_cancels_after_grace():
    """If the handler doesn't finish in time, revoke cancels."""
    src = InMemoryPartitionedSource()
    src.push("p0", _event("p0", 0))

    started = asyncio.Event()

    async def never_finishes(e: Event) -> None:
        started.set()
        # sleep past the grace period
        await asyncio.sleep(60)

    ing = ParallelIngestor(
        src, never_finishes,
        poll_timeout_s=0.01, idle_sleep_s=0.001,
        rebalance_grace_s=0.1,
    )
    await ing.on_partitions_assigned(["p0"])
    await asyncio.wait_for(started.wait(), timeout=1.0)

    await asyncio.wait_for(ing.on_partitions_revoked(["p0"]), timeout=1.0)
    assert ing.running_partitions() == []


# ---------- lifecycle via run() ----------

@pytest.mark.asyncio
async def test_run_uses_initial_partitions_and_shuts_down_on_stop():
    src = InMemoryPartitionedSource()
    src.push("p0", _event("p0", 0))
    src.push("p1", _event("p1", 0))

    seen: list[str] = []
    lock = asyncio.Lock()

    async def handler(e: Event) -> None:
        async with lock:
            seen.append(e.id)

    ing = ParallelIngestor(
        src, handler,
        poll_timeout_s=0.01, idle_sleep_s=0.001, retry_backoff_base_s=0.0,
    )

    async def stopper():
        await asyncio.sleep(0.3)
        ing.stop()

    asyncio.create_task(stopper())
    await asyncio.wait_for(ing.run(), timeout=2.0)

    assert sorted(seen) == ["p0-0", "p1-0"]
    assert ing.running_partitions() == []
    assert ing.stats.total("committed") == 2


# ---------- dynamic rebalance during run ----------

@pytest.mark.asyncio
async def test_rebalance_adds_partition_during_run():
    src = InMemoryPartitionedSource()
    src.push("p0", _event("p0", 0))

    seen: list[str] = []
    lock = asyncio.Lock()

    async def handler(e: Event) -> None:
        async with lock:
            seen.append(e.id)

    ing = ParallelIngestor(
        src, handler,
        poll_timeout_s=0.01, idle_sleep_s=0.001, retry_backoff_base_s=0.0,
    )

    async def orchestrator():
        await asyncio.sleep(0.15)
        # simulate a rebalance: add p1 mid-run
        src.push("p1", _event("p1", 0))
        await ing.on_partitions_assigned(["p1"])
        await asyncio.sleep(0.2)
        ing.stop()

    asyncio.create_task(orchestrator())
    await asyncio.wait_for(ing.run(), timeout=3.0)

    assert "p0-0" in seen
    assert "p1-0" in seen
    assert ing.stats.total("committed") == 2
    assert "p1" in ing.stats.per_partition


# ---------- validation ----------

def test_grace_validated():
    src = InMemoryPartitionedSource()
    with pytest.raises(ValueError, match="rebalance_grace_s"):
        ParallelIngestor(src, _noop, rebalance_grace_s=0)

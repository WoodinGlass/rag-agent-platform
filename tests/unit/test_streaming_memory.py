"""Offline tests for the in-memory EventSource."""
import asyncio

import pytest

from app.streaming.events import Event
from app.streaming.memory_source import InMemoryQueueSource
from app.streaming.source import EventSource


def _event(i: int) -> Event:
    return Event(id=f"e{i}", data=f"doc {i}".encode(), metadata={"k": i})


# ---------- protocol ----------

def test_memory_source_satisfies_protocol():
    assert isinstance(InMemoryQueueSource(), EventSource)


# ---------- basic poll/commit ----------

@pytest.mark.asyncio
async def test_poll_returns_events_in_order():
    src = InMemoryQueueSource()
    for i in range(3):
        src.push(_event(i))

    ids = []
    for _ in range(3):
        e = await src.poll(timeout_s=0.1)
        assert e is not None
        ids.append(e.id)
    assert ids == ["e0", "e1", "e2"]


@pytest.mark.asyncio
async def test_poll_timeout_returns_none():
    src = InMemoryQueueSource()
    e = await src.poll(timeout_s=0.05)
    assert e is None


@pytest.mark.asyncio
async def test_commit_records_id():
    src = InMemoryQueueSource()
    await src.commit("abc")
    await src.commit("abc")  # idempotent
    assert src.committed_ids == ["abc", "abc"]


@pytest.mark.asyncio
async def test_close_unblocks_poll():
    src = InMemoryQueueSource()

    async def poller():
        return await src.poll(timeout_s=5.0)

    task = asyncio.create_task(poller())
    await asyncio.sleep(0.05)
    await src.close()
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result is None


@pytest.mark.asyncio
async def test_poll_after_close_returns_none():
    src = InMemoryQueueSource()
    await src.close()
    assert await src.poll(timeout_s=0.05) is None


# ---------- backpressure / capacity ----------

def test_maxsize_enforced():
    src = InMemoryQueueSource(maxsize=2)
    src.push(_event(0))
    src.push(_event(1))
    with pytest.raises(BufferError, match="queue full"):
        src.push(_event(2))


@pytest.mark.asyncio
async def test_waiting_poller_is_woken_by_push():
    src = InMemoryQueueSource()

    async def poller():
        return await src.poll(timeout_s=1.0)

    task = asyncio.create_task(poller())
    await asyncio.sleep(0.05)
    src.push(_event(7))
    e = await asyncio.wait_for(task, timeout=1.0)
    assert e is not None
    assert e.id == "e7"


# ---------- push_bytes convenience ----------

@pytest.mark.asyncio
async def test_push_bytes_sets_id_and_metadata():
    src = InMemoryQueueSource()
    src.push_bytes(b"hi", id="x1", metadata={"tenant": "A"})
    e = await src.poll(timeout_s=0.1)
    assert e is not None
    assert e.data == b"hi"
    assert e.metadata["tenant"] == "A"
    assert e.id == "x1"


# ---------- pending introspection ----------

@pytest.mark.asyncio
async def test_pending_count():
    src = InMemoryQueueSource()
    src.push(_event(0))
    src.push(_event(1))
    assert src.pending == 2
    await src.poll(timeout_s=0.05)
    assert src.pending == 1

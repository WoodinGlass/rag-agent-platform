"""StreamingIngestor: dedup, retry, commit-on-success."""
import asyncio

import pytest

from app.streaming.events import Event
from app.streaming.ingestor import StreamingIngestor
from app.streaming.memory_source import InMemoryQueueSource


def _event(i: int) -> Event:
    return Event(id=f"e{i}", data=f"doc {i}".encode())


# ---------- happy path ----------

@pytest.mark.asyncio
async def test_processes_events_and_commits():
    src = InMemoryQueueSource()
    for i in range(3):
        src.push(_event(i))

    seen: list[str] = []

    async def handler(e: Event) -> None:
        seen.append(e.id)

    ing = StreamingIngestor(
        src, handler, poll_timeout_s=0.01, idle_sleep_s=0.001, retry_backoff_base_s=0.0
    )
    for _ in range(3):
        await ing.run_once()

    assert seen == ["e0", "e1", "e2"]
    assert ing.stats.committed == 3
    assert src.committed_ids == ["e0", "e1", "e2"]


# ---------- dedup ----------

@pytest.mark.asyncio
async def test_duplicate_event_is_deduped():
    src = InMemoryQueueSource()
    src.push(_event(1))
    src.push(_event(1))  # same id
    src.push(_event(2))

    seen: list[str] = []

    async def handler(e: Event) -> None:
        seen.append(e.id)

    ing = StreamingIngestor(
        src, handler, poll_timeout_s=0.01, idle_sleep_s=0.001, retry_backoff_base_s=0.0
    )
    for _ in range(3):
        await ing.run_once()

    assert seen == ["e1", "e2"]
    assert ing.stats.deduped == 1
    # dup is still committed (idempotent-safe)
    assert src.committed_ids.count("e1") == 2


@pytest.mark.asyncio
async def test_dedup_window_is_bounded():
    src = InMemoryQueueSource()
    ing = StreamingIngestor(
        src, _noop, dedup_window=2, poll_timeout_s=0.01, idle_sleep_s=0.001
    )

    # Push 3 unique ids; window=2 -> first id evicted
    for i in range(3):
        src.push(_event(i))
    for _ in range(3):
        await ing.run_once()

    # Now re-push the first id; it should NOT be deduped (evicted from window)
    src.push(_event(0))
    seen: list[str] = []

    async def track(e: Event) -> None:
        seen.append(e.id)

    ing.handler = track
    await ing.run_once()
    assert seen == ["e0"]


# ---------- retry ----------

@pytest.mark.asyncio
async def test_retries_then_succeeds():
    src = InMemoryQueueSource()
    src.push(_event(1))

    calls = {"n": 0}

    async def flaky(e: Event) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")

    ing = StreamingIngestor(
        src, flaky, max_retries=3, poll_timeout_s=0.01,
        idle_sleep_s=0.001, retry_backoff_base_s=0.0,
    )
    await ing.run_once()

    assert calls["n"] == 3
    assert ing.stats.committed == 1
    assert ing.stats.retries == 2


@pytest.mark.asyncio
async def test_gives_up_after_max_retries_and_does_not_commit():
    src = InMemoryQueueSource()
    src.push(_event(1))

    async def always_fail(e: Event) -> None:
        raise RuntimeError("boom")

    ing = StreamingIngestor(
        src, always_fail, max_retries=1, poll_timeout_s=0.01,
        idle_sleep_s=0.001, retry_backoff_base_s=0.0,
    )
    await ing.run_once()

    assert ing.stats.failed == 1
    assert ing.stats.committed == 0
    assert "e1" not in src.committed_ids


# ---------- stop ----------

@pytest.mark.asyncio
async def test_stop_exits_run_loop():
    src = InMemoryQueueSource()

    async def handler(e: Event) -> None:
        pass

    ing = StreamingIngestor(src, handler, poll_timeout_s=0.01, idle_sleep_s=0.001)

    async def killer() -> None:
        await asyncio.sleep(0.05)
        ing.stop()

    asyncio.create_task(killer())
    await asyncio.wait_for(ing.run(), timeout=2.0)
    assert ing._stop.is_set()


# ---------- validation ----------

def test_invalid_max_retries():
    src = InMemoryQueueSource()
    with pytest.raises(ValueError, match="max_retries"):
        StreamingIngestor(src, _noop, max_retries=-1)


def test_invalid_dedup_window():
    src = InMemoryQueueSource()
    with pytest.raises(ValueError, match="dedup_window"):
        StreamingIngestor(src, _noop, dedup_window=0)


# ---------- helpers ----------

async def _noop(e: Event) -> None:
    pass

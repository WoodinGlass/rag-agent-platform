"""DedupStore: in-memory LRU + SQLite durable. Offline."""
import time

import pytest

from app.streaming.dedup import DedupStore, InMemoryDedupStore
from app.streaming.sqlite_dedup import SQLiteDedupStore

# ---------- protocol ----------

def test_inmemory_satisfies_protocol():
    assert isinstance(InMemoryDedupStore(), DedupStore)


def test_sqlite_satisfies_protocol(tmp_path):
    store = SQLiteDedupStore(tmp_path / "d.sqlite")
    try:
        assert isinstance(store, DedupStore)
    finally:
        store.close()


# ---------- InMemoryDedupStore ----------

def test_inmemory_first_time_is_new():
    s = InMemoryDedupStore(max_size=10)
    assert s.add_if_new("a") is True
    assert s.add_if_new("b") is True
    assert s.add_if_new("a") is False
    assert s.size == 2


def test_inmemory_lru_eviction():
    s = InMemoryDedupStore(max_size=2)
    assert s.add_if_new("a") is True
    assert s.add_if_new("b") is True
    assert s.add_if_new("c") is True
    # "a" evicted
    assert "a" not in s
    assert "b" in s and "c" in s
    assert s.add_if_new("a") is True  # new again after eviction


def test_inmemory_touch_refreshes_recency():
    s = InMemoryDedupStore(max_size=2)
    s.add_if_new("a")
    s.add_if_new("b")
    s.add_if_new("a")  # hit -> refresh -> "b" is oldest now
    s.add_if_new("c")  # evicts "b"
    assert "a" in s
    assert "b" not in s


def test_inmemory_max_size_validated():
    with pytest.raises(ValueError, match="max_size"):
        InMemoryDedupStore(max_size=0)


def test_inmemory_close_clears():
    s = InMemoryDedupStore()
    s.add_if_new("a")
    s.close()
    assert s.size == 0


# ---------- SQLiteDedupStore ----------

def test_sqlite_add_if_new(tmp_path):
    s = SQLiteDedupStore(tmp_path / "d.sqlite")
    try:
        assert s.add_if_new("x") is True
        assert s.add_if_new("y") is True
        assert s.add_if_new("x") is False
        assert s.size == 2
    finally:
        s.close()


def test_sqlite_persists_across_instances(tmp_path):
    p = tmp_path / "d.sqlite"
    s1 = SQLiteDedupStore(p)
    s1.add_if_new("keep")
    s1.close()

    s2 = SQLiteDedupStore(p)
    try:
        assert s2.add_if_new("keep") is False
        assert s2.add_if_new("new") is True
    finally:
        s2.close()


def test_sqlite_prune_removes_old(tmp_path):
    s = SQLiteDedupStore(tmp_path / "d.sqlite", ttl_s=60.0)
    try:
        now = time.time()
        s.add_if_new("old")
        # simulate an old row
        s._conn.execute("UPDATE dedup SET seen_at = ? WHERE event_id = ?",
                        (now - 3600, "old"))
        s._conn.commit()

        s.add_if_new("fresh")
        deleted = s.prune(now=now)
        assert deleted == 1
        assert "old" not in s
        assert "fresh" in s
    finally:
        s.close()


def test_sqlite_ttl_validated(tmp_path):
    with pytest.raises(ValueError, match="ttl_s"):
        SQLiteDedupStore(tmp_path / "d.sqlite", ttl_s=0)


def test_sqlite_close_idempotent(tmp_path):
    s = SQLiteDedupStore(tmp_path / "d.sqlite")
    s.close()
    s.close()  # no error


# ---------- integration with StreamingIngestor ----------

from app.streaming.events import Event
from app.streaming.ingestor import StreamingIngestor
from app.streaming.memory_source import InMemoryQueueSource


def _event(i: int) -> Event:
    return Event(id=f"e{i}", data=f"doc {i}".encode())


@pytest.mark.asyncio
async def test_ingestor_uses_explicit_dedup_store():
    src = InMemoryQueueSource()
    src.push(_event(1))
    src.push(_event(1))  # duplicate

    seen: list[str] = []

    async def handler(e: Event) -> None:
        seen.append(e.id)

    store = InMemoryDedupStore(max_size=100)
    ing = StreamingIngestor(
        src, handler,
        dedup_store=store,
        poll_timeout_s=0.01,
        idle_sleep_s=0.001,
        retry_backoff_base_s=0.0,
    )
    for _ in range(2):
        await ing.run_once()

    assert seen == ["e1"]
    assert ing.stats.deduped == 1


@pytest.mark.asyncio
async def test_ingestor_dedup_survives_store_restart(tmp_path):
    """Simulates: worker 1 processes e1, restarts; worker 2 sees e1 again
    but the durable store remembers it, so no re-process.
    """
    p = tmp_path / "d.sqlite"

    # --- worker 1 ---
    src1 = InMemoryQueueSource()
    src1.push(_event(1))

    seen1: list[str] = []

    async def handler1(e: Event) -> None:
        seen1.append(e.id)

    store1 = SQLiteDedupStore(p)
    ing1 = StreamingIngestor(
        src1, handler1,
        dedup_store=store1,
        poll_timeout_s=0.01,
        idle_sleep_s=0.001,
        retry_backoff_base_s=0.0,
    )
    await ing1.run_once()
    assert seen1 == ["e1"]
    ing1.stop()
    await ing1.run()  # ensures store.close() runs via run's finally

    # --- worker 2 (fresh process would use same file) ---
    src2 = InMemoryQueueSource()
    src2.push(_event(1))  # redelivery

    seen2: list[str] = []

    async def handler2(e: Event) -> None:
        seen2.append(e.id)

    store2 = SQLiteDedupStore(p)
    ing2 = StreamingIngestor(
        src2, handler2,
        dedup_store=store2,
        poll_timeout_s=0.01,
        idle_sleep_s=0.001,
        retry_backoff_base_s=0.0,
    )
    await ing2.run_once()
    ing2.stop()
    await ing2.run()

    assert seen2 == []  # deduped across restart
    assert ing2.stats.deduped == 1

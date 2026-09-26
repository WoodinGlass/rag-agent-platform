"""Kafka adapter shape tests. Offline; no broker.

We exercise the pure mapping and the protocol conformance by injecting
a fake consumer. Real Kafka integration is out of scope for unit tests.
"""
from __future__ import annotations

import pytest

from app.streaming.kafka_source import KafkaConfig, KafkaEventSource
from app.streaming.source import EventSource


class _FakeRecord:
    def __init__(self, value: bytes, key: bytes | None = None, offset: int = 0):
        self.value = value
        self.key = key
        self.offset = offset


class _FakeTopicPartition:
    def __init__(self, topic: str, partition: int):
        self.topic = topic
        self.partition = partition

    def __hash__(self):
        return hash((self.topic, self.partition))


class _FakeConsumer:
    """Minimal KafkaConsumer stand-in. No network."""

    def __init__(self, batches: list[dict] | None = None, **kwargs):
        self._batches = list(batches or [])
        self.commits: list[dict] = []
        self.closed = False

    def poll(self, timeout_ms: int = 1000):
        if not self._batches:
            return {}
        return self._batches.pop(0)

    def commit(self, offsets):
        self.commits.append(offsets)

    def close(self):
        self.closed = True


def _make_source(monkeypatch, batches, **kw) -> tuple[KafkaEventSource, _FakeConsumer]:
    fake = _FakeConsumer(batches=batches)
    import app.streaming.kafka_source as mod

    class _FakeKafkaConsumerFactory:
        def __call__(self, *args, **kwargs):
            return fake

    monkeypatch.setattr(
        mod,
        "KafkaConsumer",
        _FakeKafkaConsumerFactory(),
        raising=False,
    )

    # patch import inside __init__ by pre-injecting into sys.modules
    import sys
    import types

    fake_mod = types.ModuleType("kafka")
    fake_mod.KafkaConsumer = _FakeKafkaConsumerFactory()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kafka", fake_mod)

    cfg = KafkaConfig(
        bootstrap_servers="localhost:9092",
        topic="docs",
        group_id="test",
        **kw,
    )
    return KafkaEventSource(cfg), fake


# ---------- protocol conformance (import-time, no broker) ----------

def test_kafka_source_satisfies_protocol(monkeypatch):
    src, _ = _make_source(monkeypatch, [])
    assert isinstance(src, EventSource)


# ---------- event mapping ----------

@pytest.mark.asyncio
async def test_poll_maps_record_to_event(monkeypatch):
    tp = _FakeTopicPartition("docs", 0)
    rec = _FakeRecord(b"hello", key=b"k1", offset=42)
    batches = [{tp: [rec]}]

    src, _ = _make_source(monkeypatch, batches)
    e = await src.poll(timeout_s=0.1)

    assert e is not None
    assert e.data == b"hello"
    assert e.id == "docs:0:42"
    assert e.metadata["topic"] == "docs"
    assert e.metadata["partition"] == 0
    assert e.metadata["offset"] == 42
    assert e.metadata["key"] == "k1"


@pytest.mark.asyncio
async def test_poll_empty_returns_none(monkeypatch):
    src, _ = _make_source(monkeypatch, [{}])
    assert await src.poll(timeout_s=0.05) is None


@pytest.mark.asyncio
async def test_poll_buffers_extra_records(monkeypatch):
    tp = _FakeTopicPartition("docs", 0)
    r1 = _FakeRecord(b"one", offset=1)
    r2 = _FakeRecord(b"two", offset=2)
    src, _ = _make_source(monkeypatch, [{tp: [r1, r2]}])

    e1 = await src.poll(timeout_s=0.1)
    e2 = await src.poll(timeout_s=0.1)
    assert e1 is not None and e1.data == b"one"
    assert e2 is not None and e2.data == b"two"


# ---------- commit ----------

@pytest.mark.asyncio
async def test_commit_sends_next_offset(monkeypatch):
    tp = _FakeTopicPartition("docs", 0)
    rec = _FakeRecord(b"x", offset=10)
    src, fake = _make_source(monkeypatch, [{tp: [rec]}])

    e = await src.poll(timeout_s=0.1)
    assert e is not None
    await src.commit(e.id)

    assert len(fake.commits) == 1
    (offsets,) = fake.commits[0].items()
    _k, v = offsets
    assert v == 11  # offset + 1


@pytest.mark.asyncio
async def test_commit_unknown_id_is_noop(monkeypatch):
    src, fake = _make_source(monkeypatch, [])
    await src.commit("nope")
    assert fake.commits == []


# ---------- close ----------

@pytest.mark.asyncio
async def test_close_is_idempotent(monkeypatch):
    src, fake = _make_source(monkeypatch, [])
    await src.close()
    await src.close()
    assert fake.closed is True
    assert await src.poll(timeout_s=0.05) is None


# ---------- import guard ----------

def test_missing_kafka_raises_clear_import_error(monkeypatch):
    import sys

    # simulate kafka not installed
    monkeypatch.setitem(sys.modules, "kafka", None)
    import app.streaming.kafka_source as mod

    # reload the module so its ImportError path triggers on instantiation
    cfg = KafkaConfig(bootstrap_servers="localhost:9092", topic="t")
    with pytest.raises((ImportError, TypeError)):
        mod.KafkaEventSource(cfg)

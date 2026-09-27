"""Offline tests for the S3 event source.

Uses a fake S3 client; no boto3 required at runtime for these tests.
The import guard test simulates missing boto3 explicitly.
"""
from __future__ import annotations

import io
import sys

import pytest

from app.streaming.s3_source import S3Config, S3ObjectSource
from app.streaming.source import EventSource


class _FakeS3:
    """Minimal stand-in for boto3's S3 client."""

    def __init__(self, objects: dict[str, bytes]):
        self._objects = dict(objects)
        self.list_calls = 0
        self.get_calls: list[tuple[str, str]] = []
        self.fail_get_for: set[str] = set()

    def list_objects_v2(self, Bucket: str, Prefix: str = "", MaxKeys: int = 1000):
        self.list_calls += 1
        items = [
            {"Key": k, "Size": len(v)}
            for k, v in self._objects.items()
            if k.startswith(Prefix)
        ]
        items.sort(key=lambda o: o["Key"])
        return {"Contents": items[:MaxKeys]}

    def get_object(self, Bucket: str, Key: str):
        self.get_calls.append((Bucket, Key))
        if Key in self.fail_get_for:
            raise RuntimeError(f"transient error for {Key}")
        if Key not in self._objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self._objects[Key])}


def _cfg(**over) -> S3Config:
    base = {"bucket": "docs", "prefix": "", "poll_interval_s": 0.0, "max_keys": 100}
    base.update(over)
    return S3Config(**base)


# ---------- protocol conformance ----------

def test_s3_source_satisfies_protocol():
    fake = _FakeS3({})
    src = S3ObjectSource(_cfg(), s3_client=fake)
    assert isinstance(src, EventSource)


def test_bucket_required():
    with pytest.raises(ValueError, match="bucket"):
        S3ObjectSource(_cfg(bucket=""), s3_client=_FakeS3({}))


# ---------- basic poll ----------

@pytest.mark.asyncio
async def test_poll_returns_objects_in_key_order():
    fake = _FakeS3({"a.txt": b"A", "b.txt": b"B", "c.txt": b"C"})
    src = S3ObjectSource(_cfg(), s3_client=fake)

    got = []
    for _ in range(3):
        e = await src.poll(timeout_s=0.05)
        assert e is not None
        got.append((e.id, e.data))
    assert got == [
        ("s3://docs/a.txt", b"A"),
        ("s3://docs/b.txt", b"B"),
        ("s3://docs/c.txt", b"C"),
    ]


@pytest.mark.asyncio
async def test_prefix_filters_objects():
    fake = _FakeS3({"raw/a.txt": b"A", "raw/b.txt": b"B", "other/c.txt": b"C"})
    src = S3ObjectSource(_cfg(prefix="raw/"), s3_client=fake)

    ids = []
    for _ in range(2):
        e = await src.poll(timeout_s=0.05)
        assert e is not None
        ids.append(e.id)
    # third poll -> empty
    assert await src.poll(timeout_s=0.02) is None
    assert ids == ["s3://docs/raw/a.txt", "s3://docs/raw/b.txt"]


@pytest.mark.asyncio
async def test_poll_empty_returns_none():
    fake = _FakeS3({})
    src = S3ObjectSource(_cfg(), s3_client=fake)
    assert await src.poll(timeout_s=0.02) is None


@pytest.mark.asyncio
async def test_metadata_shape():
    fake = _FakeS3({"x.bin": b"12345"})
    src = S3ObjectSource(_cfg(), s3_client=fake)
    e = await src.poll(timeout_s=0.05)
    assert e is not None
    assert e.metadata["bucket"] == "docs"
    assert e.metadata["key"] == "x.bin"
    assert e.metadata["size"] == 5


# ---------- commit ----------

@pytest.mark.asyncio
async def test_commit_prevents_relisting():
    fake = _FakeS3({"a.txt": b"A", "b.txt": b"B"})
    src = S3ObjectSource(_cfg(), s3_client=fake)

    e1 = await src.poll(timeout_s=0.05)
    assert e1 is not None
    await src.commit(e1.id)

    # same process: only b.txt remains queued
    e2 = await src.poll(timeout_s=0.05)
    assert e2 is not None
    assert e2.id == "s3://docs/b.txt"

    assert "a.txt" in src.committed_keys

    # third poll -> refresh -> a.txt is filtered out
    assert await src.poll(timeout_s=0.02) is None


@pytest.mark.asyncio
async def test_commit_is_idempotent():
    fake = _FakeS3({"a.txt": b"A"})
    src = S3ObjectSource(_cfg(), s3_client=fake)
    await src.commit("s3://docs/a.txt")
    await src.commit("s3://docs/a.txt")
    assert src.committed_keys == {"a.txt"}


@pytest.mark.asyncio
async def test_commit_accepts_bare_key():
    fake = _FakeS3({})
    src = S3ObjectSource(_cfg(), s3_client=fake)
    await src.commit("bare.txt")
    assert "bare.txt" in src.committed_keys


# ---------- error recovery ----------

@pytest.mark.asyncio
async def test_get_object_error_requeues_and_recovers():
    fake = _FakeS3({"a.txt": b"A"})
    fake.fail_get_for = {"a.txt"}
    src = S3ObjectSource(_cfg(), s3_client=fake)

    # first poll: get fails, event requeued, poll returns None
    assert await src.poll(timeout_s=0.02) is None
    # now heal the backend
    fake.fail_get_for.clear()
    e = await src.poll(timeout_s=0.05)
    assert e is not None
    assert e.data == b"A"


@pytest.mark.asyncio
async def test_list_error_does_not_crash_poll():
    class BadList:
        def list_objects_v2(self, **kw):
            raise RuntimeError("s3 down")

        def get_object(self, **kw):
            raise RuntimeError("never")

    src = S3ObjectSource(_cfg(), s3_client=BadList())
    assert await src.poll(timeout_s=0.02) is None


# ---------- close ----------

@pytest.mark.asyncio
async def test_close_is_idempotent():
    fake = _FakeS3({"a.txt": b"A"})
    src = S3ObjectSource(_cfg(), s3_client=fake)
    await src.close()
    await src.close()
    assert await src.poll(timeout_s=0.02) is None


# ---------- import guard ----------

def test_missing_boto3_raises_clear_import_error(monkeypatch):
    # simulate boto3 not installed
    monkeypatch.setitem(sys.modules, "boto3", None)

    import app.streaming.s3_source as mod

    with pytest.raises((ImportError, TypeError)):
        mod.S3ObjectSource(_cfg())  # no s3_client -> tries to build one

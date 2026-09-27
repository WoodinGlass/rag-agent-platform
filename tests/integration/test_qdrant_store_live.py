"""Integration tests for QdrantStore against a real Qdrant service.

Self-skips when:
- `qdrant-client` is not installed, or
- no Qdrant service is reachable at `QDRANT_URL` (default localhost:6333).

In CI we boot a Qdrant service container (see `.github/workflows/ci.yml`).
Locally:

    docker run -p 6333:6333 qdrant/qdrant:v1.10.1

Marker: `integration` — kept out of the fast offline suite.
"""
from __future__ import annotations

import contextlib
import os
import uuid

import pytest

pytestmark = pytest.mark.integration

# Import guard: qdrant-client is optional ([qdrant] extra).
pytest.importorskip("qdrant_client")

from app.rag.chunker import chunk_text
from app.rag.embedder import FakeEmbedder
from app.rag.qdrant_store import QdrantStore

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def _reachable() -> bool:
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=QDRANT_URL, timeout=2.0).get_collections()
        return True
    except Exception:  # noqa: BLE001 -- any failure means "not reachable"
        return False


if not _reachable():
    pytest.skip(
        f"Qdrant not reachable at {QDRANT_URL} "
        "(start a service or set QDRANT_URL)",
        allow_module_level=True,
    )


@pytest.fixture()
def collection() -> str:
    """Unique collection per test so parallel runs do not collide."""
    return f"test_m9_3d_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def store(collection: str):
    s = QdrantStore(url=QDRANT_URL, collection=collection, dim=32)
    yield s
    with contextlib.suppress(Exception):
        from qdrant_client import QdrantClient

        QdrantClient(url=QDRANT_URL).delete_collection(
            collection_name=collection
        )


# ---------- full cycle ----------

def test_upsert_query_has_doc(store):
    emb = FakeEmbedder(dim=32)
    chunks = chunk_text("the cat sat on the mat", "d1", size=100)
    store.upsert(chunks, emb.embed([c.text for c in chunks]))

    assert store.has_doc("d1") is True

    hits = store.query(emb.embed_one("cat"), k=5)
    assert len(hits) >= 1
    assert hits[0].doc_id == "d1"
    assert hits[0].chunk_id == chunks[0].chunk_id


def test_upsert_is_idempotent(store):
    emb = FakeEmbedder(dim=32)
    chunks = chunk_text("hello world", "d1", size=100)
    embs = emb.embed([c.text for c in chunks])

    assert store.upsert(chunks, embs) == len(chunks)
    assert store.upsert(chunks, embs) == 0


def test_has_doc_false_for_unknown(store):
    assert store.has_doc("nope") is False


# ---------- multi-tenant isolation ----------

def test_tenant_filter_isolates_results(store):
    emb = FakeEmbedder(dim=32)
    a = chunk_text("secret tenant A", "dA", size=100)
    b = chunk_text("public tenant B", "dB", size=100)
    for c in a:
        c.metadata["tenant_id"] = "A"
    for c in b:
        c.metadata["tenant_id"] = "B"

    store.upsert(a, emb.embed([c.text for c in a]))
    store.upsert(b, emb.embed([c.text for c in b]))

    hits_a = store.query(emb.embed_one("secret"), k=10, tenant_id="A")
    hits_b = store.query(emb.embed_one("secret"), k=10, tenant_id="B")

    assert all(h.metadata.get("tenant_id") == "A" for h in hits_a)
    assert all(h.metadata.get("tenant_id") == "B" for h in hits_b)


def test_no_tenant_returns_all(store):
    emb = FakeEmbedder(dim=32)
    a = chunk_text("alpha content", "dA", size=100)
    b = chunk_text("beta content", "dB", size=100)
    for c in a:
        c.metadata["tenant_id"] = "A"
    for c in b:
        c.metadata["tenant_id"] = "B"

    store.upsert(a, emb.embed([c.text for c in a]))
    store.upsert(b, emb.embed([c.text for c in b]))

    hits = store.query(emb.embed_one("content"), k=10, tenant_id=None)
    tenants = {h.metadata.get("tenant_id") for h in hits}
    assert {"A", "B"}.issubset(tenants)


# ---------- pipeline integration ----------

def test_pipeline_with_qdrant_backend(collection):
    from app.rag.pipeline import RagPipeline

    pipe = RagPipeline(
        FakeEmbedder(dim=32),
        QdrantStore(url=QDRANT_URL, collection=collection, dim=32),
        chunk_size=200,
    )
    r1 = pipe.ingest(b"the quick brown fox jumps", source="a.txt")
    assert r1.n_chunks >= 1

    r2 = pipe.ingest(b"the quick brown fox jumps", source="a.txt")
    assert r2.skipped is True

    hits = pipe.retrieve("quick brown", k=2)
    assert len(hits) >= 1

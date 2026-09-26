"""Verify tenant isolation at the store + pipeline layer.

Offline; uses MemoryStore + FakeEmbedder.
"""
from app.rag.embedder import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.store import MemoryStore


def _pipe() -> RagPipeline:
    return RagPipeline(FakeEmbedder(dim=64), MemoryStore(), chunk_size=200)


# ---- store-level filter ----

def test_memory_store_tenant_filter():
    store = MemoryStore()
    pipe = RagPipeline(FakeEmbedder(dim=64), store, chunk_size=200)
    pipe.ingest(b"alpha tenant A content", tenant_id="A")
    pipe.ingest(b"beta tenant B content", tenant_id="B")

    hits_a = store.query(
        FakeEmbedder(dim=64).embed_one("alpha"),
        k=10,
        tenant_id="A",
    )
    hits_b = store.query(
        FakeEmbedder(dim=64).embed_one("beta"),
        k=10,
        tenant_id="B",
    )

    assert all(h.metadata.get("tenant_id") == "A" for h in hits_a)
    assert all(h.metadata.get("tenant_id") == "B" for h in hits_b)


def test_memory_store_no_filter_returns_all():
    store = MemoryStore()
    pipe = RagPipeline(FakeEmbedder(dim=64), store, chunk_size=200)
    pipe.ingest(b"alpha tenant A", tenant_id="A")
    pipe.ingest(b"beta tenant B", tenant_id="B")

    hits = store.query(FakeEmbedder(dim=64).embed_one("alpha"), k=10, tenant_id=None)
    tenants = {h.metadata.get("tenant_id") for h in hits}
    assert tenants == {"A", "B"}


def test_memory_store_empty_tenant_returns_none():
    store = MemoryStore()
    pipe = RagPipeline(FakeEmbedder(dim=64), store, chunk_size=200)
    pipe.ingest(b"only tenant A content", tenant_id="A")

    hits = store.query(FakeEmbedder(dim=64).embed_one("content"), k=10, tenant_id="Z")
    assert hits == []


# ---- pipeline-level: doc_id is tenant-scoped ----

def test_same_bytes_two_tenants_different_doc_ids():
    p = _pipe()
    ra = p.ingest(b"shared bytes", tenant_id="A")
    rb = p.ingest(b"shared bytes", tenant_id="B")
    assert ra.doc_id != rb.doc_id


def test_same_tenant_same_bytes_idempotent():
    p = _pipe()
    r1 = p.ingest(b"shared bytes", tenant_id="A")
    r2 = p.ingest(b"shared bytes", tenant_id="A")
    assert r1.doc_id == r2.doc_id
    assert r2.skipped is True


def test_tenant_isolation_prevents_cross_hits():
    p = _pipe()
    # tenant A ingests secret content
    p.ingest(b"SECRET-PROJECT-XYZ confidential", tenant_id="A")
    # tenant B queries for it
    hits_b = p.retrieve("SECRET-PROJECT-XYZ", k=5, tenant_id="B")
    assert all("SECRET-PROJECT-XYZ" not in h.text for h in hits_b)


def test_tenant_isolation_no_tenant_retrieves_all():
    p = _pipe()
    p.ingest(b"alpha content", tenant_id="A")
    p.ingest(b"beta content", tenant_id="B")
    hits = p.retrieve("alpha", k=10, tenant_id=None)
    # Without a tenant filter, both tenants are visible (single-tenant mode).
    tenants = {h.metadata.get("tenant_id") for h in hits}
    assert "A" in tenants


def test_ingest_without_tenant_stores_no_tenant_key():
    store = MemoryStore()
    pipe = RagPipeline(FakeEmbedder(dim=64), store, chunk_size=200)
    pipe.ingest(b"no tenant content")
    hits = store.query(FakeEmbedder(dim=64).embed_one("content"), k=10)
    assert hits
    assert "tenant_id" not in hits[0].metadata

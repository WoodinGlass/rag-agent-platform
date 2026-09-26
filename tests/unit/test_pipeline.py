from app.rag.embedder import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.store import MemoryStore


def make_pipeline(size=80):
    return RagPipeline(FakeEmbedder(dim=64), MemoryStore(), chunk_size=size)


def test_ingest_idempotent():
    p = make_pipeline()
    r1 = p.ingest(b"hello world", source="a.txt")
    r2 = p.ingest(b"hello world", source="a.txt")
    assert r1.doc_id == r2.doc_id
    assert r1.skipped is False
    assert r2.skipped is True
    assert r2.inserted == 0


def test_different_bytes_different_doc():
    p = make_pipeline()
    assert p.ingest(b"aaa").doc_id != p.ingest(b"bbb").doc_id


def test_different_chunker_version_different_doc():
    p = make_pipeline()
    r1 = p.ingest(b"hello world")
    p.chunker_version = "v2"
    r2 = p.ingest(b"hello world")
    assert r1.doc_id != r2.doc_id


def test_retrieve_returns_hits():
    p = make_pipeline(size=200)
    p.ingest(b"The cat sat on the mat. The dog barked loudly.", source="s1")
    p.ingest(b"Quantum computing uses qubits and superposition.", source="s2")
    hits = p.retrieve("cat on the mat", k=2)
    assert len(hits) >= 1
    assert hits[0].doc_id
    assert hits[0].score > 0


def test_empty_ingest_no_chunks():
    p = make_pipeline()
    r = p.ingest(b"   ")
    assert r.n_chunks == 0

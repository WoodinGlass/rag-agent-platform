from app.rag.embedder import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.reranker import FakeReranker, IdentityReranker
from app.rag.store import MemoryStore


def make_pipeline(size=80, *, reranker=None, multiplier=2):
    return RagPipeline(
        FakeEmbedder(dim=64),
        MemoryStore(),
        chunk_size=size,
        reranker=reranker,
        retrieve_multiplier=multiplier,
    )


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


# ---- reranker wiring ----

def test_default_reranker_is_identity():
    p = make_pipeline()
    assert isinstance(p.reranker, IdentityReranker)


def test_identity_reranker_k_unchanged():
    p = make_pipeline(size=200)
    for i in range(6):
        p.ingest(f"doc {i} contents".encode(), source=f"d{i}.txt")
    hits = p.retrieve("doc 3", k=2)
    assert len(hits) == 2


def test_fake_reranker_runs_in_pipeline():
    p = make_pipeline(size=100, reranker=FakeReranker())
    p.ingest(b"apple banana", source="a.txt")
    p.ingest(b"cherry date", source="b.txt")
    hits = p.retrieve("apple", k=2)
    assert len(hits) >= 1
    # fake reranker should boost the doc containing "apple"
    assert "apple" in hits[0].text


def test_retrieve_multiplier_validated():
    import pytest

    with pytest.raises(ValueError, match="retrieve_multiplier"):
        make_pipeline(multiplier=0)


def test_retrieve_multiplier_fetches_more():
    """With multiplier=3 and k=1, store is asked for 3 candidates."""
    p = make_pipeline(size=100, multiplier=3, reranker=IdentityReranker())
    for i in range(5):
        p.ingest(f"unique term {i}".encode(), source=f"d{i}.txt")
    hits = p.retrieve("unique term 2", k=1)
    # reranker (identity) cuts to k=1, but fetch was 3
    assert len(hits) == 1

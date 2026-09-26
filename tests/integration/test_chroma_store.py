import pytest

pytestmark = pytest.mark.integration

from app.rag.chunker import chunk_text
from app.rag.embedder import FakeEmbedder
from app.rag.store import ChromaStore


def test_chroma_upsert_and_query(tmp_path):
    store = ChromaStore(path=str(tmp_path), collection="test_m1")
    chunks = chunk_text("hello world", "d1", size=100)
    emb = FakeEmbedder(dim=32).embed([c.text for c in chunks])
    inserted = store.upsert(chunks, emb)
    assert inserted == len(chunks)
    assert store.has_doc("d1")

    hits = store.query(emb[0], k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == chunks[0].chunk_id


def test_chroma_idempotent_upsert(tmp_path):
    store = ChromaStore(path=str(tmp_path), collection="test_m1_idem")
    chunks = chunk_text("hello world", "d1", size=100)
    emb = FakeEmbedder(dim=32).embed([c.text for c in chunks])
    assert store.upsert(chunks, emb) == len(chunks)
    assert store.upsert(chunks, emb) == 0  # second call: no new inserts

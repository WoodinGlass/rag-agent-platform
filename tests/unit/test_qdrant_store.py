"""Offline tests for QdrantStore.

A fake `qdrant_client` module is injected into sys.modules so no real
service is touched. Covers:
- collection creation with payload indexes
- chunk_id -> UUIDv5 mapping
- upsert counting (new vs. existing)
- search with/without tenant filter
- has_doc via scroll
- import guard for missing qdrant-client
"""
from __future__ import annotations

import sys
import types

import pytest

from app.rag.chunker import chunk_text
from app.rag.embedder import FakeEmbedder
from app.rag.qdrant_store import QdrantStore, _to_uuid

# ---------- fake qdrant_client ----------

class _Point:
    def __init__(self, id: str, payload: dict | None = None, score: float | None = None):
        self.id = id
        self.payload = payload or {}
        self.score = score


class _Collection:
    def __init__(self, name: str):
        self.name = name


class _CollectionsList:
    def __init__(self, names: list[str]):
        self.collections = [_Collection(n) for n in names]


class _FakeQdrantClient:
    """Minimal stand-in. Records calls, keeps a simple in-memory store."""

    def __init__(self) -> None:
        # collection_name -> {point_id: (vector, payload)}
        self._collections: dict[str, dict[str, tuple[list[float], dict]]] = {}
        self.create_collection_calls: list[dict] = []
        self.create_payload_index_calls: list[dict] = []
        self.upsert_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self.scroll_calls: list[dict] = []

    # ---- metadata ----

    def get_collections(self):
        return _CollectionsList(list(self._collections.keys()))

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self.create_collection_calls.append(
            {"collection_name": collection_name, "vectors_config": vectors_config}
        )
        self._collections.setdefault(collection_name, {})

    def create_payload_index(self, collection_name: str, field_name: str, field_schema) -> None:
        self.create_payload_index_calls.append(
            {
                "collection_name": collection_name,
                "field_name": field_name,
                "field_schema": field_schema,
            }
        )

    # ---- data ----

    def retrieve(self, collection_name: str, ids, with_payload: bool = False):
        store = self._collections.get(collection_name, {})
        out = []
        for pid in ids:
            key = str(pid)
            if key in store:
                _vec, payload = store[key]
                out.append(_Point(key, payload))
        return out

    def upsert(self, collection_name: str, points) -> None:
        self.upsert_calls.append(
            {"collection_name": collection_name, "points": list(points)}
        )
        store = self._collections.setdefault(collection_name, {})
        for p in points:
            store[str(p.id)] = (list(p.vector), dict(p.payload))

    def query_points(
        self,
        collection_name: str,
        query,
        limit: int,
        query_filter=None,
        with_payload: bool = True,
    ):
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "query": list(query),
                "limit": limit,
                "query_filter": query_filter,
                "with_payload": with_payload,
            }
        )
        store = self._collections.get(collection_name, {})
        results: list[_Point] = []
        for pid, (vec, payload) in store.items():
            score = sum(a * b for a, b in zip(query, vec))
            results.append(_Point(pid, payload, score=score))
        results.sort(key=lambda p: p.score or 0.0, reverse=True)

        # apply tenant filter if present
        if query_filter is not None:
            for cond in getattr(query_filter, "must", []) or []:
                key = getattr(cond, "key", None)
                match = getattr(cond, "match", None)
                if key and match:
                    val = getattr(match, "value", None)
                    results = [p for p in results if p.payload.get(key) == val]

        class _QueryResponse:
            def __init__(self, points):
                self.points = points

        return _QueryResponse(results[:limit])

    def scroll(
        self,
        collection_name: str,
        scroll_filter=None,
        limit: int = 1,
        with_payload: bool = False,
        with_vectors: bool = False,
    ):
        self.scroll_calls.append(
            {
                "collection_name": collection_name,
                "scroll_filter": scroll_filter,
                "limit": limit,
            }
        )
        store = self._collections.get(collection_name, {})
        points = [_Point(pid, payload) for pid, (_vec, payload) in store.items()]
        if scroll_filter is not None:
            for cond in getattr(scroll_filter, "must", []) or []:
                key = getattr(cond, "key", None)
                match = getattr(cond, "match", None)
                if key and match:
                    val = getattr(match, "value", None)
                    points = [p for p in points if p.payload.get(key) == val]
        return points[:limit], None


def _install_fake_qdrant(monkeypatch) -> _FakeQdrantClient:
    """Inject a fake qdrant_client + models into sys.modules.

    Returns the fake client so tests can assert call shapes.
    """
    fake = _FakeQdrantClient()

    models = types.ModuleType("qdrant_client.models")

    class PointStruct:
        def __init__(self, id, vector, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload or {}

    class Distance:
        COSINE = "Cosine"

    class VectorParams:
        def __init__(self, size: int, distance):
            self.size = size
            self.distance = distance

    class PayloadSchemaType:
        KEYWORD = "keyword"

    class MatchValue:
        def __init__(self, value):
            self.value = value

    class FieldCondition:
        def __init__(self, key, match):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, must=None):
            self.must = must or []

    models.PointStruct = PointStruct
    models.Distance = Distance
    models.VectorParams = VectorParams
    models.PayloadSchemaType = PayloadSchemaType
    models.MatchValue = MatchValue
    models.FieldCondition = FieldCondition
    models.Filter = Filter

    top = types.ModuleType("qdrant_client")
    top.QdrantClient = lambda *args, **kwargs: fake
    top.models = models

    monkeypatch.setitem(sys.modules, "qdrant_client", top)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)
    return fake


# ---------- fixtures ----------

@pytest.fixture()
def fake_qdrant(monkeypatch):
    return _install_fake_qdrant(monkeypatch)


def _chunks(text: str, doc_id: str, size: int = 100):
    return chunk_text(text, doc_id, size=size)


def _embedder():
    return FakeEmbedder(dim=32)


# ---------- UUID mapping ----------

def test_to_uuid_is_deterministic():
    a = _to_uuid("doc1:0000")
    b = _to_uuid("doc1:0000")
    assert a == b
    # canonical UUID string shape
    assert len(a) == 36 and a.count("-") == 4


def test_to_uuid_differs_per_chunk_id():
    assert _to_uuid("a") != _to_uuid("b")


# ---------- construction ----------

def test_creates_collection_with_dim_and_cosine(fake_qdrant):
    QdrantStore(url="http://x", collection="c1", dim=16)
    assert len(fake_qdrant.create_collection_calls) == 1
    call = fake_qdrant.create_collection_calls[0]
    assert call["collection_name"] == "c1"
    assert call["vectors_config"].size == 16
    assert call["vectors_config"].distance == "Cosine"


def test_creates_payload_indexes_for_doc_id_and_tenant(fake_qdrant):
    QdrantStore(url="http://x", collection="c1", dim=16)
    fields = {c["field_name"] for c in fake_qdrant.create_payload_index_calls}
    assert fields == {"doc_id", "tenant_id"}


def test_existing_collection_is_not_recreated(fake_qdrant):
    QdrantStore(url="http://x", collection="c1", dim=8)
    before = len(fake_qdrant.create_collection_calls)
    QdrantStore(url="http://x", collection="c1", dim=8)
    after = len(fake_qdrant.create_collection_calls)
    assert after == before  # second instance sees existing collection


def test_invalid_dim_rejected(fake_qdrant):
    with pytest.raises(ValueError, match="dim must be > 0"):
        QdrantStore(url="http://x", collection="c1", dim=0)


# ---------- upsert ----------

def test_upsert_counts_new_points(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    chunks = _chunks("hello world", "d1")
    embs = _embedder().embed([c.text for c in chunks])

    assert store.upsert(chunks, embs) == len(chunks)


def test_upsert_is_idempotent(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    chunks = _chunks("hello world", "d1")
    embs = _embedder().embed([c.text for c in chunks])

    first = store.upsert(chunks, embs)
    second = store.upsert(chunks, embs)
    assert first == len(chunks)
    assert second == 0


def test_upsert_empty_returns_zero(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    assert store.upsert([], []) == 0
    assert fake_qdrant.upsert_calls == []  # no client call on empty input


def test_upsert_stores_chunk_id_in_payload(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    chunks = _chunks("hello world", "d1")
    embs = _embedder().embed([c.text for c in chunks])
    store.upsert(chunks, embs)

    point = fake_qdrant.upsert_calls[0]["points"][0]
    assert point.payload["chunk_id"] == chunks[0].chunk_id
    assert point.payload["doc_id"] == "d1"


# ---------- search ----------

def test_query_returns_hits_with_score(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    emb = _embedder()
    chunks = _chunks("the cat sat on the mat", "d1")
    store.upsert(chunks, emb.embed([c.text for c in chunks]))

    q = emb.embed_one("cat")
    hits = store.query(q, k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "d1"
    assert hits[0].chunk_id == chunks[0].chunk_id
    assert isinstance(hits[0].score, float)


def test_query_passes_limit_to_client(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    store.query([0.0] * 32, k=7)
    assert fake_qdrant.search_calls[-1]["limit"] == 7


def test_query_without_tenant_sends_no_filter(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    store.query([0.0] * 32, k=1)
    assert fake_qdrant.search_calls[-1]["query_filter"] is None


def test_query_with_tenant_sends_filter(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    store.query([0.0] * 32, k=1, tenant_id="acme")
    qf = fake_qdrant.search_calls[-1]["query_filter"]
    assert qf is not None
    assert qf.must[0].key == "tenant_id"
    assert qf.must[0].match.value == "acme"


def test_query_tenant_filter_isolates_results(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    emb = _embedder()
    chunks_a = chunk_text("alpha tenant A", "dA", size=100)
    store.upsert(chunks_a, emb.embed([c.text for c in chunks_a]))
    # Note: FakeEmbedder is deterministic; payload is what distinguishes tenants.
    # We tag chunks manually via metadata to simulate tenant scoping.
    chunks_a2 = chunk_text("secret tenant A content", "dA2", size=100)
    chunks_b2 = chunk_text("public tenant B content", "dB2", size=100)
    for c in chunks_a2:
        c.metadata["tenant_id"] = "A"
    for c in chunks_b2:
        c.metadata["tenant_id"] = "B"
    store.upsert(chunks_a2, emb.embed([c.text for c in chunks_a2]))
    store.upsert(chunks_b2, emb.embed([c.text for c in chunks_b2]))

    hits_a = store.query(emb.embed_one("content"), k=10, tenant_id="A")
    hits_b = store.query(emb.embed_one("content"), k=10, tenant_id="B")

    assert all(h.metadata.get("tenant_id") == "A" for h in hits_a)
    assert all(h.metadata.get("tenant_id") == "B" for h in hits_b)


# ---------- has_doc ----------

def test_has_doc_true_after_upsert(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    chunks = _chunks("hello world", "d1")
    embs = _embedder().embed([c.text for c in chunks])
    store.upsert(chunks, embs)
    assert store.has_doc("d1") is True


def test_has_doc_false_for_unknown(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    assert store.has_doc("nope") is False


def test_has_doc_uses_scroll_with_filter(fake_qdrant):
    store = QdrantStore(url="http://x", collection="c1", dim=32)
    store.has_doc("x")
    call = fake_qdrant.scroll_calls[-1]
    sf = call["scroll_filter"]
    assert sf.must[0].key == "doc_id"
    assert sf.must[0].match.value == "x"


# ---------- import guard ----------

def test_missing_client_raises_clear_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises((ImportError, TypeError)):
        QdrantStore(url="http://x", collection="c1", dim=8)

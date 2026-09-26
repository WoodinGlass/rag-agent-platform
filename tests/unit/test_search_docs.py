import pytest

from app.rag.store import Hit
from app.tools.search_docs import MAX_K, format_hits, make_search_docs_tool


def _hit(i: int, text: str = "chunk text", score: float = 0.9) -> Hit:
    return Hit(
        chunk_id=f"doc{i}:0000",
        doc_id=f"doc{i}",
        text=text,
        score=score,
        metadata={
            "doc_id": f"doc{i}",
            "chunk_id": f"doc{i}:0000",
            "text": text,
            "start": i * 10,
            "end": i * 10 + len(text),
        },
    )


# ---------- format_hits (pure) ----------

def test_format_hits_shape():
    out = format_hits("q", [_hit(1), _hit(2)], k=5)
    assert out["query"] == "q"
    assert out["n"] == 2
    assert out["truncated"] is False
    assert {"chunk_id", "doc_id", "score", "text", "start", "end", "_truncated"} <= set(
        out["hits"][0]
    )


def test_format_hits_truncates_text():
    long_text = "x" * 2000
    out = format_hits("q", [_hit(1, text=long_text)], k=1)
    assert len(out["hits"][0]["text"]) == 600
    assert out["hits"][0]["_truncated"] is True


def test_format_hits_truncated_flag():
    out = format_hits("q", [_hit(1), _hit(2), _hit(3)], k=2)
    assert out["n"] == 2
    assert out["truncated"] is True


def test_format_hits_scores_rounded():
    out = format_hits("q", [_hit(1, score=0.123456789)], k=1)
    assert out["hits"][0]["score"] == 0.123457


# ---------- tool wrapper (impure fn injected) ----------

def test_tool_calls_retrieve_fn():
    calls: list[tuple[str, int]] = []

    def fake_retrieve(q: str, k: int):
        calls.append((q, k))
        return [_hit(1)]

    t = make_search_docs_tool(fake_retrieve)
    out = t.call(query="hello", k=3)
    assert calls == [("hello", 3)]
    assert out["n"] == 1


def test_tool_default_k():
    got: list[int] = []

    def fake_retrieve(q: str, k: int):
        got.append(k)
        return []

    make_search_docs_tool(fake_retrieve).call(query="q")
    assert got == [5]


def test_tool_clamps_k_above_max():
    got: list[int] = []

    def fake_retrieve(q: str, k: int):
        got.append(k)
        return []

    make_search_docs_tool(fake_retrieve).call(query="q", k=999)
    assert got == [MAX_K]


def test_tool_clamps_k_below_min():
    got: list[int] = []

    def fake_retrieve(q: str, k: int):
        got.append(k)
        return []

    make_search_docs_tool(fake_retrieve).call(query="q", k=0)
    assert got == [1]


def test_tool_rejects_empty_query():
    t = make_search_docs_tool(lambda q, k: [])
    with pytest.raises(ValueError, match="non-empty"):
        t.call(query="")
    with pytest.raises(ValueError, match="non-empty"):
        t.call(query="   ")


def test_tool_spec_is_registry_compatible():
    t = make_search_docs_tool(lambda q, k: [])
    assert t.name == "search_docs"
    assert t.spec()["parameters"]["required"] == ["query"]
    assert "retrieval" in t.tags


def test_integration_with_memory_pipeline():
    """End-to-end: real RagPipeline + MemoryStore + FakeEmbedder."""
    from app.rag.embedder import FakeEmbedder
    from app.rag.pipeline import RagPipeline
    from app.rag.store import MemoryStore
    from app.tools.adapters.rag import make_retrieve_fn

    pipe = RagPipeline(FakeEmbedder(dim=64), MemoryStore(), chunk_size=200)
    pipe.ingest(b"The cat sat on the mat.", source="a.txt")
    pipe.ingest(b"Quantum computing uses qubits.", source="b.txt")

    tool = make_search_docs_tool(make_retrieve_fn(pipe))
    out = tool.call(query="cat mat", k=2)
    assert out["n"] >= 1
    assert out["hits"][0]["doc_id"]

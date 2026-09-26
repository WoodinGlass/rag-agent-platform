"""Tests for reranker interface + offline implementations."""
import pytest

from app.rag.reranker import (
    FakeReranker,
    IdentityReranker,
    get_reranker,
)
from app.rag.store import Hit


def _hit(cid: str, text: str, score: float = 0.5) -> Hit:
    return Hit(
        chunk_id=cid,
        doc_id=f"doc_{cid}",
        text=text,
        score=score,
        metadata={"chunk_id": cid, "doc_id": f"doc_{cid}"},
    )


# ---- IdentityReranker ----

def test_identity_preserves_order():
    hits = [_hit("a", "apple"), _hit("b", "banana"), _hit("c", "cherry")]
    out = IdentityReranker().rerank("apple", hits, top_n=3)
    assert [h.chunk_id for h in out] == ["a", "b", "c"]


def test_identity_cuts_to_top_n():
    hits = [_hit("a", "x"), _hit("b", "y"), _hit("c", "z")]
    out = IdentityReranker().rerank("q", hits, top_n=2)
    assert len(out) == 2
    assert [h.chunk_id for h in out] == ["a", "b"]


def test_identity_handles_empty():
    assert IdentityReranker().rerank("q", [], top_n=5) == []


def test_identity_top_n_larger_than_input():
    hits = [_hit("a", "x")]
    out = IdentityReranker().rerank("q", hits, top_n=10)
    assert len(out) == 1


# ---- FakeReranker ----

def test_fake_reorders_by_token_overlap():
    hits = [
        _hit("a", "banana cherry"),
        _hit("b", "apple fruit"),
        _hit("c", "unrelated text"),
    ]
    out = FakeReranker().rerank("apple", hits, top_n=3)
    assert out[0].chunk_id == "b"  # 'apple' overlap


def test_fake_deterministic():
    hits = [
        _hit("a", "alpha"),
        _hit("b", "beta"),
        _hit("c", "gamma"),
    ]
    a = FakeReranker().rerank("beta", hits, top_n=3)
    b = FakeReranker().rerank("beta", hits, top_n=3)
    assert [h.chunk_id for h in a] == [h.chunk_id for h in b]


def test_fake_returns_top_n():
    hits = [_hit(str(i), f"doc{i}") for i in range(10)]
    out = FakeReranker().rerank("doc3", hits, top_n=3)
    assert len(out) == 3


def test_fake_tie_break_by_score():
    # equal overlap -> retrieval score decides
    hits = [
        _hit("low", "common word", score=0.1),
        _hit("high", "common word", score=0.9),
    ]
    out = FakeReranker().rerank("common", hits, top_n=2)
    assert out[0].chunk_id == "high"


def test_fake_empty():
    assert FakeReranker().rerank("q", [], top_n=5) == []


# ---- factory ----

def test_factory_identity():
    assert isinstance(get_reranker("identity"), IdentityReranker)


def test_factory_fake():
    assert isinstance(get_reranker("fake"), FakeReranker)


def test_factory_unknown():
    with pytest.raises(ValueError, match="unknown reranker backend"):
        get_reranker("nope")

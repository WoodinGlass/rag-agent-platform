"""Reranker interface + offline implementations.

Why rerank?
- Retrieval (bi-encoder + ANN) is fast but approximate: it scores query
  and doc independently. A cross-encoder reads (query, doc) together and
  scores more accurately.
- Pattern: retrieve top-2N cheaply, rerank to top-N precisely.

Design:
- `Reranker` is a small ABC: `rerank(query, hits, top_n) -> list[Hit]`.
- `IdentityReranker` returns the input unchanged (default; zero cost).
- `FakeReranker` scores by a deterministic heuristic (no ML) — useful
  in tests without pulling `sentence-transformers`.
- `CrossEncoderReranker` lazy-imports `sentence_transformers`, so the
  package is optional (extra `rerank`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.rag.store import Hit


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        hits: Sequence[Hit],
        top_n: int,
    ) -> list[Hit]:
        """Return at most `top_n` hits, reordered by relevance to `query`."""


class IdentityReranker(Reranker):
    """No-op: keep the retriever's order and cut to top_n. Zero cost."""

    def rerank(
        self,
        query: str,
        hits: Sequence[Hit],
        top_n: int,
    ) -> list[Hit]:
        return list(hits[:top_n])


class FakeReranker(Reranker):
    """Deterministic heuristic reranker for tests.

    Scores by token overlap between query and hit text, with a small
    tie-breaker on the original retrieval score. Not meant to be good —
    meant to be *deterministic* and *non-identity* so we can assert the
    pipeline actually invoked it.
    """

    def __init__(self, *, weight_overlap: float = 1.0, weight_score: float = 0.1) -> None:
        self.weight_overlap = weight_overlap
        self.weight_score = weight_score

    def rerank(
        self,
        query: str,
        hits: Sequence[Hit],
        top_n: int,
    ) -> list[Hit]:
        q_tokens = set(query.lower().split())

        def _score(h: Hit) -> float:
            tokens = set(h.text.lower().split())
            overlap = len(q_tokens & tokens)
            return self.weight_overlap * overlap + self.weight_score * h.score

        ranked = sorted(hits, key=_score, reverse=True)
        return ranked[:top_n]


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker via sentence-transformers (lazy import).

    Install: pip install -e ".[rerank]"
    Model default: cross-encoder/ms-marco-MiniLM-L-6-v2 (small, CPU-friendly).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "sentence-transformers not installed; "
                "run: pip install 'rag-agent-platform[rerank]'"
            ) from e
        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        hits: Sequence[Hit],
        top_n: int,
    ) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(hits, scores),
            key=lambda p: float(p[1]),
            reverse=True,
        )
        return [h for h, _ in ranked[:top_n]]


def get_reranker(
    backend: str = "identity",
    **kwargs,
) -> Reranker:
    """Factory. `backend` is a config value, not a hardcoded string."""
    if backend == "identity":
        return IdentityReranker()
    if backend == "fake":
        return FakeReranker(**kwargs)
    if backend == "cross-encoder":
        return CrossEncoderReranker(**kwargs)
    raise ValueError(f"unknown reranker backend: {backend}")

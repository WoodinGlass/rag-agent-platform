"""Retrieval metrics. Pure functions — no IO, no LLM, deterministic.

All metrics operate on:
  retrieved: ordered list of doc_ids (best first)
  expected: set/list of relevant doc_ids (ground truth)

Definitions:
- hit_rate@k: 1.0 if any expected in top-k, else 0.0 (per query); averaged.
- mrr@k:      1/rank of first expected hit in top-k, else 0.0 (per query); averaged.
- precision@k: (expected ∩ retrieved[:k]) / k
- recall@k:    (expected ∩ retrieved[:k]) / len(expected)

`evaluate_retrieval(records, k)` returns both aggregate + per-query detail.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryMetrics:
    question: str
    expected: list[str]
    retrieved: list[str]
    hit: float
    reciprocal_rank: float
    precision: float
    recall: float


@dataclass(frozen=True)
class RetrievalReport:
    k: int
    n: int
    hit_rate: float
    mrr: float
    precision: float
    recall: float
    per_query: list[QueryMetrics]

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "n": self.n,
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "per_query": [
                {
                    "question": q.question,
                    "expected": q.expected,
                    "retrieved": q.retrieved,
                    "hit": q.hit,
                    "reciprocal_rank": round(q.reciprocal_rank, 4),
                    "precision": round(q.precision, 4),
                    "recall": round(q.recall, 4),
                }
                for q in self.per_query
            ],
        }


def _mrr_first_hit(retrieved: list[str], expected: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def _precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in expected) / k


def _recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    return len(top & expected) / len(expected)


def evaluate_retrieval(
    records: list[tuple[str, list[str], list[str]]],
    k: int = 5,
) -> RetrievalReport:
    """Compute aggregate retrieval metrics.

    records: list of (question, expected_doc_ids, retrieved_doc_ids_ordered)
    """
    per_query: list[QueryMetrics] = []
    for question, expected_ids, retrieved_ids in records:
        expected = set(expected_ids)
        top_k = retrieved_ids[:k]
        hit = 1.0 if (set(top_k) & expected) else 0.0
        rr = _mrr_first_hit(top_k, expected)
        p = _precision_at_k(retrieved_ids, expected, k)
        r = _recall_at_k(retrieved_ids, expected, k)
        per_query.append(
            QueryMetrics(
                question=question,
                expected=expected_ids,
                retrieved=retrieved_ids,
                hit=hit,
                reciprocal_rank=rr,
                precision=p,
                recall=r,
            )
        )

    n = len(per_query)
    if n == 0:
        return RetrievalReport(k=k, n=0, hit_rate=0.0, mrr=0.0, precision=0.0, recall=0.0, per_query=[])

    return RetrievalReport(
        k=k,
        n=n,
        hit_rate=sum(q.hit for q in per_query) / n,
        mrr=sum(q.reciprocal_rank for q in per_query) / n,
        precision=sum(q.precision for q in per_query) / n,
        recall=sum(q.recall for q in per_query) / n,
        per_query=per_query,
    )

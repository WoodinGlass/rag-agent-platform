from dataclasses import FrozenInstanceError

import pytest

from evals.metrics import (
    QueryMetrics,
    RetrievalReport,
    evaluate_retrieval,
)


def _rec(q, exp, got):
    return (q, exp, got)


def test_hit_rate_perfect():
    r = evaluate_retrieval([_rec("q", ["d1"], ["d1", "d2"])], k=2)
    assert r.hit_rate == 1.0


def test_hit_rate_miss():
    r = evaluate_retrieval([_rec("q", ["d1"], ["d2", "d3"])], k=2)
    assert r.hit_rate == 0.0


def test_hit_rate_half():
    recs = [
        _rec("q1", ["d1"], ["d1", "d2"]),
        _rec("q2", ["d1"], ["d2", "d3"]),
    ]
    r = evaluate_retrieval(recs, k=2)
    assert r.hit_rate == 0.5


def test_mrr_first_position():
    r = evaluate_retrieval([_rec("q", ["d1"], ["d1"])], k=5)
    assert r.mrr == 1.0


def test_mrr_second_position():
    r = evaluate_retrieval([_rec("q", ["d1"], ["dX", "d1"])], k=5)
    assert r.mrr == 0.5


def test_mrr_no_hit():
    r = evaluate_retrieval([_rec("q", ["d1"], ["dX", "dY"])], k=5)
    assert r.mrr == 0.0


def test_mrr_average():
    recs = [
        _rec("q1", ["d1"], ["d1"]),          # rr = 1
        _rec("q2", ["d1"], ["dX", "d1"]),    # rr = 0.5
    ]
    r = evaluate_retrieval(recs, k=5)
    assert r.mrr == pytest.approx(0.75)


def test_precision_at_k():
    r = evaluate_retrieval([_rec("q", ["d1", "d2"], ["d1", "d2", "d3"])], k=3)
    # 2 of top-3 are relevant
    assert r.precision == pytest.approx(2 / 3)


def test_precision_all_hits():
    r = evaluate_retrieval([_rec("q", ["d1", "d2"], ["d1", "d2"])], k=2)
    assert r.precision == 1.0


def test_precision_zero():
    r = evaluate_retrieval([_rec("q", ["d9"], ["d1", "d2"])], k=2)
    assert r.precision == 0.0


def test_recall_at_k():
    r = evaluate_retrieval([_rec("q", ["d1", "d2", "d3"], ["d1"])], k=3)
    assert r.recall == pytest.approx(1 / 3)


def test_recall_full():
    r = evaluate_retrieval([_rec("q", ["d1", "d2"], ["d1", "d2", "d3"])], k=3)
    assert r.recall == 1.0


def test_recall_empty_expected():
    r = evaluate_retrieval([_rec("q", [], ["d1"])], k=3)
    assert r.recall == 0.0


def test_empty_records():
    r = evaluate_retrieval([], k=5)
    assert r.n == 0
    assert r.hit_rate == 0.0
    assert r.per_query == []


def test_k_smaller_than_retrieved():
    # expected is at position 3; k=2 means it should not count
    r = evaluate_retrieval([_rec("q", ["d3"], ["d1", "d2", "d3"])], k=2)
    assert r.hit_rate == 0.0
    assert r.mrr == 0.0


def test_report_to_dict_shape():
    r = evaluate_retrieval([_rec("q", ["d1"], ["d1"])], k=1)
    d = r.to_dict()
    assert d["k"] == 1
    assert d["n"] == 1
    assert "hit_rate" in d
    assert "mrr" in d
    assert "precision" in d
    assert "recall" in d
    assert isinstance(d["per_query"], list)


def test_multiple_expected_partial_match_counts_hit():
    r = evaluate_retrieval([_rec("q", ["d1", "d2"], ["dX", "d1"])], k=2)
    # d1 in top-2, so hit
    assert r.hit_rate == 1.0
    assert r.mrr == 0.5


def test_query_metrics_frozen():
    qm = QueryMetrics(
        question="q", expected=["d1"], retrieved=["d1"],
        hit=1.0, reciprocal_rank=1.0, precision=1.0, recall=1.0,
    )
    with pytest.raises(FrozenInstanceError):
        qm.hit = 0.0  # type: ignore[misc]


def test_retrieval_report_frozen():
    r = evaluate_retrieval([_rec("q", ["d1"], ["d1"])], k=1)
    assert isinstance(r, RetrievalReport)
    with pytest.raises(FrozenInstanceError):
        r.hit_rate = 0.0  # type: ignore[misc]

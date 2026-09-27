"""Shape and determinism tests for the synthetic corpus + scale benchmark.

Offline. No network. Small N so the test runs in milliseconds.
"""
from __future__ import annotations

import pytest

from benchmarks.scale_benchmark import render_markdown, run
from benchmarks.synthetic_corpus import (
    TEMPLATE_VERSION,
    generate_corpus,
    generate_queries,
)

# ---------- generator ----------

def test_generate_corpus_size():
    docs = generate_corpus(10)
    assert len(docs) == 10


def test_generate_corpus_ids_unique_and_ordered():
    docs = generate_corpus(20)
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))
    assert ids[0] == "syn-0000"
    assert ids[-1] == "syn-0019"


def test_generate_corpus_signature_present_in_text():
    docs = generate_corpus(5)
    for i, doc in enumerate(docs):
        sig = f"sig-{i:04d}"
        # present as a standalone whitespace token so bag-of-tokens
        # embedders (FakeEmbedder) can ground on it
        assert sig in doc.text.split()
        assert sig in doc.text  # also as substring, for readability


def test_generate_corpus_first_token_is_signature():
    docs = generate_corpus(3)
    for i, doc in enumerate(docs):
        assert doc.text.split()[0] == f"sig-{i:04d}"


def test_generate_corpus_deterministic_by_seed():
    a = generate_corpus(20, seed=1)
    b = generate_corpus(20, seed=1)
    assert [d.text for d in a] == [d.text for d in b]


def test_generate_corpus_differs_across_seeds():
    a = generate_corpus(20, seed=1)
    b = generate_corpus(20, seed=2)
    assert [d.text for d in a] != [d.text for d in b]


def test_generate_corpus_rejects_zero():
    with pytest.raises(ValueError, match="n_docs"):
        generate_corpus(0)


def test_generate_corpus_template_version_present():
    assert isinstance(TEMPLATE_VERSION, str)
    assert TEMPLATE_VERSION


# ---------- queries ----------

def test_generate_queries_shape():
    docs = generate_corpus(30)
    queries = generate_queries(docs, n_queries=10)
    assert len(queries) == 10
    for q, expected in queries:
        assert q.startswith("sig-")
        assert expected.startswith("syn-")


def test_generate_queries_matches_corpus_ids():
    docs = generate_corpus(50)
    valid = {d.id for d in docs}
    queries = generate_queries(docs, n_queries=20)
    assert {exp for _, exp in queries} <= valid


def test_generate_queries_deterministic():
    docs = generate_corpus(30)
    a = generate_queries(docs, n_queries=10, seed=7)
    b = generate_queries(docs, n_queries=10, seed=7)
    assert a == b


def test_generate_queries_capped_by_corpus():
    docs = generate_corpus(5)
    queries = generate_queries(docs, n_queries=100)
    assert len(queries) == 5


def test_generate_queries_rejects_zero():
    docs = generate_corpus(5)
    with pytest.raises(ValueError, match="n_queries"):
        generate_queries(docs, n_queries=0)


# ---------- benchmark run ----------

def test_scale_benchmark_run_tiny():
    report = run(
        n_docs=20,
        n_queries=5,
        seed=1,
        k=3,
        chunk_size=200,
        embedder_name="fake",
    )
    assert report["config"]["n_docs"] == 20
    assert report["config"]["n_queries"] == 5
    assert report["ingest"]["total_ms"] > 0
    assert report["ingest"]["per_doc_ms"] > 0
    assert report["retrieve"]["latency_ms"]["p50"] > 0
    # synthetic signature is unambiguous; expect near-perfect hit rate.
    # >= 0.9 leaves headroom for any embedding quirk without hiding a
    # real regression.
    assert report["retrieve"]["quality"]["hit_rate"] >= 0.9


def test_scale_benchmark_deterministic():
    a = run(n_docs=15, n_queries=5, seed=1, k=3,
            chunk_size=200, embedder_name="fake")
    b = run(n_docs=15, n_queries=5, seed=1, k=3,
            chunk_size=200, embedder_name="fake")
    # quality is deterministic; latency varies so we do not compare it
    assert a["retrieve"]["quality"] == b["retrieve"]["quality"]
    assert a["config"] == b["config"]


def test_scale_benchmark_markdown_shape():
    report = run(n_docs=10, n_queries=3, seed=1, k=3,
                 chunk_size=200, embedder_name="fake")
    md = render_markdown(report)
    assert "Scale benchmark: 10 documents" in md
    assert "Ingest (measured)" in md
    assert "Retrieve latency" in md
    assert "hit_rate@3" in md

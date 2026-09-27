"""Shape tests for the real cross-encoder benchmark.

These tests do NOT require sentence-transformers. They verify:
- the module imports without the optional dep
- the skip path returns a well-formed report
- the markdown renderer handles both skip and success shapes
"""
from benchmarks.reranker_real import render_markdown, run


def test_run_returns_report_without_rerank_extra(tmp_path):
    """Without [rerank] installed the function must skip cleanly."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"d1","text":"x","source":"s"}\n')
    evalset = tmp_path / "evalset.jsonl"
    evalset.write_text('{"question":"q","expected_ids":["d1"]}\n')

    report = run(
        corpus_path=corpus,
        eval_set_path=evalset,
        k=3,
        chunk_size=100,
        embedder_name="fake",
        retrieve_multiplier=2,
        warmup=0,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    assert "generated_at" in report
    assert "config" in report
    if report.get("skipped"):
        assert "reason" in report
        md = render_markdown(report)
        assert "SKIPPED" in md
    else:
        assert "baseline" in report
        assert "reranked" in report
        assert "delta" in report
        md = render_markdown(report)
        assert "Cross-encoder" in md


def test_render_markdown_skip_shape():
    report = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "skipped": True,
        "reason": "no deps",
        "config": {
            "model": "x", "k": 5, "chunk_size": 200,
            "retrieve_multiplier": 2, "warmup": 2,
        },
    }
    md = render_markdown(report)
    assert "SKIPPED" in md
    assert "no deps" in md

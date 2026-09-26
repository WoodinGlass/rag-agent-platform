"""Benchmark the offline pipeline.

What we measure:
- Ingest: total + per-doc milliseconds (chunk -> embed -> store).
- Retrieve: latency percentiles over N queries with a warmup phase.

What we project (clearly labelled):
- Embedding cost of the corpus (OpenAI text-embedding-3-small, list price).
- Per-query LLM cost (gpt-4o-mini, in + out, list price).
- Token counts come from a whitespace tokenizer — an estimate, not tiktoken.

Outputs:
- benchmarks/results/latest.json  (machine-readable)
- benchmarks/results/latest.md    (human-readable)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

from app.core.logging import configure_logging
from app.rag.embedder import FakeEmbedder, get_embedder
from app.rag.pipeline import RagPipeline
from app.rag.store import MemoryStore

# ---- pricing snapshot (USD per 1M tokens), labelled in report ----
PRICES = {
    "embedding_usd_per_1m_tokens": 0.02,   # text-embedding-3-small
    "llm_input_usd_per_1m_tokens": 0.15,   # gpt-4o-mini
    "llm_output_usd_per_1m_tokens": 0.60,  # gpt-4o-mini
}

# ---- cost-model assumptions (labelled in report) ----
ASSUMED_CONTEXT_TOKENS_PER_QUERY = 200
ASSUMED_OUTPUT_TOKENS_PER_QUERY = 80


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile. p in [0, 100]."""
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (p / 100) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _summarize(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"n": 0}
    return {
        "n": len(latencies_ms),
        "mean_ms": round(statistics.fmean(latencies_ms), 3),
        "p50_ms": round(_percentile(latencies_ms, 50), 3),
        "p90_ms": round(_percentile(latencies_ms, 90), 3),
        "p95_ms": round(_percentile(latencies_ms, 95), 3),
        "p99_ms": round(_percentile(latencies_ms, 99), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
    }


def _estimate_tokens(text: str) -> int:
    """Whitespace tokenizer. Labelled as estimate in reports."""
    return len(text.split())


def _build_pipeline(embedder_name: str, chunk_size: int) -> RagPipeline:
    if embedder_name == "fake":
        embedder = FakeEmbedder(dim=256)
    else:
        embedder = get_embedder(embedder_name)
    return RagPipeline(embedder, MemoryStore(), chunk_size=chunk_size)


def run_benchmark(
    *,
    corpus_path: Path,
    eval_set_path: Path,
    n: int,
    k: int,
    chunk_size: int,
    embedder_name: str,
    warmup: int = 10,
) -> dict:
    pipe = _build_pipeline(embedder_name, chunk_size)
    corpus = _read_jsonl(corpus_path)
    eval_rows = _read_jsonl(eval_set_path)
    if not corpus or not eval_rows:
        raise ValueError("corpus and eval set must be non-empty")

    # ---- ingest (measured) ----
    ingest_t0 = time.perf_counter()
    for row in corpus:
        pipe.ingest(row["text"].encode("utf-8"), source=row.get("source", ""))
    ingest_ms = (time.perf_counter() - ingest_t0) * 1000

    # ---- retrieve warmup (not measured) ----
    for i in range(warmup):
        row = eval_rows[i % len(eval_rows)]
        pipe.retrieve(row["question"], k=k)

    # ---- retrieve measured ----
    latencies: list[float] = []
    for i in range(n):
        row = eval_rows[i % len(eval_rows)]
        t0 = time.perf_counter()
        pipe.retrieve(row["question"], k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

    # ---- cost model (projected) ----
    corpus_tokens = sum(_estimate_tokens(r["text"]) for r in corpus)
    avg_question_tokens = (
        sum(_estimate_tokens(r["question"]) for r in eval_rows) / len(eval_rows)
    )
    embed_cost_usd = (
        corpus_tokens / 1_000_000 * PRICES["embedding_usd_per_1m_tokens"]
    )
    per_query_cost_usd = (
        (avg_question_tokens + ASSUMED_CONTEXT_TOKENS_PER_QUERY)
        / 1_000_000 * PRICES["llm_input_usd_per_1m_tokens"]
        + ASSUMED_OUTPUT_TOKENS_PER_QUERY
        / 1_000_000 * PRICES["llm_output_usd_per_1m_tokens"]
    )

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": {
            "n": n,
            "k": k,
            "chunk_size": chunk_size,
            "embedder": embedder_name,
            "warmup": warmup,
            "corpus_size": len(corpus),
            "eval_set_size": len(eval_rows),
        },
        "ingest": {
            "total_ms": round(ingest_ms, 3),
            "per_doc_ms": round(ingest_ms / len(corpus), 3),
        },
        "retrieve": _summarize(latencies),
        "cost_model": {
            "note": (
                "Projection only. Measured runs use FakeEmbedder + FakeLLM "
                "(zero API cost). Token counts use whitespace tokenizer."
            ),
            "prices_usd_per_1m_tokens": PRICES,
            "assumptions": {
                "context_tokens_per_query": ASSUMED_CONTEXT_TOKENS_PER_QUERY,
                "output_tokens_per_query": ASSUMED_OUTPUT_TOKENS_PER_QUERY,
            },
            "corpus_tokens_estimated": corpus_tokens,
            "corpus_embedding_cost_usd": round(embed_cost_usd, 6),
            "avg_question_tokens": round(avg_question_tokens, 2),
            "per_query_cost_usd": round(per_query_cost_usd, 6),
            "cost_per_1k_queries_usd": round(per_query_cost_usd * 1000, 4),
        },
    }


def render_markdown(report: dict) -> str:
    cfg = report["config"]
    r = report["retrieve"]
    ing = report["ingest"]
    c = report["cost_model"]
    lines = [
        "# Benchmarks",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Corpus: {cfg['corpus_size']} docs | chunk_size={cfg['chunk_size']} | embedder=`{cfg['embedder']}`",
        f"- Retrieve: n={cfg['n']} (warmup {cfg['warmup']}) | k={cfg['k']}",
        "",
        "## Ingest (measured)",
        "",
        f"- Total: **{ing['total_ms']} ms**",
        f"- Per document: **{ing['per_doc_ms']} ms**",
        "",
        "## Retrieve latency (measured)",
        "",
        "| Metric | ms |",
        "|---|---|",
        f"| mean | {r['mean_ms']} |",
        f"| p50 | {r['p50_ms']} |",
        f"| p90 | {r['p90_ms']} |",
        f"| p95 | {r['p95_ms']} |",
        f"| p99 | {r['p99_ms']} |",
        f"| min | {r['min_ms']} |",
        f"| max | {r['max_ms']} |",
        "",
        "## Cost model (projected, not measured)",
        "",
        (
            f"- Corpus embedding: **${c['corpus_embedding_cost_usd']}** "
            f"({c['corpus_tokens_estimated']} tokens)"
        ),
        (
            f"- Per query: **${c['per_query_cost_usd']}** "
            f"(avg question {c['avg_question_tokens']} tok + "
            f"{c['assumptions']['context_tokens_per_query']} ctx + "
            f"{c['assumptions']['output_tokens_per_query']} out)"
        ),
        f"- Per 1,000 queries: **${c['cost_per_1k_queries_usd']}**",
        "",
        f"> {c['note']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="evals/data/corpus.jsonl")
    ap.add_argument("--eval-set", default="evals/data/eval_set.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--embedder", default="fake", choices=["fake", "openai"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out-dir", default="benchmarks/results")
    args = ap.parse_args(argv)

    configure_logging(level="WARNING", json_output=False)

    report = run_benchmark(
        corpus_path=Path(args.corpus),
        eval_set_path=Path(args.eval_set),
        n=args.n,
        k=args.k,
        chunk_size=args.chunk_size,
        embedder_name=args.embedder,
        warmup=args.warmup,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "latest.md").write_text(render_markdown(report))
    print(f"wrote {out_dir}/latest.json")
    print(f"wrote {out_dir}/latest.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

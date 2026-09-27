"""Scale benchmark: ingest + retrieve at N documents.

Complements `benchmarks/run.py` (12 docs) and `benchmarks/reranker_*`
(reranker deltas). This one asks: what happens when the corpus is big
enough that vector search and store overhead start to matter?

Default: N=1000, 200 retrieval queries. Deterministic: same seed ->
same numbers. Output committed to `benchmarks/results/scale_<N>.{json,md}`.

Not a load test. Single-threaded, one worker, no concurrency. The
`docs/limitations.md` file is explicit about this.
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
from benchmarks.run import _percentile
from benchmarks.synthetic_corpus import (
    TEMPLATE_VERSION,
    generate_corpus,
    generate_queries,
)
from evals.metrics import evaluate_retrieval


def _build_pipeline(embedder_name: str, chunk_size: int) -> RagPipeline:
    embedder = (
        FakeEmbedder(dim=256)
        if embedder_name == "fake"
        else get_embedder(embedder_name)
    )
    return RagPipeline(embedder, MemoryStore(), chunk_size=chunk_size)


def run(
    *,
    n_docs: int,
    n_queries: int,
    seed: int,
    k: int,
    chunk_size: int,
    embedder_name: str,
) -> dict:
    corpus = generate_corpus(n_docs, seed=seed)
    queries = generate_queries(corpus, n_queries=n_queries, seed=seed + 1)

    pipe = _build_pipeline(embedder_name, chunk_size)

    # ---- ingest (measured) ----
    # Map fixture id (syn-NNNN) to the pipeline's content-derived doc_id
    # (sha256-based). The pipeline is the source of truth for doc_id;
    # the synthetic id is only a fixture identifier.
    id_map: dict[str, str] = {}
    ingest_t0 = time.perf_counter()
    for doc in corpus:
        r = pipe.ingest(doc.text.encode("utf-8"), source=doc.source)
        id_map[doc.id] = r.doc_id
    ingest_ms = (time.perf_counter() - ingest_t0) * 1000

    # ---- warmup (not measured) ----
    for _ in range(10):
        pipe.retrieve(queries[0][0], k=k)

    # ---- retrieve (measured) ----
    records: list[tuple[str, list[str], list[str]]] = []
    latencies: list[float] = []
    for question, expected_syn_id in queries:
        expected_pipeline_id = id_map.get(expected_syn_id, expected_syn_id)
        t0 = time.perf_counter()
        hits = pipe.retrieve(question, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
        got = [h.doc_id for h in hits]
        records.append((question, [expected_pipeline_id], got))

    retrieval = evaluate_retrieval(records, k=k).to_dict()

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": {
            "n_docs": n_docs,
            "n_queries": n_queries,
            "seed": seed,
            "k": k,
            "chunk_size": chunk_size,
            "embedder": embedder_name,
            "template_version": TEMPLATE_VERSION,
        },
        "ingest": {
            "total_ms": round(ingest_ms, 3),
            "per_doc_ms": round(ingest_ms / n_docs, 4),
            "docs_per_sec": round(n_docs / (ingest_ms / 1000), 1) if ingest_ms else 0.0,
        },
        "retrieve": {
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 3),
                "p50": round(_percentile(latencies, 50), 3),
                "p90": round(_percentile(latencies, 90), 3),
                "p95": round(_percentile(latencies, 95), 3),
                "p99": round(_percentile(latencies, 99), 3),
                "min": round(min(latencies), 3),
                "max": round(max(latencies), 3),
            },
            "quality": {
                "hit_rate": retrieval["hit_rate"],
                "mrr": retrieval["mrr"],
                "recall": retrieval["recall"],
            },
        },
    }


def render_markdown(report: dict) -> str:
    cfg = report["config"]
    ing = report["ingest"]
    lat = report["retrieve"]["latency_ms"]
    q = report["retrieve"]["quality"]
    lines = [
        f"# Scale benchmark: {cfg['n_docs']} documents",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Docs: {cfg['n_docs']} | queries: {cfg['n_queries']} | k={cfg['k']}",
        (
            f"- chunk_size={cfg['chunk_size']} | embedder=`{cfg['embedder']}` | "
            f"template={cfg['template_version']} | seed={cfg['seed']}"
        ),
        "",
        "## Ingest (measured)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| total | {ing['total_ms']} ms |",
        f"| per document | {ing['per_doc_ms']} ms |",
        f"| throughput | {ing['docs_per_sec']} docs/s |",
        "",
        "## Retrieve latency (measured)",
        "",
        "| Metric | ms |",
        "|---|---|",
        f"| mean | {lat['mean']} |",
        f"| p50 | {lat['p50']} |",
        f"| p90 | {lat['p90']} |",
        f"| p95 | {lat['p95']} |",
        f"| p99 | {lat['p99']} |",
        f"| min | {lat['min']} |",
        f"| max | {lat['max']} |",
        "",
        "## Retrieval quality",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| hit_rate@{cfg['k']} | {q['hit_rate']} |",
        f"| MRR@{cfg['k']} | {q['mrr']} |",
        f"| recall@{cfg['k']} | {q['recall']} |",
        "",
        "> Single-threaded. `MemoryStore`. `FakeEmbedder(dim=256)`. See",
        "> `docs/benchmarks.md` for interpretation and honest caveats.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=1000)
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--embedder", default="fake", choices=["fake", "openai"])
    ap.add_argument("--out-dir", default="benchmarks/results")
    args = ap.parse_args(argv)

    configure_logging(level="WARNING", json_output=False)

    report = run(
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        seed=args.seed,
        k=args.k,
        chunk_size=args.chunk_size,
        embedder_name=args.embedder,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scale_{args.n_docs}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / f"{stem}.md").write_text(render_markdown(report))
    print(f"wrote {out_dir}/{stem}.json")
    print(f"wrote {out_dir}/{stem}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

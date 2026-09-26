"""Measure the impact of the reranker.

Runs the retrieval evaluation twice — once with `IdentityReranker`
(baseline) and once with `FakeReranker` — and reports the delta on
quality and latency.

Why this file exists: to make reranker claims falsifiable. If the
heuristic reranker helps, we show it. If it hurts, we show that too,
and the honest conclusion is "a real cross-encoder is required".

Output:
- benchmarks/results/reranker_delta.json
- benchmarks/results/reranker_delta.md
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
from app.rag.reranker import FakeReranker, IdentityReranker, Reranker
from app.rag.store import MemoryStore
from benchmarks.run import _percentile
from evals.metrics import evaluate_retrieval


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_pipeline(
    corpus: list[dict],
    *,
    embedder_name: str,
    chunk_size: int,
    reranker: Reranker,
    retrieve_multiplier: int,
) -> tuple[RagPipeline, dict[str, str]]:
    if embedder_name == "fake":
        embedder = FakeEmbedder(dim=256)
    else:
        embedder = get_embedder(embedder_name)

    pipe = RagPipeline(
        embedder,
        MemoryStore(),
        chunk_size=chunk_size,
        reranker=reranker,
        retrieve_multiplier=retrieve_multiplier,
    )
    id_map: dict[str, str] = {}
    for row in corpus:
        r = pipe.ingest(
            row["text"].encode("utf-8"),
            source=row.get("source", ""),
            metadata={"corpus_id": row["id"]},
        )
        id_map[row["id"]] = r.doc_id
    return pipe, id_map


def _run_once(
    corpus: list[dict],
    eval_rows: list[dict],
    *,
    embedder_name: str,
    chunk_size: int,
    k: int,
    reranker: Reranker,
    retrieve_multiplier: int,
    warmup: int,
) -> dict:
    pipe, id_map = _build_pipeline(
        corpus,
        embedder_name=embedder_name,
        chunk_size=chunk_size,
        reranker=reranker,
        retrieve_multiplier=retrieve_multiplier,
    )

    # warmup
    for i in range(warmup):
        row = eval_rows[i % len(eval_rows)]
        pipe.retrieve(row["question"], k=k)

    records: list[tuple[str, list[str], list[str]]] = []
    latencies: list[float] = []
    for row in eval_rows:
        expected_ids = [id_map[i] for i in row["expected_ids"] if i in id_map]
        t0 = time.perf_counter()
        hits = pipe.retrieve(row["question"], k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
        got_ids = [h.doc_id for h in hits]
        records.append((row["question"], expected_ids, got_ids))

    metrics = evaluate_retrieval(records, k=k).to_dict()
    metrics["latency_ms"] = {
        "mean": round(statistics.fmean(latencies), 3),
        "p50": round(_percentile(latencies, 50), 3),
        "p95": round(_percentile(latencies, 95), 3),
        "max": round(max(latencies), 3),
    }
    return metrics


def _delta(a: float, b: float) -> float:
    return round(b - a, 4)


def run(
    *,
    corpus_path: Path,
    eval_set_path: Path,
    k: int,
    chunk_size: int,
    embedder_name: str,
    retrieve_multiplier: int,
    warmup: int,
) -> dict:
    corpus = _read_jsonl(corpus_path)
    eval_rows = _read_jsonl(eval_set_path)

    baseline = _run_once(
        corpus, eval_rows,
        embedder_name=embedder_name, chunk_size=chunk_size, k=k,
        reranker=IdentityReranker(),
        retrieve_multiplier=retrieve_multiplier, warmup=warmup,
    )
    reranked = _run_once(
        corpus, eval_rows,
        embedder_name=embedder_name, chunk_size=chunk_size, k=k,
        reranker=FakeReranker(),
        retrieve_multiplier=retrieve_multiplier, warmup=warmup,
    )

    quality_delta = {
        "hit_rate": _delta(baseline["hit_rate"], reranked["hit_rate"]),
        "mrr": _delta(baseline["mrr"], reranked["mrr"]),
        "precision": _delta(baseline["precision"], reranked["precision"]),
        "recall": _delta(baseline["recall"], reranked["recall"]),
    }
    latency_delta = {
        "mean_ms": _delta(baseline["latency_ms"]["mean"], reranked["latency_ms"]["mean"]),
        "p50_ms": _delta(baseline["latency_ms"]["p50"], reranked["latency_ms"]["p50"]),
        "p95_ms": _delta(baseline["latency_ms"]["p95"], reranked["latency_ms"]["p95"]),
    }

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": {
            "k": k,
            "chunk_size": chunk_size,
            "embedder": embedder_name,
            "retrieve_multiplier": retrieve_multiplier,
            "warmup": warmup,
            "corpus_size": len(corpus),
            "eval_set_size": len(eval_rows),
        },
        "baseline": {"reranker": "identity", **baseline},
        "reranked": {"reranker": "fake", **reranked},
        "delta": {"quality": quality_delta, "latency": latency_delta},
    }


def render_markdown(report: dict) -> str:
    cfg = report["config"]
    b = report["baseline"]
    r = report["reranked"]
    q = report["delta"]["quality"]
    lat = report["delta"]["latency"]

    lines = [
        "# Reranker delta",
        "",
        f"- Generated: `{report['generated_at']}`",
        (
            f"- Corpus: {cfg['corpus_size']} docs | k={cfg['k']} | "
            f"fetch multiplier={cfg['retrieve_multiplier']}"
        ),
        f"- Embedder: `{cfg['embedder']}` | warmup {cfg['warmup']}",
        "",
        "## Quality (retrieval metrics)",
        "",
        "| Metric | Identity (baseline) | Fake (reranked) | Delta |",
        "|---|---|---|---|",
        f"| hit_rate@{cfg['k']} | {b['hit_rate']} | {r['hit_rate']} | {q['hit_rate']:+.4f} |",
        f"| MRR@{cfg['k']} | {b['mrr']} | {r['mrr']} | {q['mrr']:+.4f} |",
        f"| precision@{cfg['k']} | {b['precision']} | {r['precision']} | {q['precision']:+.4f} |",
        f"| recall@{cfg['k']} | {b['recall']} | {r['recall']} | {q['recall']:+.4f} |",
        "",
        "## Latency (per query, ms)",
        "",
        "| Metric | Identity | Fake | Delta |",
        "|---|---|---|---|",
        f"| mean | {b['latency_ms']['mean']} | {r['latency_ms']['mean']} | {lat['mean_ms']:+.3f} |",
        f"| p50 | {b['latency_ms']['p50']} | {r['latency_ms']['p50']} | {lat['p50_ms']:+.3f} |",
        f"| p95 | {b['latency_ms']['p95']} | {r['latency_ms']['p95']} | {lat['p95_ms']:+.3f} |",
        "",
        "> `FakeReranker` is a deterministic token-overlap heuristic, not a",
        "> cross-encoder. This measures the *pipeline wiring cost* and gives",
        "> a sanity check on ranking change, not a claim about cross-encoder",
        "> quality. A real cross-encoder (`sentence-transformers`) can be",
        "> plugged in via `CrossEncoderReranker` for production eval.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="evals/data/corpus.jsonl")
    ap.add_argument("--eval-set", default="evals/data/eval_set.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--embedder", default="fake", choices=["fake", "openai"])
    ap.add_argument("--retrieve-multiplier", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out-dir", default="benchmarks/results")
    args = ap.parse_args(argv)

    configure_logging(level="WARNING", json_output=False)

    report = run(
        corpus_path=Path(args.corpus),
        eval_set_path=Path(args.eval_set),
        k=args.k,
        chunk_size=args.chunk_size,
        embedder_name=args.embedder,
        retrieve_multiplier=args.retrieve_multiplier,
        warmup=args.warmup,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reranker_delta.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "reranker_delta.md").write_text(render_markdown(report))
    print(f"wrote {out_dir}/reranker_delta.json")
    print(f"wrote {out_dir}/reranker_delta.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

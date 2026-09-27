"""Real cross-encoder reranker benchmark.

Mirrors `benchmarks/reranker_delta.py` but uses the actual
`CrossEncoderReranker` (sentence-transformers) instead of the
`FakeReranker` heuristic. Purpose: produce defensible numbers for the
README claim "a real cross-encoder improves ranking".

Skip-friendly:
- If `sentence-transformers` is not installed, prints a clear message
  and exits 0 with `skipped: true` in the JSON report.
- Model is downloaded on first run; cached afterwards (~90 MB).

Usage:
    pip install -e ".[rerank]"
    python -m benchmarks.reranker_real --k 5 --retrieve-multiplier 2
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
from app.rag.reranker import IdentityReranker, Reranker
from app.rag.store import MemoryStore
from benchmarks.run import _percentile
from evals.metrics import evaluate_retrieval

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _try_build_cross_encoder(model_name: str) -> Reranker | None:
    from app.rag.reranker import CrossEncoderReranker

    try:
        return CrossEncoderReranker(model_name=model_name)
    except ImportError as e:
        print(f"sentence-transformers not installed: {e}")
        return None
    except Exception as e:  # noqa: BLE001 -- model download/runtime failure
        print(f"failed to build CrossEncoderReranker: {e}")
        return None


def _build_pipeline(
    corpus: list[dict],
    *,
    embedder_name: str,
    chunk_size: int,
    reranker: Reranker,
    retrieve_multiplier: int,
) -> tuple[RagPipeline, dict[str, str]]:
    embedder = (
        FakeEmbedder(dim=256)
        if embedder_name == "fake"
        else get_embedder(embedder_name)
    )
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
        records.append((row["question"], expected_ids, [h.doc_id for h in hits]))

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
    model_name: str,
) -> dict:
    cross = _try_build_cross_encoder(model_name)
    if cross is None:
        return {
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "skipped": True,
            "reason": (
                "sentence-transformers not installed or model unavailable; "
                "run: pip install -e '.[rerank]'"
            ),
            "config": {
                "k": k,
                "chunk_size": chunk_size,
                "embedder": embedder_name,
                "retrieve_multiplier": retrieve_multiplier,
                "warmup": warmup,
                "model": model_name,
            },
        }

    corpus = _read_jsonl(corpus_path)
    eval_rows = _read_jsonl(eval_set_path)

    # Warm the cross-encoder with a dummy call so the first real query
    # doesn't carry model-loading latency.
    _ = cross.rerank("warmup", [], top_n=1)

    baseline = _run_once(
        corpus, eval_rows,
        embedder_name=embedder_name, chunk_size=chunk_size, k=k,
        reranker=IdentityReranker(),
        retrieve_multiplier=retrieve_multiplier, warmup=warmup,
    )
    reranked = _run_once(
        corpus, eval_rows,
        embedder_name=embedder_name, chunk_size=chunk_size, k=k,
        reranker=cross,
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
        "skipped": False,
        "config": {
            "k": k,
            "chunk_size": chunk_size,
            "embedder": embedder_name,
            "retrieve_multiplier": retrieve_multiplier,
            "warmup": warmup,
            "corpus_size": len(corpus),
            "eval_set_size": len(eval_rows),
            "model": model_name,
        },
        "baseline": {"reranker": "identity", **baseline},
        "reranked": {"reranker": "cross-encoder", **reranked},
        "delta": {"quality": quality_delta, "latency": latency_delta},
    }


def render_markdown(report: dict) -> str:
    cfg = report["config"]
    lines = [
        "# Cross-encoder reranker benchmark",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Model: `{cfg['model']}`",
        (
            f"- k={cfg['k']} | chunk_size={cfg['chunk_size']} | "
            f"fetch multiplier={cfg['retrieve_multiplier']} | warmup {cfg['warmup']}"
        ),
    ]
    if report.get("skipped"):
        lines += ["", f"**SKIPPED**: {report.get('reason', '')}"]
        return "\n".join(lines) + "\n"

    b = report["baseline"]
    r = report["reranked"]
    q = report["delta"]["quality"]
    lat = report["delta"]["latency"]
    lines += [
        "",
        "## Quality",
        "",
        "| Metric | Identity | Cross-encoder | Delta |",
        "|---|---|---|---|",
        f"| hit_rate@{cfg['k']} | {b['hit_rate']} | {r['hit_rate']} | {q['hit_rate']:+.4f} |",
        f"| MRR@{cfg['k']} | {b['mrr']} | {r['mrr']} | {q['mrr']:+.4f} |",
        f"| precision@{cfg['k']} | {b['precision']} | {r['precision']} | {q['precision']:+.4f} |",
        f"| recall@{cfg['k']} | {b['recall']} | {r['recall']} | {q['recall']:+.4f} |",
        "",
        "## Latency (per query, ms)",
        "",
        "| Metric | Identity | Cross-encoder | Delta |",
        "|---|---|---|---|",
        f"| mean | {b['latency_ms']['mean']} | {r['latency_ms']['mean']} | {lat['mean_ms']:+.3f} |",
        f"| p50 | {b['latency_ms']['p50']} | {r['latency_ms']['p50']} | {lat['p50_ms']:+.3f} |",
        f"| p95 | {b['latency_ms']['p95']} | {r['latency_ms']['p95']} | {lat['p95_ms']:+.3f} |",
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
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--model", default=DEFAULT_MODEL)
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
        model_name=args.model,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reranker_real.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "reranker_real.md").write_text(render_markdown(report))
    print(f"wrote {out_dir}/reranker_real.json")
    print(f"wrote {out_dir}/reranker_real.md")
    if report.get("skipped"):
        print(f"note: skipped — {report['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

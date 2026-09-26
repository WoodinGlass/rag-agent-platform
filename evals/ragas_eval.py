"""Unified eval CLI.

Modes:
  --mode retrieval   (default) offline, deterministic metrics (hit, MRR, P, R)
  --mode ragas       generation metrics via Ragas + LLM judge
                     (requires OPENAI_API_KEY; skip-friendly if missing)

Writes:
  evals/reports/latest.json   machine-readable
  evals/reports/latest.md     human-readable summary

Ragas mode is only imported when used, so the offline path has no extra deps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from app.core.logging import configure_logging
from app.rag.embedder import FakeEmbedder, get_embedder
from app.rag.pipeline import RagPipeline
from app.rag.store import MemoryStore
from evals.metrics import evaluate_retrieval

DEFAULT_CORPUS = "evals/data/corpus.jsonl"
DEFAULT_EVAL_SET = "evals/data/eval_set.jsonl"
DEFAULT_REPORTS_DIR = "evals/reports"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_pipeline(corpus_path: Path, *, embedder_name: str, chunk_size: int) -> tuple[RagPipeline, dict[str, str]]:
    """Ingest corpus; return pipeline + {corpus_id -> doc_id}."""
    if embedder_name == "fake":
        embedder = FakeEmbedder(dim=256)
    else:
        embedder = get_embedder(embedder_name)

    pipe = RagPipeline(embedder, MemoryStore(), chunk_size=chunk_size)
    id_map: dict[str, str] = {}
    for row in _read_jsonl(corpus_path):
        r = pipe.ingest(
            row["text"].encode("utf-8"),
            source=row.get("source", ""),
            metadata={"corpus_id": row["id"]},
        )
        id_map[row["id"]] = r.doc_id
    return pipe, id_map


# ---------- retrieval mode ----------


def run_retrieval(
    *,
    corpus_path: Path,
    eval_set_path: Path,
    k: int,
    chunk_size: int,
    embedder_name: str,
) -> dict:
    pipe, id_map = _build_pipeline(
        corpus_path, embedder_name=embedder_name, chunk_size=chunk_size
    )

    records: list[tuple[str, list[str], list[str]]] = []
    raw: list[dict] = []
    for row in _read_jsonl(eval_set_path):
        expected_ids = [id_map[i] for i in row["expected_ids"] if i in id_map]
        hits = pipe.retrieve(row["question"], k=k)
        got_ids = [h.doc_id for h in hits]
        records.append((row["question"], expected_ids, got_ids))
        raw.append(
            {
                "id": row.get("id", ""),
                "question": row["question"],
                "expected_ids": row["expected_ids"],
                "expected_doc_ids": expected_ids,
                "retrieved_doc_ids": got_ids,
                "expected_answer": row.get("expected_answer", ""),
            }
        )

    report = evaluate_retrieval(records, k=k).to_dict()
    report["mode"] = "retrieval"
    report["inputs"] = {
        "corpus": str(corpus_path),
        "eval_set": str(eval_set_path),
        "chunk_size": chunk_size,
        "embedder": embedder_name,
    }
    report["queries"] = raw
    return report


# ---------- ragas mode (LLM judge) ----------


def _ragas_available() -> bool:
    try:
        import datasets  # noqa: F401
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


def run_ragas(
    *,
    corpus_path: Path,
    eval_set_path: Path,
    k: int,
    chunk_size: int,
    embedder_name: str,
    model: str,
) -> dict:
    """Ragas metrics: faithfulness, answer_relevancy, context_precision.

    Returns a dict with `skipped=True` if deps or API key are missing.
    """
    if not _ragas_available():
        return {
            "mode": "ragas",
            "skipped": True,
            "reason": "ragas/datasets not installed; run: pip install -e '.[eval]'",
        }
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "mode": "ragas",
            "skipped": True,
            "reason": "OPENAI_API_KEY not set",
        }

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    pipe, _ = _build_pipeline(
        corpus_path, embedder_name=embedder_name, chunk_size=chunk_size
    )

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for row in _read_jsonl(eval_set_path):
        question = row["question"]
        hits = pipe.retrieve(question, k=k)
        ctx = [h.text for h in hits]
        # simple generator: return the top context as the "answer" for eval
        # (in production this would call the agent)
        answer = ctx[0] if ctx else ""

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(row.get("expected_answer", ""))

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    judge_llm = ChatOpenAI(model=model, temperature=0)
    judge_emb = OpenAIEmbeddings()

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    scores = {k_: round(float(v), 4) for k_, v in result.items() if isinstance(v, (int, float))}

    return {
        "mode": "ragas",
        "skipped": False,
        "n": len(questions),
        "model": model,
        "scores": scores,
        "inputs": {
            "corpus": str(corpus_path),
            "eval_set": str(eval_set_path),
            "chunk_size": chunk_size,
            "embedder": embedder_name,
            "k": k,
        },
    }


# ---------- report writers ----------


def write_reports(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report["generated_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    json_path = out_dir / "latest.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    md_path = out_dir / "latest.md"
    md_path.write_text(_render_markdown(report))
    return json_path, md_path


def write_badge(report: dict, out_dir: Path) -> Path | None:
    """Emit a shields.io endpoint JSON for the headline metric.

    Retrieval mode -> hit_rate@k
    Ragas mode     -> faithfulness (if present)
    Returns None if no metric available.
    """
    badge: dict | None = None

    if report["mode"] == "retrieval" and not report.get("skipped"):
        value = float(report["hit_rate"])
        color = "brightgreen" if value >= 0.9 else "green" if value >= 0.8 else "yellow"
        badge = {
            "schemaVersion": 1,
            "label": f"retrieval hit-rate@{report['k']}",
            "message": f"{value:.4f}",
            "color": color,
        }
    elif report["mode"] == "ragas" and not report.get("skipped"):
        scores = report.get("scores", {})
        if "faithfulness" in scores:
            value = float(scores["faithfulness"])
            color = "brightgreen" if value >= 0.85 else "green" if value >= 0.7 else "yellow"
            badge = {
                "schemaVersion": 1,
                "label": "faithfulness",
                "message": f"{value:.4f}",
                "color": color,
            }

    if badge is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "badge.json"
    path.write_text(json.dumps(badge, indent=2) + "\n")
    return path


def _render_markdown(report: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"- Generated: `{report.get('generated_at', '')}`",
        f"- Mode: **{report.get('mode', 'unknown')}**",
    ]
    if report.get("skipped"):
        lines.append(f"- **SKIPPED**: {report.get('reason', '')}")
        lines.append("")
        return "\n".join(lines) + "\n"

    if report["mode"] == "retrieval":
        lines += [
            f"- k: `{report['k']}`",
            f"- n queries: `{report['n']}`",
            "",
            "## Aggregate",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| hit_rate@{report['k']} | {report['hit_rate']:.4f} |",
            f"| MRR@{report['k']} | {report['mrr']:.4f} |",
            f"| precision@{report['k']} | {report['precision']:.4f} |",
            f"| recall@{report['k']} | {report['recall']:.4f} |",
            "",
            "## Per query",
            "",
            "| Question | hit | rr | p@k | r@k |",
            "|---|---|---|---|---|",
        ]
        for q in report.get("queries", []):
            pq = next(
                (x for x in report.get("per_query", []) if x["question"] == q["question"]),
                None,
            )
            if pq:
                lines.append(
                    f"| {q['question']} | {pq['hit']:.0f} | {pq['reciprocal_rank']:.2f} | "
                    f"{pq['precision']:.2f} | {pq['recall']:.2f} |"
                )
    elif report["mode"] == "ragas":
        lines += [
            f"- n queries: `{report['n']}`",
            f"- judge model: `{report['model']}`",
            "",
            "## Scores (LLM-as-judge)",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        for name, value in sorted(report["scores"].items()):
            lines.append(f"| {name} | {value:.4f} |")

    lines.append("")
    return "\n".join(lines)


# ---------- CLI ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "ragas"], default="retrieval")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    ap.add_argument("--out-dir", default=DEFAULT_REPORTS_DIR)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--embedder", default="fake", choices=["fake", "openai"])
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    args = ap.parse_args(argv)

    configure_logging(level="WARNING", json_output=False)

    if args.mode == "retrieval":
        report = run_retrieval(
            corpus_path=Path(args.corpus),
            eval_set_path=Path(args.eval_set),
            k=args.k,
            chunk_size=args.chunk_size,
            embedder_name=args.embedder,
        )
    else:
        report = run_ragas(
            corpus_path=Path(args.corpus),
            eval_set_path=Path(args.eval_set),
            k=args.k,
            chunk_size=args.chunk_size,
            embedder_name=args.embedder,
            model=args.judge_model,
        )

    json_path, md_path = write_reports(report, Path(args.out_dir))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")

    badge_path = write_badge(report, Path(args.out_dir))
    if badge_path:
        print(f"wrote {badge_path}")

    if report.get("skipped"):
        print(f"note: skipped — {report['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

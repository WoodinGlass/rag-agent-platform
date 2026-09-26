"""Retrieval hit-rate@k evaluation.

CLI:
    python -m evals.run --dataset evals/data/demo.jsonl --k 5

Dataset (JSONL, 1 row per query):
    {"question": "...", "expected_ids": ["doc_rag", ...]}

Corpus (JSONL, 1 row per doc):
    {"id": "doc_rag", "text": "...", "source": "rag.txt"}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.logging import configure_logging
from app.rag.embedder import get_embedder
from app.rag.pipeline import RagPipeline
from app.rag.store import MemoryStore

DEFAULT_CORPUS = "evals/data/corpus.jsonl"
DEFAULT_DATASET = "evals/data/demo.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--embedder", default="fake", choices=["fake", "openai"])
    ap.add_argument("--out", default=None, help="path to JSON report")
    args = ap.parse_args()

    configure_logging(level="WARNING", json_output=False)

    pipe = RagPipeline(
        get_embedder(args.embedder, dim=256) if args.embedder == "fake" else get_embedder(args.embedder),
        MemoryStore(),
        chunk_size=args.chunk_size,
    )

    # ingest
    id_map: dict[str, str] = {}
    for row in _read_jsonl(Path(args.corpus)):
        r = pipe.ingest(
            row["text"].encode("utf-8"),
            source=row.get("source", ""),
            metadata={"corpus_id": row["id"]},
        )
        id_map[row["id"]] = r.doc_id

    # eval
    total = hits = 0
    per_query: list[dict] = []
    for row in _read_jsonl(Path(args.dataset)):
        expected = [id_map[i] for i in row["expected_ids"] if i in id_map]
        got = pipe.retrieve(row["question"], k=args.k)
        got_ids = {h.doc_id for h in got}
        ok = bool(got_ids & set(expected))
        total += 1
        hits += int(ok)
        per_query.append(
            {
                "question": row["question"],
                "expected": expected,
                "got": sorted(got_ids),
                "hit": ok,
            }
        )

    rate = hits / total if total else 0.0
    print(f"hit-rate@{args.k}: {hits}/{total} = {rate:.3f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"k": args.k, "hits": hits, "total": total, "rate": rate, "per_query": per_query},
                indent=2,
            )
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

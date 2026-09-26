"""Ingest the demo corpus into a local store.

Usage:
    python scripts/ingest_demo.py
    python scripts/ingest_demo.py --backend chroma --path ./data/chroma
    python scripts/ingest_demo.py --query "What is RAG?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.logging import configure_logging
from app.rag.embedder import get_embedder
from app.rag.pipeline import RagPipeline
from app.rag.store import get_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="evals/data/corpus.jsonl")
    ap.add_argument("--backend", default="memory", choices=["memory", "chroma"])
    ap.add_argument("--path", default="./data/chroma")
    ap.add_argument("--collection", default="rag_docs")
    ap.add_argument("--embedder", default="fake", choices=["fake", "openai"])
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--query", default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    configure_logging(level="INFO", json_output=False)

    if args.backend == "chroma":
        store = get_store("chroma", path=args.path, collection=args.collection)
    else:
        store = get_store("memory")

    pipe = RagPipeline(
        get_embedder(args.embedder),
        store,
        chunk_size=args.chunk_size,
    )

    n_ok = n_skip = 0
    for line in Path(args.corpus).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        r = pipe.ingest(
            row["text"].encode("utf-8"),
            source=row.get("source", ""),
            metadata={"corpus_id": row["id"]},
        )
        print(f"[{row['id']:>16}] doc={r.doc_id[:8]} chunks={r.n_chunks} skipped={r.skipped}")
        if r.skipped:
            n_skip += 1
        else:
            n_ok += 1

    print(f"\ningested={n_ok} skipped={n_skip}")

    if args.query:
        print(f"\nquery: {args.query!r} (k={args.k})")
        for i, h in enumerate(pipe.retrieve(args.query, k=args.k), 1):
            print(f"  {i}. score={h.score:.3f} doc={h.doc_id[:8]} {h.text[:70]!r}")


if __name__ == "__main__":
    main()

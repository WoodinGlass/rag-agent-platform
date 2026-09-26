"""End-to-end streaming demo. Fully offline.

Pushes a handful of documents into an in-memory queue, runs the
StreamingIngestor against it, and shows the resulting stats.

What this proves:
- The ingestor loop works end-to-end (poll -> handler -> commit).
- The handler can ingest into the RAG pipeline (tenant-scoped).
- Dedup + retry behave as documented, without a broker.

Usage:
    python scripts/stream_demo.py
    python scripts/stream_demo.py --n 20
    python scripts/stream_demo.py --duplicates 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.core.logging import configure_logging
from app.rag.embedder import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.store import MemoryStore
from app.streaming.events import Event
from app.streaming.ingestor import StreamingIngestor
from app.streaming.memory_source import InMemoryQueueSource

DEFAULT_DOCS = [
    "RAG combines retrieval with generation.",
    "Agents call tools in a loop until a goal is reached.",
    "Idempotency means re-running has the same effect as running once.",
    "Chroma is a local vector store used for prototyping.",
    "Correlation ids let you grep one request through the logs.",
    "Recursive chunking respects paragraph and sentence boundaries.",
    "Pydantic enforces schemas at the HTTP boundary.",
    "Multi-stage Docker builds keep runtime images small.",
    "LangGraph models an agent as a state machine with cycles.",
    "OpenTelemetry spans can be exported to a collector via OTLP.",
]


async def _run(n: int, duplicates: int, tenant: str) -> int:
    source = InMemoryQueueSource()
    pipeline = RagPipeline(FakeEmbedder(dim=128), MemoryStore(), chunk_size=200)

    # Producer: push docs (and some duplicates) into the queue.
    for i in range(n):
        text = DEFAULT_DOCS[i % len(DEFAULT_DOCS)]
        source.push_bytes(
            text.encode("utf-8"),
            id=f"evt-{i:04d}",
            metadata={"source": f"doc-{i}.txt", "tenant": tenant},
        )
    # Add duplicates to exercise dedup (same event id).
    for i in range(duplicates):
        text = DEFAULT_DOCS[i % len(DEFAULT_DOCS)]
        source.push_bytes(
            text.encode("utf-8"),
            id=f"evt-{i:04d}",  # same id as above
            metadata={"source": f"doc-{i}.txt", "tenant": tenant},
        )

    # Consumer: ingest each event into the pipeline, tenant-scoped.
    async def handler(event: Event) -> None:
        r = pipeline.ingest(
            event.data,
            source=str(event.metadata.get("source", "")),
            metadata={"event_id": event.id},
            tenant_id=tenant,
        )
        print(
            f"  handled {event.id}  doc={r.doc_id[:8]}  "
            f"chunks={r.n_chunks}  skipped={r.skipped}"
        )

    ingestor = StreamingIngestor(
        source,
        handler,
        poll_timeout_s=0.05,
        max_retries=2,
        dedup_window=64,
        idle_sleep_s=0.001,
        retry_backoff_base_s=0.0,
    )

    # We know exactly how many events are queued; stop when we've
    # pulled that many. In production the ingestor would run until
    # a shutdown signal (SIGTERM) is received.
    total_pushed = n + duplicates
    for _ in range(total_pushed):
        keep = await ingestor.run_once()
        if not keep:
            break

    await source.close()

    print()
    print("stats:")
    print(json.dumps(ingestor.stats.as_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="unique docs to push")
    ap.add_argument("--duplicates", type=int, default=2, help="duplicate events to push")
    ap.add_argument("--tenant", default="public")
    args = ap.parse_args(argv)

    configure_logging(level="WARNING", json_output=False)
    return asyncio.run(_run(args.n, args.duplicates, args.tenant))


if __name__ == "__main__":
    sys.exit(main())

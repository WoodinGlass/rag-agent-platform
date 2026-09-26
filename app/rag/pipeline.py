"""Ingest + retrieve pipeline.

Ingest is idempotent: same (bytes, chunker_version) -> same doc_id.
If the store already has that doc_id, ingest is a no-op.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.ids import content_hash
from app.core.ids import doc_id as make_doc_id
from app.core.logging import get_logger
from app.rag.chunker import CHUNKER_VERSION, chunk_text
from app.rag.embedder import Embedder
from app.rag.store import Hit, VectorStore

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    inserted: int
    skipped: bool
    n_chunks: int


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        chunk_size: int = 500,
        chunker_version: str = CHUNKER_VERSION,
    ):
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.chunker_version = chunker_version

    def ingest(
        self,
        data: bytes,
        *,
        source: str = "",
        metadata: dict | None = None,
    ) -> IngestResult:
        chash = content_hash(data)
        did = make_doc_id(chash, self.chunker_version)

        if self.store.has_doc(did):
            log.info("ingest.skip", extra={"doc_id": did, "reason": "already_present"})
            return IngestResult(doc_id=did, inserted=0, skipped=True, n_chunks=0)

        text = data.decode("utf-8", errors="replace")
        chunks = chunk_text(
            text,
            did,
            size=self.chunk_size,
            version=self.chunker_version,
            metadata={"source": source, **(metadata or {})},
        )
        if not chunks:
            log.info("ingest.empty", extra={"doc_id": did})
            return IngestResult(doc_id=did, inserted=0, skipped=False, n_chunks=0)

        embeddings = self.embedder.embed([c.text for c in chunks])
        inserted = self.store.upsert(chunks, embeddings)
        log.info(
            "ingest.ok",
            extra={"doc_id": did, "n_chunks": len(chunks), "inserted": inserted},
        )
        return IngestResult(doc_id=did, inserted=inserted, skipped=False, n_chunks=len(chunks))

    def retrieve(self, question: str, k: int = 5) -> list[Hit]:
        qvec = self.embedder.embed_one(question)
        hits = self.store.query(qvec, k=k)
        log.info("retrieve.ok", extra={"k": k, "n_hits": len(hits)})
        return hits

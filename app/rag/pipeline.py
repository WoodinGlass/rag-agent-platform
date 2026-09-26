"""Ingest + retrieve pipeline.

Ingest is idempotent: same (bytes, chunker_version) -> same doc_id.
If the store already has that doc_id, ingest is a no-op.

Retrieve optionally reranks:
- Fetch top `k * retrieve_multiplier` from the store (cheap, approximate).
- If a Reranker is configured, rerank to `k` (accurate, expensive).
- Default: IdentityReranker -> behaviour unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.ids import content_hash
from app.core.ids import doc_id as make_doc_id
from app.core.logging import get_logger
from app.rag.chunker import CHUNKER_VERSION, chunk_text
from app.rag.embedder import Embedder
from app.rag.reranker import IdentityReranker, Reranker
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
        reranker: Reranker | None = None,
        retrieve_multiplier: int = 2,
    ):
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.chunker_version = chunker_version
        self.reranker: Reranker = reranker or IdentityReranker()
        if retrieve_multiplier < 1:
            raise ValueError("retrieve_multiplier must be >= 1")
        self.retrieve_multiplier = retrieve_multiplier

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
        fetch_k = max(k, k * self.retrieve_multiplier)
        candidates = self.store.query(qvec, k=fetch_k)
        reranked = self.reranker.rerank(question, candidates, top_n=k)
        log.info(
            "retrieve.ok",
            extra={
                "k": k,
                "fetch_k": fetch_k,
                "n_candidates": len(candidates),
                "n_returned": len(reranked),
                "reranker": type(self.reranker).__name__,
            },
        )
        return reranked

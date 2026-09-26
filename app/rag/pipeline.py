"""Ingest + retrieve pipeline.

Ingest is idempotent: same (bytes, chunker_version, tenant_id) -> same doc_id.
Different tenant -> different doc_id, so a doc ingested by A cannot collide
with the same bytes ingested by B.

Retrieve optionally reranks:
- Fetch top `k * retrieve_multiplier` (cheap, approximate).
- If a Reranker is configured, rerank to `k` (accurate, expensive).
- Default: IdentityReranker -> behaviour unchanged.

Multi-tenant (M5.3): `tenant_id` is written into chunk metadata on ingest
and passed to the store on retrieve. When omitted, no filtering (single-
tenant behaviour).
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
        tenant_id: str | None = None,
    ) -> IngestResult:
        chash = content_hash(data)
        # Tenant-scoped id: same bytes in two tenants produce two doc_ids.
        id_namespace = self.chunker_version if tenant_id is None else f"{tenant_id}::{self.chunker_version}"
        did = make_doc_id(chash, id_namespace)

        if self.store.has_doc(did):
            log.info("ingest.skip", extra={"doc_id": did, "reason": "already_present", "tenant": tenant_id})
            return IngestResult(doc_id=did, inserted=0, skipped=True, n_chunks=0)

        text = data.decode("utf-8", errors="replace")
        chunk_meta: dict = {"source": source, **(metadata or {})}
        if tenant_id is not None:
            chunk_meta["tenant_id"] = tenant_id

        chunks = chunk_text(
            text,
            did,
            size=self.chunk_size,
            version=self.chunker_version,
            metadata=chunk_meta,
        )
        if not chunks:
            log.info("ingest.empty", extra={"doc_id": did, "tenant": tenant_id})
            return IngestResult(doc_id=did, inserted=0, skipped=False, n_chunks=0)

        embeddings = self.embedder.embed([c.text for c in chunks])
        inserted = self.store.upsert(chunks, embeddings)
        log.info(
            "ingest.ok",
            extra={"doc_id": did, "n_chunks": len(chunks), "inserted": inserted, "tenant": tenant_id},
        )
        return IngestResult(doc_id=did, inserted=inserted, skipped=False, n_chunks=len(chunks))

    def retrieve(
        self,
        question: str,
        k: int = 5,
        *,
        tenant_id: str | None = None,
    ) -> list[Hit]:
        qvec = self.embedder.embed_one(question)
        fetch_k = max(k, k * self.retrieve_multiplier)
        candidates = self.store.query(qvec, k=fetch_k, tenant_id=tenant_id)
        reranked = self.reranker.rerank(question, candidates, top_n=k)
        log.info(
            "retrieve.ok",
            extra={
                "k": k,
                "fetch_k": fetch_k,
                "n_candidates": len(candidates),
                "n_returned": len(reranked),
                "reranker": type(self.reranker).__name__,
                "tenant": tenant_id,
            },
        )
        return reranked

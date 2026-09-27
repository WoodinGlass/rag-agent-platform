"""Qdrant vector store adapter.

Lazy import so `qdrant-client` is only required when backend=qdrant.
Missing package -> clear ImportError.

Design notes:
- Qdrant point IDs must be UUIDs or unsigned integers. Our chunk_ids
  contain a colon (`doc:0001`), so we derive a deterministic UUIDv5
  from the chunk_id. Stable across runs and idempotent on upsert.
- The collection must be created with an explicit vector dimension.
  We take `dim` at construction. Mismatched dims are the user's
  responsibility (documented in the error message from Qdrant).
- Scoring: `Distance.COSINE` yields a similarity score where higher is
  better, matching our `Hit.score` convention.
- Tenant filter uses a payload `FieldCondition`. Qdrant's payload index
  for `tenant_id` and `doc_id` is created on first collection init to
  keep filters fast.
- `has_doc` uses `scroll` with a filter — O(1) with the doc_id payload
  index, so we create it at collection init.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from app.rag.chunker import Chunk
from app.rag.store import Hit, VectorStore

# Namespace for chunk_id -> UUID v5. Any stable UUID works; using NAMESPACE_URL.
_UUID_NS = uuid.NAMESPACE_URL


def _to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(_UUID_NS, chunk_id))


class QdrantStore(VectorStore):
    """Persistent/remote vector store. Requires qdrant-client (`.[qdrant]`)."""

    def __init__(
        self,
        url: str,
        collection: str = "rag_docs",
        *,
        dim: int = 1536,
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "qdrant-client not installed; run: "
                "pip install 'rag-agent-platform[qdrant]'"
            ) from e

        if dim <= 0:
            raise ValueError("dim must be > 0")

        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection
        self._dim = dim

        self._ensure_collection(
            QdrantClient=QdrantClient,
            Distance=Distance,
            VectorParams=VectorParams,
        )

    def _ensure_collection(self, *, QdrantClient, Distance, VectorParams) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
        )
        # Index the payload fields we filter on.
        from qdrant_client.models import PayloadSchemaType

        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="tenant_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    # ---------- write ----------

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        from qdrant_client.models import PointStruct

        points: list[Any] = []
        for chunk, emb in zip(chunks, embeddings):
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "chunker_version": chunk.chunker_version,
                "index": chunk.index,
                **chunk.metadata,
            }
            points.append(
                PointStruct(
                    id=_to_uuid(chunk.chunk_id),
                    vector=[float(x) for x in emb],
                    payload=payload,
                )
            )

        # Count new inserts: retrieve the point IDs first.
        if not points:
            return 0
        existing = self._client.retrieve(
            collection_name=self._collection,
            ids=[p.id for p in points],
            with_payload=False,
        )
        existing_ids = {str(r.id) for r in existing}
        inserted = sum(1 for p in points if str(p.id) not in existing_ids)

        self._client.upsert(collection_name=self._collection, points=points)
        return inserted

    # ---------- read ----------

    def query(
        self,
        embedding: Sequence[float],
        k: int = 5,
        *,
        tenant_id: str | None = None,
    ) -> list[Hit]:
        query_filter = None
        if tenant_id is not None:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id),
                    )
                ]
            )

        # `query_points` is the current API (qdrant-client >= 1.10).
        # The older `search` method was deprecated in 1.10 and removed in
        # 1.12; keeping `query_points` avoids the attr-defined mypy error.
        response = self._client.query_points(
            collection_name=self._collection,
            query=[float(x) for x in embedding],
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        results = response.points

        hits: list[Hit] = []
        for r in results:
            payload = dict(r.payload or {})
            hits.append(
                Hit(
                    chunk_id=str(payload.get("chunk_id", "")),
                    doc_id=str(payload.get("doc_id", "")),
                    text=str(payload.get("text", "")),
                    score=float(r.score),
                    metadata=payload,
                )
            )
        return hits

    def has_doc(self, doc_id: str) -> bool:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points) > 0

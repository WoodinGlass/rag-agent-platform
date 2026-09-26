"""Vector store wrappers.

`MemoryStore` for tests + ephemeral dev (zero deps).
`ChromaStore` for local persistence (`pip install .[vector]`).
Chroma import is lazy so unit tests never touch it.

Multi-tenant (M5.3): queries accept an optional `tenant_id`. When set,
only chunks whose `tenant_id` metadata matches are returned. Ingestion
stores `tenant_id` in chunk metadata; retrieval filters on it. When
`tenant_id` is None, no filter is applied (single-tenant behaviour).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.rag.chunker import Chunk


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Return number of *newly inserted* chunks (updates don't count)."""

    @abstractmethod
    def query(
        self,
        embedding: Sequence[float],
        k: int = 5,
        *,
        tenant_id: str | None = None,
    ) -> list[Hit]:
        ...

    @abstractmethod
    def has_doc(self, doc_id: str) -> bool:
        ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class MemoryStore(VectorStore):
    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._meta: dict[str, dict] = {}
        self._docs: set[str] = set()

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        inserted = 0
        for chunk, emb in zip(chunks, embeddings):
            if chunk.chunk_id not in self._vectors:
                inserted += 1
            self._vectors[chunk.chunk_id] = list(emb)
            self._meta[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "chunker_version": chunk.chunker_version,
                "index": chunk.index,
                **chunk.metadata,
            }
            self._docs.add(chunk.doc_id)
        return inserted

    def query(
        self,
        embedding: Sequence[float],
        k: int = 5,
        *,
        tenant_id: str | None = None,
    ) -> list[Hit]:
        scored: list[tuple[str, float]] = []
        for cid, vec in self._vectors.items():
            if tenant_id is not None and self._meta[cid].get("tenant_id") != tenant_id:
                continue
            scored.append((cid, _cosine(embedding, vec)))
        scored.sort(key=lambda x: (-x[1], x[0]))  # stable tiebreak

        hits: list[Hit] = []
        for cid, score in scored[:k]:
            m = self._meta[cid]
            hits.append(
                Hit(
                    chunk_id=cid,
                    doc_id=m["doc_id"],
                    text=m["text"],
                    score=score,
                    metadata=m,
                )
            )
        return hits

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self._docs


class ChromaStore(VectorStore):
    """Persistent local store. Requires chromadb (`.[vector]`).

    Note on typing: chromadb's public stubs accept a wide union that
    includes numpy arrays. We pass plain Python lists of floats — the
    runtime is fine, but `list` is invariant, so mypy can't narrow
    `list[list[float]]` to the union. We keep the payload variables
    typed as `Any` for the call site and let runtime validation handle
    shape errors.
    """

    def __init__(self, path: str, collection: str = "rag_docs") -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "chromadb not installed; run: pip install 'rag-agent-platform[vector]'"
            ) from e
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(name=collection)

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        ids = [c.chunk_id for c in chunks]
        existing = set(self._col.get(ids=ids).get("ids", []))
        inserted = sum(1 for i in ids if i not in existing)

        emb_list: Any = [[float(x) for x in e] for e in embeddings]
        docs: list[str] = [c.text for c in chunks]
        metas: Any = [
            {
                "doc_id": c.doc_id,
                "start": c.start,
                "end": c.end,
                "chunker_version": c.chunker_version,
                "index": c.index,
                **c.metadata,
            }
            for c in chunks
        ]

        self._col.upsert(
            ids=ids,
            documents=docs,
            embeddings=emb_list,
            metadatas=metas,
        )
        return inserted

    def query(
        self,
        embedding: Sequence[float],
        k: int = 5,
        *,
        tenant_id: str | None = None,
    ) -> list[Hit]:
        emb_list: Any = [float(x) for x in embedding]
        where: Any = {"tenant_id": tenant_id} if tenant_id is not None else None

        res = self._col.query(
            query_embeddings=[emb_list],
            n_results=k,
            where=where,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: list[Hit] = []
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            m = dict(meta or {})
            hits.append(
                Hit(
                    chunk_id=str(cid),
                    doc_id=str(m.get("doc_id", "")),
                    text=str(text or ""),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                    metadata=m,
                )
            )
        return hits

    def has_doc(self, doc_id: str) -> bool:
        r = self._col.get(where={"doc_id": doc_id}, limit=1)
        return bool(r.get("ids"))


def get_store(backend: str = "memory", **kwargs: Any) -> VectorStore:
    if backend == "memory":
        return MemoryStore()
    if backend == "chroma":
        return ChromaStore(**kwargs)
    raise ValueError(f"unknown vector backend: {backend}")

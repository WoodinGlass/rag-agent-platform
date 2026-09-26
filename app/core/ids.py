"""Deterministic ids & idempotency keys.

Ingest is idempotent on `(content_hash, chunker_version)`.
Re-ingesting the same bytes with the same chunker = no-op.

If you bump the chunker version, doc_id changes on purpose -> old data
remains queryable for rollback, new data is re-embedded cleanly.
"""
from __future__ import annotations

import hashlib


def content_hash(data: bytes) -> str:
    """sha256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def doc_id(content_hash_hex: str, chunker_version: str) -> str:
    """Stable doc id from (content_hash, chunker_version). 32-char hex."""
    key = f"{content_hash_hex}::{chunker_version}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:32]


def chunk_id(doc_id_hex: str, index: int) -> str:
    """Stable chunk id: doc-scoped + zero-padded positional index."""
    return f"{doc_id_hex}:{index:04d}"


def idempotency_key(*parts: str) -> str:
    """Generic composite key for non-doc operations (events, requests)."""
    joined = "::".join(parts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()

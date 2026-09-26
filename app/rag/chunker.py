"""Deterministic recursive chunker.

Guarantees:
- same input + same version -> byte-identical output
- no empty chunks
- chunk boundaries prefer paragraph > sentence > hard cut
- `start`/`end` offsets are monotonic (source span, best-effort)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CHUNKER_VERSION = "v1"

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    chunker_version: str
    index: int
    metadata: dict = field(default_factory=dict)


def _hard_cut(text: str, size: int) -> list[str]:
    out: list[str] = []
    for i in range(0, len(text), size):
        piece = text[i:i + size]
        if piece.strip():
            out.append(piece)
    return out


def _pack(pieces: list[str], size: int, joiner: str) -> list[str]:
    """Greedily pack small pieces into chunks <= size. Recursively split oversized."""
    out: list[str] = []
    buf = ""
    for p in pieces:
        if len(p) > size:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_recursive_pieces(p, size))
            continue
        if not buf:
            buf = p
        elif len(buf) + len(joiner) + len(p) <= size:
            buf = f"{buf}{joiner}{p}"
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _recursive_pieces(text: str, size: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    # 1. paragraph
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) > 1:
        return _pack(paras, size, "\n\n")

    # 2. sentence
    sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if len(sents) > 1:
        return _pack(sents, size, " ")

    # 3. hard cut
    return _hard_cut(text, size)


def chunk_text(
    text: str,
    doc_id: str,
    *,
    size: int = 500,
    version: str = CHUNKER_VERSION,
    metadata: dict | None = None,
) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be > 0")

    from app.core.ids import chunk_id as _cid

    pieces = _recursive_pieces(text, size)
    chunks: list[Chunk] = []
    cursor = 0
    n = len(text)
    for i, piece in enumerate(pieces):
        pos = text.find(piece, cursor)
        if pos == -1:
            head = piece[:30]
            pos = text.find(head, cursor) if head else cursor
        if pos == -1:
            pos = cursor
        end = min(pos + len(piece), n) if n else pos + len(piece)
        cursor = max(cursor, end)
        chunks.append(
            Chunk(
                chunk_id=_cid(doc_id, i),
                doc_id=doc_id,
                text=piece,
                start=pos,
                end=end,
                chunker_version=version,
                index=i,
                metadata=dict(metadata or {}),
            )
        )
    return chunks

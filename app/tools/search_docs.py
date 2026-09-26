"""search_docs tool.

Pure logic: format hits -> JSON-serializable dict for the LLM.
Impure: the retrieve_fn is injected (see tools/adapters/rag.py).

Output shape (stable contract, do not change casually):
    {
      "query": str,
      "n": int,
      "hits": [
        {"chunk_id": str, "doc_id": str, "score": float,
         "text": str, "start": int, "end": int}
      ],
      "truncated": bool
    }
"""
from __future__ import annotations

from collections.abc import Callable

from app.rag.store import Hit
from app.tools.base import Tool

DEFAULT_K = 5
MAX_K = 20
MAX_TEXT_CHARS = 600

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "k": {
            "type": "integer",
            "description": f"Number of hits (1..{MAX_K}).",
            "minimum": 1,
            "maximum": MAX_K,
        },
    },
    "required": ["query"],
}


def _clamp_k(k: int) -> int:
    return max(1, min(int(k), MAX_K))


def _format_hit(h: Hit) -> dict:
    text = h.text
    truncated = len(text) > MAX_TEXT_CHARS
    return {
        "chunk_id": h.chunk_id,
        "doc_id": h.doc_id,
        "score": round(float(h.score), 6),
        "text": text[:MAX_TEXT_CHARS],
        "start": int(h.metadata.get("start", 0)),
        "end": int(h.metadata.get("end", 0)),
        "_truncated": truncated,
    }


def format_hits(query: str, hits: list[Hit], k: int) -> dict:
    """Pure: hits -> JSON-safe dict. No IO."""
    k_clamped = _clamp_k(k)
    items = [_format_hit(h) for h in hits[:k_clamped]]
    return {
        "query": query,
        "n": len(items),
        "hits": items,
        "truncated": len(hits) > k_clamped,
    }


def make_search_docs_tool(
    retrieve_fn: Callable[[str, int], list[Hit]],
    *,
    name: str = "search_docs",
) -> Tool:
    """Build a Tool from an injected retrieve function."""

    def _search(query: str, k: int = DEFAULT_K) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k_clamped = _clamp_k(k)
        hits = retrieve_fn(query, k_clamped)
        return format_hits(query, hits, k_clamped)

    return Tool(
        name=name,
        description=(
            "Search the local document corpus. Returns top-k chunks with "
            "scores and source offsets. Use for factual/grounded questions."
        ),
        input_schema=INPUT_SCHEMA,
        func=_search,
        tags=("retrieval", "io"),
    )

"""Adapter: bind a RagPipeline to a callable for the search_docs tool.

The tool itself is pure (formats hits -> dict). This adapter owns the
impure RagPipeline instance and is injected at wiring time. That way
unit tests can pass a fake callable and never touch a real store.
"""
from __future__ import annotations

from app.rag.pipeline import RagPipeline
from app.rag.store import Hit


def make_retrieve_fn(pipeline: RagPipeline):
    """Return a `(query, k) -> list[Hit]` callable bound to a pipeline."""

    def _retrieve(query: str, k: int = 5) -> list[Hit]:
        return pipeline.retrieve(query, k=k)

    return _retrieve

"""Adapter: bind a RagPipeline to a callable for the search_docs tool.

The tool itself is pure (formats hits -> dict). This adapter owns the
impure RagPipeline instance and is injected at wiring time. That way
unit tests can pass a fake callable and never touch a real store.

Tenant (M6.2): the retrieve function reads the current tenant from the
request context (`app.core.tenant`) at call time and passes it to the
pipeline. This is what makes multi-tenant work through the agent loop:
the tool doesn't know about tenants; the adapter resolves it ambiently.
When no tenant is set (single-tenant mode), the pipeline is called with
`tenant_id=None`, which disables the store filter.
"""
from __future__ import annotations

from collections.abc import Callable

from app.core.tenant import get_tenant_id
from app.rag.pipeline import RagPipeline
from app.rag.store import Hit


def make_retrieve_fn(pipeline: RagPipeline) -> Callable[[str, int], list[Hit]]:
    """Return a `(query, k) -> list[Hit]` callable bound to a pipeline."""

    def _retrieve(query: str, k: int = 5) -> list[Hit]:
        tenant = get_tenant_id()
        return pipeline.retrieve(query, k=k, tenant_id=tenant)

    return _retrieve

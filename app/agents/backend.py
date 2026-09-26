"""Agent backend protocol.

Both implementations (state machine and LangGraph) expose the same
`run(question) -> AgentOutput` contract. Nothing downstream needs to
know which one is active.

Why a protocol and not an ABC?
- `Agent` (the state machine) already exists and has this exact
  signature; a Protocol lets us type-check without forcing an
  inheritance change.
- New backends (LangGraph, future) can be added without editing the
  existing class.
- `runtime_checkable` allows `isinstance(x, AgentBackend)` in
  factories, which is what the config-driven selection relies on.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.agents.schema import AgentOutput


@runtime_checkable
class AgentBackend(Protocol):
    """Contract every agent implementation must satisfy."""

    def run(self, question: str) -> AgentOutput:  # pragma: no cover - protocol
        ...

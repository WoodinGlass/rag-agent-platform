"""Structured output schema for the agent.

The agent MUST return JSON matching `AgentOutput`. This module is the
single source of truth for that contract — Pydantic validates both
at construction time and via `model_validate_json` for LLM responses.

Design:
- `final_answer` is the only required free-text field.
- `tool_calls` is an audit log (records what the agent did).
- `citations` grounds claims to retrieved chunks (doc_id + chunk_id).
- `confidence` is optional; used for downstream filtering.
- `refused` short-circuits: when True, final_answer explains why.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    quote: str = Field(default="", max_length=280)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(min_length=1)
    args: dict = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    latency_ms: int = Field(default=0, ge=0)


class AgentOutput(BaseModel):
    """Final agent answer. Enforced via Pydantic before returning to caller."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    final_answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    refused: bool = False
    reason: str = ""

    @field_validator("question", "final_answer")
    @classmethod
    def _strip_and_require_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must be a non-empty string")
        return v

    @field_validator("reason")
    @classmethod
    def _reason_required_if_refused(cls, v: str, info) -> str:
        refused = info.data.get("refused")
        if refused and not v.strip():
            raise ValueError("reason is required when refused=True")
        return v

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> AgentOutput:
        return cls.model_validate_json(raw)

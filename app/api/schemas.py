"""HTTP request/response schemas.

Design notes:
- `extra="forbid"`: unknown fields are rejected. Catches client typos early.
- Ingest takes base64 bytes (not str) so ingest() stays byte-identical ->
  idempotency key on sha256(content) remains meaningful across encodings.
- QueryResponse wraps AgentOutput with server-side timing; the output
  contract itself lives in `app.agents.schema`.
- No IO here. Pure data shapes.
"""
from __future__ import annotations

import base64
import binascii
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.schema import AgentOutput

# ---------- /ingest ----------


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_base64: str = Field(
        min_length=1,
        description="Document bytes, base64-encoded (standard alphabet).",
    )
    source: str = Field(default="", max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content_base64")
    @classmethod
    def _valid_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"content_base64 is not valid base64: {e}") from e
        return v

    def decode(self) -> bytes:
        return base64.b64decode(self.content_base64, validate=True)


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    inserted: int = Field(ge=0)
    skipped: bool
    n_chunks: int = Field(ge=0)


# ---------- /query ----------


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def _strip_and_require_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must be non-empty")
        return v


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: AgentOutput
    elapsed_ms: int = Field(default=0, ge=0)
    correlation_id: str = Field(default="", max_length=64)


# ---------- /healthz ----------


HealthStatus = Literal["ok", "degraded", "down"]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: HealthStatus = "ok"
    version: str = Field(min_length=1)
    uptime_s: float = Field(ge=0.0)
    checks: dict[str, bool] = Field(default_factory=dict)


# ---------- errors ----------


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str = Field(min_length=1)
    detail: str = ""
    correlation_id: str = Field(default="", max_length=64)

"""LLM protocol + AgentStep schema + FakeLLM for offline tests.

The agent loop asks the LLM to reply with a single JSON object matching
`AgentStep` — either a tool call or a finish. This keeps the loop
provider-agnostic and fully testable without network.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentStep(BaseModel):
    """One LLM step: either call a tool or finish."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["tool", "finish"]
    thought: str = ""

    # action == "tool"
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)

    # action == "finish"
    final_answer: str | None = None
    citations: list[dict] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    refused: bool = False
    reason: str = ""

    @field_validator("tool")
    @classmethod
    def _tool_required_when_action_tool(cls, v: str | None, info) -> str | None:
        if info.data.get("action") == "tool" and not v:
            raise ValueError("tool is required when action='tool'")
        return v

    @field_validator("final_answer")
    @classmethod
    def _answer_required_when_finish(cls, v: str | None, info) -> str | None:
        if info.data.get("action") == "finish" and not v:
            raise ValueError("final_answer is required when action='finish'")
        return v


class LLM(Protocol):
    """Minimal LLM interface. Implementations wrap OpenAI / Anthropic."""

    def complete(self, messages: list[dict], **kw: Any) -> str: ...


class FakeLLM:
    """Scripted, deterministic LLM for tests.

    Pops one scripted reply per `complete()` call and records every
    request so tests can assert what the agent sent.
    """

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], **kw: Any) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self._script:
            raise RuntimeError("FakeLLM: script exhausted")
        return self._script.pop(0)

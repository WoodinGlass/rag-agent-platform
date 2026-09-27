"""LLM protocol + AgentStep schema + implementations.

Implementations:
- `FakeLLM` - scripted, deterministic, offline (tests, demos).
- `OpenAILLM` - real OpenAI-compatible chat completions; lazy import.
  Works against any OpenAI-compatible endpoint (OpenAI, Groq, Together,
  OpenRouter, vLLM) by setting `base_url`.
- `AnthropicLLM` - real Anthropic messages; lazy import.

All implement the same `LLM` protocol: `complete(messages) -> str`.
"""
from __future__ import annotations

import os
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

# OpenAI-compatible endpoints.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq's lineup changes; this is a safe, fast default with
# JSON-mode support. See: https://console.groq.com/docs/models
GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"


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
    """Minimal LLM interface. Implementations wrap OpenAI / Anthropic / Groq."""

    def complete(self, messages: list[dict], **kw: Any) -> str: ...


class FakeLLM:
    """Scripted, deterministic LLM for tests."""

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], **kw: Any) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self._script:
            raise RuntimeError("FakeLLM: script exhausted")
        return self._script.pop(0)


class OpenAILLM:
    """OpenAI-compatible chat completions adapter.

    `base_url=None` targets official OpenAI. Set it to any OpenAI-
    compatible endpoint (Groq, Together, OpenRouter, vLLM).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        max_tokens: int | None = 512,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "openai not installed; run: pip install 'rag-agent-platform[llm]'"
            ) from e

        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout_s,
        )
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], **kw: Any) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        kwargs.update(kw)
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class AnthropicLLM:
    """Anthropic messages adapter."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        api_key: str | None = None,
        *,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "anthropic not installed; run: pip install 'rag-agent-platform[llm]'"
            ) from e

        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], **kw: Any) -> str:
        system_parts: list[str] = []
        chat_messages: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        resp = self._client.messages.create(
            model=self.model,
            system="\n\n".join(system_parts) if system_parts else "",
            messages=chat_messages,
            max_tokens=self.max_tokens,
            **kw,
        )
        parts: list[str] = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


def get_llm(provider: str = "fake", **kwargs: Any) -> LLM:
    """Factory mirroring `get_embedder` / `get_store` / `get_reranker`."""
    if provider == "fake":
        return FakeLLM(**kwargs)
    if provider == "openai":
        return OpenAILLM(**kwargs)
    if provider == "groq":
        kwargs.setdefault("base_url", GROQ_BASE_URL)
        return OpenAILLM(**kwargs)
    if provider == "anthropic":
        return AnthropicLLM(**kwargs)
    raise ValueError(f"unknown llm provider: {provider!r}")

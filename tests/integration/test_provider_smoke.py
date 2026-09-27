"""Real LLM provider smoke test.

Purpose: prove the real provider path actually works against a live API,
not just against mocks. Runs only when the `provider` marker is selected
AND a provider API key is present in the environment.

Supported providers (checked in order):
- GROQ_API_KEY    -> Groq (OpenAI-compatible; free tier)
- OPENAI_API_KEY  -> OpenAI (also enables the embeddings test)

Skips cleanly otherwise. Never runs on regular PRs. Cost per run:
free (Groq) or <$0.01 (OpenAI).

Run locally:
    GROQ_API_KEY=gsk_... pytest -m provider -v
    OPENAI_API_KEY=sk-... pytest -m provider -v
"""
from __future__ import annotations

import os

import pytest

from app.agents.llm import (
    GROQ_BASE_URL,
    GROQ_DEFAULT_MODEL,
    AgentStep,
    OpenAILLM,
)
from app.agents.schema import AgentOutput

pytestmark = pytest.mark.provider


def _detect_provider() -> tuple[str | None, str | None, str | None, str | None]:
    """Return (name, api_key, base_url, model) or (None, None, None, None)."""
    if os.getenv("GROQ_API_KEY"):
        return (
            "groq",
            os.getenv("GROQ_API_KEY"),
            GROQ_BASE_URL,
            GROQ_DEFAULT_MODEL,
        )
    if os.getenv("OPENAI_API_KEY"):
        return (
            "openai",
            os.getenv("OPENAI_API_KEY"),
            None,
            "gpt-4o-mini",
        )
    return (None, None, None, None)


_PROVIDER, _API_KEY, _BASE_URL, _MODEL = _detect_provider()
_HAVE_KEY = _PROVIDER is not None

skip_if_no_key = pytest.mark.skipif(
    not _HAVE_KEY,
    reason="no provider key (set GROQ_API_KEY or OPENAI_API_KEY)",
)

skip_if_no_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set (embeddings require OpenAI)",
)


def _make_llm() -> OpenAILLM:
    assert _API_KEY is not None
    # Keep output small: Groq's free tier caps output tokens per minute.
    return OpenAILLM(
        model=_MODEL or "gpt-4o-mini",
        api_key=_API_KEY,
        base_url=_BASE_URL,
        temperature=0.0,
        max_tokens=256,
    )


# ---------- chat completions (works with Groq or OpenAI) ----------

@skip_if_no_key
def test_provider_reports_which_key_is_active(capsys):
    """Informational: print which provider was used for the run."""
    with capsys.disabled():
        print(f"\n[provider-smoke] using: {_PROVIDER} model={_MODEL}")


@skip_if_no_key
def test_llm_emits_parseable_json_finish():
    llm = _make_llm()
    messages = [
        {
            "role": "system",
            "content": (
                "Reply with a single JSON object matching this schema:\n"
                '{"action":"finish","final_answer":str,"confidence":"low"|"medium"|"high"}\n'
                "No prose, JSON only."
            ),
        },
        {"role": "user", "content": "What is 2 + 2? Answer briefly."},
    ]
    raw = llm.complete(messages)
    step = AgentStep.model_validate_json(raw)
    assert step.action == "finish"
    assert step.final_answer
    assert step.confidence in ("low", "medium", "high")


@skip_if_no_key
def test_llm_emits_tool_call_when_asked():
    llm = _make_llm()
    messages = [
        {
            "role": "system",
            "content": (
                "You reply ONLY with a JSON object. No prose. No tool calls.\n"
                "The JSON object has exactly this shape:\n"
                '{"action":"tool","tool":"calculator","args":{"expression":"12*7"}}\n'
                "Do NOT use any native tool-calling API. Do NOT return "
                '{"name":...,"arguments":...}. Emit the JSON object '
                "described above as plain text."
            ),
        },
        {
            "role": "user",
            "content": "Compute 12 * 7 using the calculator tool. "
            "Emit the JSON object described above.",
        },
    ]
    raw = llm.complete(messages)
    step = AgentStep.model_validate_json(raw)
    assert step.action == "tool"
    assert step.tool == "calculator"


# ---------- end-to-end agent (works with Groq or OpenAI) ----------

@skip_if_no_key
def test_agent_run_with_real_llm_returns_valid_output():
    from app.agents.bootstrap import build_agent
    from app.rag.embedder import FakeEmbedder  # keep vector side offline
    from app.rag.store import MemoryStore

    llm = _make_llm()
    stack = build_agent(
        llm=llm,
        embedder=FakeEmbedder(dim=64),
        store=MemoryStore(),
        chunk_size=200,
    )
    stack.pipeline.ingest(
        b"The word 'cat' has 3 letters.", source="cat.txt"
    )

    out = stack.agent.run("How many letters are in the word 'cat'?")
    assert isinstance(out, AgentOutput)
    assert out.final_answer
    assert out.confidence in ("low", "medium", "high")


# ---------- embeddings (OpenAI only; Groq does not offer embeddings) ----------

@skip_if_no_openai
def test_openai_embedder_returns_vector_of_expected_dim():
    from app.rag.embedder import OpenAIEmbedder

    emb = OpenAIEmbedder(model="text-embedding-3-small")
    vec = emb.embed_one("hello world")
    assert isinstance(vec, list)
    assert len(vec) == emb.dim
    assert all(isinstance(x, float) for x in vec[:5])


@skip_if_no_openai
def test_openai_embedder_batch_returns_one_per_input():
    from app.rag.embedder import OpenAIEmbedder

    emb = OpenAIEmbedder(model="text-embedding-3-small")
    vecs = emb.embed(["one", "two", "three"])
    assert len(vecs) == 3
    assert all(len(v) == emb.dim for v in vecs)

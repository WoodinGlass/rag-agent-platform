# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added

**M1 — Local RAG (deterministic core)**
- `app/core`: pydantic-settings config, deterministic ids, JSON logging
- `app/rag/chunker.py`: recursive chunker, version-tagged, deterministic
- `app/rag/embedder.py`: provider-agnostic interface + `FakeEmbedder`
- `app/rag/store.py`: `VectorStore` ABC, `MemoryStore`, lazy `ChromaStore`
- `app/rag/pipeline.py`: idempotent ingest (`sha256 + chunker_version`)
- `evals/run.py`: retrieval hit-rate@k CLI
- `scripts/ingest_demo.py`: offline ingest demo

**M2 — Agent + tools**
- `app/agents/registry.py`: `ToolRegistry` (register / get / specs)
- `app/tools/base.py`: `Tool` protocol (name, description, JSON schema, callable)
- `app/tools/search_docs.py`: retrieval tool wrapping `RagPipeline`
- `app/tools/calculator.py`: ast-safe arithmetic (no `eval`, DoS caps)
- `app/tools/web_fetch.py`: HTTP GET with timeout + retry (jitter backoff)
- `app/tools/adapters/`: impure boundaries (`rag.py`, `http.py`)
- `app/agents/schema.py`: `AgentOutput` (Pydantic v2, extra=forbid, refused semantics)
- `app/agents/llm.py`: `LLM` protocol, `AgentStep` schema, `FakeLLM` (offline)
- `app/agents/prompts.py`: versioned system prompt + tool renderer
- `app/agents/agent.py`: state machine loop `plan → tool_call → observe → finish`
- `app/agents/bootstrap.py`: one-shot factory wiring 3 tools + registry + agent
- `scripts/agent_demo.py`: multi-hop demo (offline, FakeLLM)

### Verified
- `pytest -m "not integration"` green (~131 tests)
- Retrieval hit-rate@5 = **1.000** on demo set (12/12)
- Idempotent ingest: re-running `ingest_demo.py` reports `skipped=12`
- Multi-hop agent (search_docs → calculator → finish) yields valid `AgentOutput`
- Tool failures and invalid LLM JSON are captured, not fatal

### Infrastructure
- CI workflow: ruff + pytest (unit) on push/PR
- Initial repo scaffold: README, LICENSE, pyproject, .env.example, .gitignore

## [0.1.0] - 2025-XX-XX

### Added
- Initial scaffold

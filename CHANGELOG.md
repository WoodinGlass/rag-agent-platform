# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added

**M5 — Hardening (in progress)**
- `benchmarks/run.py`: offline ingest + retrieve latency (p50/p90/p95/p99), cost projection from published list prices
- `benchmarks/results/latest.{json,md}`: committed artifacts
- `docs/benchmarks.md`: measured vs projected, honest gaps
- tests: benchmark helpers (percentile, summarize, tokenizer estimate)
- `benchmarks/reranker_delta.py`: identity vs fake, quality + latency delta
- `benchmarks/results/reranker_delta.{json,md}`: committed
- `app/rag/reranker.py`: Reranker ABC, Identity/Fake/CrossEncoder impls
- `pyproject.toml`: new `[rerank]` extra (sentence-transformers)
- `pipeline.retrieve()`: fetch k*multiplier, rerank to k (default identity)

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

**M3 — API + Docker**
- `app/api/schemas.py`: HTTP request/response Pydantic (extra=forbid)
- `app/api/deps.py`: dependency wiring (agent, pipeline, settings, cid)
- `app/api/middleware.py`: `X-Request-ID` propagation + structured access log
- `app/api/routes.py`: `/ingest`, `/query`, `/healthz`, `/metrics`
- `app/core/metrics.py`: in-process counters + latency histograms
- `app/main.py`: lifespan wiring, error handler, correlation-id middleware
- `tests/integration/test_api_lifespan.py`: end-to-end API tests
- `docker/Dockerfile`: multi-stage, non-root user, healthcheck
- `docker/docker-compose.yml`: API + Qdrant
- `.dockerignore`: build context hygiene
- `.env.example`: full config documentation

### Verified
- `pytest -m "not integration"` green (~189 tests)
- `pytest -m integration` green (~9 tests)
- Retrieval hit-rate@5 = **1.000** on demo set (12/12)
- Idempotent ingest: re-running `ingest_demo.py` reports `skipped=12`
- Multi-hop agent (search_docs → calculator → finish) yields valid `AgentOutput`
- API smoke: `POST /ingest`, `POST /query`, `GET /healthz`, `GET /metrics` return valid JSON
- Docker image builds and boots; `/healthz` reachable in container

### Infrastructure
- CI: lint (ruff) + unit + integration on push/PR
- CI: docker job — compose validate + image build + container smoke test
- Initial repo scaffold: README, LICENSE, pyproject, .env.example, .gitignore

## [0.1.0] - 2025-XX-XX

### Added
- Initial scaffold

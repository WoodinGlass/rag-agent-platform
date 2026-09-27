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
- **M5.3 — multi-tenant + auth (opt-in)**
  - `app/core/auth.py`: parse `TENANT_KEYS`, resolve tenant from `X-API-Key`
  - `TENANT_AUTH_ENABLED` default `false` (single-tenant backward compat)
  - Tenant-scoped `doc_id` (same bytes in two tenants -> different ids)
  - Store-level `tenant_id` filter (MemoryStore, ChromaStore `where`)
  - `TenantDep` route dependency: `/ingest` fully isolated; 401 on bad key
  - 26 new tests (parse, resolve, HTTP, isolation)
  - ~~**Known gap:** `/query` used the startup-bound pipeline~~ — **fixed in M6.2**
- **M5.4 — OpenTelemetry traces (opt-in)**
  - `app/core/tracing.py`: minimal span interface, NoopSpan, OTelSpan
  - Lazy `opentelemetry-sdk` import; missing SDK -> warning + disabled
  - New `[otel]` extra; CI installs it
  - Instrumented: `http.ingest`, `http.query`, `pipeline.ingest`,
    `pipeline.retrieve`, `agent.run`, `tool.<name>`
  - Failures inside span never break the app
  - 10 new tests (coercion, noop, enabled path when SDK present)
  - README § Observability; `.env.example` updated


**M6.1 — LangGraph backend (opt-in)**
- `app/agents/backend.py`: runtime-checkable `AgentBackend` Protocol
- `app/agents/langgraph_backend.py`: StateGraph loop (plan -> execute -> finish)
- `pyproject.toml`: new `[langgraph]` extra
- config: `agent_backend = state_machine | langgraph` (default: state machine)
- parity tests with the state machine (same script -> same output)

**M6.2 — Tenant-per-request via contextvar**
- `app/core/tenant.py`: contextvar + get/set/reset
- `app/api/middleware.py`: TenantMiddleware resolves X-API-Key, sets context
- `/healthz`, `/metrics`, `/docs` exempt from auth
- `tools/adapters/rag.py`: reads tenant from context; agent loop unchanged
- closes the M5.3 known gap

**M6.3 — OTLP exporter + collector example**
- `pyproject.toml`: `[otel]` includes `opentelemetry-exporter-otlp-proto-http`
- `app/core/tracing.py`: exporter selection (`console` | `otlp`), never raises
- `app/core/config.py`: `otel_exporter`, `otel_otlp_endpoint`
- `docker/otel-collector.yml`: minimal collector (OTLP -> stdout)
- `docker/docker-compose.yml`: `otel-collector` service, profile `otel`
- `.env.example`, README § Observability updated
- 4 new tests (exporter selection, enable path)

**M9 — Post-MVP polish**
- **M9.1 — Real LLM provider path (smoke-tested)**
  - `app/agents/llm.py`: `OpenAILLM` gains `base_url` +
    `max_tokens`; works with any OpenAI-compatible endpoint
    (Groq, Together, OpenRouter, vLLM)
  - `GROQ_DEFAULT_MODEL = qwen/qwen3.8-27b` (chosen after testing:
    `gpt-oss-20b` -> native tool-calling conflict;
    `allam-2-7b` -> too small for multi-turn agent loop;
    `qwen3.8-27b` + `max_tokens=256` works within Groq free tier)
  - `llm_provider` accepts `groq`; `main.py` routes it through
    `OpenAILLM` with `GROQ_BASE_URL`
  - `tests/integration/test_provider_smoke.py`: real-API smoke
    test, self-skips without key; marker `provider`
  - `.github/workflows/provider-smoke.yml`: manual + weekly,
    accepts `GROQ_API_KEY` or `OPENAI_API_KEY`
  - `.env.example`: documents Groq as the free real-provider path

**M8 — Backlog (post-MVP)**
- **M8.2 — Durable dedup state**
  - `app/streaming/dedup.py`: `DedupStore` protocol + `InMemoryDedupStore`
  - `app/streaming/sqlite_dedup.py`: `SQLiteDedupStore` (stdlib only)
  - `StreamingIngestor(dedup_store=...)` — backward compat with
    `dedup_window` (builds the in-memory default)
  - config: `streaming_dedup_backend` (memory | sqlite),
    `streaming_dedup_sqlite_path`, `streaming_dedup_ttl_s`
  - 15 tests: LRU behavior, TTL prune, persistence across restart,
    ingestor integration
- **M8.1 — Bounded prefetch across partitions**
  - `ParallelIngestor(prefetch_n, prefetch_budget_total)`
  - budget divided deterministically across partitions:
    `min(prefetch_n, max(1, budget // n_partitions))`
  - per-partition prefetch size exposed in stats
  - config: `streaming_prefetch_budget_total` (0 = no total cap)
  - 9 tests: budget division, cap by N, minimum 1, disabled default,
    end-to-end with prefetch, input validation

**M7 — Post-MVP**
- **M7.1 — Prefetch queue**
  - `app/streaming/prefetch.py`: `Prefetcher` (bounded buffer + fill task)
  - backpressure preserved; commit delegates to source
  - `StreamingIngestor.run_with_prefetch(prefetch_n=N)`
  - config: `streaming_prefetch_n` (0 disables, default)
- **M7.4 — Exactly-once (documented honestly)**
  - `docs/exactly-once.md`: why "exactly-once" for a Kafka->vector-store
    pipeline is idempotency + dedup, not Kafka transactions
  - Reference to existing code: `sha256+version` doc ids, bounded dedup
    window, at-least-once commit semantics
  - Failure-mode table with the guarantees we do and do not claim
  - When Kafka transactions *would* be the right tool (Kafka->Kafka)
  - No transaction producer added (deliberate; would be misdirection)
- **M7.3 — Multi-partition parallel ingestors**
  - `app/streaming/partitioned.py`: `PartitionedSource` protocol +
    `InMemoryPartitionedSource` (offline)
  - `app/streaming/parallel_ingestor.py`: `ParallelIngestor` (one
    `StreamingIngestor` per partition, `asyncio.gather`)
  - per-partition stats, aggregated totals, error isolation
  - Kafka rebalance not implemented (partitions enumerated at start)
- **M7.2 — S3 event source adapter**
  - `app/streaming/s3_source.py`: `S3ObjectSource` (lazy `[s3]` extra)
  - polls `list_objects_v2`, filters by prefix, at-least-once
  - blocking boto3 calls offloaded via `asyncio.to_thread`
  - error isolation: list/get failures do not crash the loop
  - config: `s3_bucket`, `s3_prefix`, `s3_endpoint_url`, `s3_region`,
    `s3_poll_interval_s`, `s3_max_keys`
  - `.env.example` extended

**M6.4 — Streaming ingestion**
- `app/streaming/events.py`: `Event` envelope (id, data, metadata)
- `app/streaming/source.py`: `EventSource` protocol (poll / commit / close)
- `app/streaming/memory_source.py`: `InMemoryQueueSource` (offline, mirrors Kafka)
- `app/streaming/ingestor.py`: `StreamingIngestor` — at-least-once, bounded
  dedup window (LRU), retry with backoff, one-in-flight backpressure
- `app/streaming/kafka_source.py`: `KafkaEventSource` (lazy `[kafka]` extra)
- `pyproject.toml`: new `[kafka]` extra; `pytest-asyncio` + `asyncio_mode=auto`
- `app/core/config.py`: `streaming_backend`, `streaming_poll_timeout_s`,
  `streaming_max_retries`, `streaming_dedup_window`, `streaming_idle_sleep_s`
- `scripts/stream_demo.py`: offline end-to-end demo (producer + consumer)
- `.env.example`: `STREAMING_*`, `KAFKA_*` knobs
- README § Ingestion modes
- 20+ tests: protocol conformance, poll/commit, dedup, backpressure,
  retry, give-up, buffered records, import guard

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

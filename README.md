# rag-agent-platform

> A production-minded RAG + AI agent backend with tool-calling, structured outputs, and a reproducible evaluation loop.

[![CI](https://github.com/WoodinGlass/rag-agent-platform/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WoodinGlass/rag-agent-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Problem (one sentence)

Developers need a **reusable, testable, and observable RAG + agent backend** that ingests arbitrary documents, retrieves grounded context, executes tools, and returns **structured JSON** — instead of one-off notebooks glued to a single LLM provider.

## What this is / is not

| This is | This is not |
|---|---|
| Backend service (FastAPI) + evaluation harness | A UI / chatbot frontend |
| Provider-agnostic LLM layer (OpenAI / Anthropic) | Tied to a single vendor SDK |
| Deterministic ingestion + retrieval, non-deterministic generation | Fully deterministic pipeline |
| Portfolio-grade with CI, tests, Docker, evals | Production SaaS with billing / multi-tenant |

---

## Tech stack

- **Runtime:** Python 3.11+, FastAPI, Uvicorn
- **Agent loop:** deterministic state machine (offline-testable); LLM-as-decider. Interface is LangGraph-ready — swap-in later without changing `Agent.run()`
- **LLM providers:** OpenAI, Anthropic (pluggable via `LLM` protocol; `FakeLLM` for tests)
- **Vector store:** Chroma (local dev) → Qdrant (prod); `MemoryStore` for tests
- **Schemas:** Pydantic v2 (`extra="forbid"` for contract enforcement)
- **Eval:** retrieval hit-rate@k now; Ragas (faithfulness + relevancy) in M4
- **Observability:** structured JSON logs + correlation ID; metrics endpoint in M3
- **Packaging:** `pyproject.toml` (single source of truth), `pip install -e`
- **Infra:** Docker + docker-compose (M3), GitHub Actions CI

---

## Architecture

```text
 [docs] --> [Ingest] --> chunk --> embed --> [Vector Store]
                                                    |
                                                    v
 [query] --> [FastAPI] --> [RAG Pipeline] --> [Agent Loop]
                                                    |
                                          plan -> tool_call -> observe -> finish
                                                    |
                                                    v
                                        [Structured JSON output]
                                        (AgentOutput, Pydantic-validated)
```

### Design decisions (Phase 0 pre-flight)

| Decision | Choice | Rationale |
|---|---|---|
| **Deterministic vs non-deterministic** | Ingest + retrieval deterministic (hash-based idempotency); generation non-deterministic | Enables re-ingest without duplicates |
| **Payload vs pipeline** | Payload = documents + embeddings; Pipeline = chunk → embed → retrieve → generate | Clear separation for testing |
| **Storage tier** | Small structured: SQLite (dev) / Postgres (prod). Unstructured: local FS → S3. Vectors: Chroma → Qdrant | No refactor when scaling |
| **Ingestion mode** | Batch first (simple), streaming later behind flag | MVP velocity |
| **Idempotency key** | `sha256(file_bytes) + chunker_version` | Re-ingest is a no-op |
| **Agent loop** | State machine, not LangGraph (yet) | Fully offline-testable; deterministic; `Agent.run()` signature stable |
| **Pure vs IO** | Tools pure; IO in `tools/adapters/` | Unit tests never touch network/disk |
| **Failure modes** | LLM timeout → retry w/ jitter + fallback model. Vector store down → circuit breaker, cached retrieval. API down → 503 + correlation ID in logs | See `docs/runbooks/` |

---

## Repository layout

```text
rag-agent-platform/
├── app/
│   ├── agents/              # agent loop, registry, schema, prompts
│   │   ├── agent.py         # state machine: plan -> tool_call -> observe -> finish
│   │   ├── bootstrap.py     # one-shot factory: pipeline + 3 tools + agent
│   │   ├── llm.py           # LLM protocol, AgentStep schema, FakeLLM
│   │   ├── prompts.py       # versioned system prompt + tool renderer
│   │   ├── registry.py      # ToolRegistry (register / get / specs)
│   │   └── schema.py        # AgentOutput, Citation, ToolCall (Pydantic)
│   ├── rag/                 # chunker, embedder, store, pipeline
│   │   ├── chunker.py       # recursive, version-tagged, deterministic
│   │   ├── embedder.py      # Embedder ABC + FakeEmbedder + lazy OpenAIEmbedder
│   │   ├── store.py         # VectorStore ABC + MemoryStore + lazy ChromaStore
│   │   └── pipeline.py      # idempotent ingest + retrieve
│   ├── tools/
│   │   ├── adapters/        # impure boundaries (rag.py, http.py)
│   │   ├── base.py          # Tool protocol (name, description, schema, callable)
│   │   ├── calculator.py    # ast-safe arithmetic (no eval)
│   │   ├── search_docs.py   # wraps RagPipeline.retrieve
│   │   └── web_fetch.py     # HTTP GET w/ timeout + retry
│   ├── api/                 # FastAPI routers, schemas, deps (M3)
│   ├── core/                # config, logging, ids
│   └── main.py              # application entrypoint (M3)
├── evals/                   # hit-rate@k CLI + demo corpus + reports
├── tests/
│   ├── unit/                # fast, offline (default)
│   └── integration/         # needs Docker / DB (marker: integration)
├── docker/                  # Dockerfile + docker-compose.yml (M3)
├── docs/
│   ├── architecture.md
│   └── runbooks/            # 3 most common failures (M4)
├── scripts/
│   ├── agent_demo.py        # offline multi-hop demo
│   └── ingest_demo.py       # offline ingest demo
├── .github/workflows/       # ci.yml
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Quickstart (local)

```bash
# 1. install
pip install -e ".[dev,http]"

# 2. config
cp .env.example .env
# edit .env if you want a real LLM (default is offline/fake)

# 3. ingest demo corpus
python scripts/ingest_demo.py --query "What is RAG?"

# 4. run the agent end-to-end (offline, no API key needed)
python scripts/agent_demo.py

# 5. run API (M3)
uvicorn app.main:app --reload

# 6. query
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Quickstart (Docker)

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## Testing

```bash
pytest -m "not integration"   # fast, offline (default)
pytest -m integration         # spins up Docker (testcontainers)
```

Every tool call is boundary-guarded: tool exceptions and invalid LLM JSON become
`ToolCall(ok=False)` or a corrective feedback message — the agent loop never crashes.

## Evaluation

```bash
python -m evals.run --dataset evals/data/demo.jsonl --out evals/reports/
```

Latest report: `evals/reports/retrieval_latest.json`

| Metric | Target | Latest |
|---|---|---|
| Retrieval hit-rate@5 (M1) | ≥ 0.80 | **1.000** (12/12) |
| Multi-hop agent success (M2) | pass | ✅ (`scripts/agent_demo.py`) |
| Faithfulness (M4) | ≥ 0.85 | – |
| Answer relevancy (M4) | ≥ 0.80 | – |
| Context precision (M4) | ≥ 0.75 | – |
| p95 latency (query) | ≤ 2.5 s | – |
| Cost / 1k queries | ≤ $0.50 | – |

---

## Failure modes & runbooks (M4)

See `docs/runbooks/`:

1. `llm-timeout.md` — retry with jitter, fallback model, cache hit
2. `vector-store-down.md` — circuit breaker, degraded retrieval
3. `schema-violation.md` — hard fail, correlation ID, sample payload

---

## Contributing

Conventional commits. One PR = one milestone checkbox. `pre-commit` runs ruff + mypy.

---

## Roadmap (production-scale)

Each milestone ships **runnable, tested, and documented** code — not stubs.

### M1 — Local RAG (deterministic core) ✅

- [x] `app/rag/`: chunker (recursive + version), embedder (provider-agnostic), Chroma store
- [x] Idempotent ingest: `sha256(file) + chunker_version` as key
- [x] `scripts/ingest_demo.py` ingests a sample corpus
- [x] Retrieval returns top-k with scores + source spans
- [x] Unit tests (chunker edge cases, embedder mock)
- [x] **Exit criteria:** `pytest -m "not integration"` green, retrieval hit-rate@5 = **1.000** on demo set

### M2 — Agent + tools ✅

- [x] Agent loop with tool registry (`register(tool)`), driven by an `LLM` protocol
- [x] 3 tools: `search_docs`, `calculator`, `web_fetch` (timeout + retry, jitter backoff)
- [x] Structured JSON output enforced via `AgentOutput` (Pydantic v2, `extra="forbid"`)
- [x] Pure logic vs IO separated (`app/tools/*.py` pure; `app/tools/adapters/*.py` IO)
- [x] Offline-testable: `FakeLLM` + `FakeEmbedder` + `MemoryStore`
- [x] **Exit criteria:** multi-hop demo (`search_docs → calculator → finish`) returns valid `AgentOutput`; schema round-trip passes

### M3 — API + Docker

- [ ] FastAPI routes: `/ingest`, `/query`, `/healthz`, `/metrics`
- [ ] Correlation ID middleware, structured JSON logs
- [ ] `docker/Dockerfile` (multi-stage), `docker-compose.yml` (API + Qdrant)
- [ ] `.env.example` fully documents every knob
- [ ] Config via `pydantic-settings`, no hardcoded strings
- [ ] **Exit criteria:** `docker compose up` → `curl /query` returns valid JSON

### M4 — CI + eval report

- [ ] GitHub Actions: lint (ruff) → type (mypy) → unit test → build → smoke
- [ ] Integration tests behind `integration` marker (testcontainers)
- [ ] Ragas eval job: faithfulness, answer relevancy, context precision
- [ ] `evals/reports/` committed as artifacts; README badge shows latest
- [ ] `docs/runbooks/` for 3 top failures (LLM timeout, store down, bad schema)
- [ ] **Exit criteria:** CI green on `main`, eval report published, changelog updated

### M5+ (backlog — post-MVP)

- [ ] Streaming ingestion (Kafka / S3 events)
- [ ] Reranker (cross-encoder) behind flag
- [ ] Multi-tenant isolation + auth
- [ ] Cost model + benchmarks (`docs/benchmarks.md`)
- [ ] OpenTelemetry traces
- [ ] Optional: swap state machine → LangGraph (interface already compatible)

---

## License

MIT — see [`LICENSE`](LICENSE).

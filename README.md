# rag-agent-platform

> A production-minded RAG + AI agent backend with tool-calling, structured outputs, and a reproducible evaluation loop.

[![CI](https://github.com/WoodinGlass/rag-agent-platform/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WoodinGlass/rag-agent-platform/actions/workflows/ci.yml)
[![Eval](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/WoodinGlass/rag-agent-platform/main/evals/reports/badge.json)](https://github.com/WoodinGlass/rag-agent-platform/blob/main/evals/reports/latest.md)
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
- **Agent loop:** deterministic state machine (offline-testable); LangGraph-ready interface
- **LLM providers:** OpenAI, Anthropic (pluggable via `LLM` protocol; `FakeLLM` for tests)
- **Vector store:** Chroma (local) / Qdrant (service) / MemoryStore (tests)
- **Schemas:** Pydantic v2 (`extra="forbid"` for contract enforcement)
- **Eval:** offline retrieval metrics (hit-rate, MRR, precision, recall) on every PR; Ragas (faithfulness, relevancy, context precision) via LLM judge on weekly/manual runs
- **Observability:** structured JSON logs + correlation ID; `/metrics` endpoint
- **Packaging:** `pyproject.toml` (single source of truth)
- **Infra:** Docker + docker-compose, GitHub Actions CI (lint / type / coverage / docker smoke) + Eval workflow

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
| **Deterministic vs non-deterministic** | Ingest + retrieval deterministic; generation non-deterministic | Enables re-ingest without duplicates |
| **Payload vs pipeline** | Payload = documents + embeddings; Pipeline = chunk → embed → retrieve → generate | Clear separation for testing |
| **Storage tier** | Small structured: SQLite (dev) / Postgres (prod). Vectors: Chroma → Qdrant | No refactor when scaling |
| **Ingestion mode** | Batch first; streaming later behind flag | MVP velocity |
| **Idempotency key** | `sha256(file_bytes) + chunker_version` | Re-ingest is a no-op |
| **Agent loop** | State machine, not LangGraph (yet) | Fully offline-testable; `Agent.run()` signature stable |
| **Pure vs IO** | Tools pure; IO in `tools/adapters/` | Unit tests never touch network/disk |
| **Failure modes** | LLM timeout → retry w/ jitter. Store down → circuit breaker. API → 503 + CID in logs | See `docs/runbooks/` |

---

## Repository layout

```text
rag-agent-platform/
├── app/
│   ├── agents/              # agent loop, registry, schema, prompts, bootstrap
│   ├── rag/                 # chunker, embedder, store, pipeline
│   ├── tools/               # tool implementations + IO adapters
│   ├── api/                 # FastAPI routes, schemas, deps, middleware
│   ├── core/                # config, logging, ids, metrics
│   └── main.py              # application entrypoint (lifespan)
├── evals/
│   ├── metrics.py           # pure retrieval metrics (hit, MRR, P, R)
│   ├── ragas_eval.py        # unified CLI (retrieval | ragas modes)
│   ├── data/                # corpus.jsonl, demo.jsonl, eval_set.jsonl
│   └── reports/             # latest.{json,md}, badge.json (committed)
├── tests/
│   ├── unit/                # fast, offline (default)
│   └── integration/         # end-to-end (marker: integration)
├── docker/                  # Dockerfile + docker-compose.yml
├── docs/                    # architecture.md + runbooks/
├── scripts/                 # agent_demo.py, ingest_demo.py
├── .github/workflows/       # ci.yml (lint/type/coverage/docker), eval.yml
├── .env.example
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

# 5. run API
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

## API

All requests accept an optional `X-Request-ID` header; the value (or a generated one) is echoed back and attached to every log line.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Base64 bytes -> chunk -> embed -> store (idempotent on `sha256+version`) |
| `POST` | `/query` | Run agent loop -> `AgentOutput` (structured JSON) |
| `GET`  | `/healthz` | Liveness + shallow check of agent/pipeline wiring |
| `GET`  | `/metrics` | In-process counters + latency histograms |
| `GET`  | `/docs` | OpenAPI UI |

### Example

```bash
# ingest
CONTENT=$(printf 'The cat sat on the mat.' | base64 -w0)
curl -s -X POST localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d "{\"content_base64\":\"$CONTENT\",\"source\":\"cat.txt\"}"

# query
curl -s -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-1' \
  -d '{"question":"What did the cat do?"}'

# metrics
curl -s localhost:8000/metrics
```

### Error format

All 4xx/5xx return a JSON body:

```json
{"error": "internal_error", "detail": "...", "correlation_id": "abc123"}
```

Validation errors follow FastAPI's default `{"detail": [...]}` shape.

---

## Docker

```bash
# build + run API (MemoryStore, FakeLLM - fully offline)
docker compose -f docker/docker-compose.yml up --build

# in another shell
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"hi"}'
```

The compose stack also boots **Qdrant** on ports `6333` (HTTP/dashboard) and `6334` (gRPC). Set `VECTOR_BACKEND=qdrant` once the Qdrant adapter lands (M5+); today the API defaults to `memory` so the stack works with zero config.

Config surface: see `.env.example`. All knobs come from env — nothing is hardcoded. The image runs as a non-root user and ships a `HEALTHCHECK`.

---

## Testing

```bash
pytest -m "not integration"   # fast, offline (default)
pytest -m integration         # end-to-end (in-process)
```

Coverage (threshold 85%) and type checking (mypy) run in CI. Every tool call is boundary-guarded: tool exceptions and invalid LLM JSON become `ToolCall(ok=False)` or a corrective feedback message — the agent loop never crashes.

## Evaluation

Two modes, run separately:

```bash
# offline retrieval metrics (deterministic, no API key)
python -m evals.ragas_eval --mode retrieval --k 5 --out-dir evals/reports

# generation metrics via Ragas + LLM judge (needs OPENAI_API_KEY)
python -m evals.ragas_eval --mode ragas --k 5 --out-dir evals/reports
```

Latest report: [`evals/reports/latest.md`](evals/reports/latest.md)

| Metric | Target | Latest |
|---|---|---|
| Retrieval hit-rate@5 | >= 0.80 | **1.0000** (15/15) |
| Retrieval MRR@5 | >= 0.60 | **0.7189** |
| Retrieval recall@5 | >= 0.80 | **0.9667** |
| Retrieval precision@5 | informational | 0.2533 |
| Faithfulness (Ragas) | >= 0.85 | *set `OPENAI_API_KEY` to enable* |
| Answer relevancy (Ragas) | >= 0.80 | *set `OPENAI_API_KEY` to enable* |
| Context precision (Ragas) | >= 0.75 | *set `OPENAI_API_KEY` to enable* |
| p95 latency (query) | <= 2.5 s | - |
| Cost / 1k queries | <= $0.50 | - |

Retrieval eval runs on every PR (offline, free). Generation eval runs weekly + on-demand via `.github/workflows/eval.yml`; it skips cleanly when `OPENAI_API_KEY` is absent.

---

## Failure modes & runbooks

Three failure modes are documented end-to-end (symptom -> triage ->
mitigation -> prevention -> signals):

1. [`llm-timeout.md`](docs/runbooks/llm-timeout.md) — provider slow or down;
   retry with jitter, fallback to `fake`, cache hit
2. [`vector-store-down.md`](docs/runbooks/vector-store-down.md) — store
   unreachable; restart, snapshot, or fall to `memory` (degraded)
3. [`schema-violation.md`](docs/runbooks/schema-violation.md) — client
   contract or LLM output violates schema; hard fail with cid

Architecture overview: [`docs/architecture.md`](docs/architecture.md).

---

## Contributing

Conventional commits. One PR = one milestone checkbox. `pre-commit` runs ruff + mypy.

---

## Roadmap (production-scale)

Each milestone ships **runnable, tested, and documented** code — not stubs.

### M1 — Local RAG (deterministic core) [done]

- [x] `app/rag/`: chunker (recursive + version), embedder (provider-agnostic), Chroma store
- [x] Idempotent ingest: `sha256(file) + chunker_version` as key
- [x] `scripts/ingest_demo.py` ingests a sample corpus
- [x] Retrieval returns top-k with scores + source spans
- [x] Unit tests (chunker edge cases, embedder mock)
- [x] **Exit criteria:** `pytest -m "not integration"` green, retrieval hit-rate@5 = **1.000** on demo set

### M2 — Agent + tools [done]

- [x] Agent loop with tool registry (`register(tool)`), driven by an `LLM` protocol
- [x] 3 tools: `search_docs`, `calculator`, `web_fetch` (timeout + retry, jitter backoff)
- [x] Structured JSON output enforced via `AgentOutput` (Pydantic v2, `extra="forbid"`)
- [x] Pure logic vs IO separated (`app/tools/*.py` pure; `app/tools/adapters/*.py` IO)
- [x] Offline-testable: `FakeLLM` + `FakeEmbedder` + `MemoryStore`
- [x] **Exit criteria:** multi-hop demo (`search_docs -> calculator -> finish`) returns valid `AgentOutput`

### M3 — API + Docker [done]

- [x] FastAPI routes: `/ingest`, `/query`, `/healthz`, `/metrics`
- [x] Correlation ID middleware, structured JSON logs
- [x] `docker/Dockerfile` (multi-stage), `docker-compose.yml` (API + Qdrant)
- [x] `.env.example` fully documents every knob
- [x] Config via `pydantic-settings`, no hardcoded strings
- [x] **Exit criteria:** `docker compose up` -> `curl /query` returns valid JSON (verified in CI)

### M4 — CI + eval report [done]

- [x] GitHub Actions: lint (ruff) -> type (mypy) -> coverage -> docker smoke
- [x] Integration tests behind `integration` marker (offline, in-process)
- [x] Ragas eval job: faithfulness, answer relevancy, context precision (skip-friendly without API key)
- [x] `evals/reports/` committed; README shows endpoint badge + latest report link
- [x] `docs/runbooks/` for 3 failures (LLM timeout, store down, schema violation)
- [x] **Exit criteria:** CI green on `main`, eval report published, changelog updated

### M5+ (backlog — post-MVP)

- [ ] Streaming ingestion (Kafka / S3 events)
- [ ] Reranker (cross-encoder) behind flag
- [ ] Multi-tenant isolation + auth
- [ ] Cost model + benchmarks (`docs/benchmarks.md`)
- [ ] OpenTelemetry traces
- [ ] Swap state machine -> LangGraph (interface already compatible)

---

## License

MIT — see [`LICENSE`](LICENSE).

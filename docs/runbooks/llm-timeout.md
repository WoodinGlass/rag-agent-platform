# Runbook: LLM Timeout

## Symptom

- `/query` returns `500` with `correlation_id` in the body.
- Log line: `request.error` with `detail="LLM timeout after Ns"` or
  `"network error after N tries"`.
- Client sees elevated latency before failure (near `LLM_TIMEOUT_S`).
- `/metrics` shows `query.latency_ms` histogram bucket at the top end
  climbing.

## Blast radius

| Component | Impact |
|---|---|
| `/query` | Degraded — some requests fail |
| `/ingest` | **Unaffected** (no LLM call) |
| `/healthz` | Still `ok` (LLM not part of the shallow check) |
| Retrieval | **Unaffected** (embedder is local/independent) |

## Triage (first 3 checks)

1. **Correlate the failure in logs.**
   ```bash
   # grep the cid from the client error body
   docker logs rag-api | grep '"cid":"<CID>"' | tail -20
Confirm it's the LLM provider, not us.

bash
# quick provider ping (OpenAI example)
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" \
  https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
Check /metrics for the shape.

bash
curl -s localhost:8000/metrics | jq '.histograms."query.latency_ms"'
All buckets hit at 10000 → provider-wide outage.

Mostly <=100 with a few outliers → transient; retries absorbed most.

Mitigation
Fast path (no code change):

Bump timeout + retries via env and restart:

bash
LLM_TIMEOUT_S=60 LLM_MAX_RETRIES=5 \
  docker compose -f docker/docker-compose.yml up -d api
Downgrade to fake LLM to keep the API serving a valid (refused)
response while you investigate:

bash
LLM_PROVIDER=fake \
  docker compose -f docker/docker-compose.yml up -d api
If the outage is provider-wide and sustained: put the API in
read-only mode by flipping LLM_PROVIDER=fake in .env. /ingest stays
functional; /query returns a structured refused=true answer — no 500s.

Prevention
Retry with jitter already implemented in app/agents/agent.py
(exponential backoff, capped).

Fallback model (roadmap M5+): try gpt-4o-mini, fall back to
claude-haiku if primary provider 5xx > threshold.

Circuit breaker (roadmap M5+): after N consecutive failures, short-circuit
to refused=true for a cooldown window — prevents request pile-up.

Cache (roadmap M5+): identical question + retriever context → cached answer.

Signals
Signal  Where   Meaning
request.error log with cid  stdout / log collector  Which request failed
query.latency_ms top-bucket count   /metrics    Provider slowness, not our code
query.refused counter   /metrics    Fallback engaged
Provider status page    external    Third-party health
Related
app/agents/agent.py — retry wrapper

app/tools/adapters/http.py — same pattern for outbound HTTP

docs/architecture.md (M4.4) — failure-mode map

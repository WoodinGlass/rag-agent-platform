# Runbook: Vector Store Down

## Symptom

- `/query` returns `500` with `detail` mentioning the store or the
  Chroma/Qdrant client.
- `/ingest` returns `500` (ingestion requires the store).
- `/healthz` may report `status="degraded"` if `app.state.pipeline`
  init failed at startup.
- Log line: `request.error` around `app/rag/store.py`.

## Blast radius

| Component | Impact |
|---|---|
| `/ingest` | **Down** (needs store write) |
| `/query` | **Down** (needs store read) |
| `/healthz` | `degraded` |
| `/metrics` | Still serving (in-process) |
| `/docs` | Still serving |

**If using `MemoryStore` (default):** the store cannot go down — it's
in-process. This runbook applies to `ChromaStore` (local files) and a
future `QdrantStore` (remote service).

## Triage

1. **Which backend?**
   ```bash
   curl -s localhost:8000/healthz | jq .checks
   # or check env
   grep VECTOR_BACKEND .env
If chroma (local files): check disk and permissions.

bash
df -h /app/data
ls -la /app/data/chroma 2>/dev/null || echo "path missing"
If qdrant (service): check container + port.

bash
docker ps --filter name=rag-qdrant
curl -sS http://localhost:6333/readyz && echo " qdrant up"
docker logs rag-qdrant | tail -50
Mitigation
A. Qdrant is down / unhealthy

bash
docker restart rag-qdrant
# wait for ready
until curl -fsS http://localhost:6333/readyz; do sleep 1; done
If it stays unhealthy, inspect volume:

bash
docker volume inspect rag-agent-platform_qdrant-data
# last resort: back up the volume, then recreate
B. Chroma path missing / corrupted

bash
# back up first (never delete before copying)
cp -a /app/data/chroma /app/data/chroma.bak.$(date +%s) || true
# then re-ingest from source of truth:
python scripts/ingest_demo.py
Ingestion is idempotent on sha256(content) + chunker_version —
re-ingesting is safe and cheap. A fresh store from source is fully
consistent.

C. Emergency degraded mode

Set VECTOR_BACKEND=memory and restart. /query will return
refused=true (no docs, no evidence) instead of 500. Use only as a
stopgap to stop alerting while you fix the real backend.

Prevention
Healthcheck endpoint on Qdrant container (already in compose).

Circuit breaker (roadmap M5+): fail fast with refused=true when
store latency > threshold, before requests pile up.

Read replica / snapshot (roadmap M5+): keep a cold standby; swap on
failover.

Backups: Qdrant supports snapshots (POST /collections/{name}/snapshots).
Schedule via cron and ship to object storage (S3).

Ingest is a no-op on duplicates, so recovery = re-run the pipeline.

Signals
Signal  Where   Meaning
healthz.status=degraded /healthz    Stack not fully wired
docker ps shows rag-qdrant unhealthy    host    Service-level issue
Qdrant readyz returns non-200   Qdrant API  Store unavailable
ingest.requests vs ingest.inserted_docs gap /metrics    Silent write failures
Related
app/rag/store.py — MemoryStore, ChromaStore, future QdrantStore

docker/docker-compose.yml — Qdrant service + healthcheck

app/rag/pipeline.py — idempotent ingest path

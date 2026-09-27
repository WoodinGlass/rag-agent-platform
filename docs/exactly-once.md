# Exactly-once in a RAG ingestion pipeline

This document explains why "exactly-once" for a RAG ingestion pipeline is
**idempotency plus dedup**, not Kafka transactions — and what would change
if the sink were itself Kafka.

## TL;DR

- Kafka's exactly-once semantics (EOS) covers **Kafka-to-Kafka** flows:
  consume, process, produce, commit offsets — all inside one transaction.
- Our pipeline's sink is a **vector store** (and optionally an object
  store). Kafka transactions cannot make writes to those systems atomic
  with offset commits.
- The correct contract for this pipeline is **at-least-once delivery +
  idempotent handler**. We already implement both:
  - Ingestion is idempotent on `sha256(content) + chunker_version`
    (`app/rag/pipeline.py:ingest`).
  - The streaming ingestor is at-least-once with a bounded dedup window
    (`app/streaming/ingestor.py`).
- "Effectively once" is what we promise. That is the strongest honest
  guarantee for this topology.

## What Kafka EOS actually covers

Kafka's transactional producer + `read_committed` consumer gives you:

1. **Atomic offset commit + produced records.** Either both happen or
   neither, within one Kafka cluster.
2. **`read_committed` isolation.** Consumers see only records whose
   transaction committed.
3. **Epoch fencing.** A zombie producer cannot write after a new one
   takes over.

Constraints (see the Kafka docs on transactional messaging):

- Transactions are scoped to **Kafka as both source and sink**.
- Any external side effect (DB write, HTTP call, vector upsert) is
  **outside** the transaction. It can be applied, then the transaction
  can fail — leaving an orphan side effect with no committed offset.

## Why that does not fit this pipeline

Our write path is:
EventSource (Kafka/S3/memory)
│ poll
▼
StreamingIngestor
│ handler(event)
▼
RagPipeline.ingest(...) ──► VectorStore.upsert(...)
│ (MemoryStore / Chroma / Qdrant)
│ commit(event_id)
▼
EventSource.commit


The atomic unit we need is: *"this document is fully ingested in the
vector store, and this offset is committed."* Kafka cannot express that
because the vector store is not Kafka.

If we wrapped the handler in a Kafka transaction and the vector upsert
succeeded but the transaction aborted, we would have to roll back the
vector store — which is exactly the transactional guarantee our vector
store does not offer.

## What we do instead (and where it lives)

Two mechanisms compose into effectively-once:

### 1. Idempotent ingestion

`app/rag/pipeline.py` derives `doc_id` from
`sha256(content) + chunker_version`. Re-ingesting the same bytes with the
same chunker is a no-op: the store reports `skipped=True`. So even if
the same event is delivered twice, the vector store converges to the same
state.

Tested by:
- `tests/unit/test_pipeline.py::test_ingest_idempotent`
- `tests/unit/test_streaming_ingestor.py::test_duplicate_event_is_deduped`

### 2. At-least-once delivery with a bounded dedup window

`app/streaming/ingestor.py` commits an event **only after** the handler
returns successfully. On failure, the event is not committed; the source
redelivers. A bounded LRU of recently-seen event ids (`dedup_window`)
absorbs redeliveries cheaply.

Tested by:
- `tests/unit/test_streaming_ingestor.py::test_retries_then_succeeds`
- `tests/unit/test_streaming_ingestor.py::test_gives_up_after_max_retries_and_does_not_commit`
- `tests/unit/test_streaming_ingestor.py::test_dedup_window_is_bounded`

### 3. Tenant isolation composes with both

`doc_id` is tenant-scoped (`app/rag/pipeline.py`), so two tenants sending
the same bytes get different ids. Dedup operates on the event id, which
the source owns; the ingestor does not confuse cross-tenant events.

## What we do NOT claim

- **We do not claim exactly-once.** We claim *effectively once*: at-least-
  once delivery plus an idempotent write path. That is a different
  guarantee with different failure modes.
- **We do not claim crash-safe dedup state.** The dedup window is
  in-memory. A worker restart re-emits some events; the pipeline
  tolerates that because ingestion is idempotent. Durable dedup state
  is a follow-up (would need a small KV store or a compacted topic).

## Failure mode walkthrough

| Failure | What happens | Why we're fine |
|---|---|---|
| Handler raises (bad bytes, transient store error) | Event retried up to `max_retries`; on give-up, **not committed** | Source redelivers; ingestor dedup absorbs duplicates |
| Worker crashes after `ingest` succeeds but before `commit` | Source redelivers the same event | Ingestion is idempotent → second run reports `skipped=True` |
| Worker crashes mid-`ingest` (partial write to store) | Store may have partial state | `doc_id` is deterministic; next run re-ingests the whole document cleanly |
| Source redelivers a batch | Ingestor sees same ids within `dedup_window` | Dedup skips them; commit is safe to repeat |
| Dedup window overflows under long replay | Older ids evicted → re-ingested | Idempotency makes re-ingest a no-op at the store |

## When would Kafka transactions be the right tool?

If the pipeline were **Kafka-to-Kafka** — e.g., consuming raw document
events and producing embedding events back to Kafka — then a
transactional producer plus `read_committed` consumer would give a
genuine exactly-once contract *within Kafka*, and downstream consumers
could rely on it. That is a legitimate design for a different topology.

In this repository, the topology is Kafka → **vector store**, so the
honest tool is idempotency, not a transaction.

## References

- Ingestion idempotency: `app/rag/pipeline.py:ingest`
- Deterministic doc ids: `app/core/ids.py`
- Ingestor (at-least-once + dedup): `app/streaming/ingestor.py`
- Kafka adapter: `app/streaming/kafka_source.py`
- Kafka transactional messaging docs:
  <https://kafka.apache.org/documentation/#transactions>

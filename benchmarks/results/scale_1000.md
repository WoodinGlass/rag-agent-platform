# Scale benchmark: 1000 documents

- Generated: `2026-09-27T21:29:34+00:00`
- Docs: 1000 | queries: 200 | k=5
- chunk_size=200 | embedder=`fake` | template=v1 | seed=42

## Ingest (measured)

| Metric | Value |
|---|---|
| total | 1060.744 ms |
| per document | 1.0607 ms |
| throughput | 942.7 docs/s |

## Retrieve latency (measured)

| Metric | ms |
|---|---|
| mean | 154.994 |
| p50 | 119.993 |
| p90 | 217.163 |
| p95 | 237.981 |
| p99 | 415.619 |
| min | 101.788 |
| max | 487.937 |

## Retrieval quality

| Metric | Value |
|---|---|
| hit_rate@5 | 0.96 |
| MRR@5 | 0.9163 |
| recall@5 | 0.96 |

> Single-threaded. `MemoryStore`. `FakeEmbedder(dim=256)`. See
> `docs/benchmarks.md` for interpretation and honest caveats.

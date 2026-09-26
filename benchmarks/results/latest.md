# Benchmarks

- Generated: `2026-09-26T17:33:19+00:00`
- Corpus: 12 docs | chunk_size=200 | embedder=`fake`
- Retrieve: n=100 (warmup 10) | k=5

## Ingest (measured)

- Total: **3.469 ms**
- Per document: **0.289 ms**

## Retrieve latency (measured)

| Metric | ms |
|---|---|
| mean | 2.188 |
| p50 | 2.067 |
| p90 | 2.257 |
| p95 | 3.168 |
| p99 | 4.154 |
| min | 1.834 |
| max | 4.18 |

## Cost model (projected, not measured)

- Corpus embedding: **$7e-06** (369 tokens)
- Per query: **$7.9e-05** (avg question 7.0 tok + 200 ctx + 80 out)
- Per 1,000 queries: **$0.079**

> Projection only. Measured runs use FakeEmbedder + FakeLLM (zero API cost). Token counts use whitespace tokenizer.

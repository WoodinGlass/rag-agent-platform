# Kafka consumer group rebalance

This document explains how `ParallelIngestor` reacts to consumer group
rebalances and how to wire it to a real Kafka consumer.

## Why rebalance happens

Kafka assigns partitions to consumers in a group. A rebalance is
triggered whenever:

- A consumer joins the group.
- A consumer leaves (graceful or crash).
- A partition count changes.
- A subscription changes.

During a rebalance, ownership of every partition is re-decided. Kafka
guarantees that at any moment exactly one consumer owns a partition, and
it will not process events for a partition it no longer owns.

## What the consumer is told

The Kafka client invokes two callbacks on the consumer:

- **`on_revoke(partitions)`** — you are losing these partitions. Stop
  processing, commit offsets, release resources.
- **`on_assign(partitions)`** — you now own these partitions. Start
  consuming from the last committed offset.

The callbacks are synchronous from the client's point of view. Blocking
inside them pauses the client; work must be quick.

## How ParallelIngestor models this

Two methods mirror the Kafka callbacks:

- `on_partitions_assigned(partitions: list[str])`
  Starts one `StreamingIngestor` per new partition. Idempotent per
  partition: assigning an already-running partition is a no-op.

- `on_partitions_revoked(partitions: list[str])`
  Signals each per-partition ingestor to stop, waits up to
  `rebalance_grace_s` (default 5 s) for the in-flight event to finish,
  then cancels. Snapshots stats. Removes the partition from the active
  set. Unknown partitions are ignored.

Both methods take a lock, so concurrent rebalance events do not race.

## Wiring to a real consumer

Kafka's callbacks are synchronous; our methods are `async`. Bridge them
with `asyncio.run_coroutine_threadsafe` if the consumer runs in its own
thread (the common case for `kafka-python` + asyncio):

```python
import asyncio
from app.streaming.kafka_source import KafkaConfig, KafkaEventSource
from app.streaming.parallel_ingestor import ParallelIngestor

loop = asyncio.get_running_loop()
ingestor = ParallelIngestor(source=..., handler=...)

def on_assign(consumer, partitions):
    names = [f"{p.topic}:{p.partition}" for p in partitions]
    fut = asyncio.run_coroutine_threadsafe(
        ingestor.on_partitions_assigned(names), loop
    )
    # optional: fut.result(timeout=...)

def on_revoke(consumer, partitions):
    names = [f"{p.topic}:{p.partition}" for p in partitions]
    fut = asyncio.run_coroutine_threadsafe(
        ingestor.on_partitions_revoked(names), loop
    )
    fut.result(timeout=ingestor.rebalance_grace_s + 1.0)

consumer.subscribe(
    topics=["rag-docs"],
    on_assign=on_assign,
    on_revoke=on_revoke,
)
Grace period and at-least-once
When a revoke arrives, the ingestor may be in the middle of handling an
event. We do not abort mid-handler: the event must be committed
correctly or not at all.

If the handler finishes within rebalance_grace_s, we commit and
finish cleanly.

If it does not, we cancel the task. The event is not committed;
the new owner of the partition will redeliver it. Our idempotent
ingest makes reprocessing safe.

This preserves the at-least-once guarantee documented in
docs/exactly-once.md.

What we do NOT do
Static membership / cooperative-sticky assignment. We react to
whatever assignment Kafka decides; we do not influence it.

Cross-partition barrier. Each partition is independent. There is
no ordering guarantee across partitions.

Offset flush on revoke. The ingestor commits after every
successful event, so there is nothing to flush at revoke time.

Tests
tests/unit/test_streaming_rebalance.py simulates rebalance by calling
the two hook methods directly against an InMemoryPartitionedSource:

assign starts ingestors; idempotent for known partitions

revoke stops them; stats preserved; unknown partitions ignored

revoke waits for an in-flight handler; cancels after grace

dynamic assign during run() picks up new work

run() uses initial partitions and shuts down on stop

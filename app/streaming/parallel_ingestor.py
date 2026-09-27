"""ParallelIngestor: run one StreamingIngestor per partition.

Design:
- One ingestor per partition. Each has its own dedup window and its own
  commit path. This is what Kafka requires for correctness: commits are
  scoped to a partition.
- `asyncio.gather(..., return_exceptions=True)` for isolation: if one
  partition's loop raises, the others keep running. We record errors in
  aggregate stats and log them.
- Stopping: `stop()` sets a flag on every ingestor. The orchestrator's
  `run()` returns when all per-partition loops finish.

Not implemented (deliberate):
- Dynamic rebalance: partitions() is called once at run() start. Adding
  or removing partitions at runtime is a follow-up (Kafka consumer
  group rebalance would drive it).
- Backpressure across partitions: each partition has its own prefetch
  (if enabled) and its own in-flight event. Overall memory is bounded
  by partitions * per_partition_budget.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.core.tracing import span
from app.streaming.ingestor import EventHandler, IngestorStats, StreamingIngestor
from app.streaming.partitioned import PartitionedSource

log = get_logger(__name__)


@dataclass
class ParallelStats:
    per_partition: dict[str, dict] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def total(self, key: str) -> int:
        return sum(
            int(stats.get(key, 0)) for stats in self.per_partition.values()
        )

    def as_dict(self) -> dict:
        return {
            "partitions": list(self.per_partition),
            "totals": {
                "pulled": self.total("pulled"),
                "committed": self.total("committed"),
                "deduped": self.total("deduped"),
                "failed": self.total("failed"),
                "retries": self.total("retries"),
                "handler_errors": self.total("handler_errors"),
            },
            "errors": dict(self.errors),
            "per_partition": dict(self.per_partition),
        }


class ParallelIngestor:
    def __init__(
        self,
        source: PartitionedSource,
        handler: EventHandler,
        *,
        poll_timeout_s: float = 1.0,
        max_retries: int = 3,
        dedup_window: int = 1024,
        idle_sleep_s: float = 0.05,
        retry_backoff_base_s: float = 0.1,
    ) -> None:
        self.source = source
        self.handler = handler
        self.poll_timeout_s = poll_timeout_s
        self.max_retries = max_retries
        self.dedup_window = dedup_window
        self.idle_sleep_s = idle_sleep_s
        self.retry_backoff_base_s = retry_backoff_base_s

        self.stats = ParallelStats()
        self._ingestors: dict[str, StreamingIngestor] = {}
        self._stop = asyncio.Event()

    # ---------- run ----------

    async def run(self) -> None:
        """Run all partitions until stop() is called."""
        partitions = self.source.partitions()
        if not partitions:
            log.warning("parallel.no_partitions")
            return

        log.info("parallel.start", extra={"partitions": partitions})

        tasks: list[asyncio.Task[None]] = []
        for name in partitions:
            partition_source = self.source.source_for(name)
            ing = StreamingIngestor(
                partition_source,
                self.handler,
                poll_timeout_s=self.poll_timeout_s,
                max_retries=self.max_retries,
                dedup_window=self.dedup_window,
                idle_sleep_s=self.idle_sleep_s,
                retry_backoff_base_s=self.retry_backoff_base_s,
            )
            self._ingestors[name] = ing

            async def _runner(partition=name, ingestor=ing):
                try:
                    with span("parallel.partition", partition=partition):
                        await ingestor.run()
                except Exception as e:  # noqa: BLE001 -- isolation boundary
                    self.stats.errors[partition] = str(e)
                    log.error(
                        "parallel.partition_crashed",
                        extra={"partition": partition, "error": str(e)[:200]},
                    )

            tasks.append(asyncio.create_task(_runner(), name=f"ingestor-{name}"))

        # Supervisor: stop everything when the external stop flag fires.
        async def _watch_stop() -> None:
            await self._stop.wait()
            for ing in self._ingestors.values():
                ing.stop()

        stopper = asyncio.create_task(_watch_stop(), name="parallel.stop-watch")

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        finally:
            stopper.cancel()
            try:
                await stopper
            except asyncio.CancelledError:
                pass

            # Snapshot stats regardless of how we exited.
            for name, ing in self._ingestors.items():
                self.stats.per_partition[name] = ing.stats.as_dict()

            log.info("parallel.stop", extra={"n_partitions": len(self._ingestors)})

    # ---------- control ----------

    def stop(self) -> None:
        """Signal every partition loop to finish after the current event."""
        self._stop.set()

    # ---------- introspection ----------

    def ingestor_for(self, partition: str) -> StreamingIngestor | None:
        return self._ingestors.get(partition)

    @property
    def ingestor_stats(self) -> dict[str, IngestorStats]:
        return {name: ing.stats for name, ing in self._ingestors.items()}

"""ParallelIngestor: one StreamingIngestor per partition, plus rebalance.

Design:
- One ingestor per partition. Each has its own dedup store and its own
  commit path (Kafka commit is partition-scoped).
- `asyncio.gather` per partition with `return_exceptions=False` and
  try/except inside the task; a crashed partition is recorded, others
  keep running.
- Rebalance hooks (`on_partitions_assigned` / `on_partitions_revoked`)
  match the semantics Kafka's consumer group callbacks use. Call them
  from the consumer's `on_assign` / `on_revoke` (see
  `docs/kafka-rebalance.md`).

Rebalance policy:
- Assign: start an ingestor task per new partition, if not already running.
- Revoke: signal stop, wait up to `rebalance_grace_s` for the current
  event to finish, then cancel. Snapshot stats. Remove from dicts.
- Idempotent: assigning an already-running partition is a no-op;
  revoking an unknown partition is a no-op.

Not implemented (deliberate):
- Committing offsets on revoke: the ingestor already commits after each
  successful event; there is nothing extra to flush.
- Cross-partition ordering: Kafka guarantees per-partition order, which
  is what we preserve.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.core.tracing import span
from app.streaming.ingestor import EventHandler, IngestorStats, StreamingIngestor
from app.streaming.partitioned import PartitionedSource

log = get_logger(__name__)

DEFAULT_REBALANCE_GRACE_S = 5.0


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
        prefetch_n: int = 0,
        prefetch_budget_total: int = 0,
        rebalance_grace_s: float = DEFAULT_REBALANCE_GRACE_S,
    ) -> None:
        if prefetch_n < 0:
            raise ValueError("prefetch_n must be >= 0")
        if prefetch_budget_total < 0:
            raise ValueError("prefetch_budget_total must be >= 0")
        if rebalance_grace_s <= 0:
            raise ValueError("rebalance_grace_s must be > 0")

        self.source = source
        self.handler = handler
        self.poll_timeout_s = poll_timeout_s
        self.max_retries = max_retries
        self.dedup_window = dedup_window
        self.idle_sleep_s = idle_sleep_s
        self.retry_backoff_base_s = retry_backoff_base_s
        self.prefetch_n = prefetch_n
        self.prefetch_budget_total = prefetch_budget_total
        self.rebalance_grace_s = rebalance_grace_s

        self.stats = ParallelStats()
        self._ingestors: dict[str, StreamingIngestor] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._per_partition_prefetch: dict[str, int] = {}
        self._rebalance_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        # Per-partition prefetch size. Computed once in run() from the
        # initial partition count so all partitions see the same value.
        # Dynamic assignments during the run reuse this same value.
        self._partition_prefetch_size: int = 0

    # ---------- prefetch sizing ----------

    def _per_partition_prefetch_size(self, n_partitions: int) -> int:
        if self.prefetch_n <= 0:
            return 0
        if self.prefetch_budget_total <= 0:
            return self.prefetch_n
        per = self.prefetch_budget_total // max(1, n_partitions)
        return max(1, min(self.prefetch_n, per))

    # ---------- public lifecycle ----------

    async def run(self) -> None:
        """Assign initial partitions, wait for stop, then revoke all."""
        initial = self.source.partitions()
        if not initial:
            log.warning("parallel.no_partitions")

        self._partition_prefetch_size = self._per_partition_prefetch_size(
            len(initial)
        )
        log.info(
            "parallel.start",
            extra={
                "partitions": initial,
                "prefetch_per_partition": self._partition_prefetch_size,
            },
        )
        await self.on_partitions_assigned(initial)
        await self._stop.wait()

        # graceful shutdown: revoke everything still running
        await self.on_partitions_revoked(list(self._ingestors))
        log.info("parallel.stop", extra={"n_partitions": len(self.stats.per_partition)})

    def stop(self) -> None:
        """Signal every partition loop and the supervisor to finish."""
        self._stop.set()

    # ---------- rebalance API ----------

    async def on_partitions_assigned(self, partitions: list[str]) -> None:
        """Start ingestors for new partitions. Idempotent per partition.

        Safe to call from an asyncio callback while `run()` is blocked on
        stop. Uses a lock so concurrent rebalances don't race.
        """
        async with self._rebalance_lock:
            for name in partitions:
                if name in self._ingestors:
                    continue
                await self._assign_partition(name)
            log.info(
                "parallel.assigned",
                extra={"requested": partitions, "running": list(self._ingestors)},
            )

    async def on_partitions_revoked(self, partitions: list[str]) -> None:
        """Stop and await ingestors for the given partitions.

        Waits up to `rebalance_grace_s` for in-flight events to finish;
        cancels if they do not. Snapshots stats. Unknown partitions are
        ignored.
        """
        async with self._rebalance_lock:
            for name in partitions:
                await self._revoke_partition(name)
            log.info(
                "parallel.revoked",
                extra={"requested": partitions, "running": list(self._ingestors)},
            )

    # ---------- internal partition lifecycle ----------

    async def _assign_partition(self, name: str) -> None:
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

        prefetch_size = self._partition_prefetch_size
        self._per_partition_prefetch[name] = prefetch_size

        task = asyncio.create_task(
            self._run_partition(name, ing, prefetch_size),
            name=f"ingestor-{name}",
        )
        self._tasks[name] = task
        log.info(
            "parallel.partition_started",
            extra={"partition": name, "prefetch_n": prefetch_size},
        )

    async def _revoke_partition(self, name: str) -> None:
        ing = self._ingestors.get(name)
        task = self._tasks.get(name)
        if ing is None:
            return

        ing.stop()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=self.rebalance_grace_s)
            except TimeoutError:
                log.warning(
                    "parallel.rebalance_grace_exceeded",
                    extra={"partition": name, "grace_s": self.rebalance_grace_s},
                )
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # expected: we just cancelled it
                except Exception:
                    log.debug(
                        "parallel.task_error_after_cancel",
                        extra={"partition": name},
                        exc_info=True,
                    )

        snapshot = ing.stats.as_dict()
        snapshot["prefetch_n"] = self._per_partition_prefetch.get(name, 0)
        self.stats.per_partition[name] = snapshot
        self._ingestors.pop(name, None)
        self._tasks.pop(name, None)
        self._per_partition_prefetch.pop(name, None)
        log.info("parallel.partition_stopped", extra={"partition": name})

    async def _run_partition(
        self,
        name: str,
        ing: StreamingIngestor,
        prefetch_size: int,
    ) -> None:
        try:
            with span("parallel.partition", partition=name):
                if prefetch_size > 0:
                    await ing.run_with_prefetch(prefetch_n=prefetch_size)
                else:
                    await ing.run()
        except asyncio.CancelledError:
            # Expected during forced revoke.
            raise
        except Exception as e:  # noqa: BLE001 -- isolation boundary
            self.stats.errors[name] = str(e)
            log.error(
                "parallel.partition_crashed",
                extra={"partition": name, "error": str(e)[:200]},
            )

    # ---------- introspection ----------

    def ingestor_for(self, partition: str) -> StreamingIngestor | None:
        return self._ingestors.get(partition)

    def running_partitions(self) -> list[str]:
        return sorted(self._ingestors)

    @property
    def ingestor_stats(self) -> dict[str, IngestorStats]:
        return {name: ing.stats for name, ing in self._ingestors.items()}

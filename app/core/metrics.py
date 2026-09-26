"""Minimal in-process metrics.

We deliberately avoid pulling `prometheus-client` for the MVP: the
surface we need is small (counters + histograms for latency), and
in-process state is enough to render a `/metrics` payload.

Upgrade path (M5+): swap `Counter`/`Histogram` for `prometheus_client`
without changing call sites here.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _Histogram:
    buckets_ms: tuple[float, ...] = (5, 25, 100, 500, 2500, 10_000)
    counts: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    total: int = 0
    sum_ms: float = 0.0

    def observe(self, ms: float) -> None:
        self.total += 1
        self.sum_ms += ms
        for b in self.buckets_ms:
            if ms <= b:
                self.counts[b] += 1

    def snapshot(self) -> dict:
        return {
            "count": self.total,
            "sum_ms": round(self.sum_ms, 3),
            "avg_ms": round(self.sum_ms / self.total, 3) if self.total else 0.0,
            "buckets": {str(b): self.counts[b] for b in self.buckets_ms},
        }


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._hists: dict[str, _Histogram] = defaultdict(_Histogram)

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def observe_ms(self, name: str, ms: float) -> None:
        with self._lock:
            self._hists[name].observe(ms)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {k: v.snapshot() for k, v in self._hists.items()},
            }


_metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics

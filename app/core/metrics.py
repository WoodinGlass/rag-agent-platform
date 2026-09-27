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

    def reset(self) -> None:
        """Clear all counters and histograms. For tests and reinit only."""
        with self._lock:
            self._counters.clear()
            self._hists.clear()




# ---------- Prometheus exposition format ----------

# Metric names must match [a-zA-Z_:][a-zA-Z0-9_:]*. Our counters use
# dots (ingest.requests) which are not allowed; we sanitize to
# underscores (ingest_requests). Same for histogram names.
_NAME_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:"
)


def _sanitize_name(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if i == 0 and ch.isdigit():
            out.append("_")
            out.append(ch)
        elif ch in _NAME_ALLOWED:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _sanitize_label(value: str) -> str:
    # Prometheus label values escape backslash, double-quote, newline.
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def render_prometheus(snapshot: dict) -> str:
    """Render a `Metrics.snapshot()` dict as Prometheus text.

    - Counters become `<name>_total` (Prometheus convention).
    - Histograms are cumulative: bucket counts are monotonically
      non-decreasing and end with `+Inf` == total count.
    - `_sum` and `_count` accompany each histogram.
    - Names are sanitized to `[a-zA-Z_:][a-zA-Z0-9_:]*`.
    """
    lines: list[str] = []

    counters = snapshot.get("counters", {}) or {}
    for raw_name in sorted(counters):
        name = _sanitize_name(raw_name) + "_total"
        value = counters[raw_name]
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")

    hists = snapshot.get("histograms", {}) or {}
    for raw_name in sorted(hists):
        h = hists[raw_name]
        name = _sanitize_name(raw_name)
        lines.append(f"# TYPE {name} histogram")

        # cumulative buckets. Metrics._Histogram.buckets stores
        # inclusive counts per upper bound; ensure +Inf present.
        raw_buckets = h.get("buckets", {}) or {}
        # sort numerically by bound; "+Inf" always last
        def _bound_key(kv):
            k = kv[0]
            try:
                return (0, float(k))
            except (TypeError, ValueError):
                return (1, 0.0)

        running = 0
        for bound, count in sorted(raw_buckets.items(), key=_bound_key):
            running += int(count)
            lines.append(
                f'{name}_bucket{{le="{_sanitize_label(str(bound))}"}} {running}'
            )

        total = int(h.get("count", 0))
        lines.append(f'{name}_bucket{{le="+Inf"}} {total}')

        sum_ms = h.get("sum_ms", 0.0)
        # Prometheus expects float; keep ms as the unit (documented).
        lines.append(f"{name}_sum {sum_ms}")
        lines.append(f"{name}_count {total}")

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


_metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics

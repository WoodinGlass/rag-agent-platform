"""Tests for benchmark helpers. Offline, deterministic."""
import pytest

from benchmarks.run import (
    _estimate_tokens,
    _percentile,
    _summarize,
)

# ---- percentile ----

def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_single_value():
    assert _percentile([5.0], 50) == 5.0


def test_percentile_min():
    assert _percentile([1.0, 2.0, 3.0], 0) == 1.0


def test_percentile_max():
    assert _percentile([1.0, 2.0, 3.0], 100) == 3.0


def test_percentile_median_odd():
    assert _percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_percentile_median_even():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_interpolates():
    # 95th percentile of 1..100 should be near 95
    values = [float(i) for i in range(1, 101)]
    assert _percentile(values, 95) == pytest.approx(95.05, abs=0.2)


# ---- summarize ----

def test_summarize_empty():
    assert _summarize([]) == {"n": 0}


def test_summarize_basic():
    s = _summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5
    assert s["mean_ms"] == 3.0
    assert s["min_ms"] == 1.0
    assert s["max_ms"] == 5.0
    assert s["p50_ms"] == 3.0


def test_summarize_monotonic_percentiles():
    values = [float(i) for i in range(1, 101)]
    s = _summarize(values)
    assert s["p50_ms"] <= s["p90_ms"] <= s["p95_ms"] <= s["p99_ms"]


# ---- token estimate ----

def test_estimate_tokens_empty():
    assert _estimate_tokens("") == 0


def test_estimate_tokens_simple():
    assert _estimate_tokens("hello world") == 2


def test_estimate_tokens_collapses_whitespace():
    assert _estimate_tokens("  hello   world  ") == 2

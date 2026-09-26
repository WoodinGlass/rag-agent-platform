"""Tests for the reranker delta helper. Offline."""
from benchmarks.reranker_delta import _delta


def test_delta_positive():
    assert _delta(0.5, 0.8) == 0.3


def test_delta_negative():
    assert _delta(0.8, 0.5) == -0.3


def test_delta_zero():
    assert _delta(0.75, 0.75) == 0.0


def test_delta_rounded():
    # 0.1 + 0.2 in float is 0.30000000000000004; rounding cleans it up
    assert _delta(0.1, 0.1 + 0.2) == 0.2

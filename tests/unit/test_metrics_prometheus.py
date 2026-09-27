"""Prometheus exposition format: shape, cumulative buckets, sanitization.

Offline. Exercises `render_prometheus` directly plus the HTTP route via
TestClient so the media type and route wiring are covered too.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import CorrelationIdMiddleware
from app.api.routes import router
from app.core.metrics import render_prometheus

# ---------- unit: renderer ----------

def _empty() -> dict:
    return {"counters": {}, "histograms": {}}


def test_render_empty_produces_empty_string():
    assert render_prometheus(_empty()) == ""


def test_render_counter_gets_total_suffix():
    snap = {"counters": {"ingest.requests": 5}, "histograms": {}}
    out = render_prometheus(snap)
    assert "# TYPE ingest_requests_total counter" in out
    assert "ingest_requests_total 5" in out


def test_render_counter_sanitizes_dots_and_dashes():
    snap = {"counters": {"my-cool.metric.name": 3}, "histograms": {}}
    out = render_prometheus(snap)
    assert "my_cool_metric_name_total 3" in out


def test_render_counter_leading_digit_gets_underscore_prefix():
    snap = {"counters": {"1st.metric": 1}, "histograms": {}}
    out = render_prometheus(snap)
    assert "_1st_metric_total 1" in out


def test_render_histogram_buckets_are_cumulative():
    """Buckets must be non-decreasing; last is +Inf == count."""
    snap = {
        "counters": {},
        "histograms": {
            "query.latency_ms": {
                "count": 10,
                "sum_ms": 123.4,
                "avg_ms": 12.34,
                "buckets": {"5": 2, "25": 5, "100": 3, "10000": 0},
            }
        },
    }
    out = render_prometheus(snap)
    # cumulative: 2, then 2+5=7, then 7+3=10, then +0
    assert 'query_latency_ms_bucket{le="5"} 2' in out
    assert 'query_latency_ms_bucket{le="25"} 7' in out
    assert 'query_latency_ms_bucket{le="100"} 10' in out
    assert 'query_latency_ms_bucket{le="10000"} 10' in out
    assert 'query_latency_ms_bucket{le="+Inf"} 10' in out
    assert "query_latency_ms_sum 123.4" in out
    assert "query_latency_ms_count 10" in out


def test_render_histogram_type_line_present():
    snap = {
        "counters": {},
        "histograms": {
            "x": {"count": 1, "sum_ms": 1.0, "buckets": {"1": 1}},
        },
    }
    out = render_prometheus(snap)
    assert "# TYPE x histogram" in out


def test_render_sorts_metric_names_stable():
    snap = {
        "counters": {"b": 1, "a": 2, "c": 3},
        "histograms": {"z": {"count": 0, "sum_ms": 0.0, "buckets": {}}},
    }
    out = render_prometheus(snap)
    # a before b before c
    ia = out.index("a_total")
    ib = out.index("b_total")
    ic = out.index("c_total")
    assert ia < ib < ic


def test_render_histogram_non_numeric_bound_sorts_last():
    """Any non-numeric bucket key sorts after numeric; +Inf still appended."""
    snap = {
        "counters": {},
        "histograms": {
            "x": {
                "count": 4,
                "sum_ms": 4.0,
                "buckets": {"weird": 1, "5": 3},
            }
        },
    }
    out = render_prometheus(snap)
    # numeric first (5), then weird, then +Inf
    i5 = out.index('x_bucket{le="5"}')
    iw = out.index('x_bucket{le="weird"}')
    iinf = out.index('x_bucket{le="+Inf"}')
    assert i5 < iw < iinf
    # weird is cumulative after 5 -> 3 + 1 = 4
    assert 'x_bucket{le="weird"} 4' in out
    assert 'x_bucket{le="+Inf"} 4' in out


def test_render_ends_with_newline():
    snap = {"counters": {"x": 1}, "histograms": {}}
    assert render_prometheus(snap).endswith("\n")


# ---------- integration: HTTP route + media type ----------

def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    return app


def test_metrics_prom_route_returns_plaintext_media_type():
    c = TestClient(_app())
    r = c.get("/metrics/prom")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    # no metrics recorded yet -> empty body is valid
    assert r.text == ""


def test_metrics_prom_route_reflects_counters_after_activity():
    # Use the module-level Metrics singleton via a direct increment so we
    # do not need to hit real routes.
    from app.core.metrics import get_metrics

    get_metrics().inc("test.requests", 7)
    c = TestClient(_app())
    body = c.get("/metrics/prom").text
    assert "test_requests_total 7" in body


def test_json_metrics_route_still_works():
    c = TestClient(_app())
    r = c.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert "counters" in r.json()

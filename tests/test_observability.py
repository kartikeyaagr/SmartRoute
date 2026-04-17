"""
Tests for observability.py — Prometheus metrics and structured logging.
No real OTel collector or Prometheus server required.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from smartroute.observability import (
    CACHE_HITS_TOTAL,
    COST_TOTAL,
    ESCALATIONS_TOTAL,
    REQUEST_LATENCY,
    REQUEST_TOTAL,
    record_decision,
)
from smartroute.router import RoutingDecision


def _decision(**kwargs) -> RoutingDecision:
    defaults = dict(
        request_id="obs-001",
        prompt_hash="abc",
        difficulty_tier="MEDIUM",
        difficulty_score=0.7,
        final_model="groq/llama-3.1-8b-instant",
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=0.0002,
        latency_ms=300.0,
        classifier_backend="keyword",
        escalated=False,
        cache_hit=False,
        cache_similarity=None,
    )
    defaults.update(kwargs)
    return RoutingDecision(**defaults)


# ---------------------------------------------------------------------------
# record_decision — Prometheus counters update
# ---------------------------------------------------------------------------

def test_record_decision_increments_request_total():
    before = REQUEST_TOTAL.labels(
        tier="EASY", final_model="groq/llama-3.1-8b-instant",
        escalated="false", cache_hit="false",
    )._value.get()
    record_decision(_decision(difficulty_tier="EASY", escalated=False, cache_hit=False))
    after = REQUEST_TOTAL.labels(
        tier="EASY", final_model="groq/llama-3.1-8b-instant",
        escalated="false", cache_hit="false",
    )._value.get()
    assert after == before + 1


def test_record_decision_increments_escalation_counter():
    before = ESCALATIONS_TOTAL.labels(from_tier="MEDIUM")._value.get()
    record_decision(_decision(difficulty_tier="MEDIUM", escalated=True))
    after = ESCALATIONS_TOTAL.labels(from_tier="MEDIUM")._value.get()
    assert after == before + 1


def test_record_decision_no_escalation_counter_when_not_escalated():
    before = ESCALATIONS_TOTAL.labels(from_tier="EASY")._value.get()
    record_decision(_decision(difficulty_tier="EASY", escalated=False))
    after = ESCALATIONS_TOTAL.labels(from_tier="EASY")._value.get()
    assert after == before  # unchanged


def test_record_decision_increments_cache_hit_counter():
    before = CACHE_HITS_TOTAL.labels(backend="memory")._value.get()
    record_decision(_decision(cache_hit=True, cache_similarity=1.0))
    after = CACHE_HITS_TOTAL.labels(backend="memory")._value.get()
    assert after == before + 1


def test_record_decision_increments_cost_counter():
    before = COST_TOTAL.labels(
        tier="HARD", final_model="groq/llama-3.3-70b-versatile"
    )._value.get()
    record_decision(_decision(
        difficulty_tier="HARD",
        final_model="groq/llama-3.3-70b-versatile",
        estimated_cost_usd=0.005,
    ))
    after = COST_TOTAL.labels(
        tier="HARD", final_model="groq/llama-3.3-70b-versatile"
    )._value.get()
    assert after == pytest.approx(before + 0.005)


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------

def test_metrics_endpoint_returns_prometheus_text():
    from smartroute.server import app
    with TestClient(app) as client:
        r = client.get("/metrics")
    assert r.status_code == 200
    assert "smartroute_requests_total" in r.text
    assert "smartroute_request_latency_seconds" in r.text


# ---------------------------------------------------------------------------
# configure_logging — emits JSON
# ---------------------------------------------------------------------------

def test_configure_logging_sets_json_formatter():
    from smartroute.observability import configure_logging
    configure_logging("WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert any(h.formatter is not None for h in root.handlers)

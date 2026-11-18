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


# ---------------------------------------------------------------------------
# Tracing: generation spans + Langfuse exporter config
# ---------------------------------------------------------------------------

async def test_call_model_emits_generation_span(monkeypatch):
    """Every model call becomes a span Langfuse renders as a generation."""
    import json
    from types import SimpleNamespace

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from smartroute import providers
    from tests.conftest import CHEAP_ID

    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(providers, "_trace", SimpleNamespace(get_tracer=tp.get_tracer))

    fake = MagicMock()
    fake.usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3)
    fake.choices = [SimpleNamespace(message=SimpleNamespace(content="hi"))]
    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        r = await providers.call_model(CHEAP_ID, [{"role": "user", "content": "x"}])

    (span,) = exporter.get_finished_spans()
    a = span.attributes
    assert a["langfuse.observation.type"] == "generation"
    assert a["langfuse.observation.model.name"] == CHEAP_ID
    assert json.loads(a["langfuse.observation.usage_details"]) == {"input": 12, "output": 3}
    assert json.loads(a["langfuse.observation.cost_details"])["total"] == r.estimated_cost_usd
    assert "langfuse.observation.input" not in a  # content capture is opt-in


def test_langfuse_keys_build_otlp_http_exporter(monkeypatch):
    from smartroute import observability

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr("smartroute.config.settings.langfuse_public_key", "pk-lf-1")
    monkeypatch.setattr("smartroute.config.settings.langfuse_secret_key", "sk-lf-2")
    with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as exp:
        observability._exporter()
    kwargs = exp.call_args.kwargs
    assert kwargs["endpoint"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert kwargs["headers"]["Authorization"] == "Basic cGstbGYtMTpzay1sZi0y"  # b64("pk-lf-1:sk-lf-2")


def test_no_tracing_config_means_no_exporter(monkeypatch):
    from smartroute import observability

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr("smartroute.config.settings.langfuse_public_key", "")
    assert observability._exporter() is None

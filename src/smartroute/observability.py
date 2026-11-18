"""
Observability for SmartRoute: Prometheus metrics + structured JSON logging + OTel traces.

Prometheus metrics (exposed at GET /metrics):
  smartroute_requests_total          — counter, labels: tier, model, escalated, cache_hit
  smartroute_request_latency_seconds — histogram, labels: tier
  smartroute_cost_usd_total          — counter, labels: tier, model
  smartroute_escalations_total       — counter, labels: tier
  smartroute_cache_hits_total        — counter, labels: backend
  smartroute_guardrail_total         — counter, labels: reason, mode

OTel traces:
  Each route_async call is a span with routing decision attributes.
  Each model call is a child span marked as a Langfuse "generation" (providers.py).
  Ship to any OTLP/HTTP backend via OTEL_EXPORTER_OTLP_ENDPOINT, or to Langfuse by
  setting LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY. Neither set = spans dropped.
"""

import logging
import os

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_TOTAL = Counter(
    "smartroute_requests_total",
    "Total routed requests",
    ["tier", "final_model", "escalated", "cache_hit"],
)

REQUEST_LATENCY = Histogram(
    "smartroute_request_latency_seconds",
    "End-to-end request latency",
    ["tier"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

COST_TOTAL = Counter(
    "smartroute_cost_usd_total",
    "Cumulative estimated LLM cost in USD",
    ["tier", "final_model"],
)

ESCALATIONS_TOTAL = Counter(
    "smartroute_escalations_total",
    "Number of cascade escalations",
    ["from_tier"],
)

GUARDRAIL_TOTAL = Counter(
    "smartroute_guardrail_total",
    "Requests flagged by the input guardrail (mode=block were rejected, mode=log passed)",
    ["reason", "mode"],
)

CACHE_HITS_TOTAL = Counter(
    "smartroute_cache_hits_total",
    "Cache hits by backend type",
    ["backend"],
)


def record_decision(decision) -> None:
    """Update all Prometheus metrics from a completed RoutingDecision."""
    tier = decision.difficulty_tier
    model = decision.final_model or "unknown"
    escalated = str(decision.escalated).lower()
    cache_hit = str(decision.cache_hit).lower()

    REQUEST_TOTAL.labels(
        tier=tier, final_model=model, escalated=escalated, cache_hit=cache_hit
    ).inc()

    REQUEST_LATENCY.labels(tier=tier).observe(decision.latency_ms / 1000)

    COST_TOTAL.labels(tier=tier, final_model=model).inc(decision.estimated_cost_usd)

    if decision.escalated:
        ESCALATIONS_TOTAL.labels(from_tier=tier).inc()

    if decision.cache_hit:
        backend = "pgvector" if (decision.cache_similarity or 1.0) < 1.0 else "memory"
        CACHE_HITS_TOTAL.labels(backend=backend).inc()


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

def configure_logging(log_level: str = "INFO") -> None:
    """
    Replace the root handler with a JSON formatter.
    All subsequent log calls emit structured JSON with timestamp + trace_id.
    """
    from pythonjsonlogger.json import JsonFormatter  # type: ignore[import]

    fmt = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# OpenTelemetry traces
# ---------------------------------------------------------------------------

def _exporter():
    """
    OTLP/HTTP exporter, or None when nothing is configured.

    HTTP, not gRPC: Langfuse's OTLP endpoint only speaks HTTP, and every OTel
    collector accepts both (HTTP on :4318). Precedence:
      1. OTEL_EXPORTER_OTLP_ENDPOINT (+ OTEL_EXPORTER_OTLP_HEADERS) — any backend;
         the exporter reads both from the environment and appends /v1/traces.
      2. LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY — Langfuse, auth built for you.
    """
    import base64

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    from smartroute.config import settings

    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return OTLPSpanExporter()
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        auth = base64.b64encode(
            f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
        ).decode()
        return OTLPSpanExporter(
            endpoint=settings.langfuse_base_url.rstrip("/") + "/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
        )
    return None


def configure_tracing(service_name: str = "smartroute") -> None:
    """
    Set up OTel tracing. With no exporter configured the provider has no
    processors, so spans are created and dropped (near-zero overhead).
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = _exporter()
    if exporter is not None:
        # Batch: export happens off the request path, in a background thread.
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def get_tracer():
    from opentelemetry import trace
    return trace.get_tracer("smartroute")

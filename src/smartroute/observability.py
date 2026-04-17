"""
Observability for SmartRoute: Prometheus metrics + structured JSON logging + OTel traces.

Prometheus metrics (exposed at GET /metrics):
  smartroute_requests_total          — counter, labels: tier, model, escalated, cache_hit
  smartroute_request_latency_seconds — histogram, labels: tier
  smartroute_cost_usd_total          — counter, labels: tier, model
  smartroute_escalations_total       — counter, labels: tier
  smartroute_cache_hits_total        — counter, labels: backend

OTel traces:
  Each route_async call is a span with routing decision attributes.
  Configure OTEL_EXPORTER_OTLP_ENDPOINT to ship to a collector.
  Defaults to no-op when endpoint is not set.
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

def configure_tracing(service_name: str = "smartroute") -> None:
    """
    Set up OTel tracing. Exports to OTEL_EXPORTER_OTLP_ENDPOINT when set,
    otherwise uses a no-op tracer (zero overhead, no side effects).
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        # BatchSpanProcessor for production — efficient async export
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    # No exporter configured → no-op (don't add console noise in production)

    trace.set_tracer_provider(provider)


def get_tracer():
    from opentelemetry import trace
    return trace.get_tracer("smartroute")

"""
SmartRoute FastAPI server — OpenAI-compatible chat completions endpoint.

Endpoints:
    GET  /health                  — liveness check
    GET  /metrics                 — Prometheus metrics
    POST /v1/chat/completions     — OpenAI-compatible, stream supported

SmartRoute routing metadata is returned in the X-SmartRoute-Meta response header
as a JSON-encoded object, keeping the response body 100% OpenAI-compatible.
"""

import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict
from sse_starlette.sse import EventSourceResponse

from smartroute import db as _db
from smartroute import observability as _obs
from smartroute.config import settings
from smartroute.providers import ProviderError
from smartroute.router import Router

_bearer = HTTPBearer(auto_error=False)


def _require_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """Validate Bearer token if SMARTROUTE_SERVER_API_KEY is configured."""
    if not settings.server_api_key:
        return
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, settings.server_api_key
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "smartroute"
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class _Choice(BaseModel):
    index: int = 0
    message: _ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[_Choice]
    usage: _Usage


class _DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class _ChunkChoice(BaseModel):
    index: int = 0
    delta: _DeltaMessage
    finish_reason: str | None = None


class _ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[_ChunkChoice]


class _SmartRouteMeta(BaseModel):
    difficulty_tier: str
    difficulty_score: float
    cascade_path: list[str]
    verifier_score: int | None
    escalated: bool
    estimated_cost_usd: float
    latency_ms: float
    classifier_backend: str
    # Two-layer decision fields. Defaulted so a cascade-mode router still serialises.
    route_path: str = ""
    gate_reason: str = ""
    projected_cost_usd: float = 0.0
    subtask_count: int = 0


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _obs.configure_logging(settings.log_level)
    _obs.configure_tracing()
    if settings.database_url:
        await _db.init_pool(settings.database_url)
    app.state.router = Router()
    yield
    await _db.close_pool()


app = FastAPI(title="SmartRoute", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    msg = str(exc)
    if "AuthenticationError" in msg:
        status = 401
    elif "RateLimitError" in msg:
        status = 429
    elif "ServiceUnavailableError" in msg:
        status = 503
    elif "timed out" in msg:
        status = 504
    else:
        status = 502
    return JSONResponse(
        status_code=status,
        content={"error": {"message": msg, "type": "provider_error", "code": status}},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions", dependencies=[Security(_require_auth)])
async def chat_completions(body: ChatCompletionRequest) -> JSONResponse:
    tracer = _obs.get_tracer()
    with tracer.start_as_current_span("smartroute.route") as span:
        router: Router = app.state.router
        content, decision = await router.route_async(body.messages)

        span.set_attribute("smartroute.tier", decision.difficulty_tier)
        span.set_attribute("smartroute.model", decision.final_model)
        span.set_attribute("smartroute.escalated", decision.escalated)
        span.set_attribute("smartroute.latency_ms", decision.latency_ms)
        span.set_attribute("smartroute.cost_usd", decision.estimated_cost_usd)
        span.set_attribute("smartroute.cache_hit", decision.cache_hit)
        span.set_attribute("smartroute.route_path", decision.route_path)
        span.set_attribute("smartroute.subtask_count", decision.subtask_count)

    await _db.insert_decision(decision)
    _obs.record_decision(decision)

    meta = _SmartRouteMeta(
        difficulty_tier=decision.difficulty_tier,
        difficulty_score=decision.difficulty_score,
        cascade_path=decision.cascade_path,
        verifier_score=decision.verifier_score,
        escalated=decision.escalated,
        estimated_cost_usd=decision.estimated_cost_usd,
        latency_ms=decision.latency_ms,
        classifier_backend=decision.classifier_backend,
        route_path=decision.route_path,
        gate_reason=decision.gate_reason,
        projected_cost_usd=decision.projected_cost_usd,
        subtask_count=decision.subtask_count,
    )

    if body.stream:
        return _stream(content, decision, meta)

    response = ChatCompletionResponse(
        id=f"chatcmpl-{decision.request_id}",
        created=int(time.time()),
        model=decision.final_model,
        choices=[_Choice(message=_ChoiceMessage(content=content))],
        usage=_Usage(
            prompt_tokens=decision.input_tokens,
            completion_tokens=decision.output_tokens,
            total_tokens=decision.input_tokens + decision.output_tokens,
        ),
    )
    return JSONResponse(
        content=response.model_dump(),
        headers={"X-SmartRoute-Meta": meta.model_dump_json()},
    )


def _stream(content: str, decision: Any, meta: _SmartRouteMeta) -> EventSourceResponse:
    """Fake-streaming: emit the complete response as SSE chunks."""
    chunk_id = f"chatcmpl-{decision.request_id}"
    created = int(time.time())
    model = decision.final_model

    async def generate():
        yield _ChatCompletionChunk(
            id=chunk_id, created=created, model=model,
            choices=[_ChunkChoice(delta=_DeltaMessage(role="assistant"))],
        ).model_dump_json()

        yield _ChatCompletionChunk(
            id=chunk_id, created=created, model=model,
            choices=[_ChunkChoice(delta=_DeltaMessage(content=content))],
        ).model_dump_json()

        yield _ChatCompletionChunk(
            id=chunk_id, created=created, model=model,
            choices=[_ChunkChoice(delta=_DeltaMessage(), finish_reason="stop")],
        ).model_dump_json()

        yield "[DONE]"

    return EventSourceResponse(
        generate(),
        headers={"X-SmartRoute-Meta": meta.model_dump_json()},
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run(
        "smartroute.server:app",
        host=settings.server_host,
        port=settings.server_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

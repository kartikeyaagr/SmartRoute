"""
Server endpoint tests. Uses TestClient (sync ASGI) + monkeypatching to avoid real API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from smartroute.router import RoutingDecision


def _make_decision(**kwargs) -> RoutingDecision:
    defaults = dict(
        request_id="abc123",
        prompt_hash="deadbeef",
        difficulty_tier="EASY",
        difficulty_score=0.8,
        final_model="groq/llama-3.1-8b-instant",
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=0.0001,
        latency_ms=120.0,
        classifier_backend="keyword",
        cascade_path=["groq/llama-3.1-8b-instant"],
        escalated=False,
        verifier_score=None,
    )
    defaults.update(kwargs)
    return RoutingDecision(**defaults)


@pytest.fixture()
def client():
    # Import here so lifespan runs inside the test context
    from smartroute.server import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def authed_client(monkeypatch):
    monkeypatch.setattr("smartroute.server.settings", MagicMock(
        server_api_key="test-key",
        server_host="0.0.0.0",
        server_port=8000,
    ))
    from smartroute.server import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /v1/chat/completions — no auth configured
# ---------------------------------------------------------------------------

def test_chat_completions_success(client):
    decision = _make_decision()
    with patch.object(
        client.app.state.router, "route_async", new=AsyncMock(return_value=("Hello!", decision))
    ):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert body["usage"]["total_tokens"] == 15
    assert "X-SmartRoute-Meta" in r.headers


def test_chat_completions_provider_error(client):
    from smartroute.providers import ProviderError
    with patch.object(
        client.app.state.router, "route_async",
        new=AsyncMock(side_effect=ProviderError("AuthenticationError: bad key")),
    ):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

def test_auth_no_key_configured_allows_any_request(client):
    """When SERVER_API_KEY is empty, no Bearer token is required."""
    decision = _make_decision()
    with patch.object(
        client.app.state.router, "route_async", new=AsyncMock(return_value=("ok", decision))
    ):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
    assert r.status_code == 200


def test_auth_rejects_missing_token(monkeypatch):
    """When SERVER_API_KEY is set, missing Bearer → 401."""
    from smartroute import server as srv
    monkeypatch.setattr(srv.settings, "server_api_key", "secret")
    from smartroute.server import app
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
    assert r.status_code == 401


def test_auth_rejects_wrong_token(monkeypatch):
    from smartroute import server as srv
    monkeypatch.setattr(srv.settings, "server_api_key", "correct-key")
    from smartroute.server import app
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-key"},
            json={"messages": [{"role": "user", "content": "test"}]},
        )
    assert r.status_code == 401


def test_auth_accepts_correct_token(monkeypatch):
    from smartroute import server as srv
    monkeypatch.setattr(srv.settings, "server_api_key", "correct-key")
    decision = _make_decision()
    from smartroute.server import app
    with TestClient(app) as c:
        with patch.object(
            c.app.state.router, "route_async", new=AsyncMock(return_value=("ok", decision))
        ):
            r = c.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer correct-key"},
                json={"messages": [{"role": "user", "content": "test"}]},
            )
    assert r.status_code == 200


def test_meta_header_carries_decision_fields():
    """The X-SmartRoute-Meta contract must expose why a request went where it did."""
    import json

    from smartroute.server import app

    decision = _make_decision(
        difficulty_tier="CHEAP",
        route_path="cheap",
        gate_reason="lookup p=0.99 >= 0.85",
        projected_cost_usd=0.000066,
        p_lookup=0.99,
        subtask_count=0,
    )
    with TestClient(app) as client:
        with patch.object(
            client.app.state.router, "route_async",
            new=AsyncMock(return_value=("Paris", decision)),
        ):
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    assert response.status_code == 200
    meta = json.loads(response.headers["X-SmartRoute-Meta"])
    assert meta["route_path"] == "cheap"
    assert "lookup" in meta["gate_reason"]
    assert meta["projected_cost_usd"] > 0
    assert meta["subtask_count"] == 0

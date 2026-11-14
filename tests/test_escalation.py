"""Jev-verified escalation of middle answers. No network."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from smartroute.decision import RouteDecision, TriageSignals
from smartroute.escalation import JevVerifier, derived_threshold
from smartroute.providers import ModelResponse, ProviderError
from smartroute.router import Router
from smartroute.taskroute import DECISIONS_PATH, OPENROUTER_BASE, TaskRouteError


def _client(p_correct=None, status=200, calls=None):
    calls = calls if calls is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path == DECISIONS_PATH
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json={
            "answers": {"answers_correctly": {"type": "noul", "noul": p_correct}},
            "usage": {"cost": 0.00003},
        })
    return httpx.AsyncClient(base_url=OPENROUTER_BASE, transport=httpx.MockTransport(handler))


def _verifier(p_correct=0.9, threshold=0.78, status=200, calls=None):
    return JevVerifier("k", threshold=threshold, client=_client(p_correct, status, calls))


class TestThreshold:
    def test_derived_from_lambda_and_frontier_cost(self):
        assert derived_threshold(0.0002, 0.001) == pytest.approx(0.8)

    def test_frontier_dearer_than_a_wrong_answer_never_escalates(self):
        assert derived_threshold(0.01, 0.001) == 0.0

    def test_zero_lambda_never_escalates(self):
        assert derived_threshold(0.0002, 0.0) == 0.0


class TestVerifier:
    async def test_confident_answer_is_kept(self):
        r = await _verifier(0.95).check("2+2?", "4")
        assert r.escalate is False and r.p_correct == 0.95
        assert r.cost_usd == pytest.approx(0.00003)

    async def test_doubtful_answer_escalates(self):
        r = await _verifier(0.30).check("2+2?", "5")
        assert r.escalate is True

    async def test_request_shape(self):
        calls = []
        await _verifier(calls=calls).check("q", "a")
        body = json.loads(calls[0].content)
        assert body["state"] == {"user_message": "q", "assistant_response": "a"}
        assert body["questions"]["answers_correctly"]["type"] == "noul"

    async def test_empty_answer_escalates_without_a_call(self):
        calls = []
        r = await _verifier(calls=calls).check("q", "   ")
        assert r.escalate is True and calls == []

    async def test_outage_keeps_middle_answer(self):
        r = await _verifier(status=503).check("q", "a")
        assert r.escalate is False and r.p_correct is None

    def test_missing_key_rejected(self):
        with pytest.raises(TaskRouteError):
            JevVerifier("", threshold=0.5)


class _MiddleLayer:
    """Decision layer stub that always routes to the middle tier."""

    def decide(self, prompt):
        return RouteDecision(
            path="middle", reason="atomic", signals=TriageSignals(0.1, 0.1, "stub"),
        )


def _resp(model):
    return ModelResponse(model=model, content=f"answer from {model}", input_tokens=10,
                         output_tokens=20, estimated_cost_usd=0.001, latency_ms=1.0)


class TestRouterEscalation:
    async def _route(self, verifier, side_effect=None):
        router = Router(dispatch="decision", decision_layer=_MiddleLayer(), escalator=verifier)
        fake = AsyncMock(side_effect=side_effect or (lambda m, *a, **k: _resp(m)))
        with patch("smartroute.router.call_model", new=fake):
            return await router.route_async([{"role": "user", "content": "q"}]), fake

    async def test_kept_answer_stays_on_middle(self, test_catalog):
        (content, d), fake = await self._route(_verifier(0.95))
        assert d.final_model == test_catalog.by_role("middle").id
        assert d.escalated is False
        assert d.escalation_p_correct == 0.95
        assert d.estimated_cost_usd == pytest.approx(0.001 + 0.00003)
        assert fake.await_count == 1

    async def test_failed_check_escalates_to_frontier(self, test_catalog):
        (content, d), fake = await self._route(_verifier(0.2))
        frontier = test_catalog.by_role("frontier").id
        assert d.final_model == frontier
        assert content == f"answer from {frontier}"
        assert d.escalated is True
        assert d.cheap_response == f"answer from {test_catalog.by_role('middle').id}"
        assert d.estimated_cost_usd == pytest.approx(0.002 + 0.00003)
        assert "jev p_correct=0.20" in d.gate_reason

    async def test_frontier_failure_returns_middle_answer(self, test_catalog):
        frontier = test_catalog.by_role("frontier").id

        def side(m, *a, **k):
            if m == frontier:
                raise ProviderError("down")
            return _resp(m)

        (content, d), _ = await self._route(_verifier(0.2), side_effect=side)
        assert content == f"answer from {test_catalog.by_role('middle').id}"
        assert "frontier failed" in d.gate_reason

    async def test_disabled_by_default(self):
        router = Router(dispatch="decision", decision_layer=_MiddleLayer())
        with patch("smartroute.router.call_model", new=AsyncMock(side_effect=lambda m, *a, **k: _resp(m))):
            _, d = await router.route_async([{"role": "user", "content": "q"}])
        assert d.escalation_p_correct is None

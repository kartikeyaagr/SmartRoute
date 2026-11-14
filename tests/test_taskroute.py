"""Task-ranked routing: decision model -> OpenRouter rankings -> cost-tier cap. No network."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from smartroute.providers import ModelResponse, ProviderError
from smartroute.router import Router
from smartroute.taskroute import (
    COST_TIERS,
    DECISIONS_PATH,
    OPENROUTER_BASE,
    DecisionsClient,
    OpenRouterData,
    TaskRanker,
    TaskRouteConfig,
    TaskRouteError,
)

BENCHMARKS = {
    "data": [
        # coding / agentic / intelligence; prices come from /models
        {"model_permaslug": "anthropic/claude-opus-5.5-20261001", "coding_index": 80, "agentic_index": 75, "intelligence_index": 78},  # frontier
        {"model_permaslug": "openai/gpt-6.1-sol", "coding_index": 74, "agentic_index": 79, "intelligence_index": 80},  # premium
        {"model_permaslug": "deepseek/deepseek-v4.1-flash", "coding_index": 60, "agentic_index": 50, "intelligence_index": 62},  # budget
        {"model_permaslug": "openai/gpt-6-luna", "coding_index": 55, "intelligence_index": 64},  # budget, no agentic score
        {"model_permaslug": "nvidia/nemotron:free", "coding_index": 70, "intelligence_index": 50},
        {"model_permaslug": "mystery/unlisted", "coding_index": 99, "intelligence_index": 99},
        {"model_permaslug": "nulls/only", "coding_index": None},
    ],
    "meta": {"as_of": "2026-11-13T00:00:00Z", "source": "artificial-analysis"},
}

MODELS = {
    "data": [
        {"id": "anthropic/claude-opus-5.5", "canonical_slug": "anthropic/claude-opus-5.5-20261001",
         "pricing": {"prompt": "0.000005", "completion": "0.000025"}},
        {"id": "openai/gpt-6.1-sol", "pricing": {"prompt": "0.000002", "completion": "0.000008"}},
        {"id": "deepseek/deepseek-v4.1-flash", "pricing": {"prompt": "0.00000005", "completion": "0.0000002"}},
        {"id": "openai/gpt-6-luna", "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}},
        {"id": "nvidia/nemotron:free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
}


def _answers(task="coding", task_conf=0.9, effort=3, effort_conf=0.9):
    probs = {str(i): (effort_conf if i == effort else (1 - effort_conf) / 3) for i in range(4)}
    return {
        "task": {"type": "choice", "choice": task, "confidence": task_conf, "probabilities": {}},
        "effort": {"type": "score", "score": float(effort), "confidence": effort_conf, "probabilities": probs},
    }


def _transport(answers, calls, fail_data=False):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/benchmarks":
            assert request.url.params["source"] == "artificial-analysis"
            return httpx.Response(503) if fail_data else httpx.Response(200, json=BENCHMARKS)
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json=MODELS)
        if request.url.path == DECISIONS_PATH:
            return httpx.Response(200, json={"answers": answers, "usage": {"cost": 0.00002}})
        return httpx.Response(404)
    return handler


def _ranker(answers=None, calls=None, config=None, fail_data=False):
    calls = calls if calls is not None else []
    cfg = config or TaskRouteConfig()
    client = httpx.AsyncClient(
        base_url=OPENROUTER_BASE, transport=httpx.MockTransport(_transport(answers or _answers(), calls, fail_data))
    )
    return TaskRanker(
        api_key="test",
        config=cfg,
        data=OpenRouterData("test", cfg, client=client),
        decisions=DecisionsClient("test", cfg, client=client),
    )


class TestSelection:
    async def test_hard_coding_prompt_gets_best_coding_score(self):
        d = await _ranker(_answers(effort=3)).select("design a distributed lock service")
        assert d.task_tag == "coding"
        assert d.max_tier_name == "frontier"
        # permaslug resolved to the routable id via canonical_slug
        assert d.candidates[0].model_id == "anthropic/claude-opus-5.5"
        assert d.candidates[0].quality == 80
        assert d.decision_cost_usd == pytest.approx(0.00002)

    async def test_ranking_follows_the_task_index(self):
        d = await _ranker(_answers(task="agentic", effort=3)).select("book my travel")
        assert [c.model_id for c in d.candidates] == [
            "openai/gpt-6.1-sol", "anthropic/claude-opus-5.5", "deepseek/deepseek-v4.1-flash",
        ]  # luna has no agentic score, so it is not a candidate

    async def test_trivial_prompt_is_capped_to_budget(self):
        d = await _ranker(_answers(effort=0)).select("write hello world in python")
        assert d.max_tier_name == "budget"
        assert [c.model_id for c in d.candidates] == [
            "deepseek/deepseek-v4.1-flash", "openai/gpt-6-luna",
        ]
        assert "anthropic/claude-opus-5.5" in d.rejected

    async def test_moderate_effort_allows_premium_but_not_frontier(self):
        d = await _ranker(_answers(effort=2)).select("refactor this module")
        assert d.candidates[0].model_id == "openai/gpt-6.1-sol"
        assert all(c.tier <= COST_TIERS.index("premium") for c in d.candidates)

    async def test_low_effort_confidence_bumps_one_tier(self):
        d = await _ranker(_answers(effort=0, effort_conf=0.4)).select("x")
        assert d.max_tier_name == "standard"

    async def test_free_and_unroutable_models_are_excluded(self):
        d = await _ranker(_answers(effort=3)).select("x")
        assert d.rejected["nvidia/nemotron:free"] == "free variant"
        assert d.rejected["mystery/unlisted"] == "not routable on OpenRouter"

    async def test_low_task_confidence_falls_back_to_intelligence_index(self):
        d = await _ranker(_answers(task="coding", task_conf=0.1, effort=0)).select("x")
        assert d.task_tag == "general"
        assert "low task confidence" in d.task_label
        assert d.candidates[0].model_id == "openai/gpt-6-luna"  # 64 > deepseek 62

    async def test_all_candidates_over_cap_takes_cheapest(self):
        cfg = TaskRouteConfig(tier_caps_per_mtok=(0.0001, 0.0002, 0.0003))
        d = await _ranker(_answers(effort=0), config=cfg).select("x")
        assert [c.model_id for c in d.candidates] == ["deepseek/deepseek-v4.1-flash"]

    async def test_one_decisions_call_with_both_questions(self):
        calls = []
        await _ranker(calls=calls).select("explain monads")
        dec = [c for c in calls if c.url.path == DECISIONS_PATH]
        assert len(dec) == 1
        body = json.loads(dec[0].content)
        assert body["model"] == "~typesafe/jev-latest"
        assert body["questions"]["task"]["type"] == "choice"
        assert set(body["questions"]["task"]["criteria"]) == {"coding", "agentic", "general"}
        assert body["questions"]["effort"]["type"] == "score"
        assert len(body["questions"]["effort"]["criteria"]) == 4
        assert body["state"]["user_message"] == "explain monads"

    async def test_benchmarks_are_cached(self):
        calls = []
        r = _ranker(calls=calls)
        await r.select("a")
        await r.select("b")
        assert sum(c.url.path == "/api/v1/benchmarks" for c in calls) == 1

    async def test_benchmark_outage_raises_taskroute_error(self):
        with pytest.raises(TaskRouteError):
            await _ranker(fail_data=True).select("x")

    def test_missing_key_is_rejected(self):
        with pytest.raises(TaskRouteError):
            TaskRanker(api_key="")


def _resp(model, content="ok"):
    return ModelResponse(model=model, content=content, input_tokens=10, output_tokens=20,
                         estimated_cost_usd=0.001, latency_ms=5.0)


class TestRouterTaskRank:
    async def test_routes_to_selected_model_via_openrouter(self):
        router = Router(dispatch="taskrank", task_ranker=_ranker(_answers(effort=3)))
        with patch("smartroute.router.call_model", new=AsyncMock(side_effect=lambda m, *a, **k: _resp(m))) as cm:
            content, d = await router.route_async([{"role": "user", "content": "hard"}])
        assert content == "ok"
        assert cm.call_args.args[0] == "openrouter/anthropic/claude-opus-5.5"
        assert d.final_model == "openrouter/anthropic/claude-opus-5.5"
        assert d.task_tag == "coding"
        assert d.effort_level == 3
        assert d.cost_tier_cap == "frontier"
        # decision call is billed into the total, not hidden
        assert d.estimated_cost_usd == pytest.approx(0.001 + 0.00002)

    async def test_fails_over_to_next_candidate(self):
        async def fake(model, *a, **k):
            if model.endswith("claude-opus-5.5"):
                raise ProviderError("down")
            return _resp(model)
        router = Router(dispatch="taskrank", task_ranker=_ranker(_answers(effort=3)))
        with patch("smartroute.router.call_model", new=AsyncMock(side_effect=fake)):
            _, d = await router.route_async([{"role": "user", "content": "hard"}])
        assert d.final_model == "openrouter/openai/gpt-6.1-sol"
        assert d.escalated is True

    async def test_selection_outage_falls_back_to_default_tier(self, test_catalog):
        router = Router(dispatch="taskrank", task_ranker=_ranker(fail_data=True))
        with patch("smartroute.router.call_model", new=AsyncMock(side_effect=lambda m, *a, **k: _resp(m))):
            _, d = await router.route_async([{"role": "user", "content": "x"}])
        assert d.final_model == test_catalog.default.id
        assert "taskrank unavailable" in d.gate_reason

    def test_invalid_dispatch_still_rejected(self):
        with pytest.raises(ValueError):
            Router(dispatch="nope")

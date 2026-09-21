"""
Tests for layer 2.

The guards are the point. Decomposition costs ~2.4x a direct middle answer before a
single sub-task runs, so every path that stops it from running is load-bearing:
a malformed plan, the planner's own atomic veto, a single-sub-task plan, and the
projected-cost bail-out.
"""

import json
from unittest.mock import patch

import pytest

from smartroute.decomposer import (
    MAX_SUBTASKS,
    Decomposer,
    Plan,
    SubTask,
    parse_plan,
)
from smartroute.providers import ModelResponse, ProviderError


def response(content: str, tokens: int = 10) -> ModelResponse:
    return ModelResponse(
        model="m", content=content, input_tokens=tokens, output_tokens=tokens,
        estimated_cost_usd=0.0, latency_ms=1.0,
    )


def plan_json(*subtasks: tuple[str, str], atomic: bool = False) -> str:
    return json.dumps(
        {"atomic": atomic, "subtasks": [{"task": t, "tier": tier} for t, tier in subtasks]}
    )


class TestParsePlan:
    def test_atomic(self):
        assert parse_plan(plan_json(atomic=True)).atomic

    def test_subtasks_are_parsed_with_tiers(self):
        plan = parse_plan(plan_json(("find the director", "cheap"), ("compare them", "middle")))
        assert [s.tier for s in plan.subtasks] == ["cheap", "middle"]
        assert plan.should_execute

    def test_json_embedded_in_prose_is_recovered(self):
        """Chat models wrap JSON in commentary; that should not cost a request."""
        assert parse_plan(f"Sure!\n{plan_json(atomic=True)}\nHope that helps").atomic

    @pytest.mark.parametrize("text", ["", "I cannot do that", "{not json", None])
    def test_unparseable_is_malformed_not_an_exception(self, text):
        assert parse_plan(text).malformed

    def test_invalid_tier_is_dropped(self):
        assert parse_plan(plan_json(("a", "ultra"))).malformed

    def test_empty_task_text_is_dropped(self):
        assert parse_plan(plan_json(("   ", "cheap"))).malformed

    def test_fan_out_is_capped(self):
        many = tuple((f"task {i}", "cheap") for i in range(20))
        assert len(parse_plan(plan_json(*many)).subtasks) == MAX_SUBTASKS

    def test_single_subtask_is_not_worth_executing(self):
        """One sub-task plus synthesis is strictly worse than answering directly."""
        assert not parse_plan(plan_json(("only thing", "cheap"))).should_execute


class TestGuards:
    async def test_malformed_plan_falls_back_to_direct_middle(self):
        with patch("smartroute.decomposer.call_model", side_effect=[
            response("gibberish, no json here"), response("direct answer"),
        ]):
            result = await Decomposer().run("q", [{"role": "user", "content": "q"}])
        assert result.route == "direct-middle"
        assert result.content == "direct answer"
        assert "malformed" in result.reason

    async def test_planner_atomic_veto_answers_directly(self):
        """The gate is probabilistic; the planner gets a second, better-informed veto."""
        with patch("smartroute.decomposer.call_model", side_effect=[
            response(plan_json(atomic=True)), response("direct answer"),
        ]):
            result = await Decomposer().run("q", [{"role": "user", "content": "q"}])
        assert result.route == "direct-middle"
        assert "atomic" in result.reason

    async def test_expensive_plan_bails_out_to_one_frontier_call(self):
        """
        A plan of frontier sub-tasks costs more than the frontier call it replaces,
        so the premise for splitting is gone and we make that call instead.
        """
        expensive = plan_json(*[(f"hard thing {i}", "frontier") for i in range(5)])
        with patch("smartroute.decomposer.call_model", side_effect=[
            response(expensive), response("frontier answer"),
        ]):
            result = await Decomposer().run("q" * 400, [{"role": "user", "content": "q"}])
        assert result.route == "bailout-frontier"
        assert "frontier" in result.reason

    async def test_cheap_plan_is_executed(self):
        cheap = plan_json(("find a", "cheap"), ("find b", "cheap"))
        with patch("smartroute.decomposer.call_model", side_effect=[
            response(cheap), response("A"), response("B"), response("final"),
        ]):
            result = await Decomposer().run("q" * 400, [{"role": "user", "content": "q"}])
        assert result.route == "decomposed"
        assert result.content == "final"
        assert len(result.subtasks) == 2


class TestExecution:
    async def test_subtasks_route_to_their_declared_tiers(self):
        mixed = plan_json(("lookup", "cheap"), ("reason", "middle"))
        seen = []

        async def record(model, messages, **kw):
            seen.append(model)
            return response(mixed if len(seen) == 1 else "x")

        with patch("smartroute.decomposer.call_model", side_effect=record):
            await Decomposer().run("q" * 400, [{"role": "user", "content": "q"}])

        from smartroute.catalog import get_catalog

        catalog = get_catalog()
        # plan(middle), subtask(cheap), subtask(middle), synthesis(middle)
        assert seen[1] == catalog.by_role("cheap").id
        assert seen[2] == catalog.by_role("middle").id
        assert seen[3] == catalog.by_role("middle").id

    async def test_one_failed_subtask_does_not_sink_the_request(self):
        cheap = plan_json(("a", "cheap"), ("b", "cheap"))
        calls = {"n": 0}

        async def flaky(model, messages, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return response(cheap)
            if calls["n"] == 2:
                raise ProviderError("sub-task blew up")
            return response("synthesised anyway")

        with patch("smartroute.decomposer.call_model", side_effect=flaky):
            result = await Decomposer().run("q" * 400, [{"role": "user", "content": "q"}])

        assert result.route == "decomposed"
        assert result.content == "synthesised anyway"

    async def test_cost_and_tokens_accumulate_across_every_call(self):
        cheap = plan_json(("a", "cheap"), ("b", "cheap"))
        with patch("smartroute.decomposer.call_model", side_effect=[
            response(cheap, tokens=5), response("A", tokens=5),
            response("B", tokens=5), response("final", tokens=5),
        ]):
            result = await Decomposer().run("q" * 400, [{"role": "user", "content": "q"}])
        assert result.input_tokens == 20  # 4 calls x 5
        assert len(result.models_used) == 4


class TestPricing:
    def test_frontier_heavy_plans_price_above_cheap_ones(self):
        d = Decomposer()
        prompt = "q" * 400
        cheap_plan = Plan(atomic=False, subtasks=[SubTask("a", "cheap"), SubTask("b", "cheap")])
        costly_plan = Plan(atomic=False, subtasks=[SubTask("a", "frontier"), SubTask("b", "frontier")])
        assert d.price_plan(prompt, costly_plan) > d.price_plan(prompt, cheap_plan)

    def test_price_grows_with_fan_out(self):
        d = Decomposer()
        prompt = "q" * 400
        two = Plan(atomic=False, subtasks=[SubTask(f"t{i}", "cheap") for i in range(2)])
        five = Plan(atomic=False, subtasks=[SubTask(f"t{i}", "cheap") for i in range(5)])
        assert d.price_plan(prompt, five) > d.price_plan(prompt, two)

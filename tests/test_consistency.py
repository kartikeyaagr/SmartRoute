"""Self-consistency checker. No network."""

from unittest.mock import AsyncMock, patch

import pytest

from smartroute.decision import RouteDecision, TriageSignals
from smartroute.escalation import SelfConsistencyVerifier, answers_agree, extract_answer
from smartroute.providers import ModelResponse, ProviderError
from smartroute.router import Router, build_checker


@pytest.mark.parametrize("text,expected", [
    ("A", ("letter", "A")),
    ("(B) because it is", ("letter", "B")),
    ("The answer is C.", ("letter", "C")),
    ("[D]", ("letter", "D")),
    ("A man walks his dog", ("text", "a man walks his dog")),
    ("so 18 eggs\n#### 18", ("number", "18")),
    ("It costs 3,000 dollars", ("number", "3000")),
    ("```python\ndef f(x):\n    return x + 1\n```", ("text", "python def f x return x 1")),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_agreement_rules():
    assert answers_agree("#### 18", "The total is 18")
    assert not answers_agree("A", "B")
    assert not answers_agree("A", "42")                   # letter vs number
    assert answers_agree("Paris is the capital of France", "The capital of France is Paris")
    assert not answers_agree("Paris", "Berlin is the capital")


def _sampler(answers, cost=0.0005):
    async def sample(messages, n):
        return [(a, cost) for a in answers[:n]]
    return sample


class TestVerifier:
    async def test_full_agreement_keeps_answer(self):
        r = await SelfConsistencyVerifier(_sampler(["A", "A"]), threshold=0.75).check("q", "A")
        assert r.p_correct == 1.0 and r.escalate is False
        assert r.cost_usd == pytest.approx(0.001)

    async def test_split_vote_escalates(self):
        r = await SelfConsistencyVerifier(_sampler(["A", "C"]), threshold=0.75).check("q", "A")
        assert r.p_correct == 0.5 and r.escalate is True

    async def test_sampler_failure_keeps_answer(self):
        async def boom(messages, n):
            raise ProviderError("down")
        r = await SelfConsistencyVerifier(boom, threshold=0.75).check("q", "A")
        assert r.p_correct is None and r.escalate is False

    async def test_messages_are_forwarded(self):
        seen = {}

        async def sample(messages, n):
            seen["m"] = messages
            return [("A", 0.0)] * n
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        await SelfConsistencyVerifier(sample, threshold=0.5).check("q", "A", msgs)
        assert seen["m"] == msgs

    def test_zero_samples_rejected(self):
        with pytest.raises(ValueError):
            SelfConsistencyVerifier(_sampler([]), threshold=0.5, samples=0)


class TestBuildChecker:
    def test_comma_list_builds_ensemble(self, monkeypatch):
        from smartroute.escalation import EnsembleVerifier
        from smartroute.router import settings
        monkeypatch.setattr(settings, "openrouter_api_key", "k")
        v = build_checker("jev,consistency", 0.7, sampler=_sampler(["A"]))
        assert isinstance(v, EnsembleVerifier)

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError):
            build_checker("jev,vibes", 0.7)


class _MiddleLayer:
    def decide(self, prompt):
        return RouteDecision(path="middle", reason="atomic", signals=TriageSignals(0.1, 0.1, "stub"))


class TestRouter:
    async def _route(self, test_catalog, sample_answers):
        middle = test_catalog.by_role("middle").id
        frontier = test_catalog.by_role("frontier").id
        samples = iter(sample_answers)
        calls = []

        async def fake(model, messages, **kw):
            calls.append((model, kw.get("temperature")))
            content = "A" if "temperature" not in kw else next(samples)
            if model == frontier:
                content = "FRONTIER"
            return ModelResponse(model=model, content=content, input_tokens=1, output_tokens=1,
                                 estimated_cost_usd=0.001, latency_ms=1.0)

        router = Router(dispatch="decision", decision_layer=_MiddleLayer(), jev_escalation=True)
        router._escalator = SelfConsistencyVerifier(router._sample_tier, threshold=0.75, samples=2)
        with patch("smartroute.router.call_model", new=AsyncMock(side_effect=fake)):
            out = await router.route_async([{"role": "user", "content": "q"}])
        return out, calls, middle, frontier

    async def test_consistent_answer_stays_on_middle(self, test_catalog):
        (content, d), calls, middle, _ = await self._route(test_catalog, ["A", "A"])
        assert content == "A" and d.final_model == middle and not d.escalated
        assert d.escalation_p_correct == 1.0
        assert [m for m, t in calls if t is not None] == [middle, middle]  # 2 resamples at temp
        assert d.estimated_cost_usd == pytest.approx(0.003)               # answer + 2 samples

    async def test_inconsistent_answer_escalates(self, test_catalog):
        (content, d), _, _, frontier = await self._route(test_catalog, ["B", "C"])
        assert content == "FRONTIER" and d.final_model == frontier and d.escalated
        assert d.escalation_p_correct == 0.0

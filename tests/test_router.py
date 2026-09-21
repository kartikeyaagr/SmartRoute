"""
Tests for verifier.py and router.py.

All tests use mocked providers — no real API calls.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartroute.classifier import DifficultyTier
from smartroute.providers import ModelResponse, ProviderError
from smartroute.router import Router, RoutingDecision
from smartroute.verifier import CascadeVerifier, _pick_verifier, _parse_score

from tests.conftest import CHEAP_ID, FRONTIER_ID, JUDGE_ID, MIDDLE_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages(text: str = "What is 2+2?") -> list[dict]:
    return [{"role": "user", "content": text}]


def _mock_response(content: str = "4", model: str = CHEAP_ID) -> ModelResponse:
    return ModelResponse(
        model=model,
        content=content,
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=0.0001,
        latency_ms=120.0,
    )


def _make_classifier(tier: DifficultyTier, score: float = 0.80):
    clf = MagicMock()
    clf.classify.return_value = (tier, score)
    clf.backend.return_value = "keyword"
    return clf


def _make_verifier(score: int) -> CascadeVerifier:
    v = MagicMock(spec=CascadeVerifier)
    v.score_async = AsyncMock(return_value=score)
    return v


# ---------------------------------------------------------------------------
# Verifier unit tests
# ---------------------------------------------------------------------------

def test_pick_verifier_uses_catalog_judge():
    # llama-8b cheap → qwen3-32b verifier (Alibaba arch, different from Meta Llama)
    assert _pick_verifier(CHEAP_ID).id == JUDGE_ID


def test_pick_verifier_unknown_uses_default():
    assert _pick_verifier("unknown/model").id == JUDGE_ID


def test_parse_score_valid():
    assert _parse_score("4") == 4
    assert _parse_score("The score is 3.") == 3
    assert _parse_score("  5  ") == 5


def test_parse_score_invalid():
    assert _parse_score("no number here") is None
    assert _parse_score("") is None


# ---------------------------------------------------------------------------
# Router tests — the 13 plan tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_easy_route_no_verifier():
    """EASY tier: cheap model called, verifier never called."""
    clf = _make_classifier(DifficultyTier.EASY)
    verifier = _make_verifier(score=5)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response("Paris")
        router = Router(dispatch="cascade", classifier=clf, verifier=verifier)
        content, decision = await router.route_async(_messages("What is the capital of France?"))

    assert content == "Paris"
    assert decision.verifier_score is None
    assert decision.escalated is False
    assert decision.final_model == CHEAP_ID
    verifier.score_async.assert_not_called()


@pytest.mark.asyncio
async def test_medium_route_verifier_passes():
    """MEDIUM tier: cheap called, verifier returns ≥4, accept cheap response."""
    clf = _make_classifier(DifficultyTier.MEDIUM)
    verifier = _make_verifier(score=4)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response("Backprop works by...")
        router = Router(dispatch="cascade", classifier=clf, verifier=verifier)
        content, decision = await router.route_async(_messages("Explain backpropagation"))

    assert content == "Backprop works by..."
    assert decision.verifier_score == 4
    assert decision.escalated is False
    assert decision.final_model == CHEAP_ID


@pytest.mark.asyncio
async def test_medium_route_verifier_escalates():
    """MEDIUM tier: verifier returns <4, escalates to frontier."""
    clf = _make_classifier(DifficultyTier.MEDIUM)
    verifier = _make_verifier(score=2)

    cheap_resp = _mock_response("bad answer", model=CHEAP_ID)
    frontier_resp = _mock_response("good answer", model=FRONTIER_ID)

    call_sequence = [cheap_resp, frontier_resp]

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = call_sequence
        router = Router(dispatch="cascade", classifier=clf, verifier=verifier)
        content, decision = await router.route_async(_messages("Explain backpropagation"))

    assert content == "good answer"
    assert decision.verifier_score == 2
    assert decision.escalated is True
    assert decision.final_model == FRONTIER_ID


@pytest.mark.asyncio
async def test_hard_route_direct_frontier():
    """HARD tier: goes directly to frontier, no cheap model attempted."""
    clf = _make_classifier(DifficultyTier.HARD)
    verifier = _make_verifier(score=5)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response("Proof by induction...", model=FRONTIER_ID)
        router = Router(dispatch="cascade", classifier=clf, verifier=verifier)
        content, decision = await router.route_async(_messages("Prove infinitely many primes"))

    assert content == "Proof by induction..."
    assert decision.final_model == FRONTIER_ID
    assert decision.verifier_score is None
    verifier.score_async.assert_not_called()
    # Should not have tried any cheap model
    assert CHEAP_ID not in decision.cascade_path


@pytest.mark.asyncio
async def test_cheap_tier_unavailable_fallthrough():
    """Cheap model fails → escalates to frontier."""
    clf = _make_classifier(DifficultyTier.EASY)
    verifier = _make_verifier(score=5)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [
            ProviderError("llama-8b down"),
            _mock_response("frontier saved us", model=FRONTIER_ID),
        ]
        router = Router(dispatch="cascade", classifier=clf, verifier=verifier)
        content, decision = await router.route_async(_messages())

    assert content == "frontier saved us"
    assert decision.final_model == FRONTIER_ID
    assert CHEAP_ID in decision.cascade_path
    assert decision.escalated is True


@pytest.mark.asyncio
async def test_frontier_unavailable_raises():
    """All frontier models fail → ProviderError raised."""
    clf = _make_classifier(DifficultyTier.HARD)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = ProviderError("all down")
        router = Router(dispatch="cascade", classifier=clf)
        with pytest.raises(ProviderError):
            await router.route_async(_messages("Hard question"))


@pytest.mark.asyncio
async def test_verifier_parse_failure_escalates():
    """Verifier returns score=2 (parse failure) → escalates to frontier."""
    clf = _make_classifier(DifficultyTier.MEDIUM)
    # Simulate parse failure: verifier returns 2 (confidence=2 sentinel)
    verifier = _make_verifier(score=2)

    cheap_resp = _mock_response("meh answer", model=CHEAP_ID)
    frontier_resp = _mock_response("great answer", model=FRONTIER_ID)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [cheap_resp, frontier_resp]
        router = Router(dispatch="cascade", classifier=clf, verifier=verifier)
        content, decision = await router.route_async(_messages("Explain entropy"))

    assert content == "great answer"
    assert decision.escalated is True
    assert decision.final_model == FRONTIER_ID


@pytest.mark.asyncio
async def test_cascade_path_in_routing_decision():
    """RoutingDecision.cascade_path lists all models attempted in order."""
    clf = _make_classifier(DifficultyTier.MEDIUM)
    verifier = _make_verifier(score=2)  # force escalation

    cheap_resp = _mock_response("cheap", model=CHEAP_ID)
    frontier_resp = _mock_response("frontier", model=FRONTIER_ID)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [cheap_resp, frontier_resp]
        router = Router(dispatch="cascade", classifier=clf, verifier=verifier)
        _, decision = await router.route_async(_messages("Medium question"))

    assert CHEAP_ID in decision.cascade_path
    assert FRONTIER_ID in decision.cascade_path
    assert decision.cascade_path.index(CHEAP_ID) < decision.cascade_path.index(FRONTIER_ID)


@pytest.mark.asyncio
async def test_routing_decision_no_prompt_logged():
    """RoutingDecision stores prompt_hash (SHA-256), never raw prompt."""
    clf = _make_classifier(DifficultyTier.EASY)
    msg = _messages("My secret prompt")

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response("answer")
        router = Router(dispatch="cascade", classifier=clf)
        _, decision = await router.route_async(msg)

    # hash is a 64-char hex string
    assert len(decision.prompt_hash) == 64
    assert all(c in "0123456789abcdef" for c in decision.prompt_hash)
    # raw prompt not anywhere in the decision
    import dataclasses
    decision_dict = dataclasses.asdict(decision)
    assert "secret" not in str(decision_dict)


def test_sync_route_in_async_context():
    """route() (sync) works even when called from within a running event loop."""

    async def _inner():
        clf = _make_classifier(DifficultyTier.EASY)
        with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_response("sync works")
            router = Router(dispatch="cascade", classifier=clf)
            # route() must not fail with "event loop already running"
            content, decision = router.route(_messages())
        assert content == "sync works"

    asyncio.run(_inner())


@pytest.mark.asyncio
async def test_verifier_timeout_escalates():
    """Verifier raises ProviderError (timeout) → router treats as score=2, escalates."""
    clf = _make_classifier(DifficultyTier.MEDIUM)

    verifier = MagicMock(spec=CascadeVerifier)
    verifier.score_async = AsyncMock(side_effect=ProviderError("timeout"))

    cheap_resp = _mock_response("cheap answer", model=CHEAP_ID)
    frontier_resp = _mock_response("frontier answer", model=FRONTIER_ID)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [cheap_resp, frontier_resp]
        router = Router(dispatch="cascade", classifier=clf, verifier=verifier)
        content, decision = await router.route_async(_messages("Medium question"))

    assert content == "frontier answer"
    assert decision.escalated is True
    assert decision.verifier_score == 2
    assert decision.final_model == FRONTIER_ID


@pytest.mark.asyncio
async def test_verifier_timeout_escalates_internal():
    """
    CascadeVerifier.score_async: ProviderError on call → returns 2.
    This is where the timeout/failure handling lives.
    """
    with patch("smartroute.verifier.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = ProviderError("connection timeout")
        verifier = CascadeVerifier()
        score = await verifier.score_async("prompt", "response", "claude-haiku-4-5")

    assert score == 2


@pytest.mark.asyncio
async def test_cheap_sequence_fallthrough():
    """Cheap model fails → escalates to frontier with WARNING."""
    clf = _make_classifier(DifficultyTier.EASY)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [
            ProviderError("llama-8b down"),
            _mock_response("frontier saved us", model=FRONTIER_ID),
        ]
        router = Router(dispatch="cascade", classifier=clf)
        content, decision = await router.route_async(_messages())

    assert content == "frontier saved us"
    assert decision.escalated is True
    assert decision.final_model == FRONTIER_ID
    assert CHEAP_ID in decision.cascade_path


@pytest.mark.asyncio
async def test_easy_refusal_escalates():
    """EASY tier: cheap model returns empty response → escalates to frontier."""
    clf = _make_classifier(DifficultyTier.EASY)

    empty_resp = _mock_response("", model=CHEAP_ID)
    frontier_resp = _mock_response("Real answer", model=FRONTIER_ID)

    with patch("smartroute.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [empty_resp, frontier_resp]
        router = Router(dispatch="cascade", classifier=clf)
        content, decision = await router.route_async(_messages("Refused question"))

    assert content == "Real answer"
    assert decision.escalated is True
    assert decision.final_model == FRONTIER_ID


# ---------------------------------------------------------------------------
# Two-layer dispatch (dispatch="decision", the default)
# ---------------------------------------------------------------------------

class _StubTriage:
    def __init__(self, p_lookup=0.0, p_decompose=0.0):
        from smartroute.decision import TriageSignals

        self._signals = TriageSignals(p_lookup, p_decompose, "stub")

    def signals(self, prompt):
        return self._signals

    def backend(self):
        return "stub"


def _decision_router(p_lookup=0.0, p_decompose=0.0, **kw):
    from smartroute.decision import DecisionLayer

    return Router(
        decision_layer=DecisionLayer(triage=_StubTriage(p_lookup, p_decompose), **kw),
    )


async def test_decision_mode_is_the_default():
    assert Router()._dispatch_mode == "decision"


async def test_rejects_unknown_dispatch_mode():
    with pytest.raises(ValueError, match="dispatch must be"):
        Router(dispatch="sideways")


async def test_confident_lookup_goes_to_the_cheap_tier():
    router = _decision_router(p_lookup=0.99)
    with patch("smartroute.router.call_model", return_value=_mock_response("Paris")) as call:
        content, decision = await router.route_async([{"role": "user", "content": "capital of France?"}])

    assert content == "Paris"
    assert decision.route_path == "cheap"
    assert decision.final_model == CHEAP_ID
    assert call.call_args.args[0] == CHEAP_ID


async def test_everything_else_defaults_to_middle_not_frontier():
    """The premise of the architecture: the frontier is never a default."""
    router = _decision_router(p_lookup=0.1, p_decompose=0.1)
    with patch("smartroute.router.call_model", return_value=_mock_response("answer")):
        _, decision = await router.route_async([{"role": "user", "content": "a question"}])

    assert decision.route_path == "middle"
    assert decision.final_model == MIDDLE_ID


async def test_decomposable_request_reaches_layer_2():
    router = _decision_router(p_lookup=0.0, p_decompose=0.99, lambda_wrong_usd=0.01)
    plan = '{"atomic": false, "subtasks": [{"task":"a","tier":"cheap"},{"task":"b","tier":"cheap"}]}'
    with patch("smartroute.router.call_model", return_value=_mock_response("x")), \
         patch("smartroute.decomposer.call_model", side_effect=[
             _mock_response(plan), _mock_response("A"), _mock_response("B"), _mock_response("final"),
         ]):
        content, decision = await router.route_async(
            [{"role": "user", "content": "q" * 400}]
        )

    assert decision.route_path == "decompose"
    assert content == "final"
    assert decision.subtask_count == 2


async def test_decision_fields_are_recorded_for_audit():
    router = _decision_router(p_lookup=0.99)
    with patch("smartroute.router.call_model", return_value=_mock_response("Paris")):
        _, decision = await router.route_async([{"role": "user", "content": "capital?"}])

    assert decision.gate_reason
    assert decision.p_lookup == 0.99
    assert decision.projected_cost_usd > 0
    assert decision.classifier_backend == "stub"


async def test_legacy_meta_fields_still_populated():
    """db.py, observability.py and X-SmartRoute-Meta all read difficulty_tier."""
    router = _decision_router(p_lookup=0.99)
    with patch("smartroute.router.call_model", return_value=_mock_response("Paris")):
        _, decision = await router.route_async([{"role": "user", "content": "capital?"}])

    assert decision.difficulty_tier == "CHEAP"
    assert isinstance(decision.difficulty_score, float)


async def test_provider_error_is_logged_then_raised():
    router = _decision_router(p_lookup=0.99)
    with patch("smartroute.router.call_model", side_effect=ProviderError("down")):
        with pytest.raises(ProviderError):
            await router.route_async([{"role": "user", "content": "capital?"}])


async def test_cache_short_circuits_before_the_decision_layer():
    from smartroute.cache import InMemoryLRUCache

    cache = InMemoryLRUCache()
    await cache.put("capital of France?", "Paris")
    router = _decision_router(p_lookup=0.99)
    router._cache = cache

    with patch("smartroute.router.call_model") as call:
        content, decision = await router.route_async(
            [{"role": "user", "content": "capital of France?"}]
        )

    assert content == "Paris"
    assert decision.cache_hit
    call.assert_not_called()

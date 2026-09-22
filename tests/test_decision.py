"""
Tests for the decision layer.

The behaviours worth protecting here are economic, not just functional: the frontier
must never be a default, uncertainty must resolve to middle, and decomposition must
be gated on arithmetic rather than on "does this look decomposable".
"""

import pytest

from smartroute.decision import (
    DECOMPOSE,
    DEFAULT,
    LOOKUP,
    CostModel,
    DecisionLayer,
    EmbeddingTriage,
    HeuristicTriage,
    TriageSignals,
)


class FixedTriage:
    """Triage stub so decision-rule tests are independent of classifier quality."""

    def __init__(self, p_lookup=0.0, p_decompose=0.0):
        self._signals = TriageSignals(p_lookup, p_decompose, "fixed")

    def signals(self, prompt):
        return self._signals

    def backend(self):
        return "fixed"


def layer(**kw):
    kw.setdefault("triage", FixedTriage())
    return DecisionLayer(**kw)


class TestHeuristicTriage:
    @pytest.mark.parametrize(
        "prompt",
        [
            "Who was the man behind The Chipmunks?",
            "What year did the Berlin Wall fall?",
            "How tall is Mount Everest?",
            "Define entropy",
        ],
    )
    def test_recognises_google_substitute_lookups(self, prompt):
        assert HeuristicTriage().signals(prompt).p_lookup >= 0.75

    @pytest.mark.parametrize(
        "prompt",
        [
            "Kanji Company had sales last year of $265 million, including cash sales of "
            "$25 million. If its average collection period was 36 days, its ending "
            "accounts receivable balance is closest to",
            "Explain why the Engle-Granger test rejects the null hypothesis here",
            "Write a function that reverses a linked list in place",
        ],
    )
    def test_declarative_and_reasoning_prompts_are_not_lookups(self, prompt):
        """These are what the old regex classifier misfiled; they must not go cheap."""
        assert HeuristicTriage().signals(prompt).p_lookup < 0.75

    @pytest.mark.parametrize(
        "prompt",
        [
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            "Which is older, the Eiffel Tower or the Statue of Liberty? Compare them.",
            "Find the director's birthplace and then the population of that city",
        ],
    )
    def test_flags_multi_part_prompts(self, prompt):
        assert HeuristicTriage().signals(prompt).p_decompose >= 0.60

    def test_lookup_opener_with_multi_part_is_not_a_lookup(self):
        """'Who ... and ... both ...' opens like a lookup but is not one."""
        signals = HeuristicTriage().signals(
            "Who directed both Sinister and Doctor Strange, and compare their box office?"
        )
        assert signals.p_lookup < 0.75


class TestLayer1:
    def test_confident_lookup_goes_cheap(self):
        decision = layer(triage=FixedTriage(p_lookup=0.9)).decide("q")
        assert decision.path == LOOKUP

    def test_borderline_lookup_falls_through_to_middle(self):
        """Asymmetric errors: a false lookup is far costlier than a missed one."""
        decision = layer(triage=FixedTriage(p_lookup=0.74)).decide("q")
        assert decision.path == DEFAULT

    def test_threshold_is_configurable(self):
        assert layer(triage=FixedTriage(p_lookup=0.5), lookup_threshold=0.4).decide("q").path == LOOKUP


class TestGate:
    def test_atomic_prompt_stops_before_layer_2(self):
        """The gate must short-circuit, not decompose and discover it was pointless."""
        decision = layer(triage=FixedTriage(p_decompose=0.1)).decide("q")
        assert decision.path == DEFAULT
        assert "atomic" in decision.reason

    def test_decomposable_and_worth_it_goes_to_layer_2(self):
        decision = layer(triage=FixedTriage(p_decompose=0.9), lambda_wrong_usd=0.01).decide("q")
        assert decision.path == DECOMPOSE

    def test_decomposable_but_not_worth_it_answers_directly(self):
        """
        With a low cost-of-being-wrong, paying 2.4x to split a task cannot pay off.
        This is the case the gate exists for.
        """
        decision = layer(triage=FixedTriage(p_decompose=0.9), lambda_wrong_usd=0.0).decide("q")
        assert decision.path != DECOMPOSE
        assert "E[cost] favours" in decision.reason

    def test_expensive_subtask_mix_suppresses_decomposition(self):
        """If sub-tasks are expected to land on the frontier, splitting loses."""
        all_frontier = CostModel(subtask_tier_mix=(0.0, 0.0, 1.0))
        decision = layer(
            triage=FixedTriage(p_decompose=0.9), lambda_wrong_usd=0.01, cost_model=all_frontier
        ).decide("q")
        assert decision.path != DECOMPOSE

    def test_alternatives_are_reported_for_auditability(self):
        decision = layer(triage=FixedTriage(p_decompose=0.9)).decide("q")
        assert set(decision.alternatives) == {DEFAULT, "frontier", DECOMPOSE}


class TestEconomics:
    def test_frontier_is_never_a_default_destination(self):
        """Across the whole signal space, no confident-free prompt lands on frontier."""
        for p_lookup in (0.0, 0.3, 0.5):
            for p_decompose in (0.0, 0.2, 0.5):
                decision = layer(triage=FixedTriage(p_lookup, p_decompose)).decide("q")
                assert decision.path in (LOOKUP, DEFAULT), decision.reason

    def test_decomposition_is_costed_above_a_direct_middle_answer(self):
        """The 2.4x overhead is the reason the gate has to exist."""
        d = layer()
        assert d.estimate_decomposition_cost() > 2 * d._direct_cost("middle")

    def test_every_path_carries_a_projected_cost(self):
        for signals in (FixedTriage(0.9, 0.0), FixedTriage(0.0, 0.1), FixedTriage(0.0, 0.9)):
            assert layer(triage=signals).decide("q").projected_cost_usd > 0

    def test_decisions_are_free_and_deterministic(self):
        """decide() makes no model calls, so it can run over a whole corpus offline."""
        d = layer(triage=HeuristicTriage())
        first = d.decide("Who invented the telephone?")
        second = d.decide("Who invented the telephone?")
        assert first.path == second.path == LOOKUP


class TestEmbeddingTriage:
    def test_falls_back_to_heuristic_without_a_trained_model(self, tmp_path):
        triage = EmbeddingTriage(model_path=tmp_path / "absent.joblib")
        assert triage.backend() == "heuristic"
        assert triage.signals("Who was the man behind The Chipmunks?").p_lookup >= 0.75

    def test_corrupt_model_file_degrades_gracefully(self, tmp_path):
        bad = tmp_path / "triage.joblib"
        bad.write_bytes(b"not a joblib file")
        assert EmbeddingTriage(model_path=bad).backend() == "heuristic"

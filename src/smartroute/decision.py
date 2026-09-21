"""
The decision layer: where a request goes, and why.

Measured motivation
-------------------
The shipped KeywordBaselineClassifier sends 92.8% of MMLU to MEDIUM (dataset labels
say 43%), because almost every stem contains "calculate"/"explain"/"determine".
Reordering the regexes does not help — EASY patterns are anchored `.match()` on
openers like "what is", and real prompts are declarative stems, so nothing matches
either way. The regex approach cannot separate these prompts at all.

The response is not a cleverer regex. It is:

  1. a learned estimate of what kind of request this is, and
  2. a decision rule that spends money only when the arithmetic says to.

Architecture
------------
    prompt
      |
      +-- LAYER 1  confident "google-substitute" lookup?  --> CHEAP
      |
      +-- GATE     sub-tasks needed AND worth it?         --> MIDDLE   (the default)
      |
      +-- LAYER 2  decompose, route each sub-task         --> see decomposer.py

The frontier model is never a default destination. You cannot route what you cannot
classify, so uncertainty resolves to MIDDLE rather than to the most expensive option.

Why the gate is load-bearing
----------------------------
Priced from the catalog at 128 in / 300 out, decompose + synthesise costs ~2.4x
simply answering with the middle model. It only pays when it replaces work that would
otherwise need the frontier model AND the sub-tasks stay cheap:

    3 cheap sub-tasks        0.52x a frontier call   pays
    1 cheap + 1 mid + 1 front 1.21x a frontier call   loses

So the gate asks "is decomposing *worth it*", not "is it possible".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from smartroute.catalog import Catalog, get_catalog

logger = logging.getLogger(__name__)

# Routing destinations.
LOOKUP = "cheap"
DEFAULT = "middle"
DECOMPOSE = "decompose"


@dataclass(frozen=True)
class TriageSignals:
    """What layer 1 believes about a prompt."""

    p_lookup: float
    p_decompose: float
    backend: str


@dataclass(frozen=True)
class RouteDecision:
    """Where the request goes, with the evidence that sent it there."""

    path: str
    reason: str
    signals: TriageSignals
    projected_cost_usd: float = 0.0
    alternatives: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer 1 — triage
# ---------------------------------------------------------------------------

class Triage(Protocol):
    def signals(self, prompt: str) -> TriageSignals: ...
    def backend(self) -> str: ...


# A lookup is the question someone types instead of opening a search box: short,
# single-hop, factual. These patterns are the cold-start signal used before the
# learned heads have training data, and the permanent fallback when
# sentence-transformers is not installed.
_LOOKUP_OPENERS = re.compile(
    r"^\s*(who|what|when|where|which|how many|how much|how tall|how long|how old|"
    r"name the|define|in what year|whose)\b",
    re.IGNORECASE,
)
_MULTI_PART = re.compile(
    r"\band then\b|\bafter (?:that|which)\b|\bboth\b|\bcompare\b|\bversus\b|\bvs\.?\b"
    r"|\beach of\b|\ball of the\b|\bstep by step\b|\bfirst.*then\b"
    r"|\bwho (?:is|was) older\b|\bsame (?:nationality|country|year)\b",
    re.IGNORECASE,
)
_REASONING = re.compile(
    r"\bwhy\b|\bexplain\b|\banalyz|\bevaluate\b|\bderive\b|\bprove\b|\bcalculate\b"
    r"|\bdesign\b|\bimplement\b|\bwrite (?:a|an|the)\b|\bsummar",
    re.IGNORECASE,
)
_LOOKUP_MAX_WORDS = 20


class HeuristicTriage:
    """
    Regex cold-start / fallback.

    Deliberately conservative on p_lookup: a false lookup sends a hard question to the
    weakest model, while a missed lookup costs a fraction of a cent in forgone savings.
    The errors are not symmetric, so neither is the heuristic.
    """

    def signals(self, prompt: str) -> TriageSignals:
        text = prompt.strip()
        words = len(text.split())

        multi = bool(_MULTI_PART.search(text))
        reasoning = bool(_REASONING.search(text))
        opener = bool(_LOOKUP_OPENERS.match(text))

        if opener and words <= _LOOKUP_MAX_WORDS and not multi and not reasoning:
            p_lookup = 0.85
        elif opener and words <= _LOOKUP_MAX_WORDS:
            p_lookup = 0.40
        else:
            p_lookup = 0.05

        p_decompose = 0.80 if multi else (0.35 if words > 40 and reasoning else 0.10)
        return TriageSignals(p_lookup=p_lookup, p_decompose=p_decompose, backend=self.backend())

    def backend(self) -> str:
        return "heuristic"


class EmbeddingTriage:
    """
    Two logistic-regression heads over a sentence embedding.

    The encoder is all-MiniLM-L6-v2 — already loaded by cache.py for the pgvector
    semantic cache, so on a cache-enabled deployment the prompt is embedded once and
    used twice, making routing's marginal cost effectively zero.

    Falls back to HeuristicTriage when sentence-transformers is unavailable or no
    trained model has been fitted yet.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_path: Path | None = None) -> None:
        self._encoder = None
        self._heads = None
        self.calibration: dict[str, list[dict]] = {}
        self._fallback = HeuristicTriage()
        self._model_path = model_path or (Path(__file__).parent / "data" / "triage.joblib")
        self._load()

    def _load(self) -> None:
        if not self._model_path.exists():
            logger.info(
                "no trained triage model at %s — using heuristic fallback. "
                "Train with: uv run harness/train_triage.py",
                self._model_path,
            )
            return
        try:
            import joblib

            bundle = joblib.load(self._model_path)
            self._heads = bundle["heads"]
            self.calibration = bundle.get("calibration", {})
        except Exception as exc:
            logger.warning("failed to load triage model: %s — using heuristic fallback", exc)

    def _encode(self, text: str):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._MODEL_NAME)
        return self._encoder.encode([text], normalize_embeddings=True)

    def signals(self, prompt: str) -> TriageSignals:
        if self._heads is None:
            return self._fallback.signals(prompt)
        try:
            features = self._encode(prompt)
            return TriageSignals(
                p_lookup=float(self._heads["lookup"].predict_proba(features)[0][1]),
                p_decompose=float(self._heads["decompose"].predict_proba(features)[0][1]),
                backend=self.backend(),
            )
        except Exception as exc:
            logger.warning("triage inference failed: %s — falling back", exc)
            return self._fallback.signals(prompt)

    def backend(self) -> str:
        return "embedding" if self._heads is not None else "heuristic"


# ---------------------------------------------------------------------------
# The decision rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """
    Token-count expectations used to price a path before taking it.

    Defaults are the measured MMLU/HotpotQA shape; Phase 4 recalibrates them from
    observed runs. They live here, in one place, rather than as magic numbers spread
    through the dispatch logic.
    """

    prompt_tokens: int = 128
    answer_tokens: int = 300
    decompose_out_tokens: int = 150
    subtask_in_tokens: int = 60
    subtask_out_tokens: int = 200
    synthesis_out_tokens: int = 400
    expected_subtasks: int = 3
    # Share of sub-tasks expected to land on each tier, given the gate only fires when
    # decomposition looks worthwhile. If this drifts toward frontier, decomposition
    # stops paying — which is what the bail-out in decomposer.py catches at runtime.
    subtask_tier_mix: tuple[float, float, float] = (0.7, 0.3, 0.0)


class DecisionLayer:
    """
    Layer 1 + the decomposition gate.

    `decide()` is pure and free — no network, no model calls — so it can run on every
    request and be evaluated over a whole corpus offline.
    """

    def __init__(
        self,
        catalog: Catalog | None = None,
        triage: Triage | None = None,
        lookup_threshold: float | None = None,
        decompose_threshold: float = 0.75,
        lambda_wrong_usd: float | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self._catalog = catalog or get_catalog()
        self._triage = triage or EmbeddingTriage()
        self._decompose_threshold = decompose_threshold
        # Defaults to the catalog's knob so the economics live in one file alongside
        # the prices they are compared against.
        self._lambda = (
            lambda_wrong_usd
            if lambda_wrong_usd is not None
            else getattr(self._catalog, "lambda_wrong_answer_usd", 0.001)
        )
        self._costs = cost_model or CostModel()
        self._lookup_threshold = (
            lookup_threshold
            if lookup_threshold is not None
            else self._derive_lookup_threshold()
        )

    # -- threshold derivation ----------------------------------------------

    def required_lookup_precision(self) -> float:
        """
        How often a lookup divert must be right for the path to be worth taking.

        Diverting saves (middle - cheap) dollars when correct, and costs lambda when
        the prompt was not really a lookup and the cheapest model fumbles it. Breaking
        even needs:

            precision * saving = (1 - precision) * lambda
            precision = lambda / (saving + lambda)

        The saving is small — about $0.00013 on the shipped catalog — so this rises
        steeply with lambda: 88% at lambda=$0.001, 98.7% at lambda=$0.01. That is a
        real property of the economics, not a tuning artefact, and it is why layer 1
        is deliberately reluctant to fire.
        """
        saving = self._direct_cost("middle") - self._direct_cost("cheap")
        if saving <= 0:
            return 1.0  # cheap tier saves nothing; never divert
        return self._lambda / (saving + self._lambda)

    def _derive_lookup_threshold(self) -> float:
        """
        Lowest threshold whose *measured* precision clears the economic bar.

        Falls back to a conservative 0.85 when no calibration curve is available —
        untrained heads should not be trusted with the cheapest model.
        """
        required = self.required_lookup_precision()
        curve = getattr(self._triage, "calibration", {}).get("lookup") or []
        for point in sorted(curve, key=lambda p: p["threshold"]):
            if point["precision"] >= required:
                logger.debug(
                    "lookup threshold %.2f (measured precision %.2f >= required %.2f)",
                    point["threshold"], point["precision"], required,
                )
                return point["threshold"]
        if curve:
            logger.warning(
                "no threshold reaches the required lookup precision %.2f — "
                "layer 1 disabled. Lower lambda_wrong_usd or improve the head.",
                required,
            )
            return 1.01  # unreachable: never divert to cheap
        return 0.85

    # -- cost estimates ----------------------------------------------------

    def _direct_cost(self, tier: str) -> float:
        spec = self._catalog.by_role(tier)
        return spec.cost(self._costs.prompt_tokens, self._costs.answer_tokens)

    def estimate_decomposition_cost(self) -> float:
        """decompose call + N sub-task calls + synthesis, priced from the catalog."""
        c = self._costs
        middle = self._catalog.by_role("middle")

        decompose = middle.cost(c.prompt_tokens + 80, c.decompose_out_tokens)
        synthesis = middle.cost(
            c.prompt_tokens + c.expected_subtasks * c.subtask_out_tokens, c.synthesis_out_tokens
        )
        per_subtask = sum(
            share * self._catalog.by_role(role).cost(c.subtask_in_tokens, c.subtask_out_tokens)
            for share, role in zip(c.subtask_tier_mix, ("cheap", "middle", "frontier"))
        )
        return decompose + synthesis + c.expected_subtasks * per_subtask

    # -- the rule ----------------------------------------------------------

    def decide(self, prompt: str) -> RouteDecision:
        signals = self._triage.signals(prompt)

        # Layer 1: divert only on confidence. Asymmetric by design — see HeuristicTriage.
        if signals.p_lookup >= self._lookup_threshold:
            return RouteDecision(
                path=LOOKUP,
                reason=f"lookup p={signals.p_lookup:.2f} >= {self._lookup_threshold}",
                signals=signals,
                projected_cost_usd=self._direct_cost("cheap"),
            )

        # Gate: cheap to ask, so ask before paying for a decomposition round-trip.
        if signals.p_decompose < self._decompose_threshold:
            return RouteDecision(
                path=DEFAULT,
                reason=f"atomic p_decompose={signals.p_decompose:.2f} < {self._decompose_threshold}",
                signals=signals,
                projected_cost_usd=self._direct_cost("middle"),
            )

        # The prompt looks decomposable. Does the arithmetic agree?
        #
        # Hard dollar guard first, before any quality weighting. Decomposition's entire
        # justification is doing frontier-grade work without a frontier call, so if its
        # projected *spend* already exceeds a direct frontier call there is no argument
        # left for it at any quality prior. This guard matters because lambda is often
        # larger than the amounts being compared, and without it the rule collapses into
        # whichever path was assigned the smallest hand-picked failure multiplier.
        decomposition_spend = self.estimate_decomposition_cost()
        frontier_spend = self._direct_cost("frontier")
        if decomposition_spend >= frontier_spend:
            return RouteDecision(
                path="frontier",
                reason=(
                    f"decomposable p={signals.p_decompose:.2f} but projected decomposition "
                    f"spend ${decomposition_spend:.6f} >= one frontier call "
                    f"${frontier_spend:.6f} — splitting cannot pay off"
                ),
                signals=signals,
                projected_cost_usd=frontier_spend,
                alternatives={DECOMPOSE: decomposition_spend, "frontier": frontier_spend},
            )

        #   E[cost | path] = dollars spent + P(wrong) * lambda
        #
        # lambda is the operator's stated cost of a wrong answer, and is what converts
        # a quality judgement into the same units as a bill so they can be compared.
        #
        # The failure multipliers below are priors, not measurements — Phase 4
        # replaces them with per-path rates observed on the corpora.
        p_fail_direct = signals.p_decompose  # multi-part work is what one-shot answers botch
        direct = self._direct_cost("middle") + p_fail_direct * self._lambda
        frontier = frontier_spend + 0.5 * p_fail_direct * self._lambda
        decomposed = decomposition_spend + 0.3 * p_fail_direct * self._lambda

        options = {DEFAULT: direct, "frontier": frontier, DECOMPOSE: decomposed}
        best = min(options, key=options.get)

        if best == DECOMPOSE:
            return RouteDecision(
                path=DECOMPOSE,
                reason=(
                    f"decompose p={signals.p_decompose:.2f}, "
                    f"E[cost]=${decomposed:.6f} beats middle ${direct:.6f} "
                    f"and frontier ${frontier:.6f}"
                ),
                signals=signals,
                projected_cost_usd=decomposition_spend,
                alternatives=options,
            )

        # Decomposable but not worth decomposing: answer directly with the winner.
        tier = "middle" if best == DEFAULT else "frontier"
        return RouteDecision(
            path=tier,
            reason=(
                f"decomposable p={signals.p_decompose:.2f} but E[cost] favours {tier} "
                f"(${options[best]:.6f} vs decompose ${decomposed:.6f})"
            ),
            signals=signals,
            projected_cost_usd=self._direct_cost(tier),
            alternatives=options,
        )

"""
Task-ranked routing: decision model -> OpenRouter task rankings -> effort-based cost cap.

Pipeline
--------
    prompt
      |
      +-- 1. ONE Decisions API call (Jev by default), two questions answered in parallel:
      |        task   : Choice over OpenRouter's own task-classification tags
      |        effort : Score on a 4-level effort scale
      |
      +-- 2. candidates = models OpenRouter's community traffic uses most for that tag
      |        (GET /api/v1/classifications/task), priced from GET /api/v1/models
      |
      +-- 3. effort -> max cost tier; drop candidates above it
      |
      +-- 4. call the highest-ranked survivor through OpenRouter; next survivor on error

Why the task buckets are OpenRouter's tags
------------------------------------------
Inventing our own buckets would need a hand-maintained mapping onto OpenRouter's
taxonomy that drifts the moment they add a tag. Classifying straight into the tags the
rankings are keyed on makes step 2 an exact lookup.

What the rankings measure — read this
-------------------------------------
`tag_usage_share` is *share of requests*, not quality. "Most used for code generation"
is a popularity signal, confounded by price, defaults in popular apps, and free tiers.
It is a reasonable prior, not evidence that a model solves the task best. The harness
(harness/eval_routing.py) is the way to check whether this pays on your traffic.

Rate limits
-----------
The Data API allows 30 req/min per key and 500 req/day per account, so rankings and
pricing are cached (default 1h TTL), never fetched per request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from smartroute.catalog import ModelSpec, Price

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai"
DECISIONS_PATH = "/api/alpha/decisions"
CLASSIFICATIONS_PATH = "/api/v1/classifications/task"
MODELS_PATH = "/api/v1/models"

DEFAULT_DECISION_MODEL = "~typesafe/jev-latest"

# Ordered low -> high. Score answers are indices into this list.
EFFORT_LEVELS: tuple[str, ...] = (
    "Trivial: greeting, single-fact lookup, or a mechanical rewrite or format conversion",
    "Simple: short answer or small self-contained task needing little reasoning",
    "Moderate: multi-step reasoning, non-trivial code, or careful structured writing",
    "Hard: expert-level reasoning, complex debugging or system design, long multi-part work",
)

# Cost tiers, ordered cheap -> expensive. Upper bound is blended $/M tokens.
COST_TIERS: tuple[str, ...] = ("budget", "standard", "premium", "frontier")


class TaskRouteError(Exception):
    """Raised when task-ranked routing cannot produce a candidate."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskRouteConfig:
    decision_model: str = DEFAULT_DECISION_MODEL
    # Upper bound of each COST_TIERS entry in blended $/M tokens; the last tier is open.
    tier_caps_per_mtok: tuple[float, float, float] = (0.50, 2.00, 8.00)
    # effort level index -> highest cost tier index allowed.
    effort_to_max_tier: tuple[int, int, int, int] = (0, 1, 2, 3)
    # Blend weight of output price. Assistant answers usually run ~3x the prompt.
    output_weight: float = 0.75
    # Below this Score confidence, allow one tier higher: under-provisioning a hard
    # prompt costs a wrong answer, over-provisioning an easy one costs fractions of a cent.
    effort_confidence_floor: float = 0.6
    # Below this Choice confidence the task label is not trusted; use the overall
    # most-used models across all tags instead.
    task_confidence_floor: float = 0.35
    # `:free` variants are rate-limited and frequently unavailable; off by default.
    allow_free: bool = False
    max_candidates: int = 3
    cache_ttl_s: float = 3600.0
    timeout_s: float = 10.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskTag:
    tag: str
    display_name: str
    macro_category: str
    models: tuple[tuple[str, float], ...]  # (model id, tag_usage_share), ranked


@dataclass(frozen=True)
class Candidate:
    model_id: str            # OpenRouter id, e.g. "openai/gpt-4.1-mini"
    rank_share: float        # tag_usage_share
    price: Price
    blended_per_mtok: float
    tier: int                # index into COST_TIERS

    def spec(self) -> ModelSpec:
        """A catalog entry built on the fly so call_model prices it correctly."""
        return ModelSpec(
            alias=f"openrouter:{self.model_id}",
            id=f"openrouter/{self.model_id}",
            role="middle",
            price=self.price,
            env_key="OPENROUTER_API_KEY",
        )


@dataclass(frozen=True)
class TaskRouteDecision:
    task_tag: str
    task_label: str
    task_confidence: float
    effort_level: int
    effort_score: float
    effort_confidence: float
    max_tier: int
    candidates: list[Candidate]
    decision_cost_usd: float
    reason: str
    rejected: dict[str, str] = field(default_factory=dict)

    @property
    def max_tier_name(self) -> str:
        return COST_TIERS[self.max_tier]


# ---------------------------------------------------------------------------
# OpenRouter clients
# ---------------------------------------------------------------------------

class OpenRouterData:
    """Cached access to the task rankings and model pricing."""

    def __init__(self, api_key: str, config: TaskRouteConfig, client: httpx.AsyncClient | None = None):
        self._key = api_key
        self._cfg = config
        self._client = client
        self._tags: list[TaskTag] | None = None
        self._prices: dict[str, Price] | None = None
        self._as_of: str = ""
        self._fetched_at = 0.0

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=OPENROUTER_BASE, timeout=self._cfg.timeout_s)
        return self._client

    @property
    def as_of(self) -> str:
        return self._as_of

    def _fresh(self) -> bool:
        return self._tags is not None and (time.monotonic() - self._fetched_at) < self._cfg.cache_ttl_s

    async def refresh(self, force: bool = False) -> None:
        if self._fresh() and not force:
            return
        headers = {"Authorization": f"Bearer {self._key}"}
        try:
            cls = await self._http().get(CLASSIFICATIONS_PATH, params={"window": "7d"}, headers=headers)
            cls.raise_for_status()
            models = await self._http().get(MODELS_PATH, headers=headers)
            models.raise_for_status()
        except httpx.HTTPError as exc:
            if self._tags is not None:
                # Stale rankings beat no rankings; they move over days, not minutes.
                logger.warning("rankings refresh failed (%s) — serving stale snapshot from %s", exc, self._as_of)
                return
            raise TaskRouteError(f"could not fetch OpenRouter rankings: {exc}") from exc

        self._tags = parse_classifications(cls.json())
        self._prices = parse_prices(models.json())
        self._as_of = (cls.json().get("data") or {}).get("as_of", "")
        self._fetched_at = time.monotonic()
        logger.info("loaded %d task tags, %d priced models (as of %s)",
                    len(self._tags), len(self._prices), self._as_of)

    async def tags(self) -> list[TaskTag]:
        await self.refresh()
        return self._tags or []

    async def prices(self) -> dict[str, Price]:
        await self.refresh()
        return self._prices or {}


def parse_classifications(payload: dict) -> list[TaskTag]:
    data = payload.get("data") or {}
    out = []
    for c in data.get("classifications") or []:
        models = tuple(
            (m["id"], float(m.get("tag_usage_share") or 0.0))
            for m in (c.get("models") or [])
            if m.get("id")
        )
        out.append(TaskTag(
            tag=c["tag"],
            display_name=c.get("display_name") or c["tag"],
            macro_category=c.get("macro_category") or "",
            models=tuple(sorted(models, key=lambda m: -m[1])),
        ))
    return out


def parse_prices(payload: dict) -> dict[str, Price]:
    """Index /api/v1/models pricing by both `id` and `canonical_slug`. Per-token USD strings."""
    out: dict[str, Price] = {}
    for m in payload.get("data") or []:
        pricing = m.get("pricing") or {}
        try:
            price = Price(
                input_per_token=float(pricing["prompt"]),
                output_per_token=float(pricing["completion"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if price.input_per_token < 0 or price.output_per_token < 0:
            continue  # OpenRouter uses -1 for variable-priced routers
        for key in (m.get("id"), m.get("canonical_slug")):
            if key:
                out[key] = price
    return out


class DecisionsClient:
    """POST /api/alpha/decisions — typed answers with probabilities."""

    def __init__(self, api_key: str, config: TaskRouteConfig, client: httpx.AsyncClient | None = None):
        self._key = api_key
        self._cfg = config
        self._client = client

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=OPENROUTER_BASE, timeout=self._cfg.timeout_s)
        return self._client

    async def decide(self, state: dict, questions: dict) -> tuple[dict, float]:
        """Returns (answers, cost_usd)."""
        try:
            resp = await self._http().post(
                DECISIONS_PATH,
                headers={"Authorization": f"Bearer {self._key}"},
                json={"model": self._cfg.decision_model, "state": state, "questions": questions},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise TaskRouteError(f"decisions API call failed: {exc}") from exc
        body = resp.json()
        return body.get("answers") or {}, float((body.get("usage") or {}).get("cost") or 0.0)


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------

_PROMPT_CHARS = 16_000  # Jev's window is 262k tokens; this keeps the per-call cost bounded


class TaskRanker:
    """Classifies a prompt and returns an ordered, cost-capped candidate list."""

    def __init__(
        self,
        api_key: str,
        config: TaskRouteConfig | None = None,
        data: OpenRouterData | None = None,
        decisions: DecisionsClient | None = None,
    ) -> None:
        if not api_key:
            raise TaskRouteError("OPENROUTER_API_KEY is required for task-ranked routing")
        self.config = config or TaskRouteConfig()
        self._data = data or OpenRouterData(api_key, self.config)
        self._decisions = decisions or DecisionsClient(api_key, self.config)

    # -- pricing -----------------------------------------------------------

    def blended(self, price: Price) -> float:
        w = self.config.output_weight
        return ((1 - w) * price.input_per_token + w * price.output_per_token) * 1_000_000

    def tier_of(self, blended_per_mtok: float) -> int:
        for i, cap in enumerate(self.config.tier_caps_per_mtok):
            if blended_per_mtok <= cap:
                return i
        return len(COST_TIERS) - 1

    # -- questions ---------------------------------------------------------

    @staticmethod
    def _questions(tags: list[TaskTag]) -> dict:
        return {
            "task": {
                "type": "choice",
                "instructions": "Which kind of task is the user's latest message asking for?",
                "criteria": {
                    t.tag: f"{t.display_name} ({t.macro_category})" if t.macro_category else t.display_name
                    for t in tags
                },
            },
            "effort": {
                "type": "score",
                "instructions": (
                    "How much capability does answering the user's latest message well require?"
                ),
                "criteria": list(EFFORT_LEVELS),
            },
        }

    def _effort(self, answer: dict) -> tuple[int, float, float]:
        score = float(answer.get("score", 1.0))
        confidence = float(answer.get("confidence", 0.0))
        probs = answer.get("probabilities") or {}
        if probs:
            level = int(max(probs, key=lambda k: float(probs[k])))
        else:
            level = int(round(score))
        level = max(0, min(level, len(EFFORT_LEVELS) - 1))
        return level, score, confidence

    # -- selection ---------------------------------------------------------

    def _overall(self, tags: list[TaskTag]) -> tuple[tuple[str, float], ...]:
        """Usage-weighted ranking across every tag, for when the task label is unsure."""
        totals: dict[str, float] = {}
        for t in tags:
            for mid, share in t.models:
                totals[mid] = totals.get(mid, 0.0) + share
        return tuple(sorted(totals.items(), key=lambda m: -m[1]))

    def rank_candidates(
        self,
        ranked: tuple[tuple[str, float], ...],
        prices: dict[str, Price],
        max_tier: int,
    ) -> tuple[list[Candidate], dict[str, str]]:
        kept: list[Candidate] = []
        rejected: dict[str, str] = {}
        priced: list[Candidate] = []
        for mid, share in ranked:
            if mid.endswith(":free") and not self.config.allow_free:
                rejected[mid] = "free variant"
                continue
            price = prices.get(mid)
            if price is None:
                rejected[mid] = "no OpenRouter price"
                continue
            b = self.blended(price)
            c = Candidate(model_id=mid, rank_share=share, price=price, blended_per_mtok=b, tier=self.tier_of(b))
            priced.append(c)
            if c.tier > max_tier:
                rejected[mid] = f"{COST_TIERS[c.tier]} > {COST_TIERS[max_tier]} cap (${b:.2f}/M)"
                continue
            kept.append(c)

        if not kept and priced:
            # Every ranked model is above the cap. Take the cheapest rather than nothing:
            # the cap is a cost preference, not a reason to fail the request.
            cheapest = min(priced, key=lambda c: c.blended_per_mtok)
            rejected.pop(cheapest.model_id, None)
            kept = [cheapest]
        return kept[: self.config.max_candidates], rejected

    async def select(self, prompt: str) -> TaskRouteDecision:
        tags = await self._data.tags()
        if not tags:
            raise TaskRouteError("OpenRouter returned no task classifications")
        prices = await self._data.prices()

        answers, cost = await self._decisions.decide(
            state={"user_message": prompt[:_PROMPT_CHARS]},
            questions=self._questions(tags),
        )

        task = answers.get("task") or {}
        tag_key = task.get("choice")
        task_conf = float(task.get("confidence", 0.0))
        by_tag = {t.tag: t for t in tags}

        level, score, eff_conf = self._effort(answers.get("effort") or {})
        max_tier = self.config.effort_to_max_tier[level]
        bumped = eff_conf < self.config.effort_confidence_floor and max_tier < len(COST_TIERS) - 1
        if bumped:
            max_tier += 1

        if tag_key in by_tag and task_conf >= self.config.task_confidence_floor:
            tag = by_tag[tag_key]
            ranked, label = tag.models, tag.display_name
        else:
            ranked, label = self._overall(tags), "overall (low task confidence)"
            tag_key = tag_key or "unknown"

        candidates, rejected = self.rank_candidates(ranked, prices, max_tier)
        if not candidates:
            raise TaskRouteError(f"no priced candidates for task {tag_key!r}")

        reason = (
            f"task={tag_key} ({label}) conf={task_conf:.2f}; "
            f"effort={level} score={score:.2f} conf={eff_conf:.2f}"
            f"{' (+1 tier, low confidence)' if bumped else ''}; "
            f"cap={COST_TIERS[max_tier]}; pick={candidates[0].model_id} "
            f"(${candidates[0].blended_per_mtok:.2f}/M, share={candidates[0].rank_share:.2f})"
        )
        return TaskRouteDecision(
            task_tag=tag_key,
            task_label=label,
            task_confidence=task_conf,
            effort_level=level,
            effort_score=score,
            effort_confidence=eff_conf,
            max_tier=max_tier,
            candidates=candidates,
            decision_cost_usd=cost,
            reason=reason,
            rejected=rejected,
        )


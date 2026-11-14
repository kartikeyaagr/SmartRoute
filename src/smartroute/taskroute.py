"""
Task-ranked routing: decision model -> benchmark quality per task -> effort-based cost cap.

Pipeline
--------
    prompt
      |
      +-- 1. ONE Decisions API call (Jev by default), two questions answered in parallel:
      |        task   : Choice over the task types benchmarks score separately
      |        effort : Score on a 4-level effort scale
      |
      +-- 2. candidates = models ranked by the benchmark index for that task
      |        (GET /api/v1/benchmarks?source=artificial-analysis), priced and resolved
      |        to routable ids from GET /api/v1/models
      |
      +-- 3. effort -> max cost tier; drop candidates above it
      |
      +-- 4. call the best-scoring survivor through OpenRouter; next survivor on error

Why exactly three task buckets
------------------------------
Artificial Analysis publishes three composite indices per model: coding, agentic and
intelligence. A finer task taxonomy would only be mapped back onto those three, so
classifying straight into them keeps step 2 an exact lookup and stops the Choice
question spending probability mass on distinctions nothing downstream uses.

What the scores measure
-----------------------
These are benchmark composites, not your traffic. They beat usage share as a quality
signal (popularity is confounded by price and app defaults), but a model that tops
`coding_index` has not been shown to win on *your* coding prompts. Check with the
harness.

Rate limits
-----------
The Data API allows 30 req/min per key and 500 req/day per account, so benchmarks and
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
BENCHMARKS_PATH = "/api/v1/benchmarks"
MODELS_PATH = "/api/v1/models"

DEFAULT_DECISION_MODEL = "~typesafe/jev-latest"

# task bucket -> (Choice criterion, Artificial Analysis index field)
TASK_TYPES: dict[str, tuple[str, str]] = {
    "coding": (
        "Writing, reviewing, debugging or explaining code, scripts, queries or configs",
        "coding_index",
    ),
    "agentic": (
        "Multi-step tool use, planning and executing actions, or orchestrating a workflow",
        "agentic_index",
    ),
    "general": (
        "Knowledge questions, reasoning, math, analysis, writing, or anything else",
        "intelligence_index",
    ),
}
FALLBACK_TASK = "general"

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
    # Below this Choice confidence the task label is not trusted; rank by the general
    # intelligence index instead.
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
class BenchmarkEntry:
    permaslug: str
    display_name: str
    scores: dict[str, float]      # index field -> score; only non-null indices present
    price: Price | None           # pricing carried on the benchmark item, if any


@dataclass(frozen=True)
class Candidate:
    model_id: str            # routable OpenRouter id, e.g. "openai/gpt-4.1-mini"
    quality: float           # benchmark index score for the chosen task
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

def _price(pricing: dict | None) -> Price | None:
    """Per-token USD strings -> Price. None for missing or variable (-1) pricing."""
    try:
        price = Price(
            input_per_token=float((pricing or {})["prompt"]),
            output_per_token=float((pricing or {})["completion"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if price.input_per_token < 0 or price.output_per_token < 0:
        return None
    return price


def parse_benchmarks(payload: dict) -> list[BenchmarkEntry]:
    index_fields = {f for _, f in TASK_TYPES.values()}
    out = []
    for item in payload.get("data") or []:
        slug = item.get("model_permaslug")
        if not slug:
            continue
        scores = {
            f: float(item[f]) for f in index_fields if isinstance(item.get(f), (int, float))
        }
        if not scores:
            continue
        out.append(BenchmarkEntry(
            permaslug=slug,
            display_name=item.get("display_name") or slug,
            scores=scores,
            price=_price(item.get("pricing")),
        ))
    return out


def parse_models(payload: dict) -> tuple[dict[str, str], dict[str, Price]]:
    """
    Returns (slug -> routable id, routable id -> price).

    Benchmarks are keyed on permaslugs (often dated, e.g. `...-20241022`); chat
    completions want the model `id`. /api/v1/models carries both as `canonical_slug`
    and `id`, so it is the join table.
    """
    ids: dict[str, str] = {}
    prices: dict[str, Price] = {}
    for m in payload.get("data") or []:
        mid = m.get("id")
        if not mid:
            continue
        ids[mid] = mid
        if m.get("canonical_slug"):
            ids.setdefault(m["canonical_slug"], mid)
        price = _price(m.get("pricing"))
        if price is not None:
            prices[mid] = price
    return ids, prices


class OpenRouterData:
    """Cached access to benchmark scores and model pricing."""

    def __init__(self, api_key: str, config: TaskRouteConfig, client: httpx.AsyncClient | None = None):
        self._key = api_key
        self._cfg = config
        self._client = client
        self._bench: list[BenchmarkEntry] | None = None
        self._ids: dict[str, str] = {}
        self._prices: dict[str, Price] = {}
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
        return self._bench is not None and (time.monotonic() - self._fetched_at) < self._cfg.cache_ttl_s

    async def refresh(self, force: bool = False) -> None:
        if self._fresh() and not force:
            return
        headers = {"Authorization": f"Bearer {self._key}"}
        try:
            bench = await self._http().get(
                BENCHMARKS_PATH,
                params={"source": "artificial-analysis", "max_results": 100},
                headers=headers,
            )
            bench.raise_for_status()
            models = await self._http().get(MODELS_PATH, headers=headers)
            models.raise_for_status()
        except httpx.HTTPError as exc:
            if self._bench is not None:
                # Stale scores beat no scores; benchmark composites move over weeks.
                logger.warning("benchmark refresh failed (%s) — serving snapshot from %s", exc, self._as_of)
                return
            raise TaskRouteError(f"could not fetch OpenRouter benchmarks: {exc}") from exc

        self._bench = parse_benchmarks(bench.json())
        self._ids, self._prices = parse_models(models.json())
        self._as_of = (bench.json().get("meta") or {}).get("as_of", "")
        self._fetched_at = time.monotonic()
        logger.info("loaded %d benchmarked models, %d priced models (as of %s)",
                    len(self._bench), len(self._prices), self._as_of)

    async def benchmarks(self) -> list[BenchmarkEntry]:
        await self.refresh()
        return self._bench or []

    async def resolve(self, slug: str) -> str | None:
        await self.refresh()
        return self._ids.get(slug) or self._ids.get(slug.split(":")[0])

    async def price(self, model_id: str) -> Price | None:
        await self.refresh()
        return self._prices.get(model_id)


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
    def _questions() -> dict:
        return {
            "task": {
                "type": "choice",
                "instructions": "Which kind of task is the user's latest message asking for?",
                "criteria": {k: desc for k, (desc, _) in TASK_TYPES.items()},
            },
            "effort": {
                "type": "score",
                "instructions": (
                    "How much capability does answering the user's latest message well require?"
                ),
                "criteria": list(EFFORT_LEVELS),
            },
        }

    @staticmethod
    def _effort(answer: dict) -> tuple[int, float, float]:
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

    async def rank_candidates(
        self, bench: list[BenchmarkEntry], index_field: str, max_tier: int
    ) -> tuple[list[Candidate], dict[str, str]]:
        scored = sorted(
            (b for b in bench if index_field in b.scores),
            key=lambda b: -b.scores[index_field],
        )
        kept: list[Candidate] = []
        priced: list[Candidate] = []
        rejected: dict[str, str] = {}
        seen: set[str] = set()
        for b in scored:
            mid = await self._data.resolve(b.permaslug)
            if mid is None:
                rejected[b.permaslug] = "not routable on OpenRouter"
                continue
            if mid in seen:
                continue
            seen.add(mid)
            if mid.endswith(":free") and not self.config.allow_free:
                rejected[mid] = "free variant"
                continue
            price = await self._data.price(mid) or b.price
            if price is None:
                rejected[mid] = "no OpenRouter price"
                continue
            blended = self.blended(price)
            c = Candidate(
                model_id=mid, quality=b.scores[index_field], price=price,
                blended_per_mtok=blended, tier=self.tier_of(blended),
            )
            priced.append(c)
            if c.tier > max_tier:
                rejected[mid] = f"{COST_TIERS[c.tier]} > {COST_TIERS[max_tier]} cap (${blended:.2f}/M)"
                continue
            kept.append(c)

        if not kept and priced:
            # Every scored model is above the cap. Take the cheapest rather than nothing:
            # the cap is a cost preference, not a reason to fail the request.
            cheapest = min(priced, key=lambda c: c.blended_per_mtok)
            rejected.pop(cheapest.model_id, None)
            kept = [cheapest]
        return kept[: self.config.max_candidates], rejected

    async def select(self, prompt: str) -> TaskRouteDecision:
        bench = await self._data.benchmarks()
        if not bench:
            raise TaskRouteError("OpenRouter returned no benchmark scores")

        answers, cost = await self._decisions.decide(
            state={"user_message": prompt[:_PROMPT_CHARS]},
            questions=self._questions(),
        )

        task = answers.get("task") or {}
        chosen = task.get("choice")
        task_conf = float(task.get("confidence", 0.0))
        trusted = chosen in TASK_TYPES and task_conf >= self.config.task_confidence_floor
        task_key = chosen if trusted else FALLBACK_TASK
        index_field = TASK_TYPES[task_key][1]
        label = index_field if trusted else f"{index_field} (low task confidence on {chosen!r})"

        level, score, eff_conf = self._effort(answers.get("effort") or {})
        max_tier = self.config.effort_to_max_tier[level]
        bumped = eff_conf < self.config.effort_confidence_floor and max_tier < len(COST_TIERS) - 1
        if bumped:
            max_tier += 1

        candidates, rejected = await self.rank_candidates(bench, index_field, max_tier)
        if not candidates:
            raise TaskRouteError(f"no priced, routable candidates scored on {index_field}")

        top = candidates[0]
        reason = (
            f"task={task_key} conf={task_conf:.2f} -> {label}; "
            f"effort={level} score={score:.2f} conf={eff_conf:.2f}"
            f"{' (+1 tier, low confidence)' if bumped else ''}; "
            f"cap={COST_TIERS[max_tier]}; pick={top.model_id} "
            f"({index_field}={top.quality:.1f}, ${top.blended_per_mtok:.2f}/M)"
        )
        return TaskRouteDecision(
            task_tag=task_key,
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

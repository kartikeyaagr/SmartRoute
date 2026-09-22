"""
Model catalog: the single source of truth for which models exist, what they cost,
and which routing tier they serve.

Before this module, model IDs were hardcoded in six places (router._CHEAP_SEQUENCE,
router._FRONTIER_SEQUENCE, verifier._DEFAULT_VERIFIER, providers._MANUAL_PRICING,
bench._COST_PER_1K, bench._FRONTIER) and prices in three of them — which had already
drifted: the hand-written table claimed $0.88/M for Llama-3.3-70B where the registry
says $1.04/M. Swapping Groq -> Together AI took 4 source files and ~45 test assertions.

Now it takes editing models.yaml.

Pricing is resolved once, at load, in this order:
    1. an explicit `price:` block in the YAML (override for a wrong/missing registry entry)
    2. litellm.model_cost[id]
    3. UnpricedModelError

There is deliberately no fourth option. A model whose price we cannot establish is a
model we cannot make a cost-optimal routing decision about, so loading fails loudly
rather than silently recording $0.00 and corrupting the only number this project exists
to report.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Repo-root models.yaml. Overridable via SMARTROUTE_CATALOG_PATH (see config.Settings).
_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "models.yaml"

VALID_ROLES = frozenset({"cheap", "middle", "frontier", "judge"})


class CatalogError(Exception):
    """Raised when the catalog is structurally invalid."""


class UnpricedModelError(CatalogError):
    """Raised when a model's price cannot be resolved from the YAML or the registry."""


@dataclass(frozen=True)
class Price:
    """Per-token prices in USD."""

    input_per_token: float
    output_per_token: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.input_per_token + output_tokens * self.output_per_token

    @property
    def input_per_mtok(self) -> float:
        return self.input_per_token * 1_000_000

    @property
    def output_per_mtok(self) -> float:
        return self.output_per_token * 1_000_000


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    id: str
    role: str
    price: Price
    env_key: str | None = None
    context_window: int | None = None
    # Generation params this model rejects outright. Newer Claude models return
    # `invalid_request_error: temperature is deprecated for this model`, and an
    # unrecognised param is a hard 400, not a warning — so a benchmark that pins
    # temperature for reproducibility fails 100% against them while older models in
    # the same family sail through. Which params a model accepts is a property of the
    # model, so it belongs next to its price rather than in caller code.
    unsupported_params: frozenset[str] = frozenset()

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return self.price.cost(input_tokens, output_tokens)

    def filter_params(self, params: dict) -> dict:
        return {k: v for k, v in params.items() if k not in self.unsupported_params}


def _registry_entry(model_id: str) -> dict | None:
    """Look up a model in litellm's price registry. Imported lazily: litellm costs ~1.6s."""
    import litellm

    return litellm.model_cost.get(model_id)


def _resolve_price(alias: str, model_id: str, override: dict | None) -> Price:
    if override is not None:
        try:
            return Price(
                input_per_token=float(override["input_per_mtok"]) / 1_000_000,
                output_per_token=float(override["output_per_mtok"]) / 1_000_000,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError(
                f"model {alias!r}: `price` override must set input_per_mtok and "
                f"output_per_mtok (got {override!r})"
            ) from exc

    entry = _registry_entry(model_id)
    if entry is None or entry.get("input_cost_per_token") is None:
        raise UnpricedModelError(
            f"model {alias!r} ({model_id}) has no price in litellm's registry. "
            f"Add a `price: {{input_per_mtok: X, output_per_mtok: Y}}` block to models.yaml."
        )
    return Price(
        input_per_token=float(entry["input_cost_per_token"]),
        output_per_token=float(entry.get("output_cost_per_token") or 0.0),
    )


def _resolve_context_window(model_id: str, declared: int | None) -> int | None:
    if declared is not None:
        return int(declared)
    entry = _registry_entry(model_id)
    if entry is None:
        return None
    window = entry.get("max_input_tokens") or entry.get("max_tokens")
    return int(window) if window else None


class Catalog:
    """
    A loaded, validated model catalog.

    `tiers` is ordered cheap -> expensive, which is the order layer 2 considers when
    picking a model for a sub-task, and the order the escalation ladder walks.
    """

    def __init__(
        self,
        models: list[ModelSpec],
        tier_order: list[str],
        default_tier: str,
        lambda_wrong_answer_usd: float = 0.001,
    ) -> None:
        self._by_alias = {m.alias: m for m in models}
        self._by_id = {m.id: m for m in models}
        self._tier_order = tier_order
        self._default_tier = default_tier
        self.lambda_wrong_answer_usd = lambda_wrong_answer_usd

    # -- lookup ------------------------------------------------------------

    def resolve(self, alias_or_id: str) -> ModelSpec:
        """Resolve by alias first, then by full model id."""
        spec = self._by_alias.get(alias_or_id) or self._by_id.get(alias_or_id)
        if spec is None:
            raise CatalogError(
                f"unknown model {alias_or_id!r}. "
                f"Known aliases: {sorted(self._by_alias)}"
            )
        return spec

    def __contains__(self, alias_or_id: str) -> bool:
        return alias_or_id in self._by_alias or alias_or_id in self._by_id

    @property
    def tiers(self) -> list[ModelSpec]:
        """Routing tiers, ordered cheap -> expensive."""
        return [self._by_alias[a] for a in self._tier_order]

    @property
    def default(self) -> ModelSpec:
        """The tier every non-lookup request falls back to. Deliberately not the frontier."""
        return self._by_alias[self._default_tier]

    def by_role(self, role: str) -> ModelSpec:
        matches = [m for m in self._by_alias.values() if m.role == role]
        if not matches:
            raise CatalogError(f"no model with role {role!r} in the catalog")
        if len(matches) > 1:
            raise CatalogError(
                f"{len(matches)} models claim role {role!r}: {[m.alias for m in matches]}. "
                "Roles used for routing must be unique."
            )
        return matches[0]

    def all(self) -> list[ModelSpec]:
        return list(self._by_alias.values())

    # -- preflight ---------------------------------------------------------

    def missing_credentials(self, aliases: list[str] | None = None) -> list[str]:
        """
        Env vars that the given models need but which are not set.

        Deriving this from the catalog is what stops preflight from drifting: today
        harness/run_benchmark.py checks `openai_api_key` and reports `GROQ_API_KEY`.
        """
        import os

        # Importing config is what copies .env values into os.environ (pydantic-settings
        # reads them into model fields only). Relying on the package __init__ to have
        # done this already would make the check silently wrong when it hasn't.
        import smartroute.config  # noqa: F401

        specs = [self.resolve(a) for a in aliases] if aliases else self.all()
        return sorted({s.env_key for s in specs if s.env_key and not os.getenv(s.env_key)})

    # -- provenance --------------------------------------------------------

    def fingerprint(self) -> str:
        """
        Stable hash of the catalog's routing-relevant content.

        Recorded alongside benchmark results and cassettes: a run made against a
        different catalog is not comparable, and must not silently mix with this one.
        """
        import hashlib

        payload = "|".join(
            f"{m.alias}={m.id}@{m.price.input_per_token}/{m.price.output_per_token}"
            for m in sorted(self._by_alias.values(), key=lambda m: m.alias)
        )
        payload += f"||tiers={','.join(self._tier_order)}|default={self._default_tier}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _model_family(model_id: str) -> str:
    """
    The vendor/family token of a model id, used for the anti-self-grading check.

    together_ai/openai/gpt-oss-20b     -> openai
    together_ai/Qwen/Qwen3.5-397B-A17B -> qwen
    groq/llama-3.1-8b-instant          -> llama
    groq/gemma-7b-it                   -> gemma

    Providers with a flat namespace (groq/llama-3.1-8b-instant) carry the family in
    the model name itself, so fall back to its leading alphabetic token rather than
    the whole string — otherwise two models from the same family would never compare
    equal and the self-grading check would silently pass everything.
    """
    parts = model_id.split("/")
    segment = (parts[1] if len(parts) > 2 else parts[-1]).lower()
    leading = re.match(r"[a-z]+", segment)
    return leading.group(0) if leading else segment


def _check_judge_is_cross_family(path: Path, by_alias: dict[str, ModelSpec]) -> None:
    """
    A judge from the same family as the model it scores is grading its own homework.

    Cross-family verification is a locked-in design rule; enforcing it here means a
    catalog edit cannot quietly violate it.
    """
    judges = [m for m in by_alias.values() if m.role == "judge"]
    graded = [m for m in by_alias.values() if m.role in ("cheap", "middle")]
    for judge in judges:
        for target in graded:
            if _model_family(judge.id) == _model_family(target.id):
                raise CatalogError(
                    f"{path}: judge {judge.alias!r} ({judge.id}) is the same family as "
                    f"{target.alias!r} ({target.id}) — that is self-grading. "
                    "Pick a judge from a different vendor."
                )


def load_catalog(path: str | Path | None = None) -> Catalog:
    """Load and validate models.yaml. Raises CatalogError on any structural problem."""
    path = Path(path) if path else _DEFAULT_CATALOG_PATH
    if not path.exists():
        raise CatalogError(f"catalog not found at {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("models")
    if not entries:
        raise CatalogError(f"{path}: no `models` declared")

    models: list[ModelSpec] = []
    seen: set[str] = set()
    for entry in entries:
        alias, model_id = entry.get("alias"), entry.get("id")
        if not alias or not model_id:
            raise CatalogError(f"{path}: every model needs both `alias` and `id` (got {entry!r})")
        if alias in seen:
            raise CatalogError(f"{path}: duplicate alias {alias!r}")
        seen.add(alias)

        role = entry.get("role", "")
        if role not in VALID_ROLES:
            raise CatalogError(
                f"{path}: model {alias!r} has role {role!r}; expected one of {sorted(VALID_ROLES)}"
            )

        models.append(
            ModelSpec(
                alias=alias,
                id=model_id,
                role=role,
                price=_resolve_price(alias, model_id, entry.get("price")),
                env_key=entry.get("env_key"),
                context_window=_resolve_context_window(model_id, entry.get("context_window")),
                unsupported_params=frozenset(entry.get("unsupported_params") or ()),
            )
        )

    routing = raw.get("routing") or {}
    tier_order = routing.get("tiers") or [m.alias for m in models if m.role in ("cheap", "middle", "frontier")]
    default_tier = routing.get("default_tier") or (tier_order[len(tier_order) // 2] if tier_order else None)

    known = {m.alias for m in models}
    for alias in tier_order:
        if alias not in known:
            raise CatalogError(f"{path}: routing.tiers references unknown alias {alias!r}")
    if default_tier not in known:
        raise CatalogError(f"{path}: routing.default_tier references unknown alias {default_tier!r}")

    # A ladder must actually ascend, or "escalate" and "pick the cheapest that works"
    # are meaningless. Catching this here beats discovering it in a benchmark report.
    by_alias = {m.alias: m for m in models}
    prices = [(a, by_alias[a].price.input_per_token) for a in tier_order]
    if prices != sorted(prices, key=lambda p: p[1]):
        raise CatalogError(
            f"{path}: routing.tiers must be ordered cheap -> expensive, got "
            + " < ".join(f"{a}(${p * 1e6:.2f}/M)" for a, p in prices)
        )

    _check_judge_is_cross_family(path, by_alias)

    decision = raw.get("decision") or {}
    lambda_wrong = float(decision.get("lambda_wrong_answer_usd", 0.001))
    if lambda_wrong < 0:
        raise CatalogError(f"{path}: decision.lambda_wrong_answer_usd must be >= 0")

    catalog = Catalog(models, tier_order, default_tier, lambda_wrong)
    logger.debug("catalog loaded: %s (fingerprint %s)", tier_order, catalog.fingerprint())
    return catalog


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_catalog: Catalog | None = None


def get_catalog(path: str | Path | None = None) -> Catalog:
    """
    The process-wide catalog, loaded on first use.

    Lazy rather than module-level so that importing smartroute doesn't read a file
    and pull in litellm, and so tests can point at a fixture catalog via set_catalog().
    """
    global _catalog
    if _catalog is None:
        from smartroute.config import settings

        _catalog = load_catalog(path or settings.catalog_path or None)
    return _catalog


def set_catalog(catalog: Catalog | None) -> None:
    """Install a catalog (or None to reset). Test seam."""
    global _catalog
    _catalog = catalog

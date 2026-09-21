"""Tests for the model catalog — the single source of truth for models and prices."""

import textwrap

import pytest

from smartroute.catalog import (
    Catalog,
    CatalogError,
    ModelSpec,
    Price,
    UnpricedModelError,
    load_catalog,
)


def write_catalog(tmp_path, body: str):
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent(body))
    return p


# Two real registry models with a known price ordering, used wherever a test needs
# a valid catalog but isn't testing pricing itself.
CHEAP_ID = "together_ai/openai/gpt-oss-20b"        # $0.05/M in
MID_ID = "together_ai/openai/gpt-oss-120b"         # $0.15/M in
FRONTIER_ID = "together_ai/Qwen/Qwen3.5-397B-A17B"  # $0.60/M in

VALID = f"""
    version: 1
    models:
      - {{alias: cheap,    id: {CHEAP_ID},    role: cheap,    env_key: TOGETHERAI_API_KEY}}
      - {{alias: middle,   id: {MID_ID},      role: middle,   env_key: TOGETHERAI_API_KEY}}
      - {{alias: frontier, id: {FRONTIER_ID}, role: frontier, env_key: TOGETHERAI_API_KEY}}
    routing:
      tiers: [cheap, middle, frontier]
      default_tier: middle
"""


class TestPrice:
    def test_cost_is_linear_in_tokens(self):
        price = Price(input_per_token=1e-6, output_per_token=2e-6)
        assert price.cost(1000, 500) == pytest.approx(1e-3 + 1e-3)

    def test_per_mtok_helpers(self):
        price = Price(input_per_token=5e-8, output_per_token=2e-7)
        assert price.input_per_mtok == pytest.approx(0.05)
        assert price.output_per_mtok == pytest.approx(0.20)


class TestPriceResolution:
    def test_resolves_from_litellm_registry(self, tmp_path):
        catalog = load_catalog(write_catalog(tmp_path, VALID))
        # gpt-oss-20b is $0.05/M input in litellm's registry
        assert catalog.resolve("cheap").price.input_per_mtok == pytest.approx(0.05)

    def test_yaml_price_overrides_registry(self, tmp_path):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - alias: cheap
                id: {CHEAP_ID}
                role: cheap
                price: {{input_per_mtok: 9.99, output_per_mtok: 1.11}}
            routing: {{tiers: [cheap], default_tier: cheap}}
            """,
        )
        price = load_catalog(path).resolve("cheap").price
        assert price.input_per_mtok == pytest.approx(9.99)
        assert price.output_per_mtok == pytest.approx(1.11)

    def test_unpriced_model_raises_rather_than_recording_zero(self, tmp_path):
        """The whole point: an unknown model must fail loudly, never cost $0.00."""
        path = write_catalog(
            tmp_path,
            """
            version: 1
            models:
              - {alias: ghost, id: nonexistent/not-a-real-model, role: cheap}
            routing: {tiers: [ghost], default_tier: ghost}
            """,
        )
        with pytest.raises(UnpricedModelError, match="no price in litellm's registry"):
            load_catalog(path)

    def test_malformed_price_override_raises(self, tmp_path):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: cheap, id: {CHEAP_ID}, role: cheap, price: {{input_per_mtok: 1.0}}}}
            routing: {{tiers: [cheap], default_tier: cheap}}
            """,
        )
        with pytest.raises(CatalogError, match="input_per_mtok and output_per_mtok"):
            load_catalog(path)


class TestValidation:
    def test_tiers_must_ascend_in_price(self, tmp_path):
        """A ladder that doesn't ascend makes 'escalate' meaningless."""
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: cheap,    id: {CHEAP_ID},    role: cheap}}
              - {{alias: frontier, id: {FRONTIER_ID}, role: frontier}}
            routing: {{tiers: [frontier, cheap], default_tier: cheap}}
            """,
        )
        with pytest.raises(CatalogError, match="ordered cheap -> expensive"):
            load_catalog(path)

    def test_duplicate_alias_raises(self, tmp_path):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: cheap, id: {CHEAP_ID}, role: cheap}}
              - {{alias: cheap, id: {MID_ID},   role: middle}}
            routing: {{tiers: [cheap], default_tier: cheap}}
            """,
        )
        with pytest.raises(CatalogError, match="duplicate alias"):
            load_catalog(path)

    def test_unknown_role_raises(self, tmp_path):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: cheap, id: {CHEAP_ID}, role: banana}}
            routing: {{tiers: [cheap], default_tier: cheap}}
            """,
        )
        with pytest.raises(CatalogError, match="role 'banana'"):
            load_catalog(path)

    def test_tiers_referencing_unknown_alias_raises(self, tmp_path):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: cheap, id: {CHEAP_ID}, role: cheap}}
            routing: {{tiers: [cheap, nope], default_tier: cheap}}
            """,
        )
        with pytest.raises(CatalogError, match="unknown alias 'nope'"):
            load_catalog(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(CatalogError, match="catalog not found"):
            load_catalog(tmp_path / "absent.yaml")

    def test_empty_models_raises(self, tmp_path):
        path = write_catalog(tmp_path, "version: 1\nmodels: []\n")
        with pytest.raises(CatalogError, match="no `models` declared"):
            load_catalog(path)


class TestLookup:
    def test_resolve_by_alias_and_by_id(self, tmp_path):
        catalog = load_catalog(write_catalog(tmp_path, VALID))
        assert catalog.resolve("cheap").id == CHEAP_ID
        assert catalog.resolve(CHEAP_ID).alias == "cheap"

    def test_resolve_unknown_raises_with_known_aliases(self, tmp_path):
        catalog = load_catalog(write_catalog(tmp_path, VALID))
        with pytest.raises(CatalogError, match="Known aliases"):
            catalog.resolve("nope")

    def test_tiers_are_ordered_cheap_to_expensive(self, tmp_path):
        catalog = load_catalog(write_catalog(tmp_path, VALID))
        assert [m.alias for m in catalog.tiers] == ["cheap", "middle", "frontier"]
        prices = [m.price.input_per_token for m in catalog.tiers]
        assert prices == sorted(prices)

    def test_default_tier_is_middle_not_frontier(self, tmp_path):
        """The architecture's core premise: the frontier is never a default."""
        catalog = load_catalog(write_catalog(tmp_path, VALID))
        assert catalog.default.alias == "middle"
        assert catalog.default.role == "middle"

    def test_by_role_rejects_ambiguity(self, tmp_path):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: a, id: {CHEAP_ID}, role: cheap}}
              - {{alias: b, id: {MID_ID},   role: cheap}}
            routing: {{tiers: [a, b], default_tier: a}}
            """,
        )
        catalog = load_catalog(path)
        with pytest.raises(CatalogError, match="claim role 'cheap'"):
            catalog.by_role("cheap")

    def test_contains(self, tmp_path):
        catalog = load_catalog(write_catalog(tmp_path, VALID))
        assert "cheap" in catalog and CHEAP_ID in catalog and "nope" not in catalog


class TestPreflightAndProvenance:
    def test_missing_credentials_derived_from_catalog(self, tmp_path, monkeypatch):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: cheap, id: {CHEAP_ID}, role: cheap, env_key: SOME_MISSING_KEY}}
            routing: {{tiers: [cheap], default_tier: cheap}}
            """,
        )
        monkeypatch.delenv("SOME_MISSING_KEY", raising=False)
        assert load_catalog(path).missing_credentials() == ["SOME_MISSING_KEY"]

    def test_missing_credentials_empty_when_set(self, tmp_path, monkeypatch):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: cheap, id: {CHEAP_ID}, role: cheap, env_key: SOME_PRESENT_KEY}}
            routing: {{tiers: [cheap], default_tier: cheap}}
            """,
        )
        monkeypatch.setenv("SOME_PRESENT_KEY", "x")
        assert load_catalog(path).missing_credentials() == []

    def test_fingerprint_is_stable_and_price_sensitive(self, tmp_path):
        (tmp_path / "one").mkdir()
        (tmp_path / "two").mkdir()
        first = load_catalog(write_catalog(tmp_path / "one", VALID))
        same = load_catalog(write_catalog(tmp_path / "two", VALID))
        assert first.fingerprint() == same.fingerprint()

        (tmp_path / "three").mkdir()
        different = load_catalog(
            write_catalog(
                tmp_path / "three",
                f"""
                version: 1
                models:
                  - {{alias: cheap,  id: {CHEAP_ID}, role: cheap}}
                  - {{alias: middle, id: {MID_ID},   role: middle,
                      price: {{input_per_mtok: 99.0, output_per_mtok: 99.0}}}}
                routing: {{tiers: [cheap, middle], default_tier: middle}}
                """,
            )
        )
        assert different.fingerprint() != first.fingerprint()


class TestShippedCatalog:
    """The real models.yaml must be loadable and well-formed."""

    def test_repo_catalog_loads(self):
        catalog = load_catalog()
        assert [m.alias for m in catalog.tiers] == ["cheap", "middle", "frontier"]
        assert catalog.default.alias == "middle"

    def test_repo_catalog_every_model_is_priced(self):
        for spec in load_catalog().all():
            assert spec.price.input_per_token > 0, f"{spec.alias} has no input price"


class TestAntiSelfGrading:
    """A judge from the same family as the model it scores grades its own homework."""

    def test_same_family_judge_is_rejected(self, tmp_path):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: cheap, id: {CHEAP_ID}, role: cheap}}
              - {{alias: judge, id: {MID_ID},   role: judge}}
            routing: {{tiers: [cheap], default_tier: cheap}}
            """,
        )
        # both are together_ai/openai/* -> same family
        with pytest.raises(CatalogError, match="self-grading"):
            load_catalog(path)

    def test_cross_family_judge_is_accepted(self, tmp_path):
        path = write_catalog(
            tmp_path,
            f"""
            version: 1
            models:
              - {{alias: cheap, id: {CHEAP_ID}, role: cheap}}
              - {{alias: judge, id: together_ai/mistralai/Mistral-Small-24B-Instruct-2501, role: judge}}
            routing: {{tiers: [cheap], default_tier: cheap}}
            """,
        )
        assert load_catalog(path).by_role("judge").alias == "judge"

    def test_shipped_catalog_judge_is_cross_family(self):
        from smartroute.catalog import _model_family

        catalog = load_catalog()
        judge_family = _model_family(catalog.by_role("judge").id)
        assert judge_family != _model_family(catalog.by_role("cheap").id)
        assert judge_family != _model_family(catalog.by_role("middle").id)


class TestProviderSwapAcceptance:
    """
    The acceptance criterion for "model agnostic".

    Swapping every model to a different provider must require editing models.yaml and
    nothing else. Before the catalog, the Groq -> Together AI migration touched 4 source
    files and ~45 test assertions; this test is what stops that from recurring.
    """

    GROQ = """
        version: 1
        models:
          - {alias: cheap,    id: groq/llama-3.1-8b-instant,    role: cheap,    env_key: GROQ_API_KEY}
          - {alias: middle,   id: groq/qwen/qwen3-32b,          role: middle,   env_key: GROQ_API_KEY}
          - {alias: frontier, id: groq/llama-3.3-70b-versatile, role: frontier, env_key: GROQ_API_KEY}
          - {alias: judge,    id: groq/gemma-7b-it,             role: judge,    env_key: GROQ_API_KEY}
        routing:
          tiers: [cheap, middle, frontier]
          default_tier: middle
    """

    def test_wholesale_provider_swap_loads(self, tmp_path):
        catalog = load_catalog(write_catalog(tmp_path, self.GROQ))
        assert [m.id for m in catalog.tiers] == [
            "groq/llama-3.1-8b-instant",
            "groq/qwen/qwen3-32b",
            "groq/llama-3.3-70b-versatile",
        ]
        assert catalog.default.alias == "middle"
        assert all(m.price.input_per_token > 0 for m in catalog.all())

    async def test_router_dispatches_through_a_swapped_catalog(self, tmp_path):
        """The Router must pick up swapped models without touching src/."""
        from unittest.mock import patch

        from smartroute.decision import DecisionLayer, TriageSignals
        from smartroute.providers import ModelResponse
        from smartroute.router import Router

        catalog = load_catalog(write_catalog(tmp_path, self.GROQ))

        class _AlwaysLookup:
            def signals(self, prompt):
                return TriageSignals(p_lookup=0.99, p_decompose=0.0, backend="test")

            def backend(self):
                return "test"

        router = Router(
            catalog=catalog,
            decision_layer=DecisionLayer(catalog=catalog, triage=_AlwaysLookup()),
        )
        resp = ModelResponse(
            model="x", content="ok", input_tokens=10, output_tokens=5,
            estimated_cost_usd=0.0, latency_ms=1.0,
        )
        with patch("smartroute.router.call_model", return_value=resp):
            _, decision = await router.route_async([{"role": "user", "content": "hi"}])

        # resolved from the swapped catalog, with no source edit anywhere
        assert decision.final_model == "groq/llama-3.1-8b-instant"
        assert decision.route_path == "cheap"

    async def test_default_tier_follows_the_swapped_catalog(self):
        """The default destination is whatever the catalog says, not a constant."""
        import textwrap
        from pathlib import Path
        from unittest.mock import patch

        from smartroute.decision import DecisionLayer, TriageSignals
        from smartroute.providers import ModelResponse
        from smartroute.router import Router

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.yaml"
            path.write_text(textwrap.dedent(self.GROQ))
            catalog = load_catalog(path)

            class _Unsure:
                def signals(self, prompt):
                    return TriageSignals(p_lookup=0.1, p_decompose=0.1, backend="test")

                def backend(self):
                    return "test"

            router = Router(
                catalog=catalog,
                decision_layer=DecisionLayer(catalog=catalog, triage=_Unsure()),
            )
            resp = ModelResponse(
                model="x", content="ok", input_tokens=10, output_tokens=5,
                estimated_cost_usd=0.0, latency_ms=1.0,
            )
            with patch("smartroute.router.call_model", return_value=resp):
                _, decision = await router.route_async([{"role": "user", "content": "hi"}])

            assert decision.final_model == "groq/qwen/qwen3-32b"  # the swapped middle tier
            assert decision.route_path == "middle"

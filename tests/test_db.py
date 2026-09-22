"""
Unit tests for db.py — no real Postgres required.
Tests that insert_decision is a no-op when pool is None, and that
the server lifespan skips DB init when DATABASE_URL is empty.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from smartroute.router import RoutingDecision
import smartroute.db as db_module


def _decision():
    return RoutingDecision(
        request_id="test-001",
        prompt_hash="abc",
        difficulty_tier="EASY",
        difficulty_score=0.8,
        final_model="groq/llama-3.1-8b-instant",
        classifier_backend="keyword",
    )


@pytest.mark.asyncio
async def test_insert_noop_when_no_pool():
    """insert_decision should silently do nothing when pool is None."""
    original = db_module._pool
    db_module._pool = None
    try:
        await db_module.insert_decision(_decision())  # must not raise
    finally:
        db_module._pool = original


@pytest.mark.asyncio
async def test_insert_logs_error_on_db_failure():
    """insert_decision logs the error but never raises — DB failures don't kill the API."""
    mock_conn = AsyncMock()
    mock_conn.execute.side_effect = Exception("connection refused")
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    original = db_module._pool
    db_module._pool = mock_pool
    try:
        await db_module.insert_decision(_decision())  # must not raise
    finally:
        db_module._pool = original


@pytest.mark.asyncio
async def test_close_pool_noop_when_none():
    """close_pool should be safe to call even when pool was never initialised."""
    original = db_module._pool
    db_module._pool = None
    try:
        await db_module.close_pool()  # must not raise
    finally:
        db_module._pool = original


@pytest.mark.asyncio
async def test_server_skips_db_init_when_no_url(monkeypatch):
    """Lifespan should not call init_pool when DATABASE_URL is empty."""
    from smartroute import server as srv
    monkeypatch.setattr(srv.settings, "database_url", "")

    with patch.object(db_module, "init_pool", new=AsyncMock()) as mock_init:
        with patch.object(db_module, "close_pool", new=AsyncMock()):
            async with srv.lifespan(srv.app):
                pass
        mock_init.assert_not_called()


class TestSchemaInvariants:
    """
    insert_decision swallows exceptions so a DB failure never kills a response — which
    means a schema mistake produces one log line and total silent data loss. These
    tests are the only thing standing between a field rename and that outcome.
    """

    def test_insert_placeholders_match_column_count(self):
        import re

        from smartroute.db import _INSERT

        columns = [c.strip() for c in _INSERT.split("(", 1)[1].split(")", 1)[0].split(",")]
        placeholders = {int(p[1:]) for p in re.findall(r"\$\d+", _INSERT)}
        assert len(columns) == len(placeholders)
        assert placeholders == set(range(1, len(columns) + 1))

    def test_every_inserted_column_exists_in_create_table(self):
        from smartroute.db import _CREATE_TABLE, _INSERT

        columns = [c.strip() for c in _INSERT.split("(", 1)[1].split(")", 1)[0].split(",")]
        for column in columns:
            assert column in _CREATE_TABLE, f"{column} is inserted but not declared"

    def test_new_decision_fields_have_migrations(self):
        """
        CREATE TABLE IF NOT EXISTS does nothing to an existing table, so any column
        added after the first deploy needs an ALTER as well.
        """
        from smartroute.db import _MIGRATIONS

        migrated = " ".join(_MIGRATIONS)
        for column in (
            "route_path", "gate_reason", "p_lookup",
            "p_decompose", "projected_cost_usd", "subtask_count",
        ):
            assert column in migrated, f"{column} has no migration"

    def test_migrations_are_idempotent(self):
        from smartroute.db import _MIGRATIONS

        assert all("IF NOT EXISTS" in stmt for stmt in _MIGRATIONS)

    async def test_decision_dataclass_and_insert_agree(self):
        """A field added to RoutingDecision but not to the INSERT is silently dropped."""
        from dataclasses import fields

        from smartroute.db import _INSERT
        from smartroute.router import RoutingDecision

        columns = {c.strip() for c in _INSERT.split("(", 1)[1].split(")", 1)[0].split(",")}
        # Fields deliberately not persisted (large, or benchmark-only).
        not_persisted = {"cheap_response", "verifier_correct", "extraction_failed"}
        for f in fields(RoutingDecision):
            if f.name not in not_persisted:
                assert f.name in columns, f"RoutingDecision.{f.name} is never persisted"

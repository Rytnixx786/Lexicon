from __future__ import annotations

import os

import pytest

from lexicon.settings import eval_dsn

ALLOWED = {
    "marts.fct_outcomes",
    "marts.fct_requests",
    "marts.dim_tenant",
    "marts.dim_prompt_version",
    "marts.mart_cost_per_outcome",
    "support.tickets",
    "support.ticket_events",
    "support.agents",
    "support.sla_policies",
}


def eval_db() -> str:
    """The frozen snapshot DB, or skip (integration tests need `docker compose up -d` + `lexicon snapshot`)."""
    dsn = os.environ.get("LEXICON_TEST_DSN") or eval_dsn()
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2) as c:
            c.execute("select 1 from support.tickets limit 1")
    except Exception:
        pytest.skip("eval database not available")
    return dsn

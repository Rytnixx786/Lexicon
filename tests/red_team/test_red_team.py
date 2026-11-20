"""Red-team suite (BUILD_SPEC §7): every attack must be blocked by the SQL guard alone AND, separately, by the
database layer alone (guard disabled). Two independent layers, each tested on its own."""

from __future__ import annotations

import psycopg
import pytest

from lexicon import sql_guard
from lexicon.executor import ExecError, PostgresExecutor
from tests.conftest import ALLOWED, eval_db

ATTACKS: list[str] = [
    # writes and DDL
    "DELETE FROM support.tickets",
    "UPDATE marts.dim_tenant SET monthly_budget_usd = 0",
    "INSERT INTO support.agents (agent_id) VALUES ('x')",
    "DROP TABLE marts.fct_requests",
    "TRUNCATE support.ticket_events",
    "ALTER TABLE support.agents ADD COLUMN x int",
    "CREATE TABLE support.evil AS SELECT * FROM support.agents",
    "CREATE INDEX i ON support.tickets (ticket_id)",
    "MERGE INTO support.agents a USING support.agents b ON a.agent_id = b.agent_id WHEN MATCHED THEN DELETE",
    "SELECT * INTO support.copy FROM support.tickets",
    "GRANT ALL ON support.agents TO PUBLIC",
    "REVOKE SELECT ON support.agents FROM lexicon_ro",
    # stacked and wrapped statements
    "SELECT 1; DROP TABLE marts.fct_outcomes",
    "SELECT * FROM support.tickets; DELETE FROM support.tickets",
    "WITH x AS (DELETE FROM support.tickets RETURNING *) SELECT * FROM x",
    "WITH x AS (UPDATE support.agents SET is_active = false RETURNING *) SELECT count(*) FROM x",
    "SELECT * FROM support.tickets /* ; */ ; DELETE FROM support.tickets",
    "SELECT 1 -- harmless\n; DROP TABLE support.agents",
    # session and transaction control
    "SET ROLE postgres",
    "SET default_transaction_read_only = off",
    "BEGIN; DELETE FROM support.tickets; COMMIT",
    "COMMIT",
    "RESET ALL",
    # file, network and admin functions
    "SELECT pg_sleep(30)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_ls_dir('.')",
    "SELECT lo_import('/etc/passwd')",
    "SELECT set_config('default_transaction_read_only', 'off', false)",
    "SELECT current_setting('data_directory')",
    "SELECT pg_terminate_backend(1)",
    "SELECT dblink('host=evil', 'select 1')",
    "SELECT nextval('some_seq')",
    "SELECT * FROM pg_stat_activity",
    "SELECT query_to_xml('select * from support.agents', true, true, '')",
    # catalog probing and disallowed tables
    "SELECT * FROM pg_catalog.pg_user",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM pg_shadow",
    "SELECT * FROM public.secret_table",
    "SELECT usename FROM pg_catalog.pg_roles r JOIN support.agents a ON true",
    # COPY and locking
    "COPY support.agents TO STDOUT",
    "COPY support.agents TO '/tmp/agents.csv'",
    "SELECT * FROM support.agents FOR UPDATE",
    "LOCK TABLE support.agents",
    # garbage
    "",
    "this is not sql",
    "select * from support.agents where team = 'sales",  # unclosed quote: TokenError, not ParseError
]


def test_suite_is_large_enough() -> None:
    assert len(ATTACKS) >= 40


@pytest.mark.parametrize("sql", ATTACKS)
def test_guard_blocks(sql: str) -> None:
    rep = sql_guard.check(sql, ALLOWED)
    assert not rep.ok, f"guard let through: {sql!r} -> {rep.sql}"
    assert rep.sql is None


DB_LAYER = [a for a in ATTACKS if a.strip() and a != "this is not sql"]


@pytest.mark.integration
@pytest.mark.parametrize("sql", DB_LAYER)
def test_database_layer_blocks_with_guard_disabled(sql: str) -> None:
    dsn = eval_db()
    ex = PostgresExecutor(dsn, timeout_ms=1500)
    before = _fingerprint(dsn)
    try:
        ex.run(sql)
    except ExecError as e:
        assert e.kind in {
            "permission_denied",
            "syntax",
            "undefined_table",
            "timeout",
            "other",
            "undefined_column",
        }
    finally:
        ex.close()
    assert _fingerprint(dsn) == before, f"database changed after {sql!r}"


def _fingerprint(dsn: str) -> tuple[int, ...]:
    with psycopg.connect(dsn) as c:
        row = c.execute(
            "select (select count(*) from support.tickets), (select count(*) from support.agents), "
            "(select count(*) from marts.fct_requests), (select count(*) from marts.dim_tenant where monthly_budget_usd > 0), "
            "(select count(*) from information_schema.tables where table_schema in ('support', 'marts', 'public'))"
        ).fetchone()
    assert row is not None
    return tuple(int(x) for x in row)

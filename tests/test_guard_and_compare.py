from __future__ import annotations

import datetime as dt

import pytest
import sqlglot
from hypothesis import given, settings
from hypothesis import strategies as st

from lexicon import sql_guard
from lexicon.eval.compare import execution_match, has_outer_order_by, result_hash
from lexicon.executor import ResultFrame
from tests.conftest import ALLOWED

COLS = {
    "marts.fct_outcomes": {
        "setup": "text",
        "is_correct": "boolean",
        "split": "text",
        "total_cost_usd": "numeric",
    },
    "support.tickets": {"ticket_id": "text", "tenant_id": "text", "status": "text"},
}


# ---------------------------------------------------------------- guard


def test_limit_added_and_clamped() -> None:
    r = sql_guard.check("select setup from marts.fct_outcomes", ALLOWED)
    assert r.ok and r.limit_added and r.sql.endswith("LIMIT 1000")
    r = sql_guard.check("select setup from marts.fct_outcomes limit 99999", ALLOWED)
    assert r.ok and "LIMIT 1000" in r.sql and not r.limit_added
    r = sql_guard.check("select setup from marts.fct_outcomes limit 5", ALLOWED)
    assert "LIMIT 5" in r.sql


def test_unqualified_tables_resolve_and_cte_aliases_are_ignored() -> None:
    r = sql_guard.check("with x as (select * from tickets) select count(*) from x", ALLOWED)
    assert r.ok and "support.tickets" in r.sql and r.tables == ["support.tickets"]


def test_unknown_or_ambiguous_tables_rejected() -> None:
    assert sql_guard.check("select * from nope", ALLOWED).codes == ["table_not_allowed"]
    both = ALLOWED | {"other.tickets"}
    assert "ambiguous" in sql_guard.check("select * from tickets", both).violations[0].message


def test_undefined_column_detected_before_the_database() -> None:
    r = sql_guard.check("select tenant_id from marts.fct_outcomes", ALLOWED, columns=COLS)
    assert not r.ok and r.codes == ["undefined_column"]
    assert sql_guard.check("select setup from marts.fct_outcomes", ALLOWED, columns=COLS).ok


def test_set_operations_and_subquery_roots_allowed() -> None:
    assert sql_guard.check("select 1 union all select 2", ALLOWED).ok
    assert sql_guard.check("(select setup from marts.fct_outcomes)", ALLOWED).ok


def test_sqlite_dialect_for_bird() -> None:
    r = sql_guard.check("select `name` from frpm", {"main.frpm"}, dialect="sqlite")
    assert r.ok and "main.frpm" in r.sql


def test_table_functions_and_qualify_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlglot.errors import OptimizeError

    r = sql_guard.check("select * from generate_series(1, 5)", ALLOWED)
    assert not r.ok and r.codes == ["forbidden_function"]

    monkeypatch.setattr(
        sql_guard, "qualify", lambda *a, **kw: (_ for _ in ()).throw(OptimizeError("non-col warning"))
    )
    assert sql_guard.check("select setup from marts.fct_outcomes", ALLOWED, columns=COLS).ok

    monkeypatch.setattr(sql_guard, "qualify", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert sql_guard.check("select setup from marts.fct_outcomes", ALLOWED, columns=COLS).ok


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.sampled_from(["setup", "split", "is_correct", "total_cost_usd"]),
        min_size=1,
        max_size=4,
        unique=True,
    ),
    st.sampled_from(["", " where split = 'test'", " where is_correct"]),
    st.integers(1, 5000),
)
def test_anything_accepted_is_one_select_with_a_bounded_limit(
    cols: list[str], where: str, limit: int
) -> None:
    r = sql_guard.check(
        f"select {', '.join(cols)} from marts.fct_outcomes{where} limit {limit}", ALLOWED, columns=COLS
    )
    assert r.ok
    parsed = sqlglot.parse(r.sql, dialect="postgres")
    assert len(parsed) == 1 and isinstance(parsed[0], sqlglot.exp.Select)
    assert int(parsed[0].args["limit"].expression.this) <= sql_guard.MAX_ROWS


# ---------------------------------------------------------------- comparison


def f(cols: list[str], rows: list[tuple]) -> ResultFrame:  # type: ignore[type-arg]
    return ResultFrame(cols, rows)


def test_column_and_row_order_do_not_matter_without_order_by() -> None:
    gold = f(["a", "b"], [("x", 1.0), ("y", 2.0)])
    pred = f(["n", "s"], [(2.0, "y"), (1.0000000001, "x")])
    assert execution_match(gold, pred, "select a, b from t") == {"strict": True, "lenient": True}


def test_row_order_matters_with_outer_order_by() -> None:
    gold, pred = f(["a"], [(1,), (2,)]), f(["a"], [(2,), (1,)])
    assert not execution_match(gold, pred, "select a from t order by a")["strict"]
    assert has_outer_order_by("select a from t order by a")
    assert not has_outer_order_by("select a from t order by a limit 1")  # top-1: order is irrelevant


def test_extra_columns_only_pass_lenient() -> None:
    gold, pred = f(["s"], [("support",)]), f(["s", "budget"], [("support", 50)])
    assert execution_match(gold, pred, "select s from t") == {"strict": False, "lenient": True}


def test_nulls_decimals_and_dates() -> None:
    from decimal import Decimal

    gold = f(["d", "v"], [(dt.date(2026, 1, 1), None), (dt.date(2026, 1, 2), Decimal("0.5"))])
    pred = f(["d", "v"], [(dt.date(2026, 1, 2), 0.5), (dt.date(2026, 1, 1), None)])
    assert execution_match(gold, pred, "select d, v from t")["strict"]
    assert not execution_match(gold, f(["d", "v"], [(dt.date(2026, 1, 1), None)]), "select d, v from t")[
        "strict"
    ]


def test_result_hash_is_order_independent() -> None:
    assert result_hash(f(["a"], [(1,), (2,)])) == result_hash(f(["a"], [(2,), (1,)]))


@pytest.mark.parametrize(
    ("a", "b", "same"), [(0.1234567, 0.12345671, True), (1.0, 1.01, False), (0, 0.0, True)]
)
def test_float_tolerance(a: float, b: float, same: bool) -> None:
    assert execution_match(f(["x"], [(a,)]), f(["x"], [(b,)]), "select x")["strict"] is same

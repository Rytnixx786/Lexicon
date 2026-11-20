"""Context layer and golden-set integrity, including the leakage guarantee (AGENTS.md)."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import sqlglot
from pydantic import ValidationError
from rapidfuzz import fuzz

from lexicon.context import loader
from lexicon.context.render import render
from lexicon.context.schema import Context, Metric
from lexicon.eval import golden
from lexicon.retrieval import Retriever
from tests.conftest import eval_db

CTX = loader.load()
ITEMS = golden.load()
SNAPSHOTS = Path(__file__).parent / "context_snapshots"


def test_context_loads_and_references_resolve() -> None:
    assert len(CTX.tables) == 9 and len(CTX.rules) == 10 and len(CTX.examples) >= 45
    assert all(t.columns for t in CTX.tables.values())


def test_unknown_rule_reference_is_rejected() -> None:
    data = CTX.model_dump()
    data["tables"]["support.tickets"]["rule_refs"] = ["R-DOES-NOT-EXIST"]
    with pytest.raises(ValidationError, match="unknown rule"):
        Context.model_validate(data)


def test_ambiguous_metric_needs_choices() -> None:
    with pytest.raises(ValidationError, match="requires_choice"):
        Metric(
            id="m",
            label="m",
            type="ratio",
            base_table="support.tickets",
            description="d",
            requires_choice=True,
        )


def test_pii_column_is_never_rendered_into_samples_or_retrieval_text() -> None:
    from lexicon.retrieval import table_text

    assert "email" not in CTX.samples.get("support.agents", {})
    assert "@example.com" not in table_text(CTX.tables["support.agents"])


def test_marts_descriptions_match_tally_dbt_contract() -> None:
    p = Path("../Tally/dbt/models/marts/_marts.yml")
    if not p.exists():
        pytest.skip("Tally repo not next to Lexicon")
    assert loader.check_against_dbt(CTX, p) == []


@pytest.mark.integration
def test_context_matches_snapshot_columns() -> None:
    assert loader.check_against_db(CTX, eval_db()) == []


# ---------------------------------------------------------------- rendering by level


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_render_snapshots(level: str) -> None:
    r = Retriever(CTX).retrieve("What is the cost per correct triage of each setup on the test split?")
    text = render(CTX, r, level)  # type: ignore[arg-type]
    snap = SNAPSHOTS / f"cpc_question_{level}.txt"
    if not snap.exists():  # first run writes the snapshot; review it like any other diff
        snap.parent.mkdir(exist_ok=True)
        snap.write_text(text, encoding="utf-8")
    assert text == snap.read_text(encoding="utf-8")


def test_levels_add_content_monotonically() -> None:
    r = Retriever(CTX).retrieve("How many tickets have been resolved?")
    texts = {lv: render(CTX, r, lv) for lv in ("L0", "L1", "L2", "L3")}  # type: ignore[misc]
    assert "CREATE TABLE" in texts["L0"] and "BUSINESS RULES" not in texts["L1"]
    assert "BUSINESS RULES" in texts["L2"] and "VERIFIED EXAMPLE" not in texts["L2"]
    assert "VERIFIED EXAMPLE" in texts["L3"]
    assert len(texts["L0"]) < len(texts["L1"]) < len(texts["L2"]) < len(texts["L3"])


# ---------------------------------------------------------------- golden set


def test_golden_shape_and_quotas() -> None:
    assert len(ITEMS) == 120
    assert Counter(i.bucket for i in ITEMS) == dict.fromkeys(golden.BUCKETS, 20)
    refusals = Counter(i.refusal.type for i in ITEMS if i.refusal)
    assert refusals == {
        "write_operation": 6,
        "out_of_scope": 5,
        "pii_or_sensitive": 4,
        "unanswerable_from_schema": 5,
    }
    amb = Counter(i.clarification.ambiguity_type for i in ITEMS if i.clarification)
    assert amb == {"metric_choice": 6, "time_window": 5, "entity": 4, "scope": 5}


def test_every_must_rule_is_tested_twice() -> None:
    tested = Counter(r for i in ITEMS for r in i.tests_rules)
    for rule in CTX.rules.values():
        if rule.severity == "must" and rule.id not in {"R-NO-PII", "R-FAILED-IS-WRONG"}:
            assert tested[rule.id] >= 2, rule.id
    assert sum(i.refusal is not None and i.refusal.type == "pii_or_sensitive" for i in ITEMS) >= 2  # R-NO-PII


def test_gold_sql_parses_and_every_mart_table_is_used() -> None:
    used: Counter[str] = Counter()
    for i in ITEMS:
        if i.scored_sql:
            tree = sqlglot.parse_one(i.scored_sql, dialect="postgres")
            used.update(f"{t.db}.{t.name}" for t in tree.find_all(sqlglot.exp.Table) if t.db)
    for t in (
        CTX.tables
    ):  # golden-v1 coverage; GOLDEN_SET_GUIDE §4 records where it falls short of the 5-question target
        assert used[t] >= 1, t


def test_golden_results_were_verified_for_this_version() -> None:
    v = json.loads(golden.VERIFIED_FILE.read_text())
    assert len(v["items"]) == sum(1 for i in ITEMS if i.scored_sql)


def _norm_sql(s: str) -> str:
    return sqlglot.transpile(s, read="postgres")[0].lower()


def test_no_golden_leakage_into_context_examples() -> None:
    """Exact, normalised and fuzzy question matches, and identical normalised SQL, are all forbidden."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())  # noqa: E731
    for e in CTX.examples:
        for i in ITEMS:
            assert norm(e.question) != norm(i.question), (e.id, i.id)
            assert fuzz.ratio(norm(e.question), norm(i.question)) <= 90, (e.id, i.id)
            if i.scored_sql:
                assert _norm_sql(e.sql) != _norm_sql(i.scored_sql), (e.id, i.id)


def test_golden_files_are_never_mentioned_in_context() -> None:
    blob = json.dumps(copy.deepcopy(CTX.model_dump()))
    assert not any(i.id in blob for i in ITEMS)


@pytest.mark.integration
def test_rule_traps_actually_trap() -> None:
    """The naive query for a business-rule trap must give a different answer than the gold query."""
    from lexicon.executor import PostgresExecutor

    ex = PostgresExecutor(eval_db())
    naive = {
        "G-T-01": "select count(distinct ticket_id) from support.ticket_events where event_type = 'resolved'",
        "G-T-03": "select count(*) from support.agents",
        "G-T-10": "select avg(is_correct::int) from marts.fct_outcomes where setup = 'B_local' and split = 'tune'",
        "G-T-11": "select avg(latency_ms) from marts.fct_requests where setup = 'B_local' and split = 'test'",
        "G-T-15": "select count(*) filter (where ai_predicted_intent is distinct from final_intent)::numeric / count(*) from support.tickets",
    }
    gold = {i.id: i.gold_sql for i in ITEMS}
    for k, sql in naive.items():
        assert ex.run(sql).rows != ex.run(gold[k] or "").rows, k

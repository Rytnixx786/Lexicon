"""Read-only checks against the frozen snapshot: manifest fingerprints, golden and example verification, samples."""

from __future__ import annotations

import json

import pytest

from lexicon import snapshot
from lexicon.context import loader
from lexicon.context.schema import Context
from lexicon.eval.verify import verify_examples, verify_golden
from tests.conftest import eval_db

pytestmark = pytest.mark.integration


def test_snapshot_matches_its_manifest() -> None:
    dsn = eval_db()
    manifest = json.loads(snapshot.MANIFEST.read_text())
    assert snapshot.table_fingerprints(dsn) == manifest["tables"]
    assert snapshot.snapshot_hash() == manifest["sha256"]


def test_golden_and_examples_still_verify() -> None:
    eval_db()
    assert verify_golden() == []
    assert verify_examples(loader.load()) == []


def test_samples_match_the_committed_file_and_skip_pii() -> None:
    ctx = loader.load()
    samples = loader.sample_values(ctx, eval_db())
    assert samples == ctx.samples
    assert "email" not in samples.get("support.agents", {})


def test_check_against_db_reports_drift() -> None:
    ctx = loader.load()
    data = ctx.model_dump()
    data["tables"]["support.tickets"]["columns"].append({"name": "ghost", "type": "text", "description": "x"})
    del data["tables"]["support.agents"]["columns"][-1]
    problems = loader.check_against_db(Context.model_validate(data), eval_db())
    assert "support.tickets.ghost: column missing in database" in problems
    assert any(p.startswith("support.agents.") and "no context entry" in p for p in problems)


def test_role_cannot_write_even_inside_a_cte() -> None:
    from lexicon.executor import ExecError, PostgresExecutor

    ex = PostgresExecutor(eval_db())
    with pytest.raises(ExecError) as e:
        ex.run("with x as (delete from support.agents returning 1) select count(*) from x")
    assert e.value.kind in {"permission_denied", "other"}
    with pytest.raises(ExecError) as t:
        PostgresExecutor(eval_db(), timeout_ms=300).run("select pg_sleep(2)")
    assert t.value.kind == "timeout"

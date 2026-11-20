"""End-to-end pipeline with a scripted LLM: evaluate -> write_run -> gate -> report, on the real snapshot DB;
plus BIRD on a temporary SQLite database and Tally span export to an in-memory exporter."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lexicon import gate, publish
from lexicon.eval import bird, golden, runner
from lexicon.eval import report as rep
from lexicon.llm import LLMRequest, LLMResult
from tests.conftest import eval_db

ITEMS = {i.question: i for i in golden.load()}
ROUTE = {"should_refuse": None, "should_clarify": "ambiguous"}


class ScriptedLLM:
    """Answers each golden question 'perfectly' (gold SQL, right route), except the ids in `wrong`."""

    def __init__(self, wrong: set[str] | None = None) -> None:
        self.wrong = wrong or set()

    def call(self, req: LLMRequest, request_id: str | None = None) -> LLMResult:
        q = req.user.split("\n")[0].removeprefix("Question: ").removesuffix("\nLabel:")
        item = ITEMS[q]
        if req.choices:
            content = (
                item.refusal.type if item.refusal else ("ambiguous" if item.clarification else "answerable")
            )
        elif req.prompt_version.startswith("sql_agent.clarify"):
            content = "\n".join(item.clarification.acceptable_options) if item.clarification else "a\nb"
        else:
            sql = "select 1" if item.id in self.wrong else item.scored_sql
            content = f"```sql\n{sql}\n```\nRULES_APPLIED: none"
        return LLMResult(
            request_id or "r", req.key(), req.model, req.prompt_version, content, 1000, 50, 20.0, 1.7e9
        )


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    eval_db()
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(gate, "RUNS_DIR", tmp_path)
    monkeypatch.setattr("lexicon.settings.RECORDINGS_DIR", tmp_path / "rec")
    return tmp_path


@pytest.mark.integration
def test_evaluate_gate_and_report(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = ["G-S-01", "G-S-02", "G-M-01", "G-D-01", "G-T-01", "G-C-01", "G-C-07", "G-R-01", "G-R-12"]
    res = runner.evaluate(runner.RunSpec("qwen7b-L3", "m", "L3", 2, item_ids=ids), llm=ScriptedLLM())  # type: ignore[arg-type]
    m = res["metrics"]
    assert m["ex_strict"] == 1.0 and m["clarify_recall"] == 1.0 and m["refusal_type_accuracy"] == 1.0
    assert (
        m["over_refusal_rate"] == 0.0 and m["retrieval_recall"] > 0.5 and m["mean_cost_usd_per_question"] > 0
    )
    runner.write_run(res, runs)
    res2 = runner.evaluate(
        runner.RunSpec("qwen7b-L0", "m", "L0", 2, item_ids=ids), llm=ScriptedLLM({"G-S-01", "G-D-01"})
    )  # type: ignore[arg-type]
    runner.write_run(res2, runs)
    assert res2["metrics"]["ex_strict"] == pytest.approx(3 / 5)

    champ = runs / "champion.json"
    gate.promote(["qwen7b-L3"], champ)
    ok = gate.run_gate(["qwen7b-L3"], champ)
    assert ok["passed"] and (runs / "qwen7b-L3" / "gate_verdict.json").exists()
    bad = gate.run_gate(["qwen7b-L0"], champ)
    assert not bad["passed"] and "execution accuracy dropped" in bad["reason"]

    monkeypatch.setattr(rep, "DOCS", runs / "docs")
    readme = runs / "README.md"
    readme.write_text(f"x\n{rep.START}\nold\n{rep.END}\n", encoding="utf-8")
    monkeypatch.setattr(rep, "README", readme)
    for lv in ("L1", "L2"):
        (runs / f"qwen7b-{lv}").mkdir()
        for f in ("meta.json", "metrics.json", "per_question.jsonl"):
            (runs / f"qwen7b-{lv}" / f).write_text((runs / "qwen7b-L3" / f).read_text())
    out = rep.build(runs)
    text = out.read_text(encoding="utf-8")
    assert "Context effect" in text and "ARES gate verdicts" in text
    assert (runs / "docs" / "assets" / "ablation.png").exists()
    assert "Context effect" in readme.read_text(encoding="utf-8")


def test_bird_run_on_a_temporary_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dbdir = tmp_path / "dev_databases" / "shop"
    dbdir.mkdir(parents=True)
    con = sqlite3.connect(dbdir / "shop.sqlite")
    con.executescript("create table items (id int, price real); insert into items values (1, 2.5), (2, 4.0);")
    con.commit()
    con.close()
    qs = [
        {
            "question_id": 1,
            "db_id": "shop",
            "question": "Total price?",
            "evidence": "price is in USD",
            "SQL": "select sum(price) from items",
            "difficulty": "simple",
        },
        {
            "question_id": 2,
            "db_id": "shop",
            "question": "Delete everything",
            "evidence": "",
            "SQL": "select count(*) from items",
            "difficulty": "simple",
        },
    ]
    monkeypatch.setattr(bird, "fetch_questions", lambda: qs)
    monkeypatch.setattr(bird, "extract_databases", lambda: tmp_path / "dev_databases")
    slice_file = tmp_path / "slice.json"
    slice_file.write_text(json.dumps({"question_ids": [1, 2]}))
    monkeypatch.setattr(bird, "SLICE_FILE", slice_file)
    monkeypatch.setattr(bird, "RUNS_DIR", tmp_path / "runs")
    seen: list[str] = []

    class Fake:
        def call(self, req: LLMRequest, request_id: str | None = None) -> LLMResult:
            seen.append(req.system)
            sql = "select sum(price) from items" if "Total" in req.user else "delete from items"
            return LLMResult(
                "r", req.key(), req.model, req.prompt_version, f"```sql\n{sql}\n```", 10, 5, 1.0, 0.0
            )

    out = bird.run("m", True, "bird-test", llm=Fake())  # type: ignore[arg-type]
    assert out["metrics"]["ex"] == 0.5 and out["metrics"]["guard_blocked"] == 1
    assert "price is in USD" in seen[0]
    assert sqlite3.connect(dbdir / "shop.sqlite").execute("select count(*) from items").fetchone()[0] == 2


def test_span_export_and_prompt_registry_rows(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = [
        {
            "node": "generate",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "prompt_version": "sql_agent.generate.L3.v1",
            "input_tokens": 3000,
            "output_tokens": 80,
            "latency_ms": 900.0,
            "started_at": 1.7e9,
            "http_status": 200,
            "question_id": "G-S-01",
            "run_id": "qwen7b-L3",
            "cost_usd": 1e-5,
        }
    ]
    exp = InMemorySpanExporter()
    assert publish.export_spans(calls, "http://unused", exporter=exp) == 1
    span = exp.get_finished_spans()[0]
    assert span.attributes["tally.tenant_id"] == "lexicon" and span.attributes["tally.feature"] == "sql_agent"
    assert span.attributes["gen_ai.usage.input_tokens"] == 3000
    assert not any(k.startswith("gen_ai.input") for k in span.attributes)
    rows = publish.prompt_rows(calls)
    assert rows[0][0] == "sql_agent.generate.L3.v1" and len(rows[0][2]) == 12


def test_publish_refuses_when_tally_contract_would_break(tmp_path: Path) -> None:
    import yaml

    def contract(tests: list[object]) -> Path:
        yml = tmp_path / "marts.yml"
        doc = {
            "models": [{"name": "fct_requests", "columns": [{"name": "decision_id", "data_tests": tests}]}]
        }
        yml.write_text(yaml.safe_dump(doc), encoding="utf-8")
        return yml

    assert not publish.tally_accepts_non_triage_traffic(contract(["not_null"]))
    scoped = {"not_null": {"config": {"where": "feature = 'ticket_triage'"}}}
    assert publish.tally_accepts_non_triage_traffic(contract([scoped]))


@pytest.mark.integration
def test_evaluate_with_an_alternate_context_dir(runs: Path) -> None:
    demo = Path(__file__).resolve().parents[1] / "evals" / "gate_demo" / "context_without_reopen_rule"
    spec = runner.RunSpec("demo", "m", "L3", 1, item_ids=["G-S-01"], context_dir=str(demo))
    res = runner.evaluate(spec, llm=ScriptedLLM())
    assert res["meta"]["context_dir"] == str(demo)
    base = runner.evaluate(runner.RunSpec("base", "m", "L3", 1, item_ids=["G-S-01"]), llm=ScriptedLLM())
    assert res["meta"]["context_fingerprint"] != base["meta"]["context_fingerprint"]

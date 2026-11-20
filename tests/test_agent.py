"""Graph behaviour with a scripted LLM and an in-memory executor: routes, refusal without an LLM call, interrupt and
resume, the retry cap, and the deterministic ambiguity triggers."""

from __future__ import annotations

from typing import Any

import pytest

from lexicon.context import loader
from lexicon.executor import ExecError, ResultFrame
from lexicon.graph import Agent, AgentConfig, chart_spec, interrupt_payload, is_interrupted
from lexicon.llm import LLMRequest, LLMResult

CTX = loader.load()


class FakeLLM:
    def __init__(self, route: str = "answerable", sqls: list[str] | None = None) -> None:
        self.route, self.sqls, self.calls = route, list(sqls or ["select count(*) from support.tickets"]), []

    def call(self, req: LLMRequest, request_id: str | None = None) -> LLMResult:
        self.calls.append(req)
        if req.choices:
            content = self.route
        elif req.prompt_version.startswith("sql_agent.clarify"):
            content = "option one\noption two"
        else:
            sql = self.sqls.pop(0) if len(self.sqls) > 1 else self.sqls[0]
            content = f"```sql\n{sql}\n```\nRULES_APPLIED: R-EXCLUDE-TEST-TENANTS"
        return LLMResult(
            "rid", req.key(), req.model, req.prompt_version, content, 100, 10, 5.0, 0.0, choice_prob=0.9
        )


class FakeExecutor:
    dialect = "postgres"

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times, self.sql = fail_times, []

    def run(self, sql_text: str) -> ResultFrame:
        self.sql.append(sql_text)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ExecError("undefined_column", 'column "nope" does not exist')
        return ResultFrame(["count"], [(42,)])


def make(
    route: str = "answerable",
    level: str = "L3",
    fail_times: int = 0,
    sqls: list[str] | None = None,
    names: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[Agent, FakeLLM, FakeExecutor]:
    llm, ex = FakeLLM(route, sqls), FakeExecutor(fail_times)
    return Agent(CTX, llm, ex, AgentConfig(model="m", level=level), agent_names=names or {}), llm, ex  # type: ignore[arg-type]


def test_answerable_path_runs_guarded_sql() -> None:
    agent, _llm, ex = make()
    _, out = agent.start("How many tickets are there in total for the sales tenant?")
    assert out["route"] == "answerable" and out["answer"]["ok"]
    assert ex.sql == ["SELECT COUNT(*) FROM support.tickets LIMIT 1000"]
    assert out["rules_applied"] == ["R-EXCLUDE-TEST-TENANTS"]
    assert [c["node"] for c in out["calls"]] == ["classify", "generate"]
    assert out["answer"]["chart"]["type"] == "kpi"


@pytest.mark.parametrize(
    "route", ["write_operation", "pii_or_sensitive", "out_of_scope", "unanswerable_from_schema"]
)
def test_refusal_makes_no_sql_and_no_further_llm_call(route: str) -> None:
    agent, llm, ex = make(route)
    _, out = agent.start("Delete every ticket")
    assert out["refusal"]["type"] == route and ex.sql == []
    assert len(llm.calls) == 1  # the classifier only


def test_classifier_ambiguity_interrupts_and_resumes() -> None:
    agent, llm, _ex = make("ambiguous")
    tid, out = agent.start("How are we doing?")
    assert is_interrupted(out) and interrupt_payload(out)["options"] == ["option one", "option two"]
    out = agent.resume(tid, "option two")
    assert out["clarification"]["answer"] == "option two" and out["answer"]["ok"]
    assert "Clarification from the user: option two" in llm.calls[-1].user


def test_clarification_answer_guides_post_clarification_retrieval() -> None:
    agent, llm, _ex = make("ambiguous")
    tid, out = agent.start("How are we doing?")
    assert is_interrupted(out)
    out = agent.resume(tid, "Show cost per correct triage by setup from marts.fct_outcomes")
    assert out["answer"]["ok"]
    assert "marts.fct_outcomes" in llm.calls[-1].system


def test_deterministic_metric_trigger_only_from_l2() -> None:
    agent, _, _ = make(level="L3")
    _, out = agent.start("What is our escalation rate?")
    assert is_interrupted(out) and "more than one definition" in interrupt_payload(out)["reason"]
    agent0, _, _ = make(level="L1")
    _, out0 = agent0.start("What is our escalation rate?")
    assert not is_interrupted(out0)


def test_disambiguated_question_does_not_trigger() -> None:
    agent, _, _ = make(level="L2")
    _, out = agent.start(
        "What share of tickets were escalated to a human at least once? escalation rate for human escalations"
    )
    assert not is_interrupted(out)


def test_vague_time_and_duplicate_names_trigger() -> None:
    names = {
        "tanvir chowdhury": [
            {"agent_id": "ag-03", "team": "support", "is_active": True},
            {"agent_id": "ag-04", "team": "support", "is_active": False},
        ]
    }
    agent, _, _ = make(names=names)
    _, out = agent.start("How many tickets came in recently?")
    assert "time window" in interrupt_payload(out)["reason"]
    agent2, _, _ = make(names=names)
    _, out2 = agent2.start("How many tickets did Tanvir Chowdhury resolve?")
    assert interrupt_payload(out2)["options"][0].startswith("ag-03")


def test_retry_feeds_back_errors_and_respects_the_cap() -> None:
    agent, llm, _ex = make(fail_times=1)
    _, out = agent.start("Count the tickets of the billing tenant")
    assert out["attempts"] == 2 and out["answer"]["ok"]
    assert "undefined_column" in llm.calls[-1].user
    agent2, _, ex2 = make(fail_times=10)
    _, out2 = agent2.start("Count the tickets of the billing tenant")
    assert out2["attempts"] == 3 and not out2["answer"]["ok"] and len(ex2.sql) == 3


def test_guard_violation_is_retried_and_never_executed() -> None:
    agent, llm, ex = make(sqls=["delete from support.tickets", "select count(*) from support.tickets"])
    _, out = agent.start("Count the tickets of the sales tenant")
    assert out["answer"]["ok"] and all("DELETE" not in s.upper() for s in ex.sql)
    assert "forbidden_statement" in llm.calls[-1].user or "not_select" in llm.calls[-1].user


def test_chart_spec() -> None:
    import datetime as dt

    assert chart_spec(["n"], [[1]])["type"] == "kpi"
    assert chart_spec(["tenant", "n"], [["a", 1], ["b", 2]]) == {"type": "bar", "x": "tenant", "y": "n"}
    assert chart_spec(["day", "n"], [[dt.date(2026, 1, 1), 1]])["type"] == "line"
    assert chart_spec(["a", "b", "c"], [[1, 2, 3]])["type"] == "table"

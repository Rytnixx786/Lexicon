"""LLM record/replay, gate payloads, cost parity with Tally, BIRD scoring, synthetic data, report statistics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from lexicon import cost, gate, llm, snapshot
from lexicon.eval import bird, runner
from lexicon.eval import report as rep

# ---------------------------------------------------------------- llm


def test_record_replay_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RECORDINGS_DIR", tmp_path)
    body = {
        "choices": [
            {
                "message": {"content": "answerable"},
                "logprobs": {
                    "content": [
                        {
                            "token": "answer",
                            "logprob": math.log(0.8),
                            "top_logprobs": [
                                {"token": "answer", "logprob": math.log(0.8)},
                                {"token": "ambig", "logprob": math.log(0.2)},
                            ],
                        },
                        {
                            "token": "able",
                            "logprob": 0.0,
                            "top_logprobs": [{"token": "able", "logprob": 0.0}],
                        },
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 2},
    }
    req = llm.LLMRequest("m", "sys", "q", 8, "sql_agent.classify.L3.v1", ("answerable", "ambiguous"))
    with respx.mock() as mock:
        route = mock.post("http://gw/v1/chat/completions").mock(return_value=httpx.Response(200, json=body))
        live = llm.LLMClient("auto", base_url="http://gw").call(req)
        sent = json.loads(route.calls[0].request.content)
        assert route.calls[0].request.headers["X-Tenant-ID"] == "lexicon"
        assert sent["guided_choice"] == ["answerable", "ambiguous"]
    assert live.content == "answerable" and live.choice_prob == pytest.approx(0.8)
    again = llm.LLMClient("replay", base_url="http://gw").call(req)
    assert again.replayed and again.content == "answerable" and again.input_tokens == 50
    with pytest.raises(llm.ReplayMissError):
        llm.LLMClient("replay").call(llm.LLMRequest("m", "sys", "other", 8, "pv"))


def test_seed_and_prompt_version_semantics_of_the_key() -> None:
    a = llm.LLMRequest("m", "s", "u", 8, "pv1")
    assert a.key() == llm.LLMRequest("m", "s", "u", 8, "pv2").key()  # label only
    assert a.key() != llm.LLMRequest("m", "s", "u", 8, "pv1", seed=1).key()


def test_hosted_gemini_spend_guard_and_json_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RECORDINGS_DIR", tmp_path)
    req = llm.LLMRequest(
        "gemini-2.5-flash-lite", "sys", "q", 12, "sql_agent.classify.L3.v1", ("answerable", "ambiguous")
    )
    assert llm.is_hosted_model("gemini-2.5-flash-lite") and not llm.is_hosted_model("Qwen/7B")
    with pytest.raises(llm.SpendNotAllowedError):
        llm.LLMClient("auto", allow_spend=False).call(req)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    body = {
        "choices": [{"message": {"content": '{"label": "ambiguous"}'}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 10},
    }
    with respx.mock() as mock:
        route = mock.post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        res = llm.LLMClient("auto", allow_spend=True).call(req)
        sent = json.loads(route.calls[0].request.content)
        assert "seed" not in sent
        assert sent["response_format"]["type"] == "json_schema"
        assert sent["max_tokens"] >= 64
        assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert res.content == "ambiguous" and res.choice_prob is None

    hosted_costs = cost.cost_calls(
        [{"model": "gemini-2.5-flash-lite", "input_tokens": 1_000_000, "output_tokens": 1_000_000}],
        wall_seconds=1.0,
        n_live=1,
        power_w=None,
    )
    assert hosted_costs[0] == pytest.approx(0.50)


# ---------------------------------------------------------------- gate


def _write_run(root: Path, rid: str, success: list[bool], gv: str = "g1") -> None:
    d = root / rid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "golden_version": gv,
                "snapshot_sha256": "s",
                "context_fingerprint": "c",
                "model": "m",
                "level": "L3",
            }
        )
    )
    recs = []
    buckets = ["simple"] * 4 + ["should_clarify", "should_refuse"]
    for i, (b, s) in enumerate(zip(buckets, success, strict=True)):
        recs.append(
            {
                "id": f"G-{i}",
                "bucket": b,
                "ex_strict": s and b == "simple",
                "ex_lenient": s,
                "asked": b == "should_clarify" and s,
                "refused": b == "should_refuse" and s,
                "refusal_type": "write_operation" if (b == "should_refuse" and s) else None,
                "success": s,
                "attempts": 1,
                "calls": [],
            }
        )
    (d / "per_question.jsonl").write_text("\n".join(json.dumps(r) for r in recs))


def test_majority_vote_and_flip_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(
        runner,
        "aggregate",
        lambda recs: {
            "ex_strict": sum(r["ex_strict"] for r in recs) / 4,
            "task_success": 0,
            "over_refusal_rate": 0,
            "false_clarify_rate": 0,
            "bucket_scores": {"should_clarify": 1.0},
            "n": len(recs),
        },
    )
    monkeypatch.setattr(gate, "aggregate", runner.aggregate)
    _write_run(tmp_path, "a", [True, True, False, False, True, True])
    _write_run(tmp_path, "b", [True, False, False, False, True, True])
    _write_run(tmp_path, "c", [True, True, True, False, True, True])
    p = gate.payload(["a", "b", "c"])
    assert p["new_metrics"]["overall_accuracy"] == 0.5  # majority: T, T, F, F
    assert p["seed_flip_rate"] == pytest.approx(2 / 6)


def test_gate_refuses_mismatched_champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "RUNS_DIR", tmp_path)
    _write_run(tmp_path, "cand", [True] * 6, gv="g2")
    champ = tmp_path / "champion.json"
    champ.write_text(json.dumps({"golden_version": "g1", "snapshot_sha256": "s"}))
    monkeypatch.setattr(
        gate,
        "aggregate",
        lambda recs: {
            "ex_strict": 1.0,
            "task_success": 1.0,
            "over_refusal_rate": 0,
            "false_clarify_rate": 0,
            "bucket_scores": {},
            "n": 6,
        },
    )
    v = gate.run_gate(["cand"], champ)
    assert v["verdict"] == "ERROR" and "golden_version" in v["reason"]


# ---------------------------------------------------------------- cost parity with Tally


def test_cost_formula_matches_tally() -> None:
    tally_src = cost.tally_root() / "src"
    if not tally_src.exists():
        pytest.skip("Tally repo not available")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(tally_src))
    try:
        from tally import cost as tcost
        from tally.config import load_cost_model
    finally:
        sys.path.remove(str(tally_src))
    tcfg = load_cost_model(cost.tally_root() / "config" / "cost_model.yaml")
    mine = cost.GpuCost.from_tally()
    tp = tcost.Throughput(n_requests=1234, wall_seconds=600.0, concurrency=8, mean_power_w=141.4)
    assert mine.per_call_mean(1234, 600.0, 141.4) == pytest.approx(
        tcost.local_cost_per_request_mean(tcfg, tp)
    )


def test_cost_calls_split_by_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cost.GpuCost, "from_tally", classmethod(lambda cls, path=None: cls(876, 1, 0.5, 200, 0, 0.5))
    )
    monkeypatch.setattr(
        cost, "tally_measured_power", lambda: None
    )  # no measurement -> configured board power
    calls = [{"input_tokens": 300, "output_tokens": 0}, {"input_tokens": 100, "output_tokens": 0}]
    out = cost.cost_calls(calls, wall_seconds=3600, n_live=3600, power_w=None)
    assert sum(out) == pytest.approx(2 * 0.2 / 1800) and out[0] == pytest.approx(3 * out[1])


# ---------------------------------------------------------------- BIRD


def test_bird_ex_is_set_equality() -> None:
    assert bird.bird_ex([(1, "a"), (1, "a")], [(1, "a")])
    assert not bird.bird_ex([(1,)], [(1,), (2,)])


def test_bird_slice_is_stratified_and_deterministic() -> None:
    qs = [
        {"question_id": i, "difficulty": d}
        for i, d in enumerate(["simple"] * 150 + ["moderate"] * 250 + ["challenging"] * 100)
    ]
    a, b = bird.make_slice(qs, 200), bird.make_slice(qs, 200)
    assert a == b and len(a) == 200
    diff = {q["question_id"]: q["difficulty"] for q in qs}
    counts = pd.Series([diff[i] for i in a]).value_counts().to_dict()
    assert counts == {"moderate": 100, "simple": 60, "challenging": 40}


# ---------------------------------------------------------------- synthetic support data


def test_synthetic_support_is_deterministic_and_has_the_traps() -> None:
    outcomes = pd.DataFrame(
        {
            "setup": "B_local",
            "split": "test",
            "arm": "primary",
            "ticket_id": [f"bx-{i:05d}" for i in range(400)],
            "tenant_id": ["support", "billing", "sales", "support"] * 100,
            "decision_id": [f"d{i}" for i in range(400)],
            "predicted_label": ["track_order"] * 300 + ["get_refund"] * 100,
            "gold_label": ["track_order"] * 400,
        }
    )
    a, b = snapshot.synthetic_support(outcomes), snapshot.synthetic_support(outcomes)
    pd.testing.assert_frame_equal(a["ticket_events"], b["ticket_events"])
    ev, tk, ag = a["ticket_events"], a["tickets"], a["agents"]
    assert (ev["event_type"] == "reopened").any() and (tk["tenant_id"] == "test-loadgen").sum() == 300
    assert (~ag["is_active"]).any() and ag["email"].str.contains("@").all()
    reopened = set(ev.loc[ev["event_type"] == "reopened", "ticket_id"])
    assert tk.loc[tk["ticket_id"].isin(reopened), "status"].isin(["open", "resolved", "closed"]).all()
    assert (ev.groupby("ticket_id")["event_type"].apply(lambda s: (s == "resolved").sum()) > 1).any()


# ---------------------------------------------------------------- report statistics


def test_thesis_statistics_on_synthetic_runs() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for m, base in (("qwen3b", 0.1), ("qwen7b", 0.2)):
        for lv, add in zip(rep.LEVELS, (0.0, 0.2, 0.4, 0.5), strict=True):
            for q in range(80):
                rows.append(
                    {
                        "model": m,
                        "level": lv,
                        "id": f"q{q}",
                        "bucket": "simple",
                        "ex": rng.uniform() < base + add,
                        "success": True,
                        "asked": False,
                        "refused": False,
                        "cost": 0.0,
                        "tokens": 0,
                    }
                )
    t = rep.thesis(pd.DataFrame(rows))
    assert t["delta_context"][0] > t["delta_model"][0] > 0
    assert t["delta_context"][1] <= t["delta_context"][0] <= t["delta_context"][2]
    assert t["decisive"]["weak_model"] == "qwen3b" and t["lr_tests"]["level"]["p_value"] < 0.001
    assert "supported" in "\n".join(rep.headline_lines(t))


def test_report_build_and_edge_cases_without_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert rep.headline_lines({}) == ["Ablation results pending."]
    assert rep.pct(None) == "n/a" and rep.pct(0.5) == "50.0%"
    empty = pd.DataFrame(columns=["model", "level", "id", "bucket", "ex"])
    assert rep.thesis(empty)["models"] == []
    const = pd.DataFrame(
        [{"model": "qwen3b", "level": lv, "id": "q0", "bucket": "simple", "ex": True} for lv in ("L0", "L3")]
    )
    assert rep.lr_tests(const) == {}

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    rng = np.random.default_rng(3)
    for m, base in (("qwen3b", 0.15), ("coder7b", 0.25)):
        for lv, add in zip(rep.LEVELS, (0.0, 0.15, 0.2, 0.45), strict=True):
            d = runs_dir / f"{m}-{lv}"
            d.mkdir()
            recs = []
            for idx in range(20):
                ok = bool(rng.uniform() < base + add)
                recs.append(
                    {
                        "id": f"G-S-{idx:02d}",
                        "bucket": "simple",
                        "ex_strict": ok,
                        "success": ok,
                        "asked": False,
                        "refused": False,
                        "cost_usd": 0.0001,
                        "calls": [{"input_tokens": 500, "output_tokens": 40}],
                    }
                )
            (d / "per_question.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
            (d / "metrics.json").write_text(
                json.dumps(
                    {
                        "ex_strict": sum(r["ex_strict"] for r in recs) / len(recs),
                        "task_success": sum(r["success"] for r in recs) / len(recs),
                        "clarify_recall": 0.8,
                        "false_clarify_rate": 0.05,
                        "refusal_type_accuracy": 0.8,
                        "over_refusal_rate": 0.02,
                        "retrieval_recall": 0.99,
                        "mean_input_tokens_per_question": 500,
                        "mean_output_tokens_per_question": 40,
                        "mean_cost_usd_per_question": 0.0001,
                        "bucket_scores": {"simple": 0.5},
                    }
                ),
                encoding="utf-8",
            )
    bd = runs_dir / "bird-qwen7b-ev"
    bd.mkdir()
    (bd / "meta.json").write_text(json.dumps({"model": "Qwen/7B", "evidence": True, "n": 2}))
    (bd / "metrics.json").write_text(json.dumps({"ex_by_difficulty": {"simple": 0.5}}))
    (bd / "per_question.jsonl").write_text('{"correct": true}\n{"correct": false}\n')
    (runs_dir / "qwen3b-L3" / "tally_cost.json").write_text(
        json.dumps(
            {
                "run_id": "qwen3b-L3",
                "calls_found_in_tally": 20,
                "n_calls": 20,
                "mean_cost_usd_per_question": 0.0001,
            }
        )
    )
    demo_dir = runs_dir / "demo-no-reopen-rule"
    demo_dir.mkdir()
    (demo_dir / "gate_verdict.json").write_text(
        json.dumps(
            {
                "verdict": "FAIL",
                "reason": "protected bucket should_refuse got worse (0.800 -> 0.750)",
                "ares_runtime": True,
                "plugin": "lexicon@1.0.0",
                "deltas": {"overall_accuracy": 0.05, "should_refuse": -0.05, "should_clarify": 0.0},
                "payload": {
                    "runs": ["demo-no-reopen-rule", "demo-no-reopen-rule-s1", "demo-no-reopen-rule-s2"],
                    "seed_flip_rate": 0.0083,
                    "new_metrics": {"overall_accuracy": 0.4125, "task_success": 0.433},
                },
            }
        )
    )
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir()
    (evals_dir / "champion.json").write_text(
        json.dumps(
            {
                "runs": ["qwen7b-L3", "qwen7b-L3-s1", "qwen7b-L3-s2"],
                "seed_flip_rate": 0.0167,
                "new_metrics": {"overall_accuracy": 0.3625, "task_success": 0.4083},
            }
        )
    )
    readme = tmp_path / "README.md"
    readme.write_text(f"head\n{rep.START}\nold\n{rep.END}\ntail\n", encoding="utf-8")
    monkeypatch.setattr(rep, "ROOT", tmp_path)
    monkeypatch.setattr(rep, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(rep, "README", readme)

    out = rep.build(runs_dir)
    md = out.read_text(encoding="utf-8")
    assert "Seed stability (ADR-021)" in md and "Gate demo: deleting `R-RESOLVED-IS-STATUS`" in md
    assert "Cost per question, measured by Tally" in md and "BIRD Mini-Dev" in md
    assert (tmp_path / "docs" / "assets" / "ablation.png").exists()
    assert "Per-bucket task success at full context (L3)" in readme.read_text(encoding="utf-8")


def test_gate_promote_run_and_standalone_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(gate, "env", lambda k, d="": "")
    monkeypatch.setattr(
        gate,
        "aggregate",
        lambda recs: {
            "ex_strict": sum(r["ex_strict"] for r in recs) / 4,
            "task_success": sum(r["success"] for r in recs) / len(recs),
            "over_refusal_rate": 0.0,
            "false_clarify_rate": 0.0,
            "bucket_scores": {
                "simple": sum(r["ex_strict"] for r in recs) / 4,
                "should_clarify": 1.0,
                "should_refuse": 1.0 if recs[-1]["success"] else 0.0,
            },
            "n": len(recs),
        },
    )
    _write_run(tmp_path, "r1", [True, True, True, True, True, True])
    _write_run(tmp_path, "r2", [True, False, False, False, True, False])
    _write_run(tmp_path, "badgv", [True] * 6, gv="g2")
    with pytest.raises(ValueError, match="disagree on golden_version"):
        gate.payload(["r1", "badgv"])
    champ = tmp_path / "champion.json"
    gate.promote(["r1"], champ)
    ok = gate.run_gate(["r1"], champ)
    assert ok["passed"] and not ok["ares_runtime"]
    fail = gate.run_gate(["r2"], champ)
    assert not fail["passed"] and "should_refuse" in fail["reason"]


def test_gpu_guard_throttling_and_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from lexicon import gpu_guard

    monkeypatch.setenv("LEXICON_GPU_GUARD", "0")
    g = gpu_guard.GpuGuard(power_cap_w=100.0, temp_max_c=70.0, duty=0.5)
    assert g.mean_power == 0.0 and not g.hot()
    g.power.extend([120.0, 110.0])
    assert g.mean_power == 115.0 and g.hot()
    g.power.clear()
    g.temp = 80.0
    assert g.hot()
    g.temp = 50.0
    g.after_call(0.05)
    slept: list[float] = []
    monkeypatch.setattr(gpu_guard.time, "sleep", lambda s: (slept.append(s), setattr(g, "_not_before", 0.0)))
    g.before_call()
    assert slept

    calls = 0

    def fake_run(*a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess([], 0, stdout="95.5, 62\n")
        g.close()
        raise subprocess.SubprocessError("boom")

    monkeypatch.setattr(gpu_guard.subprocess, "run", fake_run)
    monkeypatch.setattr(g._stop, "wait", lambda t: g._stop.is_set())
    g._poll()
    assert g.temp == 62.0 and g.mean_power == 95.5
    monkeypatch.setattr(gpu_guard, "_GUARD", None)
    assert gpu_guard.guard() is gpu_guard.guard()


def test_runner_and_verify_without_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lexicon.context import loader
    from lexicon.eval import golden, verify
    from lexicon.executor import ExecError, ResultFrame

    class FakeEx:
        dialect = "postgres"

        def __init__(self, fail_agents: bool = False) -> None:
            self.fail_agents = fail_agents

        def run(self, sql_text: str) -> ResultFrame:
            if "support.agents" in sql_text and self.fail_agents:
                raise ExecError("other", "fail")
            if "display_name" in sql_text:
                return ResultFrame(
                    ["lower", "agent_id", "team", "is_active"], [("alex", "a1", "support", True)]
                )
            if "empty" in sql_text:
                return ResultFrame(["n"], [])
            if "broken" in sql_text:
                raise ExecError("syntax", "bad sql")
            return ResultFrame(["n"], [(42,)])

    items = {i.question: i for i in golden.load()}

    class FakeLLM:
        def call(self, req: llm.LLMRequest, request_id: str | None = None) -> llm.LLMResult:
            q = req.user.split("\n")[0].removeprefix("Question: ").removesuffix("\nLabel:")
            item = items[q]
            if req.choices:
                c = (
                    item.refusal.type
                    if item.refusal
                    else ("ambiguous" if item.clarification else "answerable")
                )
            elif req.prompt_version.startswith("sql_agent.clarify"):
                c = "opt1\nopt2"
            else:
                c = f"```sql\n{item.scored_sql}\n```\nRULES_APPLIED: none"
            return llm.LLMResult(
                request_id or "r", req.key(), req.model, req.prompt_version, c, 200, 20, 5.0, 1.7e9
            )

    monkeypatch.setattr("lexicon.settings.RECORDINGS_DIR", tmp_path / "rec")
    spec = runner.RunSpec("unit-L3", "m", "L3", concurrency=2, item_ids=["G-S-01", "G-C-01", "G-R-01"])
    res = runner.evaluate(spec, llm=FakeLLM(), executor_factory=FakeEx)  # type: ignore[arg-type]
    assert res["metrics"]["ex_strict"] == 1.0 and res["metrics"]["refusal_type_accuracy"] == 1.0
    out_dir = runner.write_run(res, tmp_path / "runs")
    assert (out_dir / "meta.json").exists() and (out_dir / "per_question.jsonl").exists()
    assert runner._agent_names(FakeEx(fail_agents=True)) == {}  # type: ignore[arg-type]
    hw_replay = runner._hardware(spec, 1.0, 0, None)
    assert hw_replay["n_live_calls"] > 0

    found, probs = verify._run_all(
        [("a", "select 1"), ("b", "select empty"), ("c", "select broken")], FakeEx()
    )
    assert len(found) == 2 and any("empty" in p for p in probs) and any("bad sql" in p for p in probs)
    cmp = verify._compare({"a": {"hash": "1"}, "b": {"hash": "2"}}, {"a": {"hash": "9"}})
    assert len(cmp) == 2

    vfile = tmp_path / "verified.json"
    efile = tmp_path / "examples.json"
    monkeypatch.setattr(golden, "VERIFIED_FILE", vfile)
    monkeypatch.setattr(verify, "EXAMPLES_FILE", efile)
    assert verify.verify_golden(update=True, ex=FakeEx()) == []
    assert verify.verify_golden(update=False, ex=FakeEx()) == []
    ctx = loader.load()
    assert verify.verify_examples(ctx, update=True, ex=FakeEx()) == []
    assert verify.verify_examples(ctx, update=False, ex=FakeEx()) == []


def test_publish_helpers_without_tally_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lexicon import publish

    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    call = {
        "node": "generate",
        "request_id": "req-1",
        "model": "m",
        "prompt_version": "sql_agent.generate.L3.v1",
        "input_tokens": 100,
        "output_tokens": 20,
        "latency_ms": 10.0,
        "started_at": 1.7e9,
        "http_status": 200,
        "cost_usd": 0.001,
    }
    (run_dir / "per_question.jsonl").write_text(json.dumps({"id": "G-S-01", "calls": [call]}) + "\n")
    monkeypatch.setattr(publish, "RUNS_DIR", tmp_path)
    calls = publish.run_calls("r1")
    assert len(calls) == 1 and calls[0]["question_id"] == "G-S-01"

    class FakeCur:
        def __init__(self) -> None:
            self.many: list[int] = []

        def __enter__(self) -> FakeCur:
            return self

        def __exit__(self, *a: object) -> None:
            pass

        def executemany(self, sql: str, rows: list[object]) -> None:
            self.many.append(len(rows))

    class FakeConn:
        def __init__(self) -> None:
            self.cur = FakeCur()
            self.committed = False

        def __enter__(self) -> FakeConn:
            return self

        def __exit__(self, *a: object) -> None:
            pass

        def execute(self, sql: str, params: object = None) -> FakeConn:
            return self

        def fetchall(self) -> list[tuple[object, ...]]:
            return [("req-1", 0.001, 100, 20)]

        def cursor(self) -> FakeCur:
            return self.cur

        def commit(self) -> None:
            self.committed = True

    fc = FakeConn()
    publish.register(fc, calls)  # type: ignore[arg-type]
    assert fc.committed and fc.cur.many == [1, 1]
    monkeypatch.setattr(publish.psycopg, "connect", lambda dsn: fc)
    monkeypatch.setattr(publish, "env", lambda k, d="": "dsn")
    res = publish.cost_from_tally("r1")
    assert res["calls_found_in_tally"] == 1 and res["mean_cost_usd_per_question"] == pytest.approx(0.001)
    monkeypatch.setattr(publish, "tally_accepts_non_triage_traffic", lambda: False)
    with pytest.raises(publish.TallyContractError):
        publish.publish("r1")
    assert publish.check_tally_path(tmp_path)


def test_executor_classification_and_sqlite_errors(tmp_path: Path) -> None:
    import sqlite3
    from decimal import Decimal

    from psycopg import errors as pg

    from lexicon import executor

    assert executor._norm(Decimal("1.5")) == 1.5 and executor._norm("x") == "x"
    assert str(executor.ExecError("syntax", "bad")) == "syntax: bad"
    assert executor._classify(pg.QueryCanceled()) == "timeout"
    assert executor._classify(pg.UndefinedColumn()) == "undefined_column"
    assert executor._classify(pg.UndefinedTable()) == "undefined_table"
    assert executor._classify(pg.ReadOnlySqlTransaction()) == "permission_denied"
    assert executor._classify(pg.SyntaxError()) == "syntax"
    assert executor._classify(pg.OperationalError()) == "other"

    db = tmp_path / "t.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table items (id int)")
    con.execute("insert into items values (1)")
    con.commit()
    con.close()
    ex = executor.SqliteExecutor(str(db))
    assert ex.run("select id from items").rows == [(1,)]
    for sql_bad, expected_kind in (
        ("select nope from items", "undefined_column"),
        ("select * from nope", "undefined_table"),
        ("select from", "syntax"),
        ("delete from items", "permission_denied"),
    ):
        with pytest.raises(executor.ExecError) as err:
            ex.run(sql_bad)
        assert err.value.kind == expected_kind


def test_loader_helpers_without_db(tmp_path: Path) -> None:
    import yaml

    from lexicon.context import loader

    ctx = loader.load()
    assert loader.cached() is loader.cached()
    assert len(loader.to_json(ctx)) > 100
    loader.write_samples({"support.tickets": {"status": ["open"]}}, root=tmp_path)
    assert (tmp_path / loader.SAMPLES_FILE).exists()

    dbt_yml = tmp_path / "_marts.yml"
    models = []
    for t in ctx.tables.values():
        if t.dbt_model:
            models.append(
                {
                    "name": t.name,
                    "columns": [{"name": c.name, "description": c.description} for c in t.columns],
                }
            )
    dbt_yml.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")
    assert loader.check_against_dbt(ctx, dbt_yml) == []
    models[0]["columns"][0]["description"] = "drifted"
    dbt_yml.write_text(yaml.safe_dump({"models": models[:-1]}), encoding="utf-8")
    probs = loader.check_against_dbt(ctx, dbt_yml)
    assert any("differs from Tally dbt contract" in p for p in probs) and any(
        "no dbt model" in p for p in probs
    )

"""Golden-set evaluation (BUILD_SPEC §10): run the agent on every item, score it, write reports/runs/<run_id>/."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
from rich.progress import Progress
from sqlglot import exp

from lexicon.context import loader
from lexicon.context.schema import Level
from lexicon.eval import golden
from lexicon.eval.compare import execution_match
from lexicon.executor import ExecError, Executor, PostgresExecutor, ResultFrame
from lexicon.graph import Agent, AgentConfig, interrupt_payload, is_interrupted
from lexicon.llm import LLMClient
from lexicon.settings import RUNS_DIR, eval_dsn

GENERIC_ANSWER = "Use the most standard interpretation."


class _LocalExecutor:
    """One Postgres connection per worker thread."""

    dialect = "postgres"

    def __init__(self, factory: Callable[[], Executor]) -> None:
        self.factory = factory
        self.local = threading.local()

    def run(self, sql_text: str) -> ResultFrame:
        if not hasattr(self.local, "ex"):
            self.local.ex = self.factory()
        res: ResultFrame = self.local.ex.run(sql_text)
        return res


def gold_tables(sql_text: str) -> set[str]:
    tree = sqlglot.parse_one(sql_text, dialect="postgres")
    ctes = {c.alias_or_name for c in tree.find_all(exp.CTE)}
    return {f"{t.db}.{t.name}" for t in tree.find_all(exp.Table) if t.db and t.name not in ctes}


@dataclass
class RunSpec:
    run_id: str
    model: str
    level: Level
    concurrency: int = 8
    deterministic_triggers: bool = True
    limit: int | None = None
    item_ids: list[str] | None = None
    seed: int = 0
    context_dir: str | None = None  # alternate context folder (gate demo / experiments)


def evaluate(
    spec: RunSpec, llm: LLMClient | None = None, executor_factory: Callable[[], Executor] | None = None
) -> dict[str, Any]:
    ctx_dir = Path(spec.context_dir) if spec.context_dir else None
    ctx = loader.load(ctx_dir) if ctx_dir else loader.load()
    items = golden.load()
    if spec.item_ids:
        items = [i for i in items if i.id in set(spec.item_ids)]
    if spec.limit:
        items = items[: spec.limit]
    factory = executor_factory or (lambda: PostgresExecutor(eval_dsn()))
    ex = _LocalExecutor(factory)
    gold_ex = factory()
    gold: dict[str, ResultFrame] = {i.id: gold_ex.run(i.scored_sql) for i in items if i.scored_sql}
    names = _agent_names(gold_ex)
    client = llm or LLMClient()
    agent = Agent(
        ctx,
        client,
        ex,
        AgentConfig(
            model=spec.model,
            level=spec.level,
            seed=spec.seed,
            deterministic_triggers=spec.deterministic_triggers,
        ),
        agent_names=names,
    )

    def one(item: golden.Item) -> dict[str, Any]:
        t0 = time.perf_counter()
        tid, out = agent.start(item.question, f"{spec.run_id}:{item.id}")
        asked, options = False, []
        if is_interrupted(out):
            asked = True
            options = interrupt_payload(out).get("options", [])
            ans = item.clarification.simulated_answer if item.clarification else GENERIC_ANSWER
            out = agent.resume(tid, ans)
        rec = score(item, out, gold.get(item.id), asked)
        rec.update(
            {
                "options": options,
                "route": out.get("route"),
                "route_confidence": out.get("route_confidence"),
                "retrieved_tables": out.get("retrieved_tables", []),
                "sql": out.get("sql"),
                "raw_sql": out.get("raw_sql"),
                "rules_applied": out.get("rules_applied", []),
                "guard": out.get("guard"),
                "error": out.get("error"),
                "sanity": out.get("sanity", []),
                "attempts": out.get("attempts", 0),
                "calls": out.get("calls", []),
                "wall_ms": (time.perf_counter() - t0) * 1000,
            }
        )
        if item.scored_sql:
            gt = gold_tables(item.scored_sql)
            rec["retrieval_recall"] = len(gt & set(rec["retrieved_tables"])) / len(gt) if gt else 1.0
        return rec

    t_start = time.time()
    records: list[dict[str, Any]] = []
    from lexicon.cost import PowerSampler, cost_calls

    with (
        PowerSampler() as power,
        Progress(transient=True) as prog,
        ThreadPoolExecutor(spec.concurrency) as pool,
    ):
        task = prog.add_task(f"{spec.model.split('/')[-1]} {spec.level}", total=len(items))
        for rec in pool.map(one, items):
            records.append(rec)
            prog.advance(task)
    wall_s = time.time() - t_start
    records.sort(key=lambda r: r["id"])
    calls = [c for r in records for c in r["calls"]]
    live = [c for c in calls if not c["replayed"]]
    hw = _hardware(spec, wall_s, len(live), power.mean_w)
    costs = cost_calls(calls, hw["wall_seconds"], hw["n_live_calls"], hw["mean_power_w"]) if calls else []
    for c, usd in zip(calls, costs, strict=True):
        c["cost_usd"] = usd
    for r in records:
        r["cost_usd"] = sum(c["cost_usd"] for c in r["calls"])
    metrics = aggregate(records)
    meta = {
        "run_id": spec.run_id,
        "model": spec.model,
        "level": spec.level,
        "n_items": len(items),
        "golden_version": golden.version(),
        "context_fingerprint": (loader.fingerprint(ctx_dir) if ctx_dir else loader.fingerprint())[:16],
        "context_dir": spec.context_dir,
        "snapshot_sha256": _snapshot_sha(),
        "deterministic_triggers": spec.deterministic_triggers,
        "embeddings": agent.retriever.embeddings,
        "wall_seconds": wall_s,
        "n_calls": len(calls),
        "n_live_calls": len(live),
        "concurrency": spec.concurrency,
        "seed": spec.seed,
        "hardware": hw,
    }
    return {"meta": meta, "metrics": metrics, "records": records}


def _hardware(spec: RunSpec, wall_s: float, n_live: int, power_w: float | None) -> dict[str, Any]:
    """Throughput for the cost model: measured when the run was live; reused from the recording when replayed."""
    from lexicon.settings import RECORDINGS_DIR

    path = RECORDINGS_DIR / "hardware" / f"{spec.model.replace('/', '__')}__{spec.level}__s{spec.seed}.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if n_live > 0 and (stored is None or n_live >= stored["n_live_calls"]):
        hw = {
            "wall_seconds": wall_s,
            "n_live_calls": n_live,
            "mean_power_w": power_w,
            "concurrency": spec.concurrency,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(hw, indent=2) + "\n", encoding="utf-8")
        return hw
    if stored is not None:  # the measurement with more live calls is the better throughput estimate
        return dict(stored)
    return {
        "wall_seconds": wall_s,
        "n_live_calls": 0,
        "mean_power_w": power_w,
        "concurrency": spec.concurrency,
    }


def _snapshot_sha() -> str:
    from lexicon.snapshot import MANIFEST

    return str(json.loads(MANIFEST.read_text(encoding="utf-8"))["sha256"]) if MANIFEST.exists() else "unknown"


def _agent_names(ex: Executor) -> dict[str, list[dict[str, Any]]]:
    try:
        f = ex.run("select lower(display_name), agent_id, team, is_active from support.agents")
    except ExecError:
        return {}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, aid, team, active in f.rows:
        out[name].append({"agent_id": aid, "team": team, "is_active": active})
    return dict(out)


def score(item: golden.Item, out: dict[str, Any], gold: ResultFrame | None, asked: bool) -> dict[str, Any]:
    refusal = out.get("refusal")
    rec: dict[str, Any] = {
        "id": item.id,
        "bucket": item.bucket,
        "question": item.question,
        "asked": asked,
        "refused": refusal is not None,
        "refusal_type": (refusal or {}).get("type"),
    }
    res = out.get("result")
    ex_s = ex_l = False
    if gold is not None and res is not None and item.scored_sql:
        pred = ResultFrame(res["columns"], [tuple(r) for r in res["rows"]], res["truncated"])
        m = execution_match(gold, pred, item.scored_sql)
        ex_s, ex_l = m["strict"], m["lenient"]
    rec["ex_strict"], rec["ex_lenient"] = ex_s, ex_l
    if item.bucket == "should_refuse":
        assert item.refusal is not None
        rec["success"] = rec["refusal_type"] == item.refusal.type
    elif item.bucket == "should_clarify":
        rec["success"] = asked and ex_s
    else:
        rec["success"] = ex_s
    return rec


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[r["bucket"]].append(r)

    def rate(rows: list[dict[str, Any]], key: str) -> float | None:
        return sum(bool(r[key]) for r in rows) / len(rows) if rows else None

    answerable = [r for b in golden.ANSWERABLE for r in by.get(b, [])]
    clar = by.get("should_clarify", [])
    refuse = by.get("should_refuse", [])
    not_refuse = [r for r in records if r["bucket"] != "should_refuse"]
    asked_all = [r for r in records if r["asked"]]
    bucket_scores = {b: rate(by[b], "success") for b in golden.BUCKETS if by.get(b)}
    refusal_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    from lexicon.eval.golden import load as _load

    expected = {i.id: i.refusal.type for i in _load() if i.refusal}
    for r in refuse:
        refusal_confusion[expected[r["id"]]][r["refusal_type"] or "answered"] += 1
    calls = [c for r in records for c in r["calls"]]
    recall = [r["retrieval_recall"] for r in records if "retrieval_recall" in r]
    return {
        "ex_strict": rate(answerable, "ex_strict"),
        "ex_lenient": rate(answerable, "ex_lenient"),
        "task_success": rate(records, "success"),
        "bucket_scores": bucket_scores,
        "clarify_recall": rate(clar, "asked"),
        "clarify_precision": (sum(r["bucket"] == "should_clarify" for r in asked_all) / len(asked_all))
        if asked_all
        else None,
        "false_clarify_rate": rate(answerable, "asked"),
        "post_clarification_ex": rate(clar, "ex_strict"),
        "refusal_recall": rate(refuse, "refused"),
        "refusal_type_accuracy": rate(refuse, "success"),
        "over_refusal_rate": rate(not_refuse, "refused"),
        "refusal_confusion": {k: dict(v) for k, v in refusal_confusion.items()},
        "retrieval_recall": sum(recall) / len(recall) if recall else None,
        "mean_calls_per_question": len(calls) / len(records) if records else 0,
        "mean_input_tokens_per_question": sum(c["input_tokens"] for c in calls) / len(records)
        if records
        else 0,
        "mean_output_tokens_per_question": sum(c["output_tokens"] for c in calls) / len(records)
        if records
        else 0,
        "mean_attempts": sum(r["attempts"] for r in records) / len(records) if records else 0,
        "mean_cost_usd_per_question": sum(r.get("cost_usd", 0.0) for r in records) / len(records)
        if records
        else 0,
        "n": len(records),
    }


def write_run(result: dict[str, Any], root: Path = RUNS_DIR) -> Path:
    out: Path = root / str(result["meta"]["run_id"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(
        json.dumps(result["meta"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for r in result["records"]:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    return out

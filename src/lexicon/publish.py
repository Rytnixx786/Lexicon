"""Hand Lexicon's LLM traffic to Tally, so Tally measures Lexicon's cost (Lexicon BUILD_SPEC §10.4, TALLY_REQUESTS T7).

1. Register tenant `lexicon` and Lexicon's prompt versions in Tally's raw tables (CONTRACTS §2, §1).
2. Write each call's cost (Tally's cost model) to raw.call_costs.
3. Export one GenAI CLIENT span per call over OTLP to Tally's collector (tally.* attributes, no prompt text).
4. Run Tally's `dbt build` (Tally's own CLI), then read cost per question back from marts.fct_requests.
Lexicon writes data through the documented contract; it never edits Tally's code or models.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import psycopg
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind
from psycopg.types.json import Jsonb

from lexicon import __version__
from lexicon.cost import tally_root
from lexicon.llm import FEATURE, TENANT
from lexicon.settings import PROMPTS_DIR, RUNS_DIR, env


def run_calls(run_id: str) -> list[dict[str, Any]]:
    recs = [
        json.loads(x)
        for x in (RUNS_DIR / run_id / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    out = []
    for r in recs:
        for c in r.get("calls", []):
            out.append({**c, "question_id": r["id"], "run_id": run_id})
    return out


def prompt_rows(calls: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    rows = {}
    for c in calls:
        pv = c["prompt_version"]
        node = pv.split(".")[1]
        path = PROMPTS_DIR / f"{node}.v1.txt"
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "n/a"
        rows[pv] = (
            pv,
            FEATURE,
            h,
            f"Lexicon/prompts/{path.name}",
            0,
            f"Lexicon {node} prompt at context level {pv.split('.')[2]}",
        )
    return list(rows.values())


def register(conn: psycopg.Connection[Any], calls: list[dict[str, Any]]) -> None:
    conn.execute(
        "insert into raw.tenants (tenant_id, display_name, owner_project, monthly_budget_usd, traffic_mix) "
        "values ('lexicon', 'Lexicon SQL agent', 'lexicon', 5, %s) on conflict (tenant_id) do nothing",
        (Jsonb({"feature": FEATURE}),),
    )
    with conn.cursor() as cur:
        cur.executemany(
            "insert into raw.prompt_registry (prompt_version, feature, prompt_hash, template_path, n_fewshot, description) "
            "values (%s, %s, %s, %s, %s, %s) on conflict (prompt_version) do update set prompt_hash = excluded.prompt_hash",
            prompt_rows(calls),
        )
        cur.executemany(
            "insert into raw.call_costs (request_id, run_id, cost_usd, cost_method) values (%s, %s, %s, %s) "
            "on conflict (request_id) do update set cost_usd = excluded.cost_usd",
            [
                (
                    c["request_id"],
                    f"lexicon:{c['run_id']}",
                    float(c.get("cost_usd", 0.0)),
                    "owned_gpu_token_weighted",
                )
                for c in calls
            ],
        )
    conn.commit()


def export_spans(calls: list[dict[str, Any]], endpoint: str, exporter: SpanExporter | None = None) -> int:
    provider = TracerProvider(
        resource=Resource.create({"service.name": "lexicon-agent", "service.version": __version__})
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            exporter or OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"),
            max_queue_size=100_000,
            max_export_batch_size=1024,
        )
    )
    tracer = provider.get_tracer("lexicon", __version__)
    for c in calls:
        start = int(float(c["started_at"]) * 1e9)
        span = tracer.start_span(
            f"chat {c['model']}",
            kind=SpanKind.CLIENT,
            start_time=start,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "vllm",
                "gen_ai.request.model": c["model"],
                "gen_ai.usage.input_tokens": int(c["input_tokens"]),
                "gen_ai.usage.output_tokens": int(c["output_tokens"]),
                "tally.request_id": c["request_id"],
                "tally.tenant_id": TENANT,
                "tally.feature": FEATURE,
                "tally.prompt_version": c["prompt_version"],
                "tally.cache_status": "BYPASS",
                "tally.http_status": int(c["http_status"]),
                "tally.latency_ms": float(c["latency_ms"]),
                "lexicon.run_id": c["run_id"],
                "lexicon.question_id": c["question_id"],
                "lexicon.node": c["node"],
            },
        )
        span.end(end_time=start + int(float(c["latency_ms"]) * 1e6))
    provider.force_flush(120_000)
    provider.shutdown()
    return len(calls)


def tally_dbt_build() -> int:
    exe = tally_root() / ".venv" / "Scripts" / "tally.exe"
    if not exe.exists():
        exe = tally_root() / ".venv" / "bin" / "tally"
    return subprocess.call([str(exe), "dbt", "build"], cwd=str(tally_root()))


def cost_from_tally(run_id: str) -> dict[str, Any]:
    """Read cost per question back from Tally's public mart (via the read-only role)."""
    with psycopg.connect(env("WAREHOUSE_DSN")) as conn:
        rows = conn.execute(
            "select request_id::text, cost_usd::float, input_tokens, output_tokens from marts.fct_requests "
            "where tenant_id = 'lexicon' and feature = 'sql_agent'"
        ).fetchall()
    by_req = {r[0]: r for r in rows}
    calls = run_calls(run_id)
    per_q: dict[str, float] = {}
    missing = 0
    for c in calls:
        row = by_req.get(c["request_id"])
        if row is None:
            missing += 1
            continue
        per_q[c["question_id"]] = per_q.get(c["question_id"], 0.0) + row[1]
    n_q = len({c["question_id"] for c in calls})
    return {
        "run_id": run_id,
        "n_calls": len(calls),
        "calls_found_in_tally": len(calls) - missing,
        "questions": n_q,
        "mean_cost_usd_per_question": sum(per_q.values()) / max(n_q, 1),
        "total_cost_usd": sum(per_q.values()),
        "source": "Tally marts.fct_requests",
    }


class TallyContractError(RuntimeError):
    pass


def tally_accepts_non_triage_traffic(marts_yml: Path | None = None) -> bool:
    """Tally's dbt tests must scope the fct_requests.decision_id tests to triage traffic (TALLY_REQUESTS T8).
    Without that, Lexicon's calls (which are not triage decisions) fail Tally's build."""
    import yaml

    path = marts_yml or tally_root() / "dbt" / "models" / "marts" / "_marts.yml"
    models = {m["name"]: m for m in yaml.safe_load(path.read_text(encoding="utf-8"))["models"]}
    col = next(c for c in models["fct_requests"]["columns"] if c["name"] == "decision_id")
    for test in col.get("data_tests", []):
        cfg = test.get("config", {}) if isinstance(test, dict) else {}
        if test == "not_null" or (
            isinstance(test, dict) and "where" not in str(cfg) and "where" not in str(test)
        ):
            return False
    return True


def publish(run_id: str, *, build: bool = True, wait_s: int = 180, force: bool = False) -> dict[str, Any]:
    if not force and not tally_accepts_non_triage_traffic():
        raise TallyContractError(
            "Tally's contract requires every request to have a triage decision_id (not_null + relationships on "
            "fct_requests.decision_id), so Lexicon's calls would fail Tally's dbt build. Apply TALLY_REQUESTS T8 in "
            "the Tally repo first. Cost per question is still computed with Tally's cost model in each run."
        )
    calls = run_calls(run_id)
    with psycopg.connect(env("TALLY_WRITER_DSN")) as conn:
        register(conn, calls)
    export_spans(calls, env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"))
    deadline = time.time() + wait_s
    with psycopg.connect(env("TALLY_WRITER_DSN")) as conn:
        while True:
            row = conn.execute(
                "select count(*) from raw.otel_spans where attributes->>'lexicon.run_id' = %s", (run_id,)
            ).fetchone()
            if row and row[0] >= len(calls):
                break
            if time.time() > deadline:
                raise TimeoutError(f"only {row[0] if row else 0}/{len(calls)} spans reached Tally")
            time.sleep(2)
    if build and tally_dbt_build() != 0:
        raise RuntimeError("Tally dbt build failed after publishing Lexicon traffic")
    res = cost_from_tally(run_id)
    (RUNS_DIR / run_id / "tally_cost.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


def check_tally_path(p: Path | None = None) -> bool:
    return (p or tally_root()).exists()

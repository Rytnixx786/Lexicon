"""Frozen evaluation warehouse (ADR-004, ADR-005).

export   Tally `marts.*` (read through the read-only role) -> evals/snapshot/marts/<table>.parquet + schema.json
restore  parquet -> lexicon_eval.marts.*, plus the seeded synthetic `support` schema
manifest row counts and a content hash per table, so a gate run can refuse a mismatched snapshot
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from lexicon.settings import SNAPSHOT_DIR, SNAPSHOT_TODAY

MARTS = ["fct_requests", "fct_outcomes", "dim_tenant", "dim_prompt_version", "mart_cost_per_outcome"]
MARTS_DIR = SNAPSHOT_DIR / "marts"
MANIFEST = SNAPSHOT_DIR / "manifest.json"
SUPPORT_TABLES = ["agents", "sla_policies", "tickets", "ticket_events"]
SEED = 20261121


# ------------------------------------------------------------------ export from Tally


def export_marts(warehouse_dsn: str, out: Path = MARTS_DIR) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    schema: dict[str, list[dict[str, str]]] = {}
    counts: dict[str, int] = {}
    with psycopg.connect(warehouse_dsn) as conn:
        for t in MARTS:
            cols = conn.execute(
                "select column_name, data_type, numeric_precision, numeric_scale from information_schema.columns "
                "where table_schema = 'marts' and table_name = %s order by ordinal_position",
                (t,),
            ).fetchall()
            schema[t] = [{"name": c[0], "type": _pg_type(c[1], c[2], c[3])} for c in cols]
            with conn.cursor() as cur:
                cur.execute(sql.SQL("select * from marts.{} order by 1").format(sql.Identifier(t)))
                rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=[c["name"] for c in schema[t]])
            for c in schema[t]:
                if c["type"] in ("uuid", "jsonb", "date") or c["type"].startswith("numeric"):
                    df[c["name"]] = df[c["name"]].map(
                        lambda v: None if v is None else (json.dumps(v) if isinstance(v, dict) else str(v))
                    )
            df.to_parquet(out / f"{t}.parquet", index=False)
            counts[t] = len(df)
    (out / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return counts


def _pg_type(data_type: str, precision: int | None, scale: int | None) -> str:
    if data_type == "numeric" and precision is not None:
        return f"numeric({precision},{scale or 0})"
    return {"timestamp with time zone": "timestamptz", "character varying": "text"}.get(data_type, data_type)


# ------------------------------------------------------------------ restore


def _copy_frame(conn: psycopg.Connection[Any], schema_name: str, table: str, df: pd.DataFrame) -> None:
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in df.columns)
    stmt = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
        sql.Identifier(schema_name), sql.Identifier(table), cols
    )
    with conn.cursor() as cur, cur.copy(stmt) as cp:
        for row in df.itertuples(index=False, name=None):
            cp.write_row([None if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in row])


def restore(admin_dsn: str, src: Path = MARTS_DIR, seed: int = SEED) -> dict[str, Any]:
    schema = json.loads((src / "schema.json").read_text(encoding="utf-8"))
    with psycopg.connect(admin_dsn) as conn:
        conn.execute("DROP SCHEMA IF EXISTS marts CASCADE")
        conn.execute("DROP SCHEMA IF EXISTS support CASCADE")
        conn.execute("CREATE SCHEMA marts")
        for t in MARTS:
            ddl = sql.SQL("CREATE TABLE marts.{} ({})").format(
                sql.Identifier(t),
                sql.SQL(", ").join(
                    sql.SQL("{} " + c["type"]).format(sql.Identifier(c["name"])) for c in schema[t]
                ),
            )
            conn.execute(ddl)
            _copy_frame(conn, "marts", t, pd.read_parquet(src / f"{t}.parquet"))
        conn.execute("CREATE SCHEMA support")
        for name, df in synthetic_support(pd.read_parquet(src / "fct_outcomes.parquet"), seed).items():
            conn.execute(SUPPORT_DDL[name])
            _copy_frame(conn, "support", name, df)
        conn.execute("ALTER TABLE marts.fct_outcomes ADD PRIMARY KEY (decision_id)")
        conn.execute("ALTER TABLE marts.fct_requests ADD PRIMARY KEY (request_id)")
        for stmt in (
            "GRANT USAGE ON SCHEMA marts, support TO lexicon_ro",
            "GRANT SELECT ON ALL TABLES IN SCHEMA marts, support TO lexicon_ro",
            "ANALYZE",
        ):
            conn.execute(stmt)
        conn.commit()
    return write_manifest(admin_dsn)


# ------------------------------------------------------------------ synthetic ticket lifecycle

SUPPORT_DDL = {
    "agents": """CREATE TABLE support.agents (
        agent_id text PRIMARY KEY, display_name text NOT NULL, team text NOT NULL, email text NOT NULL,
        hourly_cost_usd numeric(8,2) NOT NULL, is_active boolean NOT NULL, hired_on date NOT NULL)""",
    "sla_policies": """CREATE TABLE support.sla_policies (
        tenant_id text NOT NULL, priority text NOT NULL, first_response_minutes int NOT NULL,
        resolution_hours int NOT NULL, PRIMARY KEY (tenant_id, priority))""",
    "tickets": """CREATE TABLE support.tickets (
        ticket_id text PRIMARY KEY, tenant_id text NOT NULL, decision_id uuid, channel text NOT NULL,
        priority text NOT NULL, opened_at timestamptz NOT NULL, assigned_agent_id text,
        status text NOT NULL, ai_predicted_intent text, final_intent text NOT NULL)""",
    "ticket_events": """CREATE TABLE support.ticket_events (
        event_id bigint PRIMARY KEY, ticket_id text NOT NULL, event_type text NOT NULL,
        event_at timestamptz NOT NULL, actor_type text NOT NULL, agent_id text)""",
}

TEAMS = {"support": 10, "billing": 7, "sales": 5}
FIRST = [
    "Ayesha",
    "Rahim",
    "Nadia",
    "Tanvir",
    "Farhana",
    "Imran",
    "Sadia",
    "Karim",
    "Lamia",
    "Arif",
    "Nusrat",
    "Sabbir",
]
LAST = ["Rahman", "Hossain", "Akter", "Islam", "Chowdhury", "Ahmed", "Khan", "Sarkar"]


def synthetic_support(outcomes: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Seeded, documented synthetic data keyed to Tally's decisions. Traps (GOLDEN_SET_GUIDE §4):
    reopened tickets, bot vs human resolution, a test tenant to exclude, an inactive agent, PII emails."""
    rng = np.random.default_rng(seed)
    today = datetime.fromisoformat(SNAPSHOT_TODAY).replace(tzinfo=UTC)

    agents = []
    n = 0
    for team, k in TEAMS.items():
        for _ in range(k):
            n += 1
            name = f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}"
            agents.append(
                {
                    "agent_id": f"ag-{n:02d}",
                    "display_name": name,
                    "team": team,
                    "email": f"{name.lower().replace(' ', '.')}.{n}@example.com",
                    "hourly_cost_usd": float(rng.choice([4.5, 5.0, 6.0, 7.5])),
                    "is_active": bool(rng.uniform() > 0.12),
                    "hired_on": (today - timedelta(days=int(rng.integers(60, 1400)))).date(),
                }
            )
    agents.append(
        {
            "agent_id": "ag-qa",
            "display_name": "Load Test Bot",
            "team": "test-loadgen",
            "email": "loadtest@example.com",
            "hourly_cost_usd": 0.0,
            "is_active": True,
            "hired_on": (today - timedelta(days=30)).date(),
        }
    )
    agents_df = pd.DataFrame(agents)

    sla = []
    for tenant in [*TEAMS, "test-loadgen"]:
        for pri, (fr, rh) in {"high": (15, 8), "normal": (60, 24), "low": (240, 72)}.items():
            sla.append(
                {"tenant_id": tenant, "priority": pri, "first_response_minutes": fr, "resolution_hours": rh}
            )
    sla_df = pd.DataFrame(sla)

    base = outcomes[
        (outcomes["setup"] == "B_local") & (outcomes["split"] == "test") & (outcomes["arm"] == "primary")
    ]
    base = base.sort_values("ticket_id").reset_index(drop=True)
    test_rows = pd.DataFrame(
        {
            "ticket_id": [f"lt-{i:04d}" for i in range(300)],
            "tenant_id": "test-loadgen",
            "decision_id": None,
            "predicted_label": "track_order",
            "gold_label": "track_order",
        }
    )
    rows = pd.concat(
        [base[["ticket_id", "tenant_id", "decision_id", "predicted_label", "gold_label"]], test_rows],
        ignore_index=True,
    )

    tickets, events = [], []
    eid = 0
    by_team = {t: agents_df[(agents_df["team"] == t)]["agent_id"].tolist() for t in [*TEAMS, "test-loadgen"]}
    for r in rows.itertuples(index=False):
        opened = today - timedelta(days=float(rng.uniform(1, 80)), minutes=float(rng.uniform(0, 1440)))
        priority = str(rng.choice(["low", "normal", "high"], p=[0.3, 0.55, 0.15]))
        tenant = str(r.tenant_id)
        agent = str(rng.choice(by_team[tenant])) if tenant in by_team else None
        pred: Any = r.predicted_label
        misrouted = bool(pd.isna(pred) or str(pred) != str(r.gold_label))
        evs: list[tuple[str, datetime, str, str | None]] = [
            ("opened", opened, "customer", None),
            ("triaged", opened + timedelta(seconds=float(rng.uniform(1, 5))), "bot", None),
        ]
        t = opened + timedelta(minutes=float(rng.uniform(2, 240)))
        bot_resolves = (
            (not misrouted)
            and r.gold_label in {"track_order", "delivery_period", "check_invoice", "check_payment_methods"}
            and rng.uniform() < 0.6
        )
        if not bot_resolves:
            evs.append(("assigned", t, "bot", agent))
            t += timedelta(minutes=float(rng.uniform(5, 180)))
            if misrouted and rng.uniform() < 0.7:
                evs.append(("escalated_to_human", t, "agent", agent))
                t += timedelta(hours=float(rng.uniform(1, 20)))
        status = "open" if rng.uniform() < 0.06 else "resolved"
        if status == "resolved":
            evs.append(("resolved", t, "bot" if bot_resolves else "agent", None if bot_resolves else agent))
            if rng.uniform() < (0.35 if misrouted else 0.08):
                t += timedelta(hours=float(rng.uniform(2, 72)))
                evs.append(("reopened", t, "customer", None))
                if rng.uniform() < 0.6:
                    t += timedelta(hours=float(rng.uniform(1, 30)))
                    evs.append(("resolved", t, "agent", agent))
                else:
                    status = "open"
        if status == "resolved" and rng.uniform() < 0.8:
            evs.append(("closed", t + timedelta(hours=48), "bot", None))
            status = "closed"
        tickets.append(
            {
                "ticket_id": r.ticket_id,
                "tenant_id": r.tenant_id,
                "decision_id": r.decision_id,
                "channel": str(rng.choice(["email", "chat", "web"], p=[0.4, 0.45, 0.15])),
                "priority": priority,
                "opened_at": opened,
                "assigned_agent_id": None if bot_resolves else agent,
                "status": status,
                "ai_predicted_intent": r.predicted_label,
                "final_intent": r.gold_label,
            }
        )
        for et, at, actor, ag in evs:
            if at > today:
                at = today - timedelta(minutes=1)
            eid += 1
            events.append(
                {
                    "event_id": eid,
                    "ticket_id": r.ticket_id,
                    "event_type": et,
                    "event_at": at,
                    "actor_type": actor,
                    "agent_id": ag,
                }
            )
    return {
        "agents": agents_df,
        "sla_policies": sla_df,
        "tickets": pd.DataFrame(tickets),
        "ticket_events": pd.DataFrame(events),
    }


# ------------------------------------------------------------------ manifest


def table_fingerprints(dsn: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with psycopg.connect(dsn) as conn:
        for schema_name, tables in (("marts", MARTS), ("support", SUPPORT_TABLES)):
            for t in tables:
                q = sql.SQL(
                    "select count(*), md5(coalesce(string_agg(md5(x::text), '' order by md5(x::text)), '')) from {}.{} x"
                ).format(sql.Identifier(schema_name), sql.Identifier(t))
                row = conn.execute(q).fetchone()
                assert row is not None
                out[f"{schema_name}.{t}"] = {"rows": int(row[0]), "md5": str(row[1])}
    return out


def write_manifest(dsn: str) -> dict[str, Any]:
    fp = table_fingerprints(dsn)
    digest = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()
    manifest = {
        "snapshot_today": SNAPSHOT_TODAY,
        "seed": SEED,
        "tables": fp,
        "sha256": digest,
        "source": "Tally marts, run main (exported via lexicon_ro)",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def snapshot_hash() -> str:
    return str(json.loads(MANIFEST.read_text(encoding="utf-8"))["sha256"])

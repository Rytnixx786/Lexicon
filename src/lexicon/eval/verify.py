"""Re-verify golden SQL and context examples against the snapshot; a snapshot change forces re-verification."""

from __future__ import annotations

import datetime as dt
import json

from lexicon.context.schema import Context
from lexicon.eval import golden
from lexicon.eval.compare import result_hash
from lexicon.executor import ExecError, Executor, PostgresExecutor
from lexicon.settings import CONTEXT_DIR, eval_dsn

EXAMPLES_FILE = CONTEXT_DIR / "_verified_examples.json"


def _run_all(pairs: list[tuple[str, str]], ex: Executor) -> tuple[dict[str, dict[str, object]], list[str]]:
    out: dict[str, dict[str, object]] = {}
    problems: list[str] = []
    for key, sql_text in pairs:
        try:
            r = ex.run(sql_text)
        except ExecError as e:
            problems.append(f"{key}: {e}")
            continue
        if r.row_count == 0:
            problems.append(f"{key}: empty result")
        out[key] = {"rows": r.row_count, "hash": result_hash(r)}
    return out, problems


def _compare(found: dict[str, dict[str, object]], stored: dict[str, dict[str, object]]) -> list[str]:
    problems = [
        f"{k}: result hash changed"
        for k, v in found.items()
        if k in stored and stored[k]["hash"] != v["hash"]
    ]
    problems += [f"{k}: not verified yet (run with --update)" for k in found if k not in stored]
    return problems


def verify_golden(update: bool = False, ex: Executor | None = None) -> list[str]:
    items = golden.load()
    found, problems = _run_all(
        [(i.id, i.scored_sql) for i in items if i.scored_sql], ex or PostgresExecutor(eval_dsn())
    )
    if update and not problems:
        golden.VERIFIED_FILE.write_text(
            json.dumps(
                {
                    "golden_version": golden.version(),
                    "verified_at": dt.date.today().isoformat(),
                    "items": found,
                },
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return []
    return problems + _compare(found, golden.load_verified())


def verify_examples(ctx: Context, update: bool = False, ex: Executor | None = None) -> list[str]:
    found, problems = _run_all([(e.id, e.sql) for e in ctx.examples], ex or PostgresExecutor(eval_dsn()))
    stored = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))["items"] if EXAMPLES_FILE.exists() else {}
    if update and not problems:
        EXAMPLES_FILE.write_text(
            json.dumps({"verified_at": dt.date.today().isoformat(), "items": found}, indent=1, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return []
    return problems + _compare(found, stored)

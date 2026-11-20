"""Execution-accuracy comparison (BUILD_SPEC §10.2).

strict:  same rows (multiset, or sequence when gold has an outer ORDER BY) over the same number of columns,
         matching columns by value content rather than by name or position.
lenient: every gold column appears among the predicted columns (extra predicted columns allowed).
Floats compare at rel 1e-6 / abs 1e-9; NULL equals NULL.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from itertools import permutations
from typing import Any

import sqlglot
from sqlglot import exp

from lexicon.executor import ResultFrame

REL, ABS = 1e-6, 1e-9


def _canon(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return None
        return round(f, 6) if abs(f) >= 1e-3 else float(f"{f:.6g}")
    if isinstance(v, dt.datetime):
        return v.astimezone(dt.UTC).isoformat() if v.tzinfo else v.isoformat()
    if isinstance(v, (dt.date, dt.timedelta)):
        return str(v)
    return str(v)


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=REL, abs_tol=ABS)
    if isinstance(a, float) != isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=REL, abs_tol=ABS)
        except (TypeError, ValueError):
            return False
    return bool(a == b)


def _columns(frame: ResultFrame) -> list[list[Any]]:
    return [[_canon(r[i]) for r in frame.rows] for i in range(len(frame.columns))]


def _rows_equal(gold_rows: list[tuple[Any, ...]], pred_rows: list[tuple[Any, ...]], ordered: bool) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False
    if ordered:
        return all(
            all(_close(a, b) for a, b in zip(g, p, strict=True))
            for g, p in zip(gold_rows, pred_rows, strict=True)
        )
    remaining = list(pred_rows)
    for g in sorted(gold_rows, key=repr):
        hit = next(
            (i for i, p in enumerate(remaining) if all(_close(a, b) for a, b in zip(g, p, strict=True))), None
        )
        if hit is None:
            return False
        remaining.pop(hit)
    return True


def has_outer_order_by(sql_text: str, dialect: str = "postgres") -> bool:
    try:
        tree = sqlglot.parse_one(sql_text, dialect=dialect)
    except Exception:
        return False
    if tree is None or tree.args.get("order") is None:
        return False
    limit = tree.args.get("limit")
    return not (
        isinstance(limit, exp.Limit)
        and isinstance(limit.expression, exp.Literal)
        and limit.expression.this == "1"
    )


def _match(gold: ResultFrame, pred: ResultFrame, ordered: bool, lenient: bool) -> bool:
    gcols, pcols = _columns(gold), _columns(pred)
    if len(gold.rows) != len(pred.rows):
        return False
    if not lenient and len(gcols) != len(pcols):
        return False
    if len(gcols) > len(pcols) or len(gcols) == 0:
        return len(gcols) == len(pcols) == 0
    if len(pcols) > 7:  # cap the permutation search
        idx = list(range(len(gcols)))
        cands = [tuple(idx)] if len(pcols) >= len(gcols) else []
    else:
        cands = list(permutations(range(len(pcols)), len(gcols)))
    gold_rows = [tuple(c[i] for c in gcols) for i in range(len(gold.rows))]
    for perm in cands:
        pred_rows = [tuple(pcols[j][i] for j in perm) for i in range(len(pred.rows))]
        if _rows_equal(gold_rows, pred_rows, ordered):
            return True
    return False


def execution_match(
    gold: ResultFrame, pred: ResultFrame, gold_sql: str, dialect: str = "postgres"
) -> dict[str, bool]:
    ordered = has_outer_order_by(gold_sql, dialect)
    return {"strict": _match(gold, pred, ordered, False), "lenient": _match(gold, pred, ordered, True)}


def result_hash(frame: ResultFrame) -> str:
    rows = sorted(json.dumps([_canon(v) for v in r], default=str) for r in frame.rows)
    return "sha256:" + hashlib.sha256("\n".join(rows).encode()).hexdigest()[:24]

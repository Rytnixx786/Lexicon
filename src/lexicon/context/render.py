"""Render retrieved context as text for one ablation level (CONTEXT_YAML_SPEC §7).

L0 schema only (DDL)  ·  L1 + descriptions, grain, synonyms, sample values  ·  L2 + metrics, rules, joins, guidance
L3 + verified example queries. Deterministic output; snapshot-tested.
"""

from __future__ import annotations

from lexicon.context.schema import Context, Level
from lexicon.retrieval import Retrieved

_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def at_least(level: Level, target: Level) -> bool:
    return _ORDER[level] >= _ORDER[target]


def catalog(ctx: Context) -> str:
    return "All tables in the warehouse: " + ", ".join(sorted(ctx.tables))


def render(ctx: Context, r: Retrieved, level: Level) -> str:
    out: list[str] = [catalog(ctx), ""]
    for t in r.tables:
        if level == "L0":
            cols = ",\n  ".join(f"{c.name} {c.type}" for c in t.columns)
            out.append(f"CREATE TABLE {t.table} (\n  {cols}\n);")
            continue
        out.append(f"TABLE {t.table}: {t.description}")
        out.append(f"  grain: {t.grain}")
        if t.synonyms:
            out.append(f"  also called: {', '.join(t.synonyms)}")
        if at_least(level, "L2") and t.ai_context:
            out.append(f"  guidance: {t.ai_context}")
        samples = ctx.samples.get(t.table, {})
        for c in t.columns:
            bits = [f"  - {c.name} ({c.type}): {c.description}"]
            if c.synonyms:
                bits.append(f"synonyms: {', '.join(c.synonyms)}")
            if c.value_meanings:
                bits.append("values: " + "; ".join(f"{k} = {v}" for k, v in c.value_meanings.items()))
            elif c.name in samples:
                bits.append("values: " + ", ".join(samples[c.name]))
            if c.unit:
                bits.append(f"unit: {c.unit}")
            out.append(" | ".join(bits))
        out.append("")
    if at_least(level, "L2"):
        if r.metrics:
            out.append("METRICS (use these exact definitions):")
            for m in r.metrics:
                if m.requires_choice:
                    opts = "; ".join(f"{c.id}: {c.description}" for c in m.choices)
                    out.append(f"- {m.id}: AMBIGUOUS, must ask the user which one. Options: {opts}")
                elif m.type == "ratio":
                    out.append(
                        f"- {m.id} = {m.numerator} / {m.denominator} on {m.base_table}. {m.description}"
                    )
                else:
                    out.append(f"- {m.id} = {m.expr} on {m.base_table}. {m.description}")
            out.append("")
        if r.rules:
            out.append("BUSINESS RULES (always apply):")
            out.extend(
                f"- [{x.id}] {x.statement}" + (f" Hint: {x.sql_hint}" if x.sql_hint else "") for x in r.rules
            )
            out.append("")
        names = set(r.table_names)
        joins = [j for j in ctx.joins.joins if j.left in names and j.right in names]
        if joins:
            out.append("ALLOWED JOINS:")
            out.extend(
                f"- {j.left} -> {j.right} on {j.condition} ({j.relationship})"
                + (f"; {j.note}" if j.note else "")
                for j in joins
            )
        bad = [f for f in ctx.joins.forbidden if f.pair[0] in names and f.pair[1] in names]
        if bad:
            out.extend(f"- DO NOT join {f.pair[0]} directly to {f.pair[1]}: {f.reason}" for f in bad)
        out.append("")
    if at_least(level, "L3") and r.examples:
        out.append("VERIFIED EXAMPLE QUERIES:")
        for e in r.examples:
            out.append(f"Q: {e.question}\nSQL: {' '.join(e.sql.split())}")
        out.append("")
    return "\n".join(out).strip() + "\n"

"""SQL guard (BUILD_SPEC §7): the first of two safety layers. The read-only role and transaction are the second.

Order matters; every rule appends a violation and the guard fails if there is any:
  1 exactly one statement            5 tables: allowlist only, CTE aliases excluded, unqualified names resolved
  2 root is SELECT or a set operation 6 columns must exist (when a schema is given)
  3 no write/DDL/session nodes        7 LIMIT added or clamped on the outermost query
  4 function denylist                 8 return the SQL regenerated from the checked AST
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer.qualify import qualify

MAX_ROWS = 1000

_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, n)
    for n in (
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Drop",
        "Create",
        "Alter",
        "TruncateTable",
        "Command",
        "Copy",
        "Set",
        "Transaction",
        "Commit",
        "Rollback",
        "Grant",
        "Revoke",
        "Into",
        "Lock",
        "Pragma",
        "Use",
        "Analyze",
        "Comment",
        "Attach",
        "Detach",
        "Kill",
        "Refresh",
        "Cache",
        "Uncache",
    )
    if hasattr(exp, n)
)
_DENY_EXACT = {
    "set_config",
    "current_setting",
    "nextval",
    "setval",
    "currval",
    "lastval",
    "query_to_xml",
    "query_to_xml_and_xmlschema",
    "table_to_xml",
    "database_to_xml",
    "schema_to_xml",
    "load_extension",
    "readfile",
    "writefile",
    "version",
    "current_user",
    "session_user",
    "inet_server_addr",
    "inet_client_addr",
}
_DENY_PREFIX = ("pg_", "lo_", "dblink", "txid_", "brin_", "gin_", "binary_upgrade")
_DENY_SCHEMAS = {
    "pg_catalog",
    "information_schema",
    "pg_toast",
    "sqlite_master",
    "sqlite_schema",
    "main.sqlite_master",
}


@dataclass(frozen=True)
class Violation:
    code: str  # multiple_statements | parse_error | not_select | forbidden_statement | forbidden_function | table_not_allowed | undefined_column
    message: str


@dataclass
class GuardReport:
    ok: bool
    sql: str | None
    violations: list[Violation] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    limit_added: bool = False

    @property
    def codes(self) -> list[str]:
        return [v.code for v in self.violations]


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return (node.sql_name() if hasattr(node, "sql_name") else type(node).__name__).lower()


def check(
    sql_text: str,
    allowed_tables: set[str],
    *,
    dialect: str = "postgres",
    columns: Mapping[str, Mapping[str, str]] | None = None,
    max_rows: int = MAX_ROWS,
) -> GuardReport:
    """allowed_tables: fully qualified 'schema.table' names (lowercase). columns: {'schema.table': {col: type}}."""
    v: list[Violation] = []
    text = sql_text.strip().rstrip(";").strip()
    if not text:
        return GuardReport(False, None, [Violation("parse_error", "empty SQL")])
    try:
        statements = [s for s in sqlglot.parse(text, dialect=dialect) if s is not None]
    except SqlglotError as e:  # ParseError and TokenError (e.g. an unclosed quote)
        return GuardReport(False, None, [Violation("parse_error", str(e).splitlines()[0][:300])])
    if len(statements) != 1:
        return GuardReport(
            False, None, [Violation("multiple_statements", f"expected 1 statement, got {len(statements)}")]
        )
    tree = statements[0]
    while isinstance(tree, exp.Subquery) and isinstance(tree.this, exp.Query):
        tree = tree.this
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        v.append(
            Violation("not_select", f"only SELECT queries are allowed, got {type(tree).__name__.upper()}")
        )
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            v.append(Violation("forbidden_statement", f"{type(node).__name__.upper()} is not allowed"))
        elif isinstance(node, exp.Func):
            name = _func_name(node)
            if name in _DENY_EXACT or name.startswith(_DENY_PREFIX):
                v.append(Violation("forbidden_function", f"function {name}() is not allowed"))
    if v:
        return GuardReport(False, None, v)

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    by_name: dict[str, list[str]] = {}
    for full_name in allowed_tables:
        by_name.setdefault(full_name.split(".")[-1], []).append(full_name)
    used: list[str] = []
    for t in list(tree.find_all(exp.Table)):
        if not isinstance(t.this, exp.Identifier):
            v.append(
                Violation(
                    "forbidden_function", f"table function {t.this.sql(dialect=dialect)} is not allowed"
                )
            )
            continue
        name, schema = t.name.lower(), (t.db or "").lower()
        if not schema and name in cte_names:
            continue
        if schema in _DENY_SCHEMAS or name in _DENY_SCHEMAS:
            v.append(
                Violation(
                    "table_not_allowed",
                    f"system catalog {schema + '.' if schema else ''}{name} is not allowed",
                )
            )
            continue
        if not schema:
            cands = by_name.get(name, [])
            if len(cands) == 1:
                schema = cands[0].split(".")[0]
                t.set("db", exp.to_identifier(schema))
            else:
                v.append(
                    Violation(
                        "table_not_allowed",
                        f"table {name} is not in the allowlist"
                        if not cands
                        else f"table {name} is ambiguous; qualify it",
                    )
                )
                continue
        full = f"{schema}.{name}"
        if full not in allowed_tables:
            v.append(Violation("table_not_allowed", f"table {full} is not in the allowlist"))
            continue
        used.append(full)
    if v:
        return GuardReport(False, None, v, sorted(set(used)))

    if columns is not None:
        mapping: dict[str, dict[str, dict[str, str]]] = {}
        for full, cols in columns.items():
            s, n = full.split(".")
            mapping.setdefault(s, {})[n] = dict(cols)
        try:
            qualify(
                tree.copy(),
                schema=cast(dict[str, object], mapping),
                dialect=dialect,
                validate_qualify_columns=True,
            )
        except OptimizeError as e:
            msg = str(e).splitlines()[0][:300]
            if "column" in msg.lower():
                v.append(Violation("undefined_column", msg))
        except Exception:
            pass
    if v:
        return GuardReport(False, None, v, sorted(set(used)))

    added = False
    limit = tree.args.get("limit")
    if limit is None:
        tree = cast(exp.Query, tree).limit(max_rows, copy=False)
        added = True
    else:
        lit = limit.expression if isinstance(limit, exp.Limit) else None
        if isinstance(lit, exp.Literal) and lit.is_int and int(lit.this) > max_rows:
            limit.set("expression", exp.Literal.number(max_rows))
    return GuardReport(True, tree.sql(dialect=dialect), [], sorted(set(used)), added)

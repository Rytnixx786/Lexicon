"""Execution layer (BUILD_SPEC §8): read-only role, read-only transaction, statement timeout, row cap.

Errors are classified and returned without row data, so they can be fed back to the generator safely.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from psycopg import errors as pg

from lexicon.sql_guard import MAX_ROWS


@dataclass
class ResultFrame:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool = False
    elapsed_ms: float = 0.0

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass
class ExecError(Exception):
    kind: str  # syntax | undefined_column | undefined_table | timeout | permission_denied | other
    message: str
    elapsed_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"


def _norm(v: Any) -> Any:
    return float(v) if isinstance(v, Decimal) else v


class Executor(Protocol):
    dialect: str

    def run(self, sql_text: str) -> ResultFrame: ...


class PostgresExecutor:
    dialect = "postgres"

    def __init__(self, dsn: str, *, timeout_ms: int = 5000, max_rows: int = MAX_ROWS) -> None:
        self.dsn = dsn
        self.timeout_ms = timeout_ms
        self.max_rows = max_rows
        self._conn: psycopg.Connection[Any] | None = None

    def _connection(self) -> psycopg.Connection[Any]:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=False)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def _reset(self, conn: psycopg.Connection[Any]) -> None:
        """Roll back; a connection left mid-COPY or broken cannot be reused, so it is discarded."""
        try:
            conn.rollback()
        except psycopg.Error:
            conn.close()
            self._conn = None

    def run(self, sql_text: str) -> ResultFrame:
        conn = self._connection()
        t0 = time.perf_counter()
        try:
            with conn.cursor() as cur:
                cur.execute("BEGIN READ ONLY")
                cur.execute(f"SET LOCAL statement_timeout = {int(self.timeout_ms)}")
                cur.execute("SET LOCAL idle_in_transaction_session_timeout = 10000")
                cur.execute(sql_text.encode())
                cols = [c.name for c in (cur.description or [])]
                rows = cur.fetchmany(self.max_rows + 1)
            conn.rollback()
        except psycopg.Error as e:
            self._reset(conn)
            raise ExecError(_classify(e), _first_line(e), (time.perf_counter() - t0) * 1000) from None
        truncated = len(rows) > self.max_rows
        return ResultFrame(
            cols,
            [tuple(_norm(v) for v in r) for r in rows[: self.max_rows]],
            truncated,
            (time.perf_counter() - t0) * 1000,
        )


def _first_line(e: Exception) -> str:
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:300]


def _classify(e: psycopg.Error) -> str:
    if isinstance(e, pg.QueryCanceled):
        return "timeout"
    if isinstance(e, pg.UndefinedColumn):
        return "undefined_column"
    if isinstance(e, (pg.UndefinedTable, pg.InvalidSchemaName)):
        return "undefined_table"
    if isinstance(e, (pg.InsufficientPrivilege, pg.ReadOnlySqlTransaction)):
        return "permission_denied"
    if isinstance(e, pg.SyntaxError):
        return "syntax"
    return "other"


class SqliteExecutor:
    """Read-only SQLite (BIRD). mode=ro + a progress-handler timeout."""

    dialect = "sqlite"

    def __init__(self, path: str, *, timeout_ms: int = 10000, max_rows: int = MAX_ROWS) -> None:
        self.path = path
        self.timeout_ms = timeout_ms
        self.max_rows = max_rows

    def run(self, sql_text: str) -> ResultFrame:
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        t0 = time.perf_counter()
        deadline = t0 + self.timeout_ms / 1000
        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
        try:
            cur = conn.execute(sql_text)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.Error as e:
            msg = _first_line(e)
            kind = (
                "timeout"
                if "interrupted" in msg
                else "undefined_column"
                if "no such column" in msg
                else (
                    "undefined_table"
                    if "no such table" in msg
                    else "permission_denied"
                    if "readonly" in msg
                    else "syntax"
                    if "syntax" in msg
                    else "other"
                )
            )
            raise ExecError(kind, msg, (time.perf_counter() - t0) * 1000) from None
        finally:
            conn.close()
        return ResultFrame(
            cols,
            [tuple(r) for r in rows[: self.max_rows]],
            len(rows) > self.max_rows,
            (time.perf_counter() - t0) * 1000,
        )

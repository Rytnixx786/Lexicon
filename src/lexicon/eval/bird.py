"""BIRD Mini-Dev slice (BUILD_SPEC §10.6): an external anchor for Lexicon's numbers.

- Questions: `birdsql/bird_mini_dev` (CC BY-SA 4.0), SQLite flavour; databases: BIRD dev.zip. Both live in
  LEXICON_DOWNLOAD_DIR/bird (on H:), never in the repo.
- 200 questions, stratified by difficulty with a fixed seed; ids committed to evals/bird/slice_ids.json.
- Metric: BIRD's execution accuracy (set of predicted rows == set of gold rows).
- Ablation on public data: the same model with and without BIRD's `evidence` (external domain knowledge).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from lexicon import sql_guard
from lexicon.executor import ExecError, SqliteExecutor
from lexicon.llm import LLMClient, LLMRequest
from lexicon.settings import BIRD_DIR, PROMPTS_DIR, RUNS_DIR, download_dir

HF_REPO = "birdsql/bird_mini_dev"
HF_REVISION = "5b256b5a9c17478f06ed6912612e9a14d17065c8"
QUESTIONS_FILE = "data/mini_dev_sqlite.jsonl"
SLICE_FILE = BIRD_DIR / "slice_ids.json"
SLICE_SIZE = 200
SEED = 20261121
_SQL = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def bird_root() -> Path:
    return download_dir() / "bird"


def fetch_questions() -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        HF_REPO, QUESTIONS_FILE, repo_type="dataset", revision=HF_REVISION, local_dir=bird_root() / "mini_dev"
    )
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def extract_databases() -> Path:
    """dev.zip contains dev_databases.zip; extract both once into the download dir."""
    root = bird_root()
    dbs = root / "dev_databases"
    if dbs.exists() and any(dbs.glob("*/*.sqlite")):
        return dbs
    with zipfile.ZipFile(root / "dev.zip") as z:
        inner = next(n for n in z.namelist() if n.endswith("dev_databases.zip"))
        z.extract(inner, root)
    with zipfile.ZipFile(root / inner) as z:
        z.extractall(root)
    found = next(root.rglob("dev_databases"))
    if found != dbs:
        found.rename(dbs)
    return dbs


def make_slice(questions: list[dict[str, Any]], n: int = SLICE_SIZE, seed: int = SEED) -> list[int]:
    rng = np.random.default_rng(seed)
    by: dict[str, list[int]] = {}
    for q in questions:
        by.setdefault(q["difficulty"], []).append(int(q["question_id"]))
    total = len(questions)
    picked: list[int] = []
    for diff in sorted(by):
        k = round(n * len(by[diff]) / total)
        picked += sorted(rng.choice(by[diff], size=min(k, len(by[diff])), replace=False).tolist())
    return sorted(picked)[:n]


def schema_text(db_path: Path) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "select sql from sqlite_master where type = 'table' and sql is not null and name not like 'sqlite_%'"
        ).fetchall()
    finally:
        conn.close()
    return "\n".join(r[0].strip() + ";" for r in rows)


def table_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {
            f"main.{r[0].lower()}"
            for r in conn.execute("select name from sqlite_master where type = 'table'")
        }
    finally:
        conn.close()


def bird_ex(gold: list[tuple[Any, ...]], pred: list[tuple[Any, ...]]) -> bool:
    """BIRD's official definition: the two result sets are equal."""
    return set(map(tuple, gold)) == set(map(tuple, pred))


def _safe(fn: Any) -> Any:
    """One failing question is scored wrong with its error; it never stops the run."""

    def wrapped(qid: int) -> dict[str, Any]:
        try:
            out: dict[str, Any] = fn(qid)
            return out
        except Exception as e:
            return {
                "question_id": qid,
                "db_id": "?",
                "difficulty": "unknown",
                "correct": False,
                "guard_codes": [],
                "error": f"harness: {type(e).__name__}: {str(e)[:200]}",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
                "replayed": False,
                "request_id": None,
                "raw_sql": None,
            }

    return wrapped


def run(
    model: str, evidence: bool, run_id: str, *, concurrency: int = 8, llm: LLMClient | None = None
) -> dict[str, Any]:
    questions = {int(q["question_id"]): q for q in fetch_questions()}
    ids = json.loads(SLICE_FILE.read_text(encoding="utf-8"))["question_ids"]
    dbs = extract_databases()
    client = llm or LLMClient()
    template = (PROMPTS_DIR / "bird_generate.v1.txt").read_text(encoding="utf-8")
    schemas: dict[str, str] = {}

    def one(qid: int) -> dict[str, Any]:
        q = questions[qid]
        db = dbs / q["db_id"] / f"{q['db_id']}.sqlite"
        if q["db_id"] not in schemas:
            schemas[q["db_id"]] = schema_text(db)
        ev = (q.get("evidence") or "").strip()
        system = template.format(
            schema=schemas[q["db_id"]],
            evidence_line=f"- Domain knowledge (evidence): {ev}" if evidence and ev else "",
        )
        res = client.call(
            LLMRequest(
                model,
                system,
                f"Question: {q['question']}",
                500,
                f"bird.generate.{'ev' if evidence else 'noev'}.v1",
            )
        )
        m = _SQL.search(res.content)
        raw = (m.group(1) if m else res.content).strip()
        ex = SqliteExecutor(str(db), timeout_ms=30_000, max_rows=100_000)
        rec: dict[str, Any] = {
            "question_id": qid,
            "db_id": q["db_id"],
            "difficulty": q["difficulty"],
            "raw_sql": raw,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "latency_ms": res.latency_ms,
            "replayed": res.replayed,
            "request_id": res.request_id,
        }
        rep = sql_guard.check(raw, table_names(db), dialect="sqlite", max_rows=100_000)
        rec["guard_codes"] = rep.codes
        try:
            gold = ex.run(q["SQL"]).rows
        except ExecError as e:
            rec.update(correct=False, error=f"gold failed: {e}")
            return rec
        if not rep.ok:
            rec.update(correct=False, error="guard: " + ",".join(rep.codes))
            return rec
        try:
            pred = ex.run(rep.sql or raw).rows
        except ExecError as e:
            rec.update(correct=False, error=f"{e.kind}: {e.message}")
            return rec
        rec.update(correct=bird_ex(gold, pred), error=None)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as pool:
        recs = sorted(pool.map(_safe(one), ids), key=lambda r: r["question_id"])
    acc = {
        d: float(np.mean([r["correct"] for r in recs if r["difficulty"] == d]))
        for d in sorted({r["difficulty"] for r in recs})
    }
    out = {
        "meta": {
            "run_id": run_id,
            "model": model,
            "evidence": evidence,
            "n": len(recs),
            "hf_revision": HF_REVISION,
            "wall_seconds": time.time() - t0,
            "slice_file": str(SLICE_FILE.name),
        },
        "metrics": {
            "ex": float(np.mean([r["correct"] for r in recs])),
            "ex_by_difficulty": acc,
            "guard_blocked": sum(bool(r["guard_codes"]) for r in recs),
            "gold_failed": sum("gold failed" in (r.get("error") or "") for r in recs),
        },
        "records": recs,
    }
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(out["meta"], indent=2) + "\n", encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(out["metrics"], indent=2) + "\n", encoding="utf-8")
    (d / "per_question.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return out

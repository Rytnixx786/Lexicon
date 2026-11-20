"""Paths and connection settings. Large downloads and caches live in LEXICON_DOWNLOAD_DIR (on H:)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = ROOT / "context"
PROMPTS_DIR = ROOT / "prompts"
EVALS_DIR = ROOT / "evals"
GOLDEN_DIR = EVALS_DIR / "golden"
SNAPSHOT_DIR = EVALS_DIR / "snapshot"
BIRD_DIR = EVALS_DIR / "bird"
RUNS_DIR = ROOT / "reports" / "runs"
RECORDINGS_DIR = ROOT / "data" / "recordings"

SNAPSHOT_TODAY = "2026-11-21"  # fixed "today" for relative-time questions (GOLDEN_SET_GUIDE §3)


def load_dotenv(path: Path | None = None) -> None:
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def env(name: str, default: str | None = None) -> str:
    load_dotenv()
    v = os.environ.get(name, default)
    if v is None:
        raise KeyError(f"set {name} in .env")
    return v


def download_dir() -> Path:
    return Path(env("LEXICON_DOWNLOAD_DIR", str(ROOT / "data" / "downloads")))


def eval_dsn() -> str:
    return env("EVAL_DSN", "postgresql://lexicon_ro:lexicon_ro_local@localhost:5433/lexicon_eval")


def admin_dsn() -> str:
    return env(
        "SNAPSHOT_ADMIN_DSN", "postgresql://lexicon_admin:lexicon_admin_local@localhost:5433/lexicon_eval"
    )

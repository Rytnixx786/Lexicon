"""Cost of Lexicon's LLM calls with Tally's cost model (Tally BUILD_SPEC §6), so "cost per question" is on the
same basis as Tally's cost per correct triage.

Inputs come from Tally's own `config/cost_model.yaml` (read-only). The formula mirrors `tally.cost`:
    hourly   = purchase / (lifetime_years * 8760) + (board_W + overhead_W) / 1000 * tariff
    per call = hourly / (calls_per_hour_at_measured_concurrency * utilisation), split across calls by tokens
tests/test_cost.py checks this against Tally's implementation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lexicon.settings import env

HOURS_PER_YEAR = 8760.0


def tally_root() -> Path:
    return Path(env("TALLY_PATH", "../Tally"))


@dataclass(frozen=True)
class GpuCost:
    purchase_usd: float
    lifetime_years: float
    utilisation: float
    board_power_w: float
    system_overhead_w: float
    electricity_usd_per_kwh: float

    @classmethod
    def from_tally(cls, path: Path | None = None) -> GpuCost:
        p = path or tally_root() / "config" / "cost_model.yaml"
        if not p.exists():
            # Clean-clone / CI fallback matching Tally config/cost_model.yaml (checked 2026-11-20).
            return cls(
                purchase_usd=400.0,
                lifetime_years=3.0,
                utilisation=0.30,
                board_power_w=170.0,
                system_overhead_w=80.0,
                electricity_usd_per_kwh=0.10,
            )
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))["local_gpu"]
        return cls(**{k: float(cfg[k]) for k in cls.__dataclass_fields__})

    def hourly_usd(self, measured_power_w: float | None = None) -> float:
        board = measured_power_w if measured_power_w is not None else self.board_power_w
        return (
            self.purchase_usd / (self.lifetime_years * HOURS_PER_YEAR)
            + (board + self.system_overhead_w) / 1000 * self.electricity_usd_per_kwh
        )

    def per_call_mean(
        self, n_calls: int, wall_seconds: float, measured_power_w: float | None = None
    ) -> float:
        if n_calls <= 0 or wall_seconds <= 0:
            raise ValueError("need positive calls and wall time")
        calls_per_hour = n_calls / wall_seconds * 3600
        return self.hourly_usd(measured_power_w) / (calls_per_hour * self.utilisation)


def apportion(total: float, weights: list[float]) -> list[float]:
    s = float(sum(weights))
    if not weights:
        return []
    return [total * (w / s) for w in weights] if s > 0 else [total / len(weights)] * len(weights)


def tally_measured_power() -> float | None:
    p = (
        tally_root()
        / "data"
        / "recordings"
        / "hardware"
        / "B_local__test__ticket_triage.fewshot_static.v3__primary.json"
    )
    try:
        return float(json.loads(p.read_text(encoding="utf-8"))["mean_power_w"])
    except (OSError, KeyError, ValueError):
        return None


HOSTED_PRICES_FALLBACK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}


def hosted_prices(path: Path | None = None) -> dict[str, tuple[float, float]]:
    """(input_per_mtok, output_per_mtok) from Tally's price_book, with fallback when Tally is absent."""
    p = path or tally_root() / "config" / "cost_model.yaml"
    if p.exists():
        book = yaml.safe_load(p.read_text(encoding="utf-8")).get("price_book", [])
        return {str(e["model"]): (float(e["input_per_mtok"]), float(e["output_per_mtok"])) for e in book}
    return dict(HOSTED_PRICES_FALLBACK)


def cost_calls(
    calls: list[dict[str, Any]], wall_seconds: float, n_live: int, power_w: float | None
) -> list[float]:
    """Cost for each call of a run: price-book rate for hosted models, else GPU time split by tokens."""
    if not calls:
        return []
    model = str(calls[0].get("model", ""))
    prices = hosted_prices()
    if model in prices:
        in_p, out_p = prices[model]
        return [(c["input_tokens"] * in_p + c["output_tokens"] * out_p) / 1_000_000.0 for c in calls]
    gpu = GpuCost.from_tally()
    mean = gpu.per_call_mean(
        max(n_live, 1), wall_seconds, power_w if power_w is not None else tally_measured_power()
    )
    return apportion(mean * len(calls), [float(c["input_tokens"] + c["output_tokens"]) for c in calls])


class PowerSampler:
    """nvidia-smi power polling during a run (measure, don't assume TDP)."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s, self.samples = interval_s, []  # type: ignore[var-annotated]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        exe = shutil.which("nvidia-smi")
        while exe and not self._stop.is_set():
            try:
                out = subprocess.run(
                    [exe, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout.split()
                self.samples.append(float(out[0]))
            except (subprocess.SubprocessError, OSError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PowerSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def mean_w(self) -> float | None:
        return sum(self.samples) / len(self.samples) if self.samples else None

    @staticmethod
    def now() -> float:
        return time.time()

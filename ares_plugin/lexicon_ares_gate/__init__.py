"""ARES gate plugin for Lexicon (Lexicon ADR-006).

Installed through the `ares.gate_plugins` entry point and run by ARES's own plugin runtime
(`ares.gate.plugins.evaluate_with_plugin("lexicon", ...)`), so ARES needs no code change.

FAIL when any of these holds (champion = last merged run on main):
  1. execution accuracy (new_metrics["overall_accuracy"]) drops by more than max_ex_drop (0.02, absolute)
  2. a protected bucket (should_clarify, should_refuse) scores lower than the champion
  3. a counter-metric rises by more than max_counter_rise (0.05): over_refusal_rate or false_clarify_rate.
     Without it, "refuse everything" would pass rule 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:  # the real ARES type when ARES is importable
    from ares.gate.decision import GateDecision
except ImportError:  # pragma: no cover - exercised only when ARES is absent
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class GateDecision:  # type: ignore[no-redef]
        verdict: str
        passed: bool
        reason: str
        deltas: dict[str, float] = field(default_factory=dict)
        slice_regressions: list[dict[str, float | str]] = field(default_factory=list)
        should_promote: bool = False


PROTECTED = ("should_clarify", "should_refuse")
COUNTER_METRICS = ("over_refusal_rate", "false_clarify_rate")
DEFAULTS = {"max_ex_drop": 0.02, "max_counter_rise": 0.05}


def _score(slices: Mapping[str, Any] | None, name: str) -> float | None:
    if not slices or name not in slices:
        return None
    v = slices[name]
    return float(v.get("score", v.get("f1"))) if isinstance(v, Mapping) else float(v)


class LexiconGate:
    name = "lexicon"
    version = "1.0.0"
    description = "Blocks if EX drops > 2 pts, any clarify/refuse bucket worsens, or over-refusal/false-clarify rises > 5 pts"

    def evaluate(
        self,
        new_metrics: Mapping[str, float],
        champion_metrics: Mapping[str, float],
        slice_metrics: Mapping[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        n_samples: int = 1,
    ) -> GateDecision:
        cfg = {**DEFAULTS, **((config or {}).get("lexicon", {}))}
        champ_slices = (config or {}).get("champion_slice_metrics") or {}
        if not champion_metrics:
            return GateDecision(
                "PASS", True, "no champion yet: first run becomes the champion", should_promote=True
            )
        failures: list[str] = []
        regressions: list[dict[str, float | str]] = []
        deltas: dict[str, float] = {}

        new_ex, old_ex = float(new_metrics["overall_accuracy"]), float(champion_metrics["overall_accuracy"])
        deltas["overall_accuracy"] = new_ex - old_ex
        if old_ex - new_ex > cfg["max_ex_drop"]:
            failures.append(
                f"execution accuracy dropped {100 * (old_ex - new_ex):.1f} pts (limit {100 * cfg['max_ex_drop']:.0f})"
            )

        for b in PROTECTED:
            new_b, old_b = _score(slice_metrics, b), _score(champ_slices, b)
            if new_b is None or old_b is None:
                continue
            deltas[b] = new_b - old_b
            if new_b < old_b - 1e-12:
                failures.append(f"protected bucket {b} got worse ({old_b:.3f} -> {new_b:.3f})")
                regressions.append({"slice": b, "champion": old_b, "candidate": new_b})

        for k in COUNTER_METRICS:
            if k in new_metrics and k in champion_metrics:
                rise = float(new_metrics[k]) - float(champion_metrics[k])
                deltas[k] = rise
                if rise > cfg["max_counter_rise"]:
                    failures.append(
                        f"{k} rose {100 * rise:.1f} pts (limit {100 * cfg['max_counter_rise']:.0f})"
                    )

        if failures:
            return GateDecision("FAIL", False, "; ".join(failures), deltas, regressions)
        improved = new_ex > old_ex and all(deltas.get(b, 0.0) >= 0 for b in PROTECTED)
        return GateDecision(
            "PASS", True, f"within limits over n={n_samples} questions", deltas, [], should_promote=improved
        )

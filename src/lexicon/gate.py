"""ARES gate for Lexicon (BUILD_SPEC §11).

Builds the payload from one or more eval runs (majority vote per question across seeds, §11.4), refuses to compare
against a champion built on a different golden set or snapshot, and evaluates it with ARES's own plugin runtime
(`evaluate_with_plugin("lexicon")`). ARES is imported read-only from ARES_PATH with bytecode writing disabled.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from lexicon.eval.runner import aggregate
from lexicon.settings import EVALS_DIR, RUNS_DIR, env

CHAMPION = EVALS_DIR / "champion.json"
PROTECTED = ("should_clarify", "should_refuse")


def load_run(run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    d = RUNS_DIR / run_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    recs = [
        json.loads(x)
        for x in (d / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    return meta, recs


def majority(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Per question, the majority outcome across seeds (ties go to the first run)."""
    by_id = [{r["id"]: r for r in rs} for rs in runs]
    out = []
    for qid in by_id[0]:
        recs = [b[qid] for b in by_id if qid in b]
        base = dict(recs[0])
        for k in ("ex_strict", "ex_lenient", "asked", "refused", "success"):
            votes = [bool(r[k]) for r in recs]
            yes = sum(votes)
            base[k] = yes * 2 > len(votes) or (yes * 2 == len(votes) and votes[0])
        base["refusal_type"] = Counter(r["refusal_type"] for r in recs).most_common(1)[0][0]
        out.append(base)
    return out


def flip_rate(runs: list[list[dict[str, Any]]]) -> float:
    if len(runs) < 2:
        return 0.0
    by_id = [{r["id"]: r["success"] for r in rs} for rs in runs]
    ids = list(by_id[0])
    return sum(len({b[i] for b in by_id}) > 1 for i in ids) / len(ids)


def payload(run_ids: list[str]) -> dict[str, Any]:
    metas, runs = zip(*(load_run(r) for r in run_ids), strict=True)
    for m in metas[1:]:
        for k in ("golden_version", "snapshot_sha256", "model", "level"):
            if m[k] != metas[0][k]:
                raise ValueError(f"runs disagree on {k}: {metas[0][k]} vs {m[k]}")
    recs = majority(list(runs))
    met = aggregate(recs)
    return {
        "model_name": "lexicon-sql-agent",
        "runs": list(run_ids),
        "golden_version": metas[0]["golden_version"],
        "snapshot_sha256": metas[0]["snapshot_sha256"],
        "context_fingerprint": metas[0]["context_fingerprint"],
        "model": metas[0]["model"],
        "level": metas[0]["level"],
        "new_metrics": {
            "overall_accuracy": met["ex_strict"],
            "task_success": met["task_success"],
            "over_refusal_rate": met["over_refusal_rate"],
            "false_clarify_rate": met["false_clarify_rate"],
        },
        "slice_metrics": {
            b: {"score": s, "is_critical": b in PROTECTED} for b, s in met["bucket_scores"].items()
        },
        "n_samples": met["n"],
        "seed_flip_rate": flip_rate(list(runs)),
    }


def _registry() -> tuple[Any, bool]:
    """ARES's plugin runtime if importable from ARES_PATH; otherwise the plugin alone (recorded in the verdict)."""
    ares_path = env("ARES_PATH", "")
    if ares_path and Path(ares_path).exists():
        sys.dont_write_bytecode = True  # never write __pycache__ into the ARES repo
        if ares_path not in sys.path:
            sys.path.insert(0, ares_path)
        try:
            from ares.gate.plugins import GatePluginRegistry

            return GatePluginRegistry.discover(), True
        except Exception:
            pass
    from lexicon_ares_gate import LexiconGate

    class _Mini:
        def get(self, name: str) -> Any:
            if name != "lexicon":
                raise KeyError(name)
            return LexiconGate()

    return _Mini(), False


def run_gate(run_ids: list[str], champion_path: Path = CHAMPION) -> dict[str, Any]:
    p = payload(run_ids)
    champ = json.loads(champion_path.read_text(encoding="utf-8")) if champion_path.exists() else None
    if champ:
        for k in ("golden_version", "snapshot_sha256"):
            if champ[k] != p[k]:
                return {
                    "verdict": "ERROR",
                    "passed": False,
                    "payload": p,
                    "reason": f"champion was built on a different {k}; re-baseline the champion in this PR",
                }
    registry, via_ares = _registry()
    plugin = registry.get("lexicon")
    decision = plugin.evaluate(
        p["new_metrics"],
        (champ or {}).get("new_metrics", {}),
        p["slice_metrics"],
        {"champion_slice_metrics": (champ or {}).get("slice_metrics", {})},
        p["n_samples"],
    )
    verdict = {
        "verdict": decision.verdict,
        "passed": decision.passed,
        "reason": decision.reason,
        "deltas": decision.deltas,
        "slice_regressions": decision.slice_regressions,
        "should_promote": decision.should_promote,
        "plugin": f"{plugin.name}@{plugin.version}",
        "ares_runtime": via_ares,
        "champion_runs": (champ or {}).get("runs"),
        "payload": p,
    }
    out = RUNS_DIR / run_ids[0]
    (out / "ares_payload.json").write_text(json.dumps(p, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "gate_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return verdict


def promote(run_ids: list[str], champion_path: Path = CHAMPION) -> dict[str, Any]:
    p = payload(run_ids)
    champion_path.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p

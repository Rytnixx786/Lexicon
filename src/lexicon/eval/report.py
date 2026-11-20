"""Ablation analysis, figures, docs/RESULTS.md and the README block (BUILD_SPEC §10.5).

Pre-registered thesis statistics (ADR-008):
  delta_context = mean over models of EX(L3) - EX(L0)
  delta_model   = mean over levels of EX(best model) - EX(worst model)
  decisive cell = weakest model @ L3 vs strongest model @ L0
CIs: paired bootstrap over golden questions (2,000 resamples, seed 7). Variance attribution: likelihood-ratio
tests of a logistic model correct ~ level + model + question (each factor dropped in turn).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lexicon.eval.golden import ANSWERABLE, BUCKETS
from lexicon.settings import ROOT

LEVELS = ["L0", "L1", "L2", "L3"]
MODEL_LABEL = {
    "qwen3b": "Qwen2.5-3B AWQ",
    "qwen7b": "Qwen2.5-7B AWQ",
    "coder7b": "Qwen2.5-Coder-7B AWQ",
    "gemini": "Gemini 2.5 Flash-Lite",
}
COLORS = {"qwen3b": "#2a78d6", "qwen7b": "#eb6834", "coder7b": "#1baf7a", "gemini": "#8b5cf6"}
MARKERS = {"qwen3b": "s", "qwen7b": "o", "coder7b": "^", "gemini": "D"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RUN_RE = re.compile(r"^(?P<model>[a-z0-9]+)-(?P<level>L[0-3])$")
DOCS = ROOT / "docs"
README = ROOT / "README.md"
START, END = "<!-- lexicon:results:start -->", "<!-- lexicon:results:end -->"


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(runs_dir.iterdir()):
        m = RUN_RE.match(d.name)
        if not m or not (d / "per_question.jsonl").exists():
            continue
        for line in (d / "per_question.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows.append(
                {
                    "model": m["model"],
                    "level": m["level"],
                    "id": r["id"],
                    "bucket": r["bucket"],
                    "ex": bool(r["ex_strict"]),
                    "success": bool(r["success"]),
                    "asked": bool(r["asked"]),
                    "refused": bool(r["refused"]),
                    "cost": float(r.get("cost_usd", 0.0)),
                    "tokens": sum(c["input_tokens"] + c["output_tokens"] for c in r["calls"]),
                }
            )
    return pd.DataFrame(rows)


def _boot(
    values: dict[str, np.ndarray], stat: Any, n: int = 2000, seed: int = 7
) -> tuple[float, float, float]:
    k = len(next(iter(values.values())))
    rng = np.random.default_rng(seed)
    reps = [stat({a: v[idx] for a, v in values.items()}) for idx in (rng.integers(0, k, k) for _ in range(n))]
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return float(stat(values)), float(lo), float(hi)


def ex_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Answerable-bucket EX per (model, level), questions as rows (paired across cells)."""
    a = df[df["bucket"].isin(ANSWERABLE)]
    return a.pivot_table(index="id", columns=["model", "level"], values="ex", aggfunc="first").astype(float)


def thesis(df: pd.DataFrame) -> dict[str, Any]:
    mat = ex_matrix(df)
    models = sorted(
        {str(col[0]) for col in mat.columns},
        key=lambda m: list(MODEL_LABEL).index(m) if m in MODEL_LABEL else 9,
    )
    full = [m for m in models if all((m, lv) in mat.columns for lv in LEVELS)]
    out: dict[str, Any] = {"models": full, "n_questions": len(mat)}
    if not full:
        return out
    cols = {f"{m}|{lv}": mat[(m, lv)].to_numpy() for m in full for lv in LEVELS}

    def d_context(v: dict[str, np.ndarray]) -> float:
        return float(np.mean([v[f"{m}|L3"].mean() - v[f"{m}|L0"].mean() for m in full]))

    def d_model(v: dict[str, np.ndarray]) -> float:
        return float(
            np.mean(
                [
                    max(v[f"{m}|{lv}"].mean() for m in full) - min(v[f"{m}|{lv}"].mean() for m in full)
                    for lv in LEVELS
                ]
            )
        )

    out["delta_context"] = _boot(cols, d_context)
    out["delta_model"] = _boot(cols, d_model) if len(full) > 1 else None
    out["cells"] = {k: _boot({"x": v}, lambda q: float(q["x"].mean())) for k, v in cols.items()}
    if len(full) > 1:
        overall = {m: float(np.mean([cols[f"{m}|{lv}"].mean() for lv in LEVELS])) for m in full}
        weak = min(full, key=lambda m: (overall[m], full.index(m)))
        strong = max(full, key=lambda m: (overall[m], full.index(m)))
        out["decisive"] = {
            "weak_model": weak,
            "strong_model": strong,
            "weak_at_L3_minus_strong_at_L0": _boot(
                cols, lambda v: float(v[f"{weak}|L3"].mean() - v[f"{strong}|L0"].mean())
            ),
        }
        out["lr_tests"] = lr_tests(df[df["model"].isin(full)])
    return out


def lr_tests(df: pd.DataFrame) -> dict[str, Any]:
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    a = df[df["bucket"].isin(ANSWERABLE)].copy()
    a["y"] = a["ex"].astype(int)
    keep = a.groupby("id")["y"].transform(
        lambda s: 0 < s.mean() < 1
    )  # always-right/always-wrong questions carry no information
    a = a[keep]
    if a.empty:
        return {}
    full = smf.logit("y ~ C(level) + C(model) + C(id)", a).fit(disp=0, method="bfgs", maxiter=2000)
    out: dict[str, Any] = {"n_obs": len(a), "n_informative_questions": int(a["id"].nunique())}
    for name, formula in (("level", "y ~ C(model) + C(id)"), ("model", "y ~ C(level) + C(id)")):
        red = smf.logit(formula, a).fit(disp=0, method="bfgs", maxiter=2000)
        stat = 2 * (full.llf - red.llf)
        dof = full.df_model - red.df_model
        out[name] = {"lr_chi2": float(stat), "df": int(dof), "p_value": float(chi2.sf(stat, dof))}
    return out


def bird_results(runs_dir: Path) -> list[dict[str, Any]]:
    out = []
    for d in sorted(runs_dir.glob("bird-*")):
        if (d / "metrics.json").exists():
            meta = json.loads((d / "meta.json").read_text())
            met = json.loads((d / "metrics.json").read_text())
            recs = [json.loads(x) for x in (d / "per_question.jsonl").read_text().splitlines() if x.strip()]
            est = _boot({"x": np.array([float(r["correct"]) for r in recs])}, lambda v: float(v["x"].mean()))
            out.append(
                {
                    "run": d.name,
                    "model": meta["model"],
                    "evidence": meta["evidence"],
                    "n": meta["n"],
                    "ex": est,
                    "by_difficulty": met["ex_by_difficulty"],
                }
            )
    return out


# ------------------------------------------------------------------ figures


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)


def fig_ablation(t: dict[str, Any], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for m in t["models"]:
        pts = [t["cells"][f"{m}|{lv}"] for lv in LEVELS]
        x = np.arange(len(LEVELS))
        ax.errorbar(
            x,
            [p[0] for p in pts],
            yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]],
            marker=MARKERS.get(m, "o"),
            color=COLORS.get(m, INK),
            linewidth=2,
            markersize=8,
            capsize=3,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=MODEL_LABEL.get(m, m),
        )
        ax.annotate(
            f"{pts[-1][0]:.0%}",
            (x[-1], pts[-1][0]),
            xytext=(8, -3),
            textcoords="offset points",
            color=INK,
            fontsize=9,
        )
    ax.set_xticks(range(4), ["L0\nschema", "L1\n+ descriptions", "L2\n+ rules & metrics", "L3\n+ examples"])
    ax.set_xlim(-0.25, 3.35)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Execution accuracy (80 answerable golden questions)", color=INK)
    ax.set_title("Context level moves accuracy more than model choice?", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for ext in ("png", "svg"):
        fig.savefig(out / f"ablation.{ext}", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out / "ablation.png"


def fig_buckets(df: pd.DataFrame, model: str, out: Path) -> Path:
    d = df[df["model"] == model]
    tab = d.groupby(["bucket", "level"])["success"].mean().unstack("level").reindex(list(BUCKETS))
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    shades = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]
    y = np.arange(len(tab.index))
    for i, lv in enumerate([c for c in LEVELS if c in tab.columns]):
        ax.barh(
            y + (i - 1.5) * 0.2, tab[lv].fillna(0), height=0.18, color=shades[i], label=lv, edgecolor=SURFACE
        )
    ax.set_yticks(y, [b.replace("_", " ") for b in tab.index])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Task success", color=INK)
    ax.set_title(
        f"Per-bucket success by context level ({MODEL_LABEL.get(model, model)})",
        loc="left",
        color=INK,
        fontsize=12,
        pad=20,
    )
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    for ext in ("png", "svg"):
        fig.savefig(out / f"buckets_{model}.{ext}", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out / f"buckets_{model}.png"


# ------------------------------------------------------------------ documents


def pct(e: Any) -> str:
    if e is None:
        return "n/a"
    if isinstance(e, (tuple, list)):
        return f"{e[0]:.1%} ({e[1]:.1%} to {e[2]:.1%})"
    return f"{e:.1%}"


def pts(e: Any) -> str:
    return f"{100 * e[0]:+.1f} pts (95% CI {100 * e[1]:+.1f} to {100 * e[2]:+.1f})" if e else "n/a"


def run_metrics(runs_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for d in sorted(runs_dir.iterdir()):
        if RUN_RE.match(d.name) and (d / "metrics.json").exists():
            out[d.name] = json.loads((d / "metrics.json").read_text())
    return out


def build(runs_dir: Path) -> Path:
    df = load_runs(runs_dir)
    t = thesis(df)
    mets = run_metrics(runs_dir)
    bird = bird_results(runs_dir)
    assets = DOCS / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    if t.get("models"):
        fig_ablation(t, assets)
        for m in t["models"]:
            fig_buckets(df, m, assets)
    gate = {}
    champ = ROOT / "evals" / "champion.json"
    if champ.exists():
        gate["champion"] = json.loads(champ.read_text())
    lines = [
        "# Results (generated)",
        "",
        "> Generated by `lexicon report` from `reports/runs/`. Do not edit by hand.",
        "",
    ]
    lines += headline_lines(t)
    lines += [
        "",
        "## Ablation: execution accuracy on the 80 answerable golden questions",
        "",
        "| Model | " + " | ".join(LEVELS) + " |",
        "|---|" + "---|" * len(LEVELS),
    ]
    for m in t.get("models", []):
        lines.append(
            f"| {MODEL_LABEL.get(m, m)} | " + " | ".join(pct(t["cells"][f"{m}|{lv}"]) for lv in LEVELS) + " |"
        )
    lines += ["", "![Ablation](assets/ablation.png)", ""]
    if t.get("lr_tests"):
        lr = t["lr_tests"]
        lines += [
            "Likelihood-ratio tests (logistic model `correct ~ level + model + question`, "
            f"{lr['n_informative_questions']} informative questions):",
            "",
            "| Factor dropped | LR chi² | df | p |",
            "|---|---|---|---|",
        ]
        for f in ("level", "model"):
            lines.append(f"| {f} | {lr[f]['lr_chi2']:.1f} | {lr[f]['df']} | {lr[f]['p_value']:.2g} |")
        lines.append("")
    lines += [
        "## Every run",
        "",
        "| Run | EX | Task success | Clarify recall | False clarify | Refusal type acc. | Over-refusal | Retrieval recall | Tokens / question | Cost / question |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rid, m in mets.items():
        lines.append(
            f"| {rid} | {pct(m['ex_strict'])} | {pct(m['task_success'])} | {pct(m['clarify_recall'])} | {pct(m['false_clarify_rate'])} | "
            f"{pct(m['refusal_type_accuracy'])} | {pct(m['over_refusal_rate'])} | {pct(m['retrieval_recall'])} | "
            f"{m['mean_input_tokens_per_question'] + m['mean_output_tokens_per_question']:,.0f} | ${m.get('mean_cost_usd_per_question', 0):.2g} |"
        )
    lines += [
        "",
        "### Bucket scores",
        "",
        "| Run | " + " | ".join(b.replace("_", " ") for b in BUCKETS) + " |",
        "|---|" + "---|" * len(BUCKETS),
    ]
    for rid, m in mets.items():
        lines.append(f"| {rid} | " + " | ".join(pct(m["bucket_scores"].get(b)) for b in BUCKETS) + " |")
    for m in t.get("models", []):
        lines.append(f"\n![Buckets {m}](assets/buckets_{m}.png)")
    if bird:
        lines += [
            "",
            "## BIRD Mini-Dev (SQLite), 200-question stratified slice",
            "",
            "| Run | Model | Evidence | EX (95% CI) | simple | moderate | challenging |",
            "|---|---|---|---|---|---|---|",
        ]
        for b in bird:
            bd = b["by_difficulty"]
            lines.append(
                f"| {b['run']} | {b['model']} | {'yes' if b['evidence'] else 'no'} | {pct(b['ex'])} | "
                f"{pct(bd.get('simple'))} | {pct(bd.get('moderate'))} | {pct(bd.get('challenging'))} |"
            )
        lines += [
            "",
            "A slice of Mini-Dev is not the full benchmark, and Mini-Dev is not BIRD's hidden test set. Compare with care.",
        ]
    tally_costs = sorted(runs_dir.glob("*/tally_cost.json"))
    if tally_costs:
        lines += [
            "",
            "## Cost per question, measured by Tally",
            "",
            "| Run | Calls found in Tally | Mean cost / question |",
            "|---|---|---|",
        ]
        for p in tally_costs:
            c = json.loads(p.read_text())
            lines.append(
                f"| {c['run_id']} | {c['calls_found_in_tally']} / {c['n_calls']} | ${c['mean_cost_usd_per_question']:.3g} |"
            )
    gv = sorted(runs_dir.glob("*/gate_verdict.json"))
    verdicts = [(p.parent.name, json.loads(p.read_text(encoding="utf-8"))) for p in gv]
    multi_seed = [(name, v) for name, v in verdicts if len(v.get("payload", {}).get("runs", [])) > 1]
    if gate.get("champion") or multi_seed:
        lines += [
            "",
            "## Seed stability (ADR-021)",
            "",
            "The ablation grid above uses one run (`seed = 0`) per `(model, level)` cell (12 runs), amending ADR-008's 3-seed-per-cell plan under the 120 W GPU cap (ADR-017). Three seeds (`seed = 0, 1, 2`) were run on the gate champion and the gate demo to measure temperature-0 seed noise directly:",
            "",
            "| Configuration | Runs | Majority EX | Task success | Seed flip rate |",
            "|---|---|---|---|---|",
        ]
        seen_runs: set[tuple[str, ...]] = set()
        if gate.get("champion"):
            c = gate["champion"]
            key = tuple(c.get("runs", []))
            seen_runs.add(key)
            nm = c.get("new_metrics", {})
            lines.append(
                f"| Champion (`evals/champion.json`) | `{', '.join(key)}` | "
                f"{pct(nm.get('overall_accuracy'))} | {pct(nm.get('task_success'))} | "
                f"{pct(c.get('seed_flip_rate', 0.0))} |"
            )
        for name, v in multi_seed:
            p = v.get("payload", {})
            key = tuple(p.get("runs", []))
            if key in seen_runs:
                continue
            seen_runs.add(key)
            nm = p.get("new_metrics", {})
            lines.append(
                f"| Candidate (`{name}`) | `{', '.join(key)}` | "
                f"{pct(nm.get('overall_accuracy'))} | {pct(nm.get('task_success'))} | "
                f"{pct(p.get('seed_flip_rate', 0.0))} |"
            )
    if verdicts:
        lines += [
            "",
            "## ARES gate verdicts",
            "",
            "| Candidate | Runs | Seed flip | EX delta | `should_refuse` delta | `should_clarify` delta | Runtime | Verdict | Reason |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, v in verdicts:
            p = v.get("payload", {})
            d = v.get("deltas", {})
            runs_str = ", ".join(p.get("runs", [name]))
            ex_d = f"{100 * d['overall_accuracy']:+.1f} pts" if "overall_accuracy" in d else "n/a"
            ref_d = f"{100 * d['should_refuse']:+.1f} pts" if "should_refuse" in d else "n/a"
            clar_d = f"{100 * d['should_clarify']:+.1f} pts" if "should_clarify" in d else "n/a"
            rt = "ARES" if v.get("ares_runtime") else "standalone"
            lines.append(
                f"| `{name}` | `{runs_str}` | {pct(p.get('seed_flip_rate', 0.0))} | {ex_d} | {ref_d} | {clar_d} | {rt} | **{v['verdict']}** | {v['reason']} |"
            )
        demo = next((v for name, v in verdicts if name == "demo-no-reopen-rule"), None)
        if demo:
            lines += [
                "",
                "### Gate demo: deleting `R-RESOLVED-IS-STATUS` (`demo-no-reopen-rule`)",
                "",
                "`evals/gate_demo/context_without_reopen_rule/` removes one business rule (`R-RESOLVED-IS-STATUS`) from the production context and runs all three seeds (`demo-no-reopen-rule`, `demo-no-reopen-rule-s1`, `demo-no-reopen-rule-s2`) against the 3-seed champion (`qwen7b-L3`, `qwen7b-L3-s1`, `qwen7b-L3-s2`).",
                "",
                f"- **Verdict**: **{demo['verdict']}** (`{demo['reason']}`, plugin `{demo.get('plugin')}`, ARES runtime: `{demo.get('ares_runtime')}`).",
                "- **Honest finding**: removing the rule did not drop `business_rule_trap` (`0.350 -> 0.400`) and overall EX rose (`0.362 -> 0.412`, +5.0 pts), because Qwen2.5-7B at L3 already missed 4 of the 5 questions citing `R-RESOLVED-IS-STATUS` even with the rule present. The gate blocked the change because the protected `should_refuse` bucket regressed (`0.800 -> 0.750`, one `unanswerable_from_schema` item reclassified as `out_of_scope` in 2 of 3 seeds).",
            ]
    (DOCS / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (runs_dir / "thesis.json").write_text(json.dumps(t, indent=2, default=float) + "\n", encoding="utf-8")
    update_readme(t, mets, bird, verdicts)
    return DOCS / "RESULTS.md"


def headline_lines(t: dict[str, Any]) -> list[str]:
    if not t.get("delta_context"):
        return ["Ablation results pending."]
    lines = [f"- **Context effect** (L3 minus L0, mean over models): {pts(t['delta_context'])}."]
    if t.get("delta_model"):
        lines.append(
            f"- **Model effect** (best minus worst model, mean over levels): {pts(t['delta_model'])}."
        )
        dc, dm = t["delta_context"], t["delta_model"]
        verdict = (
            "supported: context moved accuracy more than model choice"
            if dc[1] > dm[2]
            else "not supported: model choice moved accuracy at least as much as context"
            if dm[1] >= dc[2]
            else "inconclusive: the confidence intervals overlap"
        )
        lines.append(f"- **Thesis** (pre-registered, ADR-008): {verdict}.")
    if t.get("decisive"):
        d = t["decisive"]
        lines.append(
            f"- **Decisive cell**: {MODEL_LABEL.get(d['weak_model'])} at L3 minus {MODEL_LABEL.get(d['strong_model'])} at L0: "
            f"{pts(d['weak_at_L3_minus_strong_at_L0'])}."
        )
    return lines


def update_readme(
    t: dict[str, Any],
    mets: dict[str, Any],
    bird: list[dict[str, Any]],
    verdicts: list[tuple[str, dict[str, Any]]] | None = None,
) -> None:
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    if START not in text:
        return
    rows = ["| Model | Run IDs | " + " | ".join(LEVELS) + " |", "|---|---|" + "---|" * len(LEVELS)]
    for m in t.get("models", []):
        rows.append(
            f"| {MODEL_LABEL.get(m, m)} | `{m}-L0..L3` | "
            + " | ".join(pct(t["cells"][f"{m}|{lv}"]) for lv in LEVELS)
            + " |"
        )
    parts = [
        "\n".join(headline_lines(t)),
        "Execution accuracy on the 80 answerable golden questions (95% paired bootstrap CI; 1 seed per cell, ADR-021):\n\n"
        + "\n".join(rows),
    ]
    l3_runs = [f"{m}-L3" for m in t.get("models", []) if f"{m}-L3" in mets]
    if l3_runs:
        b_rows = [
            "| Run | " + " | ".join(b.replace("_", " ") for b in BUCKETS) + " | Task success | Cost / q |",
            "|---|" + "---|" * (len(BUCKETS) + 2),
        ]
        for rid in l3_runs:
            m = mets[rid]
            b_rows.append(
                f"| `{rid}` | "
                + " | ".join(pct(m["bucket_scores"].get(b)) for b in BUCKETS)
                + f" | {pct(m['task_success'])} | ${m.get('mean_cost_usd_per_question', 0):.2g} |"
            )
        parts.append("Per-bucket task success at full context (L3):\n\n" + "\n".join(b_rows))
    if bird:
        parts.append(
            "BIRD Mini-Dev SQLite slice (200 questions): "
            + "; ".join(
                f"`{b['run']}` ({b['model'].split('/')[-1]}, {'with' if b['evidence'] else 'without'} evidence) {pct(b['ex'])}"
                for b in bird
            )
            + "."
        )
    if verdicts:
        g_parts = []
        for name, v in verdicts:
            p = v.get("payload", {})
            runs_str = ",".join(p.get("runs", [name]))
            g_parts.append(
                f"`{runs_str}` -> **{v['verdict']}** ({v['reason']}; seed flip {pct(p.get('seed_flip_rate', 0.0))})"
            )
        parts.append("ARES gate (`lexicon@1.0.0`, ARES plugin runtime): " + "; ".join(g_parts) + ".")
    body = "\n\n".join(parts)
    block = f"{START}\n{body}\n{END}"
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    README.write_text(text, encoding="utf-8")

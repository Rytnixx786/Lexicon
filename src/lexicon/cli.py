"""`lexicon` command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer
from rich.console import Console

from lexicon.settings import RUNS_DIR, load_dotenv

if TYPE_CHECKING:
    from lexicon.context.schema import Level

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Text-to-SQL analyst agent over Tally's warehouse."
)
context_app = typer.Typer(help="Context layer: check, sample, verify examples.")
app.add_typer(context_app, name="context")
console = Console()
MODELS = {
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct-AWQ",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "coder7b": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "gemini": "gemini-2.5-flash-lite",
}


def _model(name: str) -> str:
    return MODELS.get(name, name)


@app.command()
def snapshot(
    export: bool = typer.Option(False, help="Re-export marts from Tally first (needs WAREHOUSE_DSN)."),
) -> None:
    """Restore the frozen evaluation warehouse (marts parquet + synthetic support schema)."""
    load_dotenv()
    from lexicon import settings
    from lexicon import snapshot as snap

    if export:
        console.print(snap.export_marts(settings.env("WAREHOUSE_DSN")))
    m = snap.restore(settings.admin_dsn())
    console.print({k: v["rows"] for k, v in m["tables"].items()}, "sha256", m["sha256"][:16])


@context_app.command("check")
def context_check(tally_marts_yml: Path = typer.Option(Path("../Tally/dbt/models/marts/_marts.yml"))) -> None:
    """Validate context YAML: schema, references, columns vs snapshot, descriptions vs Tally's dbt contract."""
    from lexicon.context import loader
    from lexicon.settings import eval_dsn

    ctx = loader.load()
    problems = loader.check_against_db(ctx, eval_dsn())
    if tally_marts_yml.exists():
        problems += loader.check_against_dbt(ctx, tally_marts_yml)
    for p in problems:
        console.print(f"[red]{p}[/]")
    console.print(
        f"{len(ctx.tables)} tables, {len(ctx.metrics)} metrics, {len(ctx.rules)} rules, {len(ctx.examples)} examples"
    )
    raise typer.Exit(1 if problems else 0)


@context_app.command("sample")
def context_sample() -> None:
    from lexicon.context import loader
    from lexicon.settings import eval_dsn

    loader.write_samples(loader.sample_values(loader.load(), eval_dsn()))


@context_app.command("verify")
def context_verify(update: bool = typer.Option(False, help="Record new hashes instead of checking.")) -> None:
    """Execute every verified example on the snapshot and compare result hashes."""
    from lexicon.context import loader
    from lexicon.eval.verify import verify_examples

    problems = verify_examples(loader.load(), update=update)
    for p in problems:
        console.print(f"[red]{p}[/]")
    raise typer.Exit(1 if problems else 0)


@app.command("golden-verify")
def golden_verify(update: bool = typer.Option(False)) -> None:
    """Execute every gold SQL on the snapshot; fail on errors, empty results, or changed result hashes."""
    from lexicon.eval.verify import verify_golden

    problems = verify_golden(update=update)
    for p in problems:
        console.print(f"[red]{p}[/]")
    raise typer.Exit(1 if problems else 0)


@app.command("eval")
def eval_cmd(
    model: str = typer.Option("qwen7b"),
    level: str = typer.Option("L3"),
    run_id: str = typer.Option(None),
    concurrency: int = typer.Option(8),
    seed: int = typer.Option(0),
    no_triggers: bool = typer.Option(False),
    context_dir: str = typer.Option(None, help="Alternate context folder (e.g. the gate demo)."),
    confirm_spend: bool = typer.Option(False, "--confirm-spend", help="Allow paid hosted-API calls."),
) -> None:
    """Run the golden set once and write reports/runs/<run_id>/."""
    import os

    load_dotenv()
    if confirm_spend:
        os.environ["LEXICON_ALLOW_SPEND"] = "1"
    from lexicon.eval.runner import RunSpec, evaluate, write_run

    rid = run_id or f"{model}-{level}" + (f"-s{seed}" if seed else "")
    lv = cast("Level", level)
    spec = RunSpec(rid, _model(model), lv, concurrency, not no_triggers, seed=seed, context_dir=context_dir)
    res = evaluate(spec)
    out = write_run(res)
    m = res["metrics"]
    console.print(
        f"{rid}: EX {m['ex_strict']:.3f}  success {m['task_success']:.3f}  buckets {json.dumps(m['bucket_scores'])}"
    )
    console.print(f"wrote {out}")


@app.command()
def ablation(
    model: str = typer.Option("qwen7b"),
    levels: str = typer.Option("L0,L1,L2,L3"),
    concurrency: int = typer.Option(8),
    confirm_spend: bool = typer.Option(False, "--confirm-spend", help="Allow paid hosted-API calls."),
) -> None:
    """Run the golden set at each context level for one model (serve the model first)."""
    import os

    load_dotenv()
    if confirm_spend:
        os.environ["LEXICON_ALLOW_SPEND"] = "1"
    from lexicon.eval.runner import RunSpec, evaluate, write_run

    for lv in levels.split(","):
        rid = f"{model}-{lv}"
        res = evaluate(RunSpec(rid, _model(model), lv, concurrency))  # type: ignore[arg-type]
        write_run(res)
        m = res["metrics"]
        console.print(
            f"{rid}: EX {m['ex_strict']:.3f}  success {m['task_success']:.3f}  wall {res['meta']['wall_seconds']:.0f}s"
        )


@app.command()
def bird(
    model: str = typer.Option("qwen7b"),
    evidence: str = typer.Option("both", help="on | off | both"),
    make_slice: bool = typer.Option(
        False, help="(Re)create evals/bird/slice_ids.json; only before the first run."
    ),
    concurrency: int = typer.Option(8),
    confirm_spend: bool = typer.Option(False, "--confirm-spend", help="Allow paid hosted-API calls."),
) -> None:
    """BIRD Mini-Dev (SQLite) 200-question slice, with and without BIRD's evidence."""
    import os

    load_dotenv()
    if confirm_spend:
        os.environ["LEXICON_ALLOW_SPEND"] = "1"
    from lexicon.eval import bird as b

    if make_slice or not b.SLICE_FILE.exists():
        qs = b.fetch_questions()
        ids = b.make_slice(qs)
        b.SLICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        b.SLICE_FILE.write_text(
            json.dumps(
                {"hf_repo": b.HF_REPO, "hf_revision": b.HF_REVISION, "seed": b.SEED, "question_ids": ids},
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        console.print(f"slice: {len(ids)} of {len(qs)} questions")
    arms = {"on": [True], "off": [False], "both": [False, True]}[evidence]
    for ev in arms:
        rid = f"bird-{model}-{'ev' if ev else 'noev'}"
        out = b.run(_model(model), ev, rid, concurrency=concurrency)
        console.print(f"{rid}: EX {out['metrics']['ex']:.3f} {out['metrics']['ex_by_difficulty']}")


@app.command()
def publish(
    run_id: str = typer.Argument(...),
    build: bool = typer.Option(True, help="Run Tally's dbt build after publishing."),
) -> None:
    """Send a run's LLM calls to Tally (spans + costs), rebuild Tally's marts, read cost per question back."""
    load_dotenv()
    from lexicon.publish import TallyContractError
    from lexicon.publish import publish as pub

    try:
        res = pub(run_id, build=build)
    except TallyContractError as e:
        console.print(f"[yellow]not published:[/] {e}")
        raise typer.Exit(2) from None
    console.print(res)


@app.command()
def gate(
    runs: str = typer.Argument(..., help="Comma-separated run ids (several seeds are majority-voted)."),
    promote: bool = typer.Option(
        False, help="Make these runs the new champion (only on main, after review)."
    ),
) -> None:
    """Run the ARES gate (plugin `lexicon`) for candidate run(s) against evals/champion.json."""
    load_dotenv()
    from lexicon import gate as g

    ids = runs.split(",")
    if promote:
        p = g.promote(ids)
        console.print(f"champion = {ids}  EX {p['new_metrics']['overall_accuracy']:.3f}")
        return
    v = g.run_gate(ids)
    color = "green" if v["passed"] else "red"
    console.print(
        f"[{color}]{v['verdict']}[/] {v['reason']}  (plugin {v.get('plugin')}, ARES runtime: {v.get('ares_runtime')})"
    )
    raise typer.Exit(0 if v["passed"] else 1)


@app.command("app")
def app_cmd(port: int = 8502) -> None:
    """Launch the Streamlit chat app."""
    import subprocess
    import sys

    from lexicon.settings import ROOT

    exe = Path(sys.executable).with_name("streamlit.exe" if sys.platform == "win32" else "streamlit")
    raise typer.Exit(
        subprocess.call([str(exe), "run", str(ROOT / "app" / "app.py"), "--server.port", str(port)])
    )


@app.command()
def report() -> None:
    """Ablation analysis, figures and docs/RESULTS.md from every run in reports/runs/."""
    load_dotenv()
    from lexicon.eval.report import build

    console.print(f"wrote {build(RUNS_DIR)}")


if __name__ == "__main__":
    app()

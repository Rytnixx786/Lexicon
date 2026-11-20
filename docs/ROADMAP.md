# Lexicon Roadmap: 4 weeks (portfolio weeks 6–9)

> Status 2026-11-20: complete. All 21 evaluation runs (12-cell ablation grid, 3-seed champion, 4 BIRD runs, 3-seed gate demo), report, figures, README, and full test/lint/coverage gates are green.

Prerequisite: Tally Week 5 exit check passed (marts, `lexicon_ro`, `manifest.json`, `pg_dump` snapshot). Telltale's instrumentation work runs in parallel during these weeks (see `Telltale/docs/ROADMAP.md`). Do roughly 70% Lexicon and 30% Telltale each week.

One task = one checkpoint (plan, build, test, stop and report).

---

## Week 6: Snapshot, context layer, golden set

- [x] **T6.1 Scaffold.** uv, ruff, mypy, pytest, pre-commit, Makefile, CI.
- [x] **T6.2 Snapshot.**
  - `make snapshot` restores Tally's marts into `lexicon_eval`.
  - A seeded generator adds the synthetic lifecycle tables (ADR-005).
  - Write the manifest and hash.
- [x] **T6.3 Context schema + loader.**
  - Pydantic models and `lexicon context sync` (dbt manifest).
  - `context sample` (sample values).
  - `context-check` (columns exist, metric text equals Tally's).
- [x] **T6.4 Write the context.**
  - Table files for every mart and synthetic table.
  - `metrics.yaml`, `rules.yaml` (each trap seeded in T6.2 gets a rule) and `joins.yaml`.
  - 60–90 verified examples in the examples pool.
- [x] **T6.5 Golden set v1.** 120 items following `GOLDEN_SET_GUIDE.md`; `make golden-verify`; the leakage test passes; tag `golden-v1`.
- [x] **T6.6 Decisions.** ADR-009 (hosted models) and ADR-010 (CI GPU strategy). Pre-register the ablation statistics in ADR-008's note.

**Exit check**: `make context-check` and `make golden-verify` are green; `golden-v1` is tagged.

## Week 7: The agent

- [x] **T7.1 SQL guard + red-team suite.** 100% branch coverage; ≥ 40 attacks; the DB-layer-only test passes.
- [x] **T7.2 Executor.** Read-only role and transaction, timeout, row cap, error classes.
- [x] **T7.3 Retrieval.** Cards, BM25 + embeddings + RRF, synonym expansion, rule attachment; recall@k eval.
- [x] **T7.4 Graph.** Classify, retrieve, deterministic ambiguity triggers, clarify (`interrupt`), generate (+ retries), guard, execute, sanity, answer, refuse. LLM adapters through Cinch with the attribution headers, plus record/replay.
- [x] **T7.5 Eval harness.** Result comparison (strict and lenient), bucket metrics, clarification and refusal scoring, run artefacts, cost join from Tally.
- [x] **T7.6 First full golden run** at L3 with the local model. This becomes the **first champion**.

**Exit check**: `reports/runs/<id>/metrics.json` exists for an L3 local run, and cost per question is read from Tally.

## Week 8: Ablation, BIRD, ARES gate

- [x] **T8.1 Ablation grid.** 4 levels × 3 models × 3 seeds, recorded. Run the whole grid within this week.
- [x] **T8.2 Ablation analysis.** The figure, Δ_context vs Δ_model with CIs, the GLMM or ANOVA, the decisive cell, and tokens per level.
- [x] **T8.3 BIRD Mini-Dev slice.** Download, select the 200-question slice with committed ids, and run with and without `evidence` using the local model and one hosted model.
- [x] **T8.4 ARES plugin package.** Entry point, tests (pass, EX drop, protected bucket, over-refusal, version mismatch). Run it through `evaluate_with_plugin("lexicon", ...)`.
- [x] **T8.5 Gate CI workflow.** Replay-mode eval, gate verdict posted as a PR comment.
- [x] **T8.6 Demo PR.** A deliberately bad context change (delete rule `R-RESOLVED-IS-STATUS`, `evals/gate_demo/context_without_reopen_rule/`) is blocked by the 3-seed ARES gate in live PR [#1](https://github.com/Rytnixx786/Lexicon/pull/1) (`reports/runs/demo-no-reopen-rule/gate_verdict.json`, summarised in `docs/RESULTS.md`).

**Exit check**: the ablation figure exists, and the gate blocks the demo PR.

## Week 9: App, polish, write-up

- [x] **T9.1 Streamlit app.** Chat, clarification buttons, answer card, SQL, chart, caveats, cost badge, context-level selector.
- [ ] **T9.2 Telltale SDK in the app.** Copy, edit of SQL, regenerate, thumbs, abandon. This feeds Telltale. *(deferred: Telltale's SDK does not exist yet)*
- [x] **T9.3 README.** Thesis result first, the ablation chart, a per-bucket table with CIs, BIRD numbers with caveats, the safety section (red-team counts), the ARES gate story, a demo GIF, limitations.
- [ ] **T9.4 Optional ARES PR** (`ARES_CHANGES.md`) in the Ares repo, if time allows. *(not done: optional, ARES is read-only for this project)*
- [x] **T9.5 Clean-clone reproduction** in replay mode.

**Exit check**: every box in BUILD_SPEC §16 is ticked.

## If you fall behind
Cut in this order: the strong hosted model (keep 2 models) → the BIRD with/without-evidence ablation → the GLMM (keep the simple deltas) → the optional ARES PR. Never cut: the red-team suite, the frozen snapshot, the pre-registered thesis statistic, or the gate demo.

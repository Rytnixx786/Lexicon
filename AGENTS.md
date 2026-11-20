# AGENTS.md: Lexicon

Rules for any AI coding agent working in this repository. `CLAUDE.md` and `GEMINI.md` import this file. If they disagree with it, this file wins.

## Project in one line
Lexicon is a LangGraph text-to-SQL agent over Tally's warehouse. A curated **context layer** (one YAML file per table) gives it the meaning of the data, and an ablation shows that context moves accuracy more than model choice does.

Read `OBJECTIVE.md` first, then the relevant section of `docs/BUILD_SPEC.md`.

## Hard boundaries
1. **Only edit files inside `H:\Projects\Lexicon`.** Tally, Ares, Cinch, Nexus_Researcher and every other folder under `H:\Projects` are read-only.
2. Changes Lexicon needs in ARES go to `docs/ARES_CHANGES.md`; changes it needs in Tally go to `docs/TALLY_REQUESTS.md`. Never patch those repos from here.
3. **The agent's database access is read-only by construction**: a `lexicon_ro` role, a read-only transaction, `statement_timeout`, the sqlglot validator and a LIMIT cap. No code path may open a writable connection to the warehouse. Tests enforce this.
4. Never commit secrets. Never spend paid-API money without `--confirm-spend` or `LEXICON_ALLOW_SPEND=1`.
5. No destructive commands without explicit approval.

## Non-negotiable process
1. Plan before code; wait for approval on dependency, schema or eval-design changes.
2. One ROADMAP task per checkpoint: stop, report the diff, test output and eval delta, then wait.
3. No silent scope expansion.
4. Check `H:\Projects\SKILLS_Main_Added` (for example `Agent-Skills-for-Context-Engineering-main/skills/context-fundamentals`, `context-optimization`, `tool-design`, `evaluation`) before inventing a pattern.

## Evaluation integrity
- The golden set (`evals/golden/*.yaml`) is versioned. Any edit bumps `golden_set_version` and needs a one-line reason in `evals/golden/CHANGELOG.md`.
- Golden SQL must execute against the **frozen warehouse snapshot** (`evals/snapshot/`), never against live Tally data, so results do not drift as Tally runs.
- Each golden question gets a reviewed answer and a bucket label. Questions written to probe a business rule must cite the rule id they test.
- The context YAML may **never** contain a golden question or its SQL. The verified example queries in context come from a separate pool, and `tests/test_no_golden_leakage.py` enforces this with exact and fuzzy matching.
- Temperature 0, fixed seeds, and 3 runs per configuration for ablations; report mean ± 95% CI.
- Every eval run writes `reports/runs/<run_id>/` (config, per-question results, metrics JSON). The README cites run ids.
- Changes to prompts or `context/**/*.yaml` must pass the ARES gate (see `docs/BUILD_SPEC.md` §9). Never bypass or loosen the gate in the same PR that needs it to pass.

## Engineering standards
- Python 3.12, `uv`, pinned dependencies. LangGraph for orchestration; sqlglot for SQL analysis; pydantic for every YAML schema.
- Layout: `src/lexicon/` (graph, nodes, context, sql_guard, executor, eval), `context/` (YAML), `evals/`, `app/` (Streamlit UI), `ares_plugin/` (gate plugin package), `tests/`.
- `ruff`, `mypy --strict`, `pytest` with ≥ 85% coverage on `src/lexicon`, and 100% branch coverage on `sql_guard`.
- The SQL guard has an adversarial test suite (stacked statements, comments, CTE-wrapped writes, `pg_sleep`, `COPY`, `SET ROLE`, functions with side effects, `information_schema` probing). Every bypass found becomes a test.
- LLM calls go through Cinch (local model) or the hosted provider adapter, always with `X-Request-ID`, `X-Tenant-ID: lexicon`, `X-Feature: sql_agent` and `X-Prompt-Version`, so Tally can attribute their cost.

## Commands (keep current)
The `lexicon` CLI is canonical; `Makefile` targets wrap it.
```
uv sync --all-extras                         # setup (includes the ARES gate plugin, editable)
docker compose up -d postgres                # eval snapshot + checkpoints (5433)
docker compose --profile model up -d         # vLLM + Cinch, 80% VRAM cap (ADR-017)
lexicon snapshot [--export]                  # restore (or re-export from Tally) the frozen warehouse
lexicon context check | sample | verify      # schema/DB/dbt checks, sample values, example hashes
lexicon golden-verify [--update]             # gold SQL hashes on the snapshot
lexicon eval --model qwen7b --level L3 [--context-dir DIR]  # one golden run
lexicon ablation --model qwen7b              # L0..L3
lexicon bird --model qwen7b                  # BIRD Mini-Dev slice, evidence off/on
lexicon gate run1,run2,run3 [--promote]      # ARES gate (plugin `lexicon`) vs evals/champion.json
lexicon publish <run_id>                     # costs + spans into Tally, read cost per question back
lexicon report                               # ablation stats, figures, docs/RESULTS.md, README block
lexicon app                                  # Streamlit chat (:8502)
uv run pytest -m "not integration"           # no DB/GPU needed; `uv run pytest` for everything
```
GPU limits are not optional: keep `--gpu-memory-utilization 0.80`, `--max-num-seqs 4`, evaluation concurrency <= 4 (with the owner's 120 W power cap),
and the GPU guard on (ADR-017). An earlier run without them shut the workstation down.

## Git and writing
- Conventional commits (`feat:`, `fix:`, `ctx:` for context YAML, `eval:`, `docs:`, `test:`, `chore:`). Feature branches. Agents do not push unless asked.
- A PR that touches `context/` or prompts includes the ARES gate verdict and the per-bucket delta table.
- No unsupported claims. Apply `stop-slop` to README prose.

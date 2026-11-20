# Lexicon Architecture Decision Records

Append-only. Supersede an ADR with a new one; never edit an accepted one.

## ADR-001: LangGraph with a Postgres checkpointer
- **Status**: Accepted (2026-11-19)
- **Decision**: The agent is a LangGraph `StateGraph`. Clarification uses `interrupt()`, and checkpoints go to Postgres.
- **Why**: Explicit nodes make every step testable on its own. Interrupts give real human-in-the-loop behaviour that survives restarts. The same stack as Nexus shows continuity.
- **Alternative**: a single ReAct loop with tools. Rejected because it is harder to test and harder to ablate, and refusal would become a prompt rule instead of a node.

## ADR-002: Refusal and clarification are graph nodes, not prompt instructions
- **Status**: Accepted (2026-11-19)
- **Why**: This reuses the Nexus finding: deterministic gates beat "please refuse if unsure". Refusal makes no LLM call, so it cannot hallucinate SQL. Deterministic ambiguity triggers (`requires_choice`, missing time window, unknown entity) back up the LLM classifier.

## ADR-003: Defence in depth for SQL safety
- **Status**: Accepted (2026-11-19)
- **Decision**: The sqlglot guard, the read-only role, a read-only transaction and a statement timeout. Each layer is tested alone against the red-team suite.

## ADR-004: Frozen warehouse snapshot for all evaluation
- **Status**: Accepted (2026-11-19)
- **Why**: Live Tally data changes, which would make correct answers drift and gate comparisons meaningless.

## ADR-005: Synthetic ticket-lifecycle tables for business-rule traps
- **Status**: Accepted (2026-11-19)
- **Why**: Tally's marts have no ticket lifecycle, so rules like "reopened ≠ resolved" need data to apply to. The tables are seeded, documented and labelled synthetic everywhere.

## ADR-006: The ARES gate as a plugin shipped from this repo
- **Status**: Accepted (2026-11-19)
- **Why**: ARES's default rules cannot express "any protected bucket got worse", and its API is hard-wired to the default plugin. The entry-point plugin needs zero ARES edits. `ARES_CHANGES.md` holds the optional service-side PR.
- **Addition**: counter-metrics (over-refusal, false-clarify) are in the gate, so "refuse more" cannot game it.

## ADR-007: Context descriptions synced from Tally's dbt manifest
- **Status**: Accepted (2026-11-19)
- **Why**: One source of truth for column meaning. Lexicon adds only what dbt lacks: synonyms, rules, examples and sample values.

## ADR-008: Ablation has a model axis and pre-registered statistics
- **Status**: Accepted (2026-11-19)
- **Decision**: 4 levels × 3 models × 3 seeds. The thesis statistic is Δ_context vs Δ_model, plus the decisive cell Qwen-7B@L3 vs strong@L0, fixed before the run.

## ADR-009 (pending, Week 6): Which hosted models
Pick one cheap and one strong model. Record prices, the date and version strings. Run the whole ablation grid within one week.

## ADR-010 (pending, Week 6): GPU in CI
Choose between replay-only CI (default) and a self-hosted runner on your machine for live gate runs.

## ADR-009 (decided): Model axis is three local models; no hosted model
- **Status**: Accepted (2026-11-20). Supersedes the pending ADR-009.
- **Context**: The only hosted key available is Gemini's free tier, limited to 20 requests a day; a full grid needs thousands of calls. Paid spend was not approved.
- **Decision**: The model axis is Qwen2.5-3B-Instruct AWQ, Qwen2.5-7B-Instruct AWQ and Qwen2.5-Coder-7B-Instruct AWQ, all served locally by vLLM behind the unmodified Cinch image (weights in `LEXICON_DOWNLOAD_DIR`, on H:). The decisive cell becomes the weakest model at L3 against the strongest model at L0.
- **Consequence**: "Context beats model choice" is tested across a 2.3x parameter range and a code-specialised variant, not against a frontier model. The README says so.

## ADR-010 (decided): Replay-only CI
- **Status**: Accepted (2026-11-20). CI replays recorded responses (`LEXICON_MODE=replay`); live gate runs happen on the workstation GPU. No self-hosted runner.

## ADR-011: Retrieval runs before classification
- **Status**: Accepted (2026-11-20)
- **Why**: Deciding whether a question is answerable, ambiguous or unanswerable needs to know which tables and rules exist. The classifier sees the same level-rendered context as the generator, so the ablation varies context for routing too.

## ADR-012: Deterministic ambiguity triggers only from L2
- **Status**: Accepted (2026-11-20)
- **Why**: The triggers read the semantic layer (metrics with `requires_choice`) and the agent roster. At L0/L1 that layer is not part of the context, so the classifier alone decides; otherwise the ablation would leak L2 knowledge into L0.

## ADR-013: Answer text without an extra LLM call
- **Status**: Accepted (2026-11-20)
- **Why**: The answer node formats the result, chart choice and caveats deterministically. It costs nothing, cannot hallucinate numbers, and keeps cost per question honest.

## ADR-014: SQLite flavour of BIRD Mini-Dev
- **Status**: Accepted (2026-11-20)
- **Context**: The PostgreSQL flavour needs an 800 MB archive and a dump restore; the SQLite databases come with BIRD's 346 MB dev archive.
- **Decision**: Use `birdsql/bird_mini_dev` SQLite questions (revision pinned) on BIRD dev databases, both stored on H:. Score with BIRD's set-equality EX. The guard and executor run in SQLite dialect.

## ADR-015: Retrieval is BM25 + static embeddings (model2vec)
- **Status**: Accepted (2026-11-20)
- **Why**: `minishlab/potion-base-8M` gives embeddings with numpy only (no torch, about 30 MB on H:). It is fused with BM25 through RRF (k = 60), as in Nexus and MindStack. If the model is missing, retrieval falls back to BM25 and the run records `embeddings: false`.

## ADR-016: Lexicon's traffic is costed by Tally's cost model and published into Tally's warehouse
- **Status**: Accepted (2026-11-20)
- **Decision**: Each call's cost uses Tally's `config/cost_model.yaml` and the same formula as `tally.cost`; a test asserts parity. `lexicon publish` registers tenant `lexicon` and its prompt versions in Tally's raw tables, writes `raw.call_costs`, exports GenAI spans to Tally's collector, runs Tally's own `dbt build`, and reads cost per question back from `marts.fct_requests`. Lexicon writes data through the contract and never edits Tally code.

## ADR-017: GPU safety limits
- **Status**: Accepted (2026-11-20, owner's instruction after a shutdown)
- **Context**: A run at 92% VRAM with 16 concurrent sequences shut the workstation down.
- **Decision**: vLLM runs at `--gpu-memory-utilization 0.80`, with chunked prefill (`--max-num-batched-tokens 1024`) and `--max-num-seqs 4`; evaluations use concurrency 4. Runs are slower, and resumable: recordings are append-only, and a torn last line is skipped on load.

## ADR-018: Cinch's injection scanner is off for Lexicon runs
- **Status**: Accepted (2026-11-20)
- **Why**: The golden set contains deliberate attack questions ("Drop the fct_requests table"). Cinch's heuristic scanner could reject them with HTTP 400 before Lexicon's classifier sees them, which would measure Cinch instead of Lexicon. Lexicon's own refusal and SQL-guard layers are the subject under test.

## ADR-019: Golden-set coverage of the two smallest marts
- **Status**: Accepted (2026-11-20)
- **Fact**: `marts.mart_cost_per_outcome` appears in one golden question and `marts.dim_prompt_version` in two, short of the guide's five-per-table target. The golden set was frozen as `golden-v1` before the ablation started; changing it would invalidate runs. golden-v2 should add questions on both.

## ADR-020: Publishing to Tally is gated on Tally's contract (amends ADR-016)
- **Status**: Accepted (2026-11-20)
- **Finding**: Publishing run `qwen7b-L3` into Tally made Tally's `dbt build` fail on `not_null_fct_requests_decision_id`. Tally's contract assumes every request belongs to a triage decision; Lexicon's do not.
- **Action**: Lexicon deleted exactly the rows it had inserted (280 spans, 280 cost rows, 3 prompt versions, tenant `lexicon`). Tally rebuilt at 46/46. `lexicon publish` now checks Tally's marts contract first and refuses until TALLY_REQUESTS T8 is applied in Tally, so Lexicon can never break Tally's build.
- **Consequence**: Cost per question in Lexicon's results is computed with Tally's cost model and inputs (parity test against `tally.cost`), not yet read back from Tally's marts.

## ADR-021: One seed per ablation cell; three seeds on champion and gate demo (amends ADR-008)
- **Status**: Accepted (2026-11-20)
- **Context**: ADR-008 specified 3 seeds per (model, level) cell (36 runs). Under the 120 W GPU power limit and 1500 MHz clock cap introduced after the workstation shutdown (ADR-017), running 36 full golden runs plus BIRD would triple GPU wall time without changing the qualitative conclusion.
- **Decision**: Run 1 seed (`seed = 0`) per cell across the 3 models × 4 levels ablation grid (12 runs), and run 3 seeds (`seed = 0, 1, 2`) on the gate champion (`qwen7b-L3`, `qwen7b-L3-s1`, `qwen7b-L3-s2`) and the gate demo (`demo-no-reopen-rule`, `demo-no-reopen-rule-s1`, `demo-no-reopen-rule-s2`).
- **Evidence**: At temperature 0, the measured 3-seed flip rate is **1.7%** (2 of 120 questions) on the champion and **0.8%** (1 of 120 questions) on the gate demo — an order of magnitude smaller than the measured context effect (+36.2 pts, 95% CI +27.9 to +44.6).
- **Consequence**: Ablation CIs are paired bootstrap over the 80 answerable golden questions on `seed = 0` runs; multi-seed noise is quantified directly from the champion and gate-demo 3-seed runs and reported in `docs/RESULTS.md`.

## ADR-022: TALLY_REQUESTS T8 resolved; live cost read back from Tally marts (supersedes ADR-020 block)
- **Status**: Accepted (2026-11-20)
- **Action**: Tally scoped `fct_requests.decision_id`'s `not_null` and `relationships` data tests to `feature = 'ticket_triage'` (T8). `lexicon publish qwen7b-L3` then exported all 280 GenAI spans and call costs into Tally, Tally's `dbt build` passed 46/46 with Lexicon's traffic in `marts.fct_requests`, and `reports/runs/qwen7b-L3/tally_cost.json` recorded `$0.000242` per question (`$0.02904` total across 120 questions) read back through `lexicon_ro`.
- **Consequence**: The `tally_accepts_non_triage_traffic()` contract check stays in place as a safeguard, and `docs/RESULTS.md` now publishes the live Tally-measured cost table.

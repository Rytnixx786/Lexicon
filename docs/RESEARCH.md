# Lexicon Research Notes

Checked 2026-11-19. **[V]** = verified on the linked source. **[U]** = unverified, so confirm it before relying on it. Leaderboards and model names move fast; re-check before citing any of them in the README.

---

## §1 BIRD
- Site: https://bird-bench.github.io/
- 12,751 question–SQL pairs over 95 databases (33.4 GB). Train has 9,428; dev has 1,534 [V]. The dev set has about 11 databases [U].
- Dev download: https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip [V]. License CC BY-SA 4.0 since April 2024 [V]. Do not commit the data.
- Metrics:
  - EX: the predicted SQL returns the same result set as the gold SQL [U, standard definition].
  - R-VES: reward-based valid efficiency score, used for test submissions since August 2024 [V on name and date].
  - Soft-F1: used in Mini-Dev [V].
- Test EX leaderboard [V]:

  | Entry | Test EX |
  |---|---|
  | Human | 92.96 |
  | Top systems | about 82–83 (AskData 83.40, GrainSQL 82.95) |
  | Best single trained model | about 80 |

- **Mini-Dev** [V]: 500 questions in SQLite, MySQL and **PostgreSQL** versions, with annotations corrected in 2026. Its leaderboard lists frontier models in the high 80s. Lexicon uses its PostgreSQL version for the 200-question slice.
- BIRD-Interact (ICLR 2026) [V]: conversational and agentic text-to-SQL with clarification. At release, frontier models scored very low (GPT-5 at about 9–17% success). Cite it when you explain why clarification matters.

## §2 Spider 2.0 [V]
https://spider2-sql.github.io/: enterprise workflows on Snowflake, BigQuery and dbt (Snow, Lite and DBT variants). It is out of scope for Lexicon, but mention it as the harder enterprise benchmark you chose not to target.

## §3 sqlglot
- Version 30.33.0 (2026-11-17) [V]: https://pypi.org/project/sqlglot/
- `parse_one(sql, dialect="postgres")` and `tree.find_all(exp.Table)` [V]: https://sqlglot.com/sqlglot.html. In v30 the expressions module is split into submodules; `exp.X` access probably still works [U]. Pin the version and test the guard against it.
- Guard recipe [U, from memory; the tests make it verified]:
  - single statement
  - root `Select` or a set operation
  - reject `Insert`, `Update`, `Delete`, `Merge`, `Drop`, `Create`, `Alter`/`AlterTable`, `TruncateTable`, `Command`
  - subtract CTE aliases from table names
  - LIMIT through `tree.args.get("limit")` / `tree.limit(n)`
  - always add a DB-level read-only layer

## §4 LangGraph
- `langgraph` 1.2.23 and `langgraph-checkpoint-postgres` 3.2.1 (both November 2026) [V]: https://pypi.org/project/langgraph/
- `PostgresSaver.from_conn_string(URI)` plus `.setup()` once. The async variant is `AsyncPostgresSaver`. Self-managed connections need `autocommit=True` and `row_factory=dict_row` [V].
- `interrupt()` and `Command(resume=...)` [V]: https://docs.langchain.com/oss/python/langgraph/interrupts
  - Needs a checkpointer and a `thread_id`.
  - The payload must be JSON-serialisable.
  - A resumed node re-runs from its start, so make it idempotent.
  - Don't wrap it in a bare try/except.
  - Several interrupts in one node match resume values by position.

## §5 Semantic-layer formats to borrow from
- **Snowflake semantic views / Cortex Analyst** [V], https://docs.snowflake.com/en/user-guide/views-semantic/semantic-view-yaml-spec:
  - `synonyms` on tables, dimensions, facts and metrics
  - `sample_values` and `is_enum`
  - `verified_queries` with `verified_by` / `verified_at`
  - This is the most useful model for Lexicon.
- **dbt ≥ 1.12** [V], https://docs.getdbt.com/docs/build/semantic-models: semantic models and metrics live under `models:`. Metrics have `name`, `type` (simple/ratio/derived/...), `agg`, `expr`, `label`, `description`.
- **Apache Ossie** (formerly OSI; Snowflake + Salesforce + dbt Labs; Apache incubating since July 2026) [V], https://github.com/apache/ossie:
  - `datasets`, `fields`, `relationships`, `metrics`, `ai_context`, `custom_extensions`
  - v0.1.1 is released
  - A `synonyms` field was not confirmed.
- **Cube** [V]: measures carry `name`, `sql`, `type`, `description`, `meta.ai_context`.

## §6 Evidence that context and semantic layers help (cite carefully; most of it is vendor-run)
- **Sequeda et al. (data.world)** 2023 [V], https://arxiv.org/abs/2311.07509: GPT-4 zero-shot went from 16% on SQL over the raw schema to 54% over a knowledge graph.
- **dbt Labs** [V, vendor, small n], https://docs.getdbt.com/blog/semantic-layer-vs-text-to-sql-2026:
  - 2023: text-to-SQL 32.7% vs semantic layer 60.5%.
  - 2026 rerun: frontier models reach 84–90% with raw text-to-SQL vs 98–100% with the semantic layer, on 11 questions × 20 runs.
- **Snowflake Cortex Analyst** [V, vendor]: 90%+ on internal sets vs 51% for GPT-4o. In a BIRD-based test, 57% without a good semantic model and 78% with one.
- An independent 2026 paired benchmark of semantic layers [U, not read]: https://arxiv.org/abs/2604.25149. Read it before writing the README.

**Where Lexicon adds something**: an open, reproducible, per-component ablation (descriptions vs rules vs examples), crossed with model choice and measured with CIs. The vendor studies above compare "semantic layer vs none" as a whole, usually without a model axis.

## §7 Ambiguity and abstention
| Dataset | Source | What it is |
|---|---|---|
| AmbiQT | https://arxiv.org/abs/2310.13659 [V] | 3,000+ Spider-derived questions with two valid SQL readings (column, table and join ambiguity) |
| AMBROSIA (NeurIPS 2024 D&B) | https://arxiv.org/abs/2406.19073 [V] | Scope ambiguity, attachment ambiguity and vagueness |
| TriageSQL | https://arxiv.org/abs/2010.12634 [V] | 5 classes (answerable, ambiguous, lack-data, unanswerable-by-SQL, small talk). RoBERTa gets about 60% F1. Basis for Lexicon's route taxonomy |
| PRACTIQ (NAACL 2025) | https://arxiv.org/abs/2410.11076 [V] | 2,800 conversations with clarification turns; 4 ambiguous and 4 unanswerable categories. Basis for the should-clarify sub-types |

## §8 ARES facts (read from `H:\Projects\Ares`, 2026-11-19)
- Default gate metrics and thresholds: see `ARES_CHANGES.md`.
- Plugin protocol: `evaluate(new_metrics, champion_metrics, slice_metrics=None, config=None, n_samples=1) -> GateDecision`, discovered through the entry-point group `ares.gate_plugins` (`ares/gate/plugins.py`, `docs/gate-plugin-guide.md`).
- The API compare endpoint is hard-wired to the `default` plugin (`ares/api/routers/evaluations.py`).
- Python client: `ares_client.AresClient(base_url, api_key).submit_evaluation(...)`.
- Coverage floor: 92% (`--cov-fail-under=92`).

## §9 Patterns reused from your own projects (read-only)
- **Nexus refusal gate**: `Nexus_Researcher/NEXUS_R_Main/backend/app/agents/nodes.py`. The refusal is a graph node, deterministic, with no LLM call, so a typed `INSUFFICIENT_CONTEXT` replaces a guess. Lexicon generalises it into typed refusals plus an `interrupt()` clarification.
- **MindStack and Nexus retrieval**: BM25 + vector search fused with RRF (k = 60), then a cross-encoder. Lexicon reuses BM25 + embeddings + RRF over context cards. A cross-encoder is optional, since there are few cards.

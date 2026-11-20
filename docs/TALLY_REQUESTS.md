# What Lexicon Needs From Tally

Lexicon never edits Tally. These are requests you fulfil in the Tally repo, most of them already in Tally's Week 5 roadmap (T5.6).

| # | Need | Why | Tally status |
|---|---|---|---|
| T1 | `lexicon_ro` role: read-only, `marts` schema only, `statement_timeout` 5s | Second safety layer for the agent | CONTRACTS §3 |
| T2 | dbt `manifest.json` published at a stable path (`Tally/dbt/target/manifest.json`, or a release asset) | `lexicon context sync` imports column descriptions | Tally BUILD_SPEC §8.3 |
| T3 | Every mart column has a human-readable `description` | It becomes L1 context | Tally BUILD_SPEC §8.3 |
| T4 | Model contracts enforced on `marts.*` | Breaking renames fail in Tally, not in Lexicon | CONTRACTS intro |
| T5 | A `pg_dump` of `marts` at a tagged Tally release | Frozen eval snapshot | Tally ROADMAP T5.6 |
| T6 | `fct_outcomes.decided_at` and `run_id` | Time-window questions and per-run questions | CONTRACTS §4 (added 2026-11-19) |
| T7 | Ingest of Lexicon's own spans (`tenant_id = lexicon`, `feature = sql_agent`) | Cost per question | **Done** (`280/280` spans landed in `marts.fct_requests`, `$0.000242/q`) |
| T8 | Scope the `fct_requests.decision_id` tests to triage traffic: add `config: {where: "feature = 'ticket_triage'"}` to its `not_null` and `relationships` tests in `dbt/models/marts/_marts.yml` | Found 2026-11-20: publishing Lexicon's calls (feature `sql_agent`, no triage decision) failed Tally's `dbt build`. Scoped in Tally `dbt/models/marts/_marts.yml` with a regression test; Tally builds 46/46 green with Lexicon traffic ingested. | **Done** |

If Tally's contract version changes, Lexicon pins the old snapshot until its context files are updated and re-verified.

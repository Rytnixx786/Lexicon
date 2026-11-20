# Context YAML Spec (v1)

The context layer is the core of Lexicon. This file defines its format field by field. `src/lexicon/context/schema.py` implements it as pydantic models; the models are the authority if this doc and the code disagree.

Borrowed conventions [RESEARCH §5]:

| Source | What Lexicon takes |
|---|---|
| Snowflake semantic views | `synonyms`, `sample_values`, `is_enum`, `verified_queries` with `verified_by` / `verified_at` |
| dbt metrics | `type: simple \| ratio \| derived`, `expr`, `label`, `description` |
| Apache Ossie (formerly OSI) | `ai_context`, `relationships` |

If you later want to export to Ossie or dbt, these fields map across directly.

---

## 1. Files

```
context/
  tables/<schema>.<table>.yaml     # one per table
  metrics.yaml                     # metric definitions (shared across tables)
  rules.yaml                       # business rules
  joins.yaml                       # allowed join paths
  examples/<topic>.yaml            # verified example queries (NOT the golden set)
  glossary.yaml                    # cross-table synonyms ("spend", "volume", "accuracy")
```

## 2. Table file

```yaml
version: 1
table: marts.fct_outcomes
grain: "one row per decision (one ticket triaged by one setup in one run)"
primary_key: [decision_id]
description: >
  Final outcome of each triage decision, with total cost across all model calls
  in the decision (a cascade escalation counts both calls).
ai_context: >
  Use this table for accuracy and cost-per-correct questions. Use fct_requests
  only for per-call questions (latency, tokens, cache status).
synonyms: [outcomes, decisions, triage results]
source:
  dbt_model: model.tally.fct_outcomes   # descriptions synced from manifest.json
  synced_at: 2027-01-10
columns:
  - name: is_correct
    type: boolean
    description: "True when predicted_label equals gold_label."   # synced from dbt
    synonyms: [correct, right, accurate]
    level: L1                     # first ablation level where this text appears
  - name: setup
    type: text
    description: "Which serving setup produced the decision."
    is_enum: true
    sample_values: [A_tfidf, B_local, C_hosted, D_cascade]
    value_meanings:
      A_tfidf: "TF-IDF + logistic regression, no LLM"
      D_cascade: "local model first, escalate to hosted when unsure"
  - name: total_cost_usd
    type: numeric
    description: "Sum of cost_usd over all model calls in the decision, in US dollars."
    synonyms: [spend, cost, money spent]
    unit: USD
    expected_range: [0, 0.05]     # used by the sanity checker
  - name: split
    type: text
    is_enum: true
    sample_values: [dev, tune, test]
    rule_refs: [R-HEADLINE-TEST-ONLY]
rule_refs: [R-HEADLINE-TEST-ONLY, R-FAILED-IS-WRONG]
metric_refs: [accuracy, cost_per_correct]
```

### Field rules
| Field | Required | Notes |
|---|---|---|
| `table`, `grain`, `primary_key`, `description` | yes | `grain` is a sentence, not a column list. Wrong-grain joins are the most common text-to-SQL bug. |
| `columns[].description` | yes | Synced from dbt where possible; a manual override needs `override_reason` |
| `columns[].sample_values` | for categorical columns | ≤ 10 values, taken from the snapshot by `lexicon context sample` |
| `columns[].level` | no (default L1) | Lets the ablation renderer include or exclude text by level |
| `expected_range` | for metric-like numeric columns | Drives the `magnitude` sanity flag |
| `rule_refs`, `metric_refs` | when applicable | Must resolve to ids in `rules.yaml` / `metrics.yaml` |

## 3. `metrics.yaml`

```yaml
version: 1
metrics:
  - id: accuracy
    label: Accuracy
    type: ratio
    numerator: "count(*) filter (where is_correct)"
    denominator: "count(*)"
    base_table: marts.fct_outcomes
    description: "Share of decisions that were correct."
    synonyms: [hit rate, correctness, precision of triage]
    expected_range: [0, 1]
  - id: cost_per_correct
    label: Cost per correct triage
    type: ratio
    numerator: "sum(total_cost_usd)"
    denominator: "count(*) filter (where is_correct)"
    base_table: marts.fct_outcomes
    description: >
      Total cost divided by number of correct decisions over the SAME set of rows.
      Never average per-row ratios.
    synonyms: [cost per correct, cpc, unit cost of a right answer]
    rule_refs: [R-CPC-NO-AVG-OF-RATIOS]
    source_of_truth: "Tally dbt/models/marts/_marts.yml#cost_per_correct_usd"   # CI diffs against this
    expected_range: [0, 1]
  - id: cost
    label: Cost
    requires_choice: true            # deterministic clarification trigger
    choices:
      - {id: cost_usd, description: "Tally cost model (GPU amortisation + API list price)"}
      - {id: billed_usd, description: "What Cinch's FinOps ledger charged the tenant"}
```

## 4. `rules.yaml`

```yaml
version: 1
rules:
  - id: R-REOPEN-NOT-RESOLVED
    statement: "A ticket that was reopened after being resolved does not count as resolved."
    applies_to: [synthetic.tickets, synthetic.ticket_events]
    sql_hint: "exclude tickets with any ticket_events.event_type = 'reopened' after the last 'resolved'"
    severity: must              # must | should
    level: L2
  - id: R-HEADLINE-TEST-ONLY
    statement: "Headline accuracy and cost-per-correct use split = 'test' only, unless the user asks for another split."
    applies_to: [marts.fct_outcomes, marts.mart_cost_per_outcome]
    sql_hint: "where split = 'test'"
    severity: must
    level: L2
  - id: R-FAILED-IS-WRONG
    statement: "Requests with http_status <> 200 count as incorrect decisions, never as missing."
    applies_to: [marts.fct_outcomes, marts.fct_requests]
    severity: must
    level: L2
  - id: R-EXCLUDE-TEST-TENANTS
    statement: "Exclude tenants whose tenant_id starts with 'test-' from all business reporting."
    applies_to: [marts.dim_tenant, marts.fct_requests, marts.fct_outcomes]
    sql_hint: "tenant_id not like 'test-%'"
    severity: must
    level: L2
```

Every `must` rule has at least two golden questions that test it (`GOLDEN_SET_GUIDE.md`), and appears in the `rule_not_applied` sanity check.

## 5. `joins.yaml`

```yaml
version: 1
joins:
  - left: marts.fct_outcomes
    right: marts.fct_requests
    condition: "fct_outcomes.final_request_id = fct_requests.request_id"  # not `on`: YAML 1.1 parses it as true
    relationship: one_to_one
    note: "Use decision_id for all calls in a decision (one_to_many)."
  - left: marts.fct_requests
    right: marts.dim_tenant
    condition: "fct_requests.tenant_id = dim_tenant.tenant_id"
    relationship: many_to_one
forbidden:
  - pair: [marts.fct_requests, marts.mart_cost_per_outcome]
    reason: "Different grains; join produces fan-out. Aggregate fct_requests first."
```

The guard (BUILD_SPEC §7) can enforce `forbidden` pairs once the allowlist is stable. Until then the fan-out sanity check catches them at runtime.

## 6. Verified example queries (`examples/*.yaml`)

```yaml
version: 1
examples:
  - id: EX-017
    question: "What was the cost per correct triage for the cascade on the test split?"
    sql: |
      select sum(total_cost_usd) / nullif(count(*) filter (where is_correct), 0) as cost_per_correct_usd
      from marts.fct_outcomes
      where setup = 'D_cascade' and split = 'test'
    tables: [marts.fct_outcomes]
    metric_refs: [cost_per_correct]
    rule_refs: [R-HEADLINE-TEST-ONLY]
    verified_by: <your name>
    verified_at: 2027-01-12
    verified_result_hash: sha256:...      # recomputed in CI against the snapshot
```

Rules:
- 10–15 examples per **main** table, and 60–90 in total.
- They must not paraphrase golden questions. `test_no_golden_leakage.py` checks exact text, normalised text, a RapidFuzz ratio above 90, and the SQL AST fingerprint (sqlglot `diff` / normalised SQL).
- `verified_result_hash` is rechecked in CI. If Tally's snapshot changes, every example must be re-verified.

## 7. Rendering by ablation level

`context/render.py::render(retrieved, level) -> str` emits a deterministic text block:
- **L0**: DDL only.
- **L1**: + table descriptions, grain, column descriptions, synonyms, sample values, value meanings.
- **L2**: + metric definitions, rules (`statement` + `sql_hint`), allowed and forbidden joins.
- **L3**: + the top-3 retrieved verified examples (question + SQL).

Snapshot-test the rendered blocks (`tests/context/__snapshots__/`), so an accidental context change shows up in review as a text diff.

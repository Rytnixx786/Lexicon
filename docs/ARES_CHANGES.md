# Changes Lexicon Would Like in ARES (optional)

Lexicon does **not** need ARES changes to work: the gate runs through ARES's plugin runtime (`ares.gate.plugins.evaluate_with_plugin`), with the plugin installed from `ares_plugin/` via the `ares.gate_plugins` entry point. This file specs a small ARES PR, which you make in `H:\Projects\Ares` under its own rules (92% coverage floor, evidence in the PR). It would turn the gate into a service with history and a dashboard.

Facts below come from reading ARES on 2026-11-19. Re-check them before starting.

## What ARES does today
- `ares/gate/rules_engine.py::evaluate()` gates on fixed names:
  - `overall_f1` fails on a drop above `gate.max_regression_f1` (0.02)
  - `overall_accuracy` fails on a drop above `gate.max_regression_accuracy` (0.015)
  - `latency_p99_ms` fails on a rise above `gate.max_latency_regression_pct`
  - `model_size_mb` fails on growth without an accuracy gain
  - each critical slice must keep `f1` ≥ `gate.critical_slice_min_f1` (0.6)
  - other metrics are ignored
- `ares/gate/plugins.py` supports named plugins through the entry-point group `ares.gate_plugins`. However, `ares/api/routers/evaluations.py` (around line 129) always calls `evaluate_with_plugin("default", ...)`.
- `CompareRequest` = `{model_name, commit_sha, new_metrics, slice_metrics: {name: {f1, is_critical}}, n_samples}`. Slice metrics are keyed on `f1`.

## Proposed PR (`feat/api-named-gate-plugins`)

| # | Change | Test |
|---|---|---|
| A1 | `CompareRequest.gate_plugin: str = "default"` | Unknown plugin → 422 |
| A2 | Server-side allowlist `gate.allowed_plugins: [default]` in `ares.config.yaml`. A plugin runs only if it is installed **and** allowlisted (plugins are trusted code, as the plugin guide says). | Installed but not allowlisted → 403 |
| A3 | Slice metrics accept a generic `score`, alongside `f1` | The default plugin still reads `f1`; the Lexicon plugin reads `score` |
| A4 | The champion is stored per `(model_name, gate_plugin)` | Two plugins on the same model name do not overwrite each other's champions |
| A5 | `ares_client.submit_evaluation(..., gate_plugin="lexicon")` | Client test with a mocked API |
| A6 | The dashboard shows the plugin name and version on each decision | Snapshot test |

Use `model_name = "lexicon-sql-agent"` and `commit_sha` = Lexicon's commit.

## Lexicon payload, once A1–A5 land
```json
{
  "model_name": "lexicon-sql-agent",
  "commit_sha": "<sha>",
  "gate_plugin": "lexicon",
  "new_metrics": {"overall_accuracy": 0.742, "over_refusal_rate": 0.03, "false_clarify_rate": 0.05},
  "slice_metrics": {
    "should_clarify": {"score": 0.80, "is_critical": true},
    "should_refuse":  {"score": 0.95, "is_critical": true},
    "simple": {"score": 0.95}, "multi_join": {"score": 0.65},
    "metric_definition": {"score": 0.70}, "business_rule_trap": {"score": 0.55}
  },
  "n_samples": 120
}
```

Until that PR merges, `make gate` runs the plugin locally and also POSTs `overall_accuracy` to the default gate, for history only. That second POST is informational; the local plugin verdict is the one CI enforces.

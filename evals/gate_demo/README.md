# Gate demo: a harmful context change must be blocked

`context_without_reopen_rule/` is the production context minus one business rule, `R-RESOLVED-IS-STATUS`
("a ticket is resolved only if its status is resolved or closed; never count 'resolved' events").
That is the kind of edit a well-meaning PR might make ("simplify the rules").

Run all three seeds to match the 3-seed champion (`qwen7b-L3`, `qwen7b-L3-s1`, `qwen7b-L3-s2`):

```bash
lexicon eval --model qwen7b --level L3 --seed 0 --context-dir evals/gate_demo/context_without_reopen_rule --run-id demo-no-reopen-rule --concurrency 4
lexicon eval --model qwen7b --level L3 --seed 1 --context-dir evals/gate_demo/context_without_reopen_rule --run-id demo-no-reopen-rule-s1 --concurrency 4
lexicon eval --model qwen7b --level L3 --seed 2 --context-dir evals/gate_demo/context_without_reopen_rule --run-id demo-no-reopen-rule-s2 --concurrency 4
lexicon gate demo-no-reopen-rule,demo-no-reopen-rule-s1,demo-no-reopen-rule-s2
```

The 3-seed majority-vote verdict is written to `reports/runs/demo-no-reopen-rule/gate_verdict.json` and summarised in `docs/RESULTS.md`.

**Observed result (honest note):** The ARES gate (`lexicon@1.0.0`, via ARES's plugin runtime) returned **FAIL** across the 3-seed majority vote (`seed_flip_rate = 0.8%`, 1/120 questions) because the protected bucket `should_refuse` regressed from `0.800` to `0.750` (one `unanswerable_from_schema` item, `G-R-19`, shifted to `out_of_scope` in 2 of 3 seeds). Overall execution accuracy actually rose (`0.362 -> 0.412`, +5.0 pts) and `business_rule_trap` did not drop (`0.350 -> 0.400`), because Qwen2.5-7B at L3 already missed 4 of the 5 questions citing `R-RESOLVED-IS-STATUS` even when the rule was present.

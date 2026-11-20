from __future__ import annotations

from lexicon_ares_gate import LexiconGate

CHAMP = {"overall_accuracy": 0.70, "over_refusal_rate": 0.02, "false_clarify_rate": 0.05}
SLICES = {"should_clarify": {"score": 0.60}, "should_refuse": {"score": 0.90}}


def run(new: dict, slices: dict, champ: dict | None = None):  # type: ignore[no-untyped-def,type-arg]
    return LexiconGate().evaluate(
        new, CHAMP if champ is None else champ, slices, {"champion_slice_metrics": SLICES}, n_samples=120
    )


def test_pass_within_limits() -> None:
    d = run({"overall_accuracy": 0.69, "over_refusal_rate": 0.02, "false_clarify_rate": 0.05}, SLICES)
    assert d.passed
    assert d.verdict == "PASS"
    assert not d.should_promote


def test_ex_drop_over_two_points_fails() -> None:
    d = run({"overall_accuracy": 0.675}, SLICES)
    assert not d.passed
    assert "execution accuracy dropped 2.5 pts" in d.reason


def test_protected_bucket_regression_fails_even_with_better_ex() -> None:
    d = run({"overall_accuracy": 0.80}, {"should_clarify": {"score": 0.55}, "should_refuse": {"score": 0.90}})
    assert not d.passed
    assert d.slice_regressions[0]["slice"] == "should_clarify"


def test_refuse_everything_is_caught_by_counter_metric() -> None:
    d = run(
        {"overall_accuracy": 0.70, "over_refusal_rate": 0.30},
        {"should_clarify": {"score": 0.6}, "should_refuse": {"score": 1.0}},
    )
    assert not d.passed
    assert "over_refusal_rate rose" in d.reason


def test_first_run_promotes() -> None:
    d = LexiconGate().evaluate({"overall_accuracy": 0.5}, {}, SLICES)
    assert d.passed
    assert d.should_promote


def test_improvement_promotes_and_accepts_f1_key() -> None:
    d = run({"overall_accuracy": 0.75}, {"should_clarify": {"f1": 0.65}, "should_refuse": {"f1": 0.95}})
    assert d.passed
    assert d.should_promote
    assert d.deltas["should_clarify"] > 0


def test_custom_threshold_from_config() -> None:
    d = LexiconGate().evaluate(
        {"overall_accuracy": 0.68},
        CHAMP,
        SLICES,
        {"champion_slice_metrics": SLICES, "lexicon": {"max_ex_drop": 0.01}},
    )
    assert not d.passed

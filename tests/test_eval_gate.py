"""The routing regression gate CI runs (harness/eval_routing.py --baseline)."""

from harness.eval_routing import regressions

BASE = {"overall_routed_right": 0.84, "cheap_leakage": {"middle": 0.017, "decompose": 0.0}}


def test_within_tolerance_passes():
    r = {"overall_routed_right": 0.83, "cheap_leakage": {"middle": 0.03, "decompose": 0.0}}
    assert regressions(r, BASE, tolerance=0.02) == []


def test_accuracy_drop_fails():
    r = {"overall_routed_right": 0.80, "cheap_leakage": BASE["cheap_leakage"]}
    assert len(regressions(r, BASE, tolerance=0.02)) == 1


def test_safety_leak_fails_even_if_accuracy_rises():
    r = {"overall_routed_right": 0.90, "cheap_leakage": {"middle": 0.10, "decompose": 0.0}}
    (msg,) = regressions(r, BASE, tolerance=0.02)
    assert "want-middle" in msg

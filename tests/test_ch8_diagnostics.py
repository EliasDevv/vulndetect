"""Tests for the pure functions in scripts/ch8_diagnostics.py on synthetic data."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN

_SPEC = importlib.util.spec_from_file_location(
    "ch8_diagnostics", Path(__file__).resolve().parents[1] / "scripts" / "ch8_diagnostics.py"
)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)

V, S = LABEL_VULN, LABEL_SAFE


def pred(i: int, gold: str, label: str | None, raw: str = "", cwe: str = "CWE-119") -> dict:
    return {"id": f"x{i}", "cwe": cwe, "gold_label": gold, "pred_label": label, "raw_output": raw}


def parse(text: str) -> str | None:
    return text if text in (V, S) else None


def test_flip_profile_counts_directions_and_correctness():
    gold = [V, V, S, S, V]
    a = [pred(0, V, S), pred(1, V, V), pred(2, S, V), pred(3, S, S), pred(4, V, None)]
    b = [pred(0, V, V), pred(1, V, S), pred(2, S, S), pred(3, S, V), pred(4, V, V)]
    assert [p["gold_label"] for p in a] == gold
    r = diag.flip_profile(a, b)
    assert r["n"] == 4
    assert r["changed"] == 4
    assert r["to_vuln"] == 2 and r["to_vuln_correct"] == 1
    assert r["to_safe"] == 2 and r["to_safe_correct"] == 1
    assert r["net_correct"] == 0
    assert r["share"] == 1.0


def test_flip_profile_identical_runs_have_zero_delta():
    a = [pred(0, V, V), pred(1, S, S), pred(2, V, S)]
    r = diag.flip_profile(a, [dict(p) for p in a])
    assert r["changed"] == 0
    assert all(abs(v) < 1e-12 for v in r["delta"].values())


def test_flip_profile_rejects_different_instances():
    with pytest.raises(ValueError):
        diag.flip_profile([pred(0, V, V)], [pred(1, V, V)])


def test_agreement_table_and_overrides():
    agents = [
        {"id": "x0", "detector_output": V},
        {"id": "x1", "detector_output": V},
        {"id": "x2", "detector_output": S},
        {"id": "x3", "detector_output": S},
        {"id": "x4", "detector_output": "??"},
    ]
    preds = [
        pred(0, V, None, raw=S),
        pred(1, S, None, raw=S),
        pred(2, V, None, raw=V),
        pred(3, S, None, raw=S),
        pred(4, V, None, raw=V),
    ]
    r = diag.agreement(agents, preds, parse)
    assert r["table"][V] == {V: 0, S: 2, "None": 0}
    assert r["table"][S] == {V: 1, S: 1, "None": 0}
    assert r["table"]["None"][V] == 1
    assert r["p_validator_vuln_given_detector_vuln"] == 0.0
    assert r["p_validator_vuln_given_detector_safe"] == 0.5
    o = r["overrides"]
    assert o["n"] == 3 and o["right"] == 2
    assert o["to_safe"] == 2 and o["to_safe_right"] == 1
    assert o["to_vuln"] == 1 and o["to_vuln_right"] == 1


def record(i: int, label: str, same: bool) -> dict:
    before = f"int f{i}() {{ return {i}; }}"
    return {
        "id": f"x{i}",
        "cwe": "CWE-119",
        "label": label,
        "func_before": before,
        "func_after": before if same else before.replace("return", "return -"),
    }


def test_identity_shortcut_rule_and_sides():
    records = [record(0, V, False), record(1, V, False), record(2, S, True), record(3, S, True)]
    run = [pred(0, V, V), pred(1, V, S), pred(2, S, S), pred(3, S, None)]
    r = diag.identity_shortcut(records, {"m": [run]})
    assert r["counts"][V] == {"unchanged": 0, "changed": 2}
    assert r["counts"][S] == {"unchanged": 2, "changed": 0}
    assert r["rule"]["mcc"] == pytest.approx(1.0)
    assert r["approaches"]["m"]["changed"] == 0.5
    assert r["approaches"]["m"]["unchanged"] == 1.0


def test_fix_described_fn_groups():
    cue = re.compile(r"fixes a vulnerability")
    base = [
        pred(0, V, S, raw="this fixes a vulnerability, not vulnerable"),
        pred(1, V, S, raw="refactoring"),
        pred(2, S, S, raw="ok"),
        pred(3, V, V, raw="fixes a vulnerability"),
    ]
    other = [pred(0, V, V), pred(1, V, S), pred(2, S, V), pred(3, V, V)]
    r = diag.fix_described_fn(base, other, cue)
    assert r["n_fn"] == 2 and r["n_fix_described"] == 1
    assert r["other_right_on_fix_described"] == 1.0
    assert r["other_right_on_other_fn"] == 0.0
    assert r["n_tn"] == 1 and r["other_right_on_tn"] == 0.0


def _approach(f1s: list[float], mcc: float) -> dict:
    return {"macro": {"f1": {"values": f1s}}, "secondary": {"macro": {"mcc": {"mean": mcc}}}}


def test_robustness_reads_primary_and_sensitivity():
    comparison = {
        "approaches": {"a": _approach([0.5, 0.6, 0.4], 0.2), "b": _approach([0.6, 0.5, 0.5], 0.1)},
        "ttests": [{"a": "a", "b": "b", "p": 0.03, "mean_diff": -0.03}],
        "verdicts": {
            "alpha": 0.05,
            "pairs": [
                {
                    "a": "a",
                    "b": "b",
                    "p_max": 0.2,
                    "direction": "b>a",
                    "significant": False,
                    "p_holm": 0.4,
                    "sensitivity": {"p_max": 0.01, "significant": True, "p_holm": 0.02},
                }
            ],
        },
    }
    (row,) = diag.robustness(comparison, (("b", "a"),))
    assert row["display"] == ["b", "a"]
    assert row["primary"]["direction"] == "b"
    assert row["primary"]["significant"] is False
    assert row["primary"]["holm_significant"] is False
    assert row["wrong"]["significant"] is True
    assert row["ttest"] == {"p": 0.03, "significant": True, "direction": "b"}
    assert row["mcc_order"] == "a"
    assert row["f1_seeds_favouring"] == {"a": 1, "b": 2}

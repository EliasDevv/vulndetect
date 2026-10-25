"""Tests for the pure functions in scripts/ch7_diagnostics.py on synthetic data."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN

_SPEC = importlib.util.spec_from_file_location(
    "ch7_diagnostics", Path(__file__).resolve().parents[1] / "scripts" / "ch7_diagnostics.py"
)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


def pred(
    i: int, gold: str, label: str | None, raw: str | None = None, cwe: str = "CWE-119"
) -> dict:
    return {
        "id": f"x{i}",
        "cwe": cwe,
        "gold_label": gold,
        "pred_label": label,
        "raw_output": raw if raw is not None else str(label),
    }


def parse(text: str) -> str | None:
    return text if text in (LABEL_VULN, LABEL_SAFE) else None


def test_round0_identical():
    a = [pred(0, LABEL_VULN, LABEL_VULN), pred(1, LABEL_SAFE, LABEL_SAFE)]
    res = diag.round0_check(a, [dict(p) for p in a])
    assert res == {"n": 2, "n_label_differs": 0, "n_raw_output_differs": 0, "identical": True}


def test_round0_detects_label_and_raw_differences():
    a = [pred(0, LABEL_VULN, LABEL_VULN), pred(1, LABEL_SAFE, LABEL_SAFE, raw="x")]
    b = [pred(0, LABEL_VULN, LABEL_SAFE), pred(1, LABEL_SAFE, LABEL_SAFE, raw="y")]
    res = diag.round0_check(a, b)
    assert res["n_label_differs"] == 1
    assert res["n_raw_output_differs"] == 2
    assert not res["identical"]


def test_round0_refuses_different_instances():
    with pytest.raises(ValueError):
        diag.round0_check([pred(0, LABEL_VULN, LABEL_VULN)], [pred(1, LABEL_VULN, LABEL_VULN)])


def test_dual_agent_fallback_only_touches_unparseable_validator():
    preds = [
        pred(0, LABEL_VULN, None, raw="garbage"),
        pred(1, LABEL_SAFE, None, raw=LABEL_VULN),
        pred(2, LABEL_VULN, None, raw=LABEL_VULN, cwe="CWE-20"),
        pred(3, LABEL_SAFE, None, raw=LABEL_SAFE, cwe="CWE-20"),
    ]
    agents = [
        {"id": "x0", "detector_output": LABEL_VULN},
        {"id": "x1", "detector_output": LABEL_SAFE},
        {"id": "x2", "detector_output": LABEL_SAFE},
        {"id": "x3", "detector_output": LABEL_SAFE},
    ]
    res = diag.dual_agent_rules(agents, preds, parse, policy="wrong")
    assert res["post_hoc"] is True
    assert res["n_validator_unparseable"] == 1
    assert res["n_validator_changed_detector"] == 2
    assert res["n_both_readable"] == 3
    assert res["validator"]["parse_failure_rate"] == pytest.approx(0.25)
    assert res["validator_then_detector"]["parse_failure_rate"] == 0.0
    assert res["validator_then_detector"]["macro_accuracy"] > res["validator"]["macro_accuracy"]


def test_pass_through_curve_counts_admitted_only():
    rounds = [
        {"phase": "A", "round": 1, "accepted": True, "detector_label": LABEL_SAFE},
        {"phase": "A", "round": 1, "accepted": False, "detector_label": None},
        {"phase": "A", "round": 2, "accepted": True, "detector_label": LABEL_VULN},
        {"phase": "B", "step": 1, "detector_label": LABEL_SAFE},
    ]
    res = diag.pass_through_curve(rounds, milestones=(1, 2))
    assert res["n_admitted"] == 2
    assert res["n_passed"] == 1
    assert res["at_round"] == {"1": 1.0, "2": 0.5}


def test_clopper_pearson_known_values():
    lo, hi = diag.clopper_pearson(0, 30)
    assert lo == 0.0
    assert hi == pytest.approx(0.1157, abs=1e-4)  # 1 - 0.025**(1/30)
    lo, hi = diag.clopper_pearson(30, 30)
    assert hi == 1.0
    assert lo == pytest.approx(1 - 0.1157, abs=1e-4)
    with pytest.raises(ValueError):
        diag.clopper_pearson(3, 2)


def test_hand_check_counts_and_refuses_unrated():
    rows = [
        {"verdict": "none", "claude_suggestion": "none", "detector_label": LABEL_VULN},
        {"verdict": "other", "claude_suggestion": "broken", "detector_label": LABEL_SAFE},
    ]
    res = diag.hand_check(rows)
    assert res["counts"] == {"intended": 0, "other": 1, "broken": 0, "none": 1}
    assert res["any_weakness_share"] == 0.5
    assert res["n_author_confirmed_preliminary"] == 1
    assert res["detector_vulnerable_by_category"]["none"] == 1
    with pytest.raises(ValueError):
        diag.hand_check([{"verdict": "", "detector_label": LABEL_SAFE}])


def test_verdict_change_on_both_parsed_only():
    a = [
        pred(0, LABEL_VULN, LABEL_VULN),
        pred(1, LABEL_VULN, None),
        pred(2, LABEL_SAFE, LABEL_SAFE),
    ]
    b = [
        pred(0, LABEL_VULN, LABEL_SAFE),
        pred(1, LABEL_VULN, LABEL_VULN),
        pred(2, LABEL_SAFE, LABEL_SAFE),
    ]
    assert diag.verdict_change(a, b) == {"changed": 1, "n": 2, "share": 0.5}


def test_regime_by_date():
    assert diag.regime(dt.datetime(2026, 8, 7, 12)) == "bf16"
    assert diag.regime(dt.datetime(2026, 8, 28, 0)) == "fp16"
    assert "check" in diag.regime(dt.datetime(2026, 9, 1), dtype="bfloat16")


def test_identifier_candidates_shapes():
    entries = [
        "Check `skb_put` bounds before copying.",
        "Verify buffer sizes in memcpyChecked and ReadFrame calls.",
        "Do not trust dev->priv or foo() results; use std::move carefully.",
        "**Disable Security Checks**: plain prose only.",
    ]
    cands = diag.identifier_candidates(entries)
    for expected in ("skb_put", "memcpyChecked", "ReadFrame", "dev->priv", "foo()", "std::move"):
        assert expected in cands
    assert "Disable" not in cands


def test_function_name_from_definition():
    assert (
        diag.function_name("static int foo_bar (struct x *y)\n{\n return bar(y);\n}") == "foo_bar"
    )
    assert diag.function_name("{}") is None


def test_leakage_reports_name_identifier_and_ngram_hits():
    test_records = [
        {"func_before": "int skb_put(struct sk_buff *skb, int len)\n{\n  return len;\n}"}
    ]
    memories = {"s_d/seed42": ["Bound `skb_put` calls.", "Never return len ; blindly"]}
    res = diag.leakage(memories, test_records)
    mem = res["memories"]["s_d/seed42"]
    assert mem["hits_function_name"] == ["skb_put"]
    assert "skb_put" in mem["hits_test_identifier"]
    assert "return len ;" in mem["hits_code_3gram_all"]
    assert "return len ;" not in mem["hits_code_3gram_identifier_like"]


def test_queue_durations_skip_failed_runs():
    rows = [
        {
            "started": "2026-09-12 01:00:00",
            "finished": "2026-09-12 03:30:00",
            "target": "b",
            "exit": "0",
        },
        {
            "started": "2026-09-12 04:00:00",
            "finished": "2026-09-12 04:10:00",
            "target": "b",
            "exit": "2",
        },
    ]
    assert diag.queue_durations(rows) == {"b": [2.5]}


def test_csv_gaps_exclude_superseded_row():
    rows = [
        {"timestamp": "2026-09-08T03:38:00", "approach": "sft", "seed": "44", "output": "o"},
        {"timestamp": "2026-09-08T05:39:28", "approach": "rag", "seed": "42", "output": "old"},
        {"timestamp": "2026-09-08T09:22:47", "approach": "dual_agent", "seed": "42", "output": "d"},
    ]
    gaps = diag.csv_gaps(rows)
    assert [g["output"] for g in gaps] == ["d"]
    assert gaps[0]["hours_since_previous_row"] == pytest.approx(5.7464, abs=1e-3)

"""Tests for vulndetect.eval: metrics on hand-computed cases, stats, run log."""

from __future__ import annotations

import csv

import pytest

from vulndetect.common.predictions import Prediction
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
from vulndetect.eval import (
    append_run,
    confusion,
    evaluate,
    format_report,
    logging_enabled,
    mcnemar,
    paired_ttest,
    scores,
    summary_metrics,
)


def pred(i: int, cwe: str, gold: str, out: str | None) -> Prediction:
    return {"id": f"p{i}", "cwe": cwe, "gold_label": gold, "pred_label": out, "raw_output": "x"}


def preds_from_counts(cwe: str, tp: int, fp: int, tn: int, fn: int, start: int = 0) -> list:
    """Build predictions realizing an exact confusion matrix."""
    out, i = [], start
    for _ in range(tp):
        out.append(pred(i, cwe, LABEL_VULN, LABEL_VULN))
        i += 1
    for _ in range(fp):
        out.append(pred(i, cwe, LABEL_SAFE, LABEL_VULN))
        i += 1
    for _ in range(tn):
        out.append(pred(i, cwe, LABEL_SAFE, LABEL_SAFE))
        i += 1
    for _ in range(fn):
        out.append(pred(i, cwe, LABEL_VULN, LABEL_SAFE))
        i += 1
    return out


def test_scores_hand_computed():
    # tp=6 fp=2 tn=8 fn=4 -> acc .70, prec .75, rec .60, f1 = 2*.45/1.35 = 2/3
    p = preds_from_counts("CWE-119", tp=6, fp=2, tn=8, fn=4)
    s = scores(confusion(p))
    assert s["accuracy"] == pytest.approx(0.70)
    assert s["precision"] == pytest.approx(0.75)
    assert s["recall"] == pytest.approx(0.60)
    assert s["f1"] == pytest.approx(2 / 3)


def test_scores_degenerate_all_safe_predictions():
    p = preds_from_counts("CWE-20", tp=0, fp=0, tn=5, fn=5)
    s = scores(confusion(p))
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["f1"] == 0.0


def test_unparseable_counts_as_wrong_and_is_reported():
    p = [
        pred(0, "CWE-119", LABEL_VULN, LABEL_VULN),
        pred(1, "CWE-119", LABEL_VULN, None),
        pred(2, "CWE-119", LABEL_SAFE, None),
        pred(3, "CWE-119", LABEL_SAFE, LABEL_SAFE),
    ]
    c = confusion(p)
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (1, 1, 1, 1)
    assert c["n_unparseable"] == 2
    report = evaluate(p)
    assert report["parse_failure_rate"] == pytest.approx(0.5)
    assert report["macro"]["accuracy"] == pytest.approx(0.5)


def test_unparseable_drop_policy_excludes_from_counts():
    """Checks that the drop policy excludes unparseable outputs from the counts."""
    p = [
        pred(0, "CWE-119", LABEL_VULN, LABEL_VULN),
        pred(1, "CWE-119", LABEL_VULN, None),
        pred(2, "CWE-119", LABEL_SAFE, None),
        pred(3, "CWE-119", LABEL_SAFE, LABEL_SAFE),
    ]
    c = confusion(p, policy="drop")
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (1, 0, 1, 0)
    assert c["n_unparseable"] == 2

    report = evaluate(p, policy="drop")
    assert report["unparseable_policy"] == "drop"
    assert report["parse_failure_rate"] == pytest.approx(0.5)
    assert report["n"] == 4
    assert report["n_scored"] == 2
    assert report["per_cwe"]["CWE-119"]["n_scored"] == 2
    assert report["macro"]["accuracy"] == pytest.approx(1.0)
    assert report["alt"]["policy"] == "wrong"
    assert report["alt"]["macro"]["accuracy"] == pytest.approx(0.5)

    report_wrong = evaluate(p)
    assert report_wrong["unparseable_policy"] == "wrong"
    assert report_wrong["alt"]["policy"] == "drop"
    assert report_wrong["alt"]["macro"]["accuracy"] == pytest.approx(1.0)


def test_confusion_unknown_policy_raises():
    with pytest.raises(ValueError, match="policy"):
        confusion([], policy="ignore")


def test_summary_metrics_carries_policy_and_alt():
    report = evaluate(preds_from_counts("CWE-200", tp=2, fp=1, tn=2, fn=1), policy="drop")
    summary = summary_metrics(report)
    assert summary["policy"] == "drop"
    assert summary["macro_f1_alt"] == pytest.approx(summary["macro_f1"])


def test_macro_vs_micro():
    p = preds_from_counts("CWE-119", tp=1, fp=0, tn=1, fn=0)
    p += preds_from_counts("CWE-20", tp=0, fp=4, tn=0, fn=4, start=10)
    report = evaluate(p)
    assert report["macro"]["accuracy"] == pytest.approx(0.5)
    assert report["micro"]["accuracy"] == pytest.approx(0.2)
    assert set(report["per_cwe"]) == {"CWE-119", "CWE-20"}
    assert report["per_cwe"]["CWE-119"]["n"] == 2


def test_evaluate_empty_raises():
    with pytest.raises(ValueError):
        evaluate([])


def test_format_report_contains_all_rows():
    text = format_report(evaluate(preds_from_counts("CWE-200", tp=2, fp=1, tn=2, fn=1)))
    for needle in ("CWE-200", "All (macro)", "All (micro)", "unparseable"):
        assert needle in text


def test_paired_ttest_detects_constant_shift():
    base = [0.60, 0.65, 0.70, 0.62, 0.68]
    better = [f + 0.10 for f in base]
    res = paired_ttest(better, base)
    assert res["df"] == 4
    assert res["mean_diff"] == pytest.approx(0.10)
    assert res["p"] < 1e-6
    assert res["ci95"][0] == pytest.approx(0.10)
    assert res["ci95"][1] == pytest.approx(0.10)


def test_paired_ttest_noisy_shift_significant():
    base = [0.60, 0.65, 0.70, 0.62, 0.68]
    better = [0.72, 0.74, 0.79, 0.75, 0.76]
    res = paired_ttest(better, base)
    assert res["t"] > 0 and res["p"] < 0.05
    assert res["ci95"][0] > 0


def test_paired_ttest_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_ttest([0.5], [0.6])


def test_mcnemar_hand_computed():
    preds_a, preds_b = [], []
    for i in range(8):
        preds_a.append(pred(i, "CWE-119", LABEL_VULN, LABEL_VULN))
        preds_b.append(pred(i, "CWE-119", LABEL_VULN, LABEL_SAFE))
    for i in range(8, 10):
        preds_a.append(pred(i, "CWE-119", LABEL_VULN, LABEL_SAFE))
        preds_b.append(pred(i, "CWE-119", LABEL_VULN, LABEL_VULN))
    for i in range(10, 14):
        preds_a.append(pred(i, "CWE-119", LABEL_SAFE, LABEL_SAFE))
        preds_b.append(pred(i, "CWE-119", LABEL_SAFE, LABEL_SAFE))
    res = mcnemar(preds_a, preds_b)
    assert (res["b"], res["c"], res["n"]) == (8, 2, 14)
    # exact two-sided binomial: 2 * P[X <= 2], X ~ Binom(10, .5) = 2*56/1024
    assert res["p"] == pytest.approx(2 * 56 / 1024)


def test_mcnemar_requires_same_ids():
    a = [pred(0, "CWE-119", LABEL_VULN, LABEL_VULN)]
    b = [pred(1, "CWE-119", LABEL_VULN, LABEL_VULN)]
    with pytest.raises(ValueError):
        mcnemar(a, b)


def test_mcnemar_no_discordant_pairs():
    a = [pred(0, "CWE-119", LABEL_VULN, LABEL_VULN)]
    assert mcnemar(a, a)["p"] == 1.0


def test_append_run_creates_header_once(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    report = evaluate(preds_from_counts("CWE-119", tp=2, fp=1, tn=2, fn=1))
    for seed in (42, 43):
        append_run(
            approach="baseline",
            model="meta-llama/Llama-3.2-3B-Instruct",
            seed=seed,
            params={"prompt_version": "fig1-v2"},
            metrics=summary_metrics(report),
            output="results/predictions/baseline",
            csv_path=csv_path,
        )
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert rows[0]["approach"] == "baseline"
    assert {r["seed"] for r in rows} == {"42", "43"}
    assert "macro_f1" in rows[0]["metrics"]


def test_logging_enabled_defaults_true():
    assert logging_enabled({}) is True
    assert logging_enabled({"output": {}}) is True
    assert logging_enabled({"output": {"log_experiments": True}}) is True
    assert logging_enabled({"output": {"log_experiments": False}}) is False


def test_mcnemar_follows_the_unparseable_policy():
    """Checks that McNemar treats unparseable pairs according to the policy."""
    a = [
        pred(0, "CWE-119", LABEL_VULN, LABEL_VULN),
        pred(1, "CWE-119", LABEL_VULN, None),
        pred(2, "CWE-119", LABEL_SAFE, LABEL_SAFE),
    ]
    b = [
        pred(0, "CWE-119", LABEL_VULN, LABEL_SAFE),
        pred(1, "CWE-119", LABEL_VULN, LABEL_VULN),
        pred(2, "CWE-119", LABEL_SAFE, None),
    ]
    res = mcnemar(a, b)
    assert (res["policy"], res["n"], res["n_excluded"], res["b"], res["c"]) == ("drop", 1, 2, 1, 0)
    res = mcnemar(a, b, policy="wrong")
    assert (res["n"], res["n_excluded"], res["b"], res["c"]) == (3, 0, 2, 1)
    with pytest.raises(ValueError, match="unknown unparseable policy"):
        mcnemar(a, b, policy="ignore")


def test_holm_hand_computed():
    from vulndetect.eval import ALPHA, holm

    # sorted: 0.01*3 = 0.03; 0.03*2 = 0.06; 0.04*1 = 0.04 -> running max 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.2]) == [0.2]
    assert holm([]) == []
    assert holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])
    assert ALPHA == 0.05
    with pytest.raises(ValueError):
        holm([0.5, 1.5])

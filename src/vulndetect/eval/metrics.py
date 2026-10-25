"""Classification metrics (per CWE, macro, micro) over canonical predictions."""

from __future__ import annotations

import math

from vulndetect.common.predictions import Prediction
from vulndetect.data.schema import LABEL_VULN

METRIC_NAMES = ("accuracy", "precision", "recall", "f1")

SECONDARY_METRIC_NAMES = ("mcc", "youden_j", "pred_positive_rate")

POLICIES = ("wrong", "drop")
DEFAULT_POLICY = "wrong"


def confusion(preds: list[Prediction], policy: str = DEFAULT_POLICY) -> dict:
    """Count tp/fp/tn/fn (positive = Vulnerable) plus unparseable outputs."""
    if policy not in POLICIES:
        raise ValueError(f"unknown unparseable policy {policy!r} — known: {POLICIES}")
    tp = fp = tn = fn = unparseable = 0
    for p in preds:
        gold_vuln = p["gold_label"] == LABEL_VULN
        pred = p["pred_label"]
        if pred is None:
            unparseable += 1
            if policy == "drop":
                continue
            pred_vuln = not gold_vuln
        else:
            pred_vuln = pred == LABEL_VULN
        if gold_vuln and pred_vuln:
            tp += 1
        elif gold_vuln:
            fn += 1
        elif pred_vuln:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_unparseable": unparseable}


def scores(counts: dict) -> dict:
    """Accuracy/precision/recall/F1 from a confusion dict (0.0 when undefined)."""
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mcc": (tp * tn - fp * fn) / denom if denom else 0.0,
        "youden_j": recall - fpr,
        "pred_positive_rate": (tp + fp) / n if n else 0.0,
    }


def _report(preds: list[Prediction], policy: str) -> dict:
    per_cwe: dict[str, dict] = {}
    for cwe in sorted({p["cwe"] for p in preds}):
        sub = [p for p in preds if p["cwe"] == cwe]
        counts = confusion(sub, policy)
        n_scored = counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"]
        per_cwe[cwe] = {"n": len(sub), "n_scored": n_scored, **counts, **scores(counts)}

    total = confusion(preds, policy)
    micro = scores(total)
    macro = {
        m: sum(row[m] for row in per_cwe.values()) / len(per_cwe)
        for m in METRIC_NAMES + SECONDARY_METRIC_NAMES
    }
    return {
        "n": len(preds),
        "n_scored": total["tp"] + total["fp"] + total["tn"] + total["fn"],
        "n_unparseable": total["n_unparseable"],
        "parse_failure_rate": total["n_unparseable"] / len(preds),
        "per_cwe": per_cwe,
        "macro": macro,
        "micro": micro,
    }


def evaluate(preds: list[Prediction], policy: str = DEFAULT_POLICY) -> dict:
    """Full metrics report (per CWE, macro, micro) plus the other policy's results as ``alt``."""
    if not preds:
        raise ValueError("no predictions to evaluate")

    report = _report(preds, policy)
    other = next(p for p in POLICIES if p != policy)
    alt = _report(preds, other)
    report["unparseable_policy"] = policy
    report["alt"] = {"policy": other, "macro": alt["macro"], "micro": alt["micro"]}
    return report


def format_report(report: dict) -> str:
    """Plain-text table mirroring the paper's Tables II-V (per-CWE + All (Avg))."""
    lines = [f"{'CWE':<12} {'n':>5} " + " ".join(f"{m:>9}" for m in METRIC_NAMES)]
    for cwe, row in report["per_cwe"].items():
        lines.append(
            f"{cwe:<12} {row['n']:>5} " + " ".join(f"{row[m]:>9.3f}" for m in METRIC_NAMES)
        )
    for name, row in (("All (macro)", report["macro"]), ("All (micro)", report["micro"])):
        lines.append(
            f"{name:<12} {report['n']:>5} " + " ".join(f"{row[m]:>9.3f}" for m in METRIC_NAMES)
        )
    for name, row in (("macro", report["macro"]), ("micro", report["micro"])):
        lines.append(
            f"  {name} secondary: "
            + "  ".join(f"{m}={row[m]:+.3f}" for m in SECONDARY_METRIC_NAMES)
        )
    policy = report.get("unparseable_policy", DEFAULT_POLICY)
    lines.append(
        f"unparseable: {report['n_unparseable']}/{report['n']}"
        f" ({report['parse_failure_rate']:.1%}, policy: {policy})"
    )
    alt = report.get("alt")
    if alt:
        lines.append(
            f"alt ({alt['policy']}): macro "
            + " ".join(f"{m}={alt['macro'][m]:.3f}" for m in METRIC_NAMES)
        )
    return "\n".join(lines)

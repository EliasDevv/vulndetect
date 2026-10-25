"""Significance tests between approaches: paired t-test, exact McNemar, Holm."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from vulndetect.common.predictions import Prediction

from .metrics import POLICIES

ALPHA = 0.05


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvalues)
    if m == 0:
        return []
    for p in pvalues:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value out of range: {p!r}")
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_ttest(f1s_a: list[float], f1s_b: list[float]) -> dict:
    """Two-tailed paired t-test of per-CWE F1s (a vs b) with the 95% CI of the mean difference."""
    if len(f1s_a) != len(f1s_b) or len(f1s_a) < 2:
        raise ValueError("need two equal-length lists with >= 2 pairs")
    diff = np.asarray(f1s_a, dtype=float) - np.asarray(f1s_b, dtype=float)
    n = len(diff)
    t, p = sps.ttest_rel(f1s_a, f1s_b)
    sem = diff.std(ddof=1) / np.sqrt(n)
    t_crit = sps.t.ppf(0.975, df=n - 1)
    return {
        "t": float(t),
        "df": n - 1,
        "p": float(p),
        "mean_diff": float(diff.mean()),
        "ci95": [float(diff.mean() - t_crit * sem), float(diff.mean() + t_crit * sem)],
    }


def _correct_by_id(preds: list[Prediction], policy: str) -> dict[str, bool | None]:
    """id -> correct?; None marks an unparseable answer excluded under "drop"."""
    out: dict[str, bool | None] = {}
    for p in preds:
        if p["id"] in out:
            raise ValueError(f"duplicate prediction id {p['id']!r}")
        if p["pred_label"] is None and policy == "drop":
            out[p["id"]] = None
        else:
            out[p["id"]] = p["pred_label"] == p["gold_label"]
    return out


def mcnemar(preds_a: list[Prediction], preds_b: list[Prediction], policy: str = "drop") -> dict:
    """Exact two-sided McNemar test over instances matched by id, under the given policy."""
    if policy not in POLICIES:
        raise ValueError(f"unknown unparseable policy {policy!r} — known: {POLICIES}")
    a, b_map = _correct_by_id(preds_a, policy), _correct_by_id(preds_b, policy)
    if set(a) != set(b_map):
        raise ValueError("prediction sets differ — both runs must cover the same instances")
    ids = [i for i in a if a[i] is not None and b_map[i] is not None]
    b = sum(1 for i in ids if a[i] and not b_map[i])
    c = sum(1 for i in ids if not a[i] and b_map[i])
    p = sps.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {
        "n": len(ids),
        "n_excluded": len(a) - len(ids),
        "policy": policy,
        "b": b,
        "c": c,
        "p": float(p),
    }

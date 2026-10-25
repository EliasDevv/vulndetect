"""Shared evaluation harness: metrics, significance tests and the experiment log."""

from __future__ import annotations

from .metrics import confusion, evaluate, format_report, scores
from .runlog import append_run, logging_enabled, summary_metrics
from .stats import ALPHA, holm, mcnemar, paired_ttest

__all__ = [
    "ALPHA",
    "append_run",
    "confusion",
    "evaluate",
    "format_report",
    "holm",
    "logging_enabled",
    "mcnemar",
    "paired_ttest",
    "scores",
    "summary_metrics",
]

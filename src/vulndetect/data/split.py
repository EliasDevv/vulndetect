"""Deterministic, leakage-free train/val/test splitting by commit."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def split_by_commit(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """Add a ``split`` column assigning whole commits to train/val/test."""
    rng = np.random.default_rng(seed)
    out = df.copy()

    assignments: dict[str, str] = {}
    for cwe, group in out.groupby("cwe", sort=True):
        commit_rows = group.groupby("commit_id").size()
        commit_ids = list(commit_rows.index)
        commit_ids = [commit_ids[i] for i in rng.permutation(len(commit_ids))]

        total = int(commit_rows.sum())
        targets = {s: ratios.get(s, 0.0) * total for s in SPLIT_NAMES}
        acc = {s: 0 for s in SPLIT_NAMES}

        for cid in commit_ids:
            n = int(commit_rows[cid])
            split = max(SPLIT_NAMES, key=lambda s: _deficit(targets[s], acc[s]))
            assignments[cid] = split
            acc[split] += n

    out["split"] = out["commit_id"].map(assignments)
    _log_summary(out)
    return out.reset_index(drop=True)


def _deficit(target: float, acc: int) -> float:
    """Relative shortfall of a split; higher means it needs rows more."""
    if target <= 0:
        return -math.inf
    return (target - acc) / target


def _log_summary(df: pd.DataFrame) -> None:
    for split in SPLIT_NAMES:
        sub = df[df["split"] == split]
        logger.info(
            "split %s: %d rows, %d commits, vuln=%d safe=%d",
            split,
            len(sub),
            sub["commit_id"].nunique(),
            int((sub["vul"] == 1).sum()),
            int((sub["vul"] == 0).sum()),
        )

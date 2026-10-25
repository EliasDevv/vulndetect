"""Per-CWE class balancing by downsampling the majority label."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .schema import TARGET_CWES

logger = logging.getLogger(__name__)


def balance_splits(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_cwes: tuple[str, ...] = TARGET_CWES,
    target_sizes: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Balance Vulnerable vs Not Vulnerable within each split and CWE, optionally capped."""
    parts: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        cap = target_sizes.get(split) if target_sizes else None
        parts.append(_balance_one_split(sub, rng, target_cwes, cap, split))

    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _balance_one_split(
    sub: pd.DataFrame,
    rng: np.random.Generator,
    target_cwes: tuple[str, ...],
    cap: int | None,
    split_name: str,
) -> pd.DataFrame:
    """Balance one split per CWE, optionally capped to ``cap`` total rows."""
    keep: dict[str, int] = {}
    for cwe in target_cwes:
        c = sub[sub["cwe"] == cwe]
        keep[cwe] = int(min((c["vul"] == 1).sum(), (c["vul"] == 0).sum()))

    total = 2 * sum(keep.values())
    if cap is not None and total > cap and total > 0:
        factor = cap / total
        keep = {cwe: int(n * factor) for cwe, n in keep.items()}

    parts: list[pd.DataFrame] = []
    for cwe in target_cwes:
        n = keep[cwe]
        if n <= 0:
            continue
        c = sub[sub["cwe"] == cwe]
        parts.append(_sample(c[c["vul"] == 1], n, rng))
        parts.append(_sample(c[c["vul"] == 0], n, rng))

    if not parts:
        logger.warning("balance_splits: %s ended up empty after balancing", split_name)
        return sub.iloc[0:0].copy()

    out = pd.concat(parts, ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    logger.info(
        "balance_splits: %s -> %d rows (%d vuln / %d safe), per-CWE balanced",
        split_name,
        len(out),
        int((out["vul"] == 1).sum()),
        int((out["vul"] == 0).sum()),
    )
    return out


def _sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Randomly pick ``n`` rows from ``frame`` using ``rng`` (no replacement)."""
    if n >= len(frame):
        return frame.copy()
    idx = rng.choice(len(frame), size=n, replace=False)
    return frame.iloc[idx].copy()

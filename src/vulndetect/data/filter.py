"""Filtering and cleaning of the raw Big-Vul frame."""

from __future__ import annotations

import logging

import pandas as pd

from .schema import TARGET_CWES

logger = logging.getLogger(__name__)


def filter_cwe(df: pd.DataFrame, target_cwes: tuple[str, ...] = TARGET_CWES) -> pd.DataFrame:
    """Keep rows whose ``CWE ID`` is exactly one of the target CWEs (stored as ``cwe``)."""
    targets = set(target_cwes)
    cwe = df["CWE ID"].astype(str).str.strip()
    mask = cwe.isin(targets)

    out = df[mask].copy()
    out["cwe"] = cwe[mask]

    logger.info(
        "filter_cwe: kept %d/%d rows (exact match to %s)",
        len(out),
        len(df),
        sorted(targets),
    )
    return out


def filter_lang(df: pd.DataFrame, langs: list[str]) -> pd.DataFrame:
    """Keep rows whose ``lang`` is in ``langs`` (case-insensitive)."""
    wanted = {l.strip().lower() for l in langs}
    mask = df["lang"].astype(str).str.strip().str.lower().isin(wanted)

    out = df[mask].copy()
    logger.info("filter_lang: kept %d/%d rows (langs=%s)", len(out), len(df), sorted(wanted))
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with empty ``func_before`` and duplicate functions within the same commit."""
    before = len(df)

    out = df.copy()
    out["func_before"] = out["func_before"].astype(str)
    out = out[out["func_before"].str.strip() != ""]
    dropped_empty = before - len(out)

    n_pre = len(out)
    out = out.drop_duplicates(subset=["commit_id", "func_before"], keep="first")
    dropped_dup = n_pre - len(out)

    logger.info(
        "clean: dropped %d empty + %d duplicate rows -> %d remain",
        dropped_empty,
        dropped_dup,
        len(out),
    )
    return out.reset_index(drop=True)

"""Load the raw Big-Vul CSV into a normalized DataFrame, eagerly or in chunks."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .schema import NA, RAW_COLUMNS

logger = logging.getLogger(__name__)

_WANTED = list(RAW_COLUMNS) + ["CVE ID"]
_TEXT_COLS = [
    "CWE ID",
    "codeLink",
    "commit_id",
    "commit_message",
    "func_after",
    "func_before",
    "lang",
    "project",
    "CVE ID",
]

DEFAULT_CHUNKSIZE = 50_000


def _usecols(path: Path) -> list[str]:
    """Validate the header and return the intersection of wanted/present columns."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8", encoding_errors="replace")
    missing = [c for c in RAW_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError(
            f"Big-Vul CSV missing expected columns {missing}. "
            f"Found columns: {sorted(header.columns)}"
        )
    return [c for c in _WANTED if c in header.columns]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Strip text columns, fill missing CVE ID with 'N/A', coerce vul to 0/1."""
    for col in _TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    if "CVE ID" not in df.columns:
        df["CVE ID"] = NA
    df["CVE ID"] = df["CVE ID"].replace("", NA)
    df["vul"] = pd.to_numeric(df["vul"], errors="coerce").fillna(0).astype(int)
    return df


def iter_raw_chunks(path: str | Path, chunksize: int = DEFAULT_CHUNKSIZE):
    """Yield normalized Big-Vul chunks with only the needed columns; invalid bytes are replaced."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Big-Vul CSV not found at {path}. Run scripts/download_data.sh first."
        )
    usecols = _usecols(path)
    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    for chunk in reader:
        yield _normalize(chunk)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Eagerly read the whole CSV (small inputs and tests only)."""
    df = pd.concat(iter_raw_chunks(path), ignore_index=True)
    logger.info("Loaded %d raw Big-Vul rows from %s", len(df), path)
    return df

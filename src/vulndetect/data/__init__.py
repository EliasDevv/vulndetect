"""Canonical dataset access for all approaches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import (
    LABEL_SAFE,
    LABEL_VULN,
    RECORD_FIELDS,
    TARGET_CWES,
    Record,
)

DEFAULT_DATASET_DIR = Path("data/processed/bigvul")
VALID_SPLITS = ("train", "val", "test")

__all__ = [
    "Record",
    "LABEL_VULN",
    "LABEL_SAFE",
    "TARGET_CWES",
    "RECORD_FIELDS",
    "load_split",
    "load_split_df",
    "DEFAULT_DATASET_DIR",
]


def _split_path(split: str, dataset_dir: str | Path) -> Path:
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {VALID_SPLITS}")
    path = Path(dataset_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the dataset first: "
            f"python -m vulndetect.data.build --config configs/data.yaml"
        )
    return path


def load_split(split: str, dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> list[Record]:
    """Return one split as a list of canonical records."""
    path = _split_path(split, dataset_dir)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_split_df(split: str, dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    """Return one split as a DataFrame with the canonical columns."""
    return pd.DataFrame(load_split(split, dataset_dir), columns=list(RECORD_FIELDS))

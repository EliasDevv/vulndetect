"""Canonical prediction format shared by every approach, plus JSONL read/write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from vulndetect.data.schema import LABELS, Record

PREDICTION_FIELDS = (
    "id",
    "cwe",
    "gold_label",
    "pred_label",
    "raw_output",
)


class Prediction(TypedDict):
    """One scored instance, written as a single JSONL line."""

    id: str
    cwe: str
    gold_label: str
    pred_label: str | None
    raw_output: str


def make_prediction(record: Record, raw_output: str, pred_label: str | None) -> Prediction:
    """Build one canonical prediction from a dataset record and model output."""
    if pred_label is not None and pred_label not in LABELS:
        raise ValueError(f"pred_label {pred_label!r} not in {LABELS} (or None)")
    return {
        "id": record["id"],
        "cwe": record["cwe"],
        "gold_label": record["label"],
        "pred_label": pred_label,
        "raw_output": raw_output,
    }


def write_predictions(preds: list[Prediction], path: str | Path) -> None:
    """Write predictions as JSON lines with the canonical fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps({k: p[k] for k in PREDICTION_FIELDS}, ensure_ascii=False) + "\n")


def read_predictions(path: str | Path) -> list[Prediction]:
    """Read a predictions.jsonl file into a list of predictions."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the approach first.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_run(
    out_dir: str | Path,
    preds: list[Prediction],
    manifest: dict,
) -> tuple[Path, Path]:
    """Write one run's predictions.jsonl and manifest.json and return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    write_predictions(preds, pred_path)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return pred_path, manifest_path

"""Append one row per experiment run to results/experiments.csv."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

DEFAULT_CSV = Path("results/experiments.csv")
HEADER = ["timestamp", "approach", "model", "seed", "params", "metrics", "output"]


def append_run(
    approach: str,
    model: str,
    seed: int,
    params: dict,
    metrics: dict,
    output: str | Path,
    csv_path: str | Path = DEFAULT_CSV,
) -> None:
    """Append one run as a row to the experiment CSV, writing the header if new."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "approach": approach,
        "model": model,
        "seed": seed,
        "params": json.dumps(params, ensure_ascii=False),
        "metrics": json.dumps(metrics, ensure_ascii=False),
        "output": str(output),
    }
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        if not exists:
            w.writeheader()
        w.writerow(row)


def logging_enabled(cfg: dict) -> bool:
    """Whether a run with this config may log to experiments.csv."""
    return bool(cfg.get("output", {}).get("log_experiments", True))


def summary_metrics(report: dict) -> dict:
    """The compact metric subset that goes into the CSV row."""
    summary = {
        "macro_f1": round(report["macro"]["f1"], 4),
        "macro_acc": round(report["macro"]["accuracy"], 4),
        "micro_f1": round(report["micro"]["f1"], 4),
        "n": report["n"],
        "parse_failure_rate": round(report["parse_failure_rate"], 4),
    }
    if "unparseable_policy" in report:
        summary["policy"] = report["unparseable_policy"]
    if "alt" in report:
        summary["macro_f1_alt"] = round(report["alt"]["macro"]["f1"], 4)
    return summary

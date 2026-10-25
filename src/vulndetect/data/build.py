"""Build the canonical Big-Vul dataset and write splits, manifest and stats."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .balance import balance_splits
from .filter import clean, filter_cwe, filter_lang
from .load import DEFAULT_CHUNKSIZE, iter_raw_chunks
from .reference import (
    apply_reference,
    build_reference,
    load_reference,
    sha256_of,
    write_reference,
)
from .schema import (
    NA,
    RECORD_FIELDS,
    TARGET_CWES,
    label_from_vul,
    make_id,
)
from .split import SPLIT_NAMES, split_by_commit

logger = logging.getLogger(__name__)


def prepare_pool(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply filter_cwe, filter_lang and clean to get the candidate pool."""
    target_cwes = tuple(cfg.get("target_cwes", TARGET_CWES))
    df = filter_cwe(df, target_cwes)
    df = filter_lang(df, cfg.get("lang_filter", ["C", "C++"]))
    return clean(df)


def assign_and_balance(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Split commits by seeded permutation and balance per split; also return all commit splits."""
    seed = int(cfg["seed"])
    target_cwes = tuple(cfg.get("target_cwes", TARGET_CWES))
    rng = np.random.default_rng(seed)

    split_cfg = cfg.get("split", {})
    assigned = split_by_commit(
        df,
        ratios=split_cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1}),
        seed=seed,
    )
    commit_split = {
        str(c): s for c, s in zip(assigned["commit_id"], assigned["split"], strict=True)
    }
    balanced = balance_splits(
        assigned, rng, target_cwes, target_sizes=split_cfg.get("target_sizes")
    )
    return balanced, commit_split


def build_dataset(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run the full in-memory pipeline and return the final labeled frame."""
    balanced, _ = assign_and_balance(prepare_pool(df, cfg), cfg)
    return balanced


def build_dataset_with_reference(
    df: pd.DataFrame, cfg: dict, reference: dict
) -> tuple[pd.DataFrame, dict]:
    """Build the dataset from a frozen reference split, appending new commits."""
    pool = prepare_pool(df, cfg)
    return apply_reference(pool, reference, cfg.get("split", {}).get("target_sizes"))


def load_and_prefilter(raw_path: str | Path, cfg: dict) -> pd.DataFrame:
    """Stream the large CSV in chunks, keeping only target-CWE rows in the wanted languages."""
    target_cwes = tuple(cfg.get("target_cwes", TARGET_CWES))
    langs = cfg.get("lang_filter", ["C", "C++"])
    chunksize = int(cfg.get("source", {}).get("chunksize", DEFAULT_CHUNKSIZE))

    parts: list[pd.DataFrame] = []
    total = 0
    for i, chunk in enumerate(iter_raw_chunks(raw_path, chunksize)):
        total += len(chunk)
        chunk = filter_cwe(chunk, target_cwes)
        if len(chunk):
            chunk = filter_lang(chunk, langs)
        if len(chunk):
            parts.append(chunk)
        if (i + 1) % 5 == 0:
            kept = sum(len(p) for p in parts)
            logger.info("prefilter: read %d rows, kept %d after CWE+lang", total, kept)

    if not parts:
        raise ValueError("No rows survived CWE/lang pre-filtering — check the CSV.")
    df = pd.concat(parts, ignore_index=True)
    logger.info("prefilter done: read %d rows, kept %d", total, len(df))
    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a built frame into canonical records (list of dicts)."""
    records: list[dict] = []
    for d in df.to_dict("records"):
        commit_id = str(d["commit_id"])
        func_before = str(d["func_before"])
        rec = {
            "id": make_id(commit_id, func_before),
            "code": func_before,
            "func_before": func_before,
            "func_after": str(d["func_after"]),
            "commit_message": str(d["commit_message"]),
            "cve_id": str(d.get("CVE ID", NA)) or NA,
            "cwe": str(d["cwe"]),
            "project": str(d["project"]),
            "lang": str(d["lang"]),
            "label": label_from_vul(d["vul"]),
            "vul": int(d["vul"]),
            "commit_id": commit_id,
            "split": str(d["split"]),
        }
        records.append({k: rec[k] for k in RECORD_FIELDS})
    return records


def compute_stats(df: pd.DataFrame, target_cwes: tuple[str, ...]) -> dict:
    """Compute per-split, per-CWE and per-label counts."""
    stats: dict = {"n_total": int(len(df)), "per_split": {}}
    for split in SPLIT_NAMES:
        sub = df[df["split"] == split]
        per_cwe = {}
        for cwe in target_cwes:
            c = sub[sub["cwe"] == cwe]
            per_cwe[cwe] = {
                "vulnerable": int((c["vul"] == 1).sum()),
                "not_vulnerable": int((c["vul"] == 0).sum()),
            }
        stats["per_split"][split] = {
            "n": int(len(sub)),
            "n_commits": int(sub["commit_id"].nunique()),
            "vulnerable": int((sub["vul"] == 1).sum()),
            "not_vulnerable": int((sub["vul"] == 0).sum()),
            "per_cwe": per_cwe,
        }
    return stats


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records to a JSON-lines file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_experiment_row(csv_path: Path, cfg: dict, stats: dict, out_dir: Path) -> None:
    """Append a one-line build record to results/experiments.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "approach", "model", "seed", "params", "metrics", "output"]
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "data_build",
        "model": NA,
        "seed": cfg["seed"],
        "params": json.dumps(
            {"target_cwes": cfg.get("target_cwes"), "split": cfg.get("split")},
            ensure_ascii=False,
        ),
        "metrics": json.dumps(
            {s: stats["per_split"][s]["n"] for s in SPLIT_NAMES}, ensure_ascii=False
        ),
        "output": str(out_dir),
    }
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def print_summary(stats: dict, target_cwes: tuple[str, ...]) -> None:
    """Print per-split row, commit and label counts plus per-CWE counts."""
    print("\n=== Big-Vul build summary ===")
    print(f"{'split':<6} {'rows':>6} {'commits':>8} {'vuln':>6} {'safe':>6}")
    for split in SPLIT_NAMES:
        s = stats["per_split"][split]
        print(
            f"{split:<6} {s['n']:>6} {s['n_commits']:>8} {s['vulnerable']:>6} {s['not_vulnerable']:>6}"
        )
    print("\nper-CWE (vuln/safe) by split:")
    for split in SPLIT_NAMES:
        cells = " ".join(
            f"{cwe}:{stats['per_split'][split]['per_cwe'][cwe]['vulnerable']}/"
            f"{stats['per_split'][split]['per_cwe'][cwe]['not_vulnerable']}"
            for cwe in target_cwes
        )
        print(f"  {split:<5} {cells}")
    print()


def run(config_path: str | Path, write_reference_path: str | Path | None = None) -> dict:
    """Load the config, build the dataset, write all artifacts and return the stats dict."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    target_cwes = tuple(cfg.get("target_cwes", TARGET_CWES))
    reference_path = cfg.get("split", {}).get("reference")
    if reference_path and write_reference_path:
        raise ValueError("--write-reference needs a config WITHOUT split.reference (legacy path)")

    raw_path = Path(cfg["source"]["raw_path"])
    df_raw = load_and_prefilter(raw_path, cfg)
    if reference_path:
        reference = load_reference(reference_path)
        df, ref_stats = build_dataset_with_reference(df_raw, cfg, reference)
        split_reference = {
            "path": str(reference_path),
            "sha256": sha256_of(reference_path),
            "origin": reference.get("origin"),
            **ref_stats,
        }
        commit_split = None
    else:
        df, commit_split = assign_and_balance(prepare_pool(df_raw, cfg), cfg)
        split_reference = None

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    records = to_records(df)
    for split in SPLIT_NAMES:
        split_recs = [r for r in records if r["split"] == split]
        write_jsonl(split_recs, out_dir / f"{split}.jsonl")
    split_hashes = {split: _sha256(out_dir / f"{split}.jsonl") for split in SPLIT_NAMES}

    stats = compute_stats(df, target_cwes)

    manifest = {
        "seed": cfg["seed"],
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "source_path": str(raw_path),
        "source_sha256": _sha256(raw_path),
        "target_cwes": list(target_cwes),
        "split_hashes": split_hashes,
        "split_reference": split_reference,
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if write_reference_path:
        origin = {
            "built": dt.date.today().isoformat(),
            "algorithm": "split_by_commit (seeded permutation + deficit rule, D04) + "
            "balance_splits (rng.choice per split/CWE, floor-scaled caps)",
            "seed": cfg["seed"],
            "config_sha256": manifest["config_sha256"],
            "source_sha256": manifest["source_sha256"],
            "lang_filter": cfg.get("lang_filter"),
            "target_sizes": cfg.get("split", {}).get("target_sizes"),
            "split_hashes": split_hashes,
        }
        path = write_reference(build_reference(df, commit_split, origin), write_reference_path)
        logger.info("froze this build as split reference %s", path)

    stats_path = Path(cfg["output"]["stats_path"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps({"seed": cfg["seed"], **stats}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    append_experiment_row(Path("results/experiments.csv"), cfg, stats, out_dir)

    print_summary(stats, target_cwes)
    logger.info("Wrote splits to %s and stats to %s", out_dir, stats_path)
    return stats


def main() -> None:
    """Parse CLI arguments and build the Big-Vul dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build the canonical Big-Vul dataset.")
    ap.add_argument("--config", default="configs/data.yaml", help="path to data build YAML")
    ap.add_argument(
        "--write-reference",
        help="freeze this (legacy-path) build into a split reference file",
    )
    args = ap.parse_args()
    run(args.config, write_reference_path=args.write_reference)


if __name__ == "__main__":
    main()

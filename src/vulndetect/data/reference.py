"""Reference-anchored splits: reproduce a frozen split and append only rows of new commits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .schema import make_id
from .split import SPLIT_NAMES

REFERENCE_VERSION = 1
_BUCKETS = 10


def hash_split(commit_id: str) -> str:
    """Split of a commit unknown to the reference: a pure function of its id."""
    bucket = int(hashlib.sha1(commit_id.encode("utf-8")).hexdigest(), 16) % _BUCKETS
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def hash_key(instance_id: str) -> str:
    """Deterministic order key for top-up candidates (no RNG involved)."""
    return hashlib.sha1(instance_id.encode("utf-8")).hexdigest()


def build_reference(balanced: pd.DataFrame, commit_split: dict[str, str], origin: dict) -> dict:
    """Freeze a legacy build: every pool commit's split + ordered ids per split."""
    selected = {}
    for split in SPLIT_NAMES:
        sub = balanced[balanced["split"] == split]
        selected[split] = [
            make_id(str(c), str(f))
            for c, f in zip(sub["commit_id"], sub["func_before"], strict=True)
        ]
    return {
        "version": REFERENCE_VERSION,
        "origin": origin,
        "commit_split": dict(sorted((str(k), v) for k, v in commit_split.items())),
        "selected": selected,
    }


def write_reference(reference: dict, path: str | Path) -> Path:
    """Write a split reference as JSON and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_reference(path: str | Path) -> dict:
    """Load a split reference file and check its version."""
    ref = json.loads(Path(path).read_text(encoding="utf-8"))
    if ref.get("version") != REFERENCE_VERSION:
        raise ValueError(f"{path}: unsupported reference version {ref.get('version')!r}")
    return ref


def sha256_of(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_reference(
    pool: pd.DataFrame, reference: dict, target_sizes: dict[str, int] | None
) -> tuple[pd.DataFrame, dict]:
    """Assign splits and select rows against a reference; return the balanced frame and stats."""
    known = reference["commit_split"]
    pool = pool.copy()
    pool["id"] = [
        make_id(str(c), str(f)) for c, f in zip(pool["commit_id"], pool["func_before"], strict=True)
    ]
    if pool["id"].duplicated().any():
        raise ValueError("candidate pool has duplicate instance ids — run clean() first")
    commit_ids = pool["commit_id"].astype(str)
    pool["split"] = [known.get(c) or hash_split(c) for c in commit_ids]
    new_commits = set(commit_ids) - set(known)
    by_id = pool.set_index("id", drop=False)

    parts: list[pd.DataFrame] = []
    stats: dict = {"n_new_commits": len(new_commits), "per_split": {}}
    for split in SPLIT_NAMES:
        ids = reference["selected"][split]
        missing = [i for i in ids if i not in by_id.index]
        if missing:
            raise ValueError(
                f"{len(missing)} reference instances of split {split!r} are missing from the "
                f"pool (first: {missing[0]!r}) — the reference is not reproducible from this source"
            )
        kept = by_id.loc[ids]
        if (kept["split"] != split).any():
            raise ValueError(f"reference instances of split {split!r} map to another split")

        cap = (target_sizes or {}).get(split)
        room_pairs = None if cap is None else max(0, (int(cap) - len(kept)) // 2)
        cand = pool[(pool["split"] == split) & commit_ids.isin(new_commits)]
        cand = cand.assign(_h=cand["id"].map(hash_key)).sort_values("_h")
        safe_pool = cand[cand["vul"] == 0]
        used: set[str] = set()
        added: list[pd.Series] = []
        for _, row in cand[cand["vul"] == 1].iterrows():
            if room_pairs is not None and len(added) // 2 >= room_pairs:
                break
            partner = safe_pool[(safe_pool["cwe"] == row["cwe"]) & ~safe_pool["id"].isin(used)]
            if partner.empty:
                continue
            used.add(partner.iloc[0]["id"])
            added += [row, partner.iloc[0]]
        appended = pd.DataFrame(added, columns=cand.columns) if added else cand.iloc[0:0]
        pairs_per_cwe = {
            cwe: int(n) for cwe, n in appended[appended["vul"] == 1]["cwe"].value_counts().items()
        }
        stats["per_split"][split] = {
            "n_reference": len(kept),
            "n_new_commits": int(cand["commit_id"].nunique()),
            "n_new_rows_available": len(cand),
            "n_added": len(appended),
            "added_pairs_per_cwe": pairs_per_cwe,
            "cap": cap,
        }
        parts.append(pd.concat([kept, appended.drop(columns="_h")], ignore_index=True))

    out = pd.concat(parts, ignore_index=True).drop(columns="id")
    return out, stats

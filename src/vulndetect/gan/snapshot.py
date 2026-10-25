"""Round-milestone snapshots of the two strategy memories S_G and S_D."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

from vulndetect.gan.prompts import PROMPT_VERSIONS
from vulndetect.gan.strategies import StrategyMemory

SNAPSHOT_FORMAT_VERSION = 1

FINAL_SNAPSHOT = "final"

PHASES = ("init", "A", "B")


def git_commit() -> str | None:
    """Current HEAD of the repository this module lives in (None if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def snapshot_name(round_no: int | None) -> str:
    """``round0050`` for a Phase-A milestone, ``final`` after Phase B."""
    return FINAL_SNAPSHOT if round_no is None else f"round{round_no:04d}"


def write_snapshot(
    snapshot_dir: str | Path,
    *,
    round_no: int | None,
    phase: str,
    s_g: StrategyMemory,
    s_d: StrategyMemory,
    seed: int,
    config_path: str | Path,
    config_sha256: str,
    probe: dict | None = None,
    stats: dict | None = None,
    prompt_versions: dict | None = None,
    detecting_party: dict | None = None,
) -> Path:
    """Write one snapshot as ``<name>.json`` and return its path."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; allowed: {PHASES}")
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    name = snapshot_name(round_no)
    payload = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "name": name,
        "round": round_no,
        "phase": phase,
        "seed": seed,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "prompt_versions": dict(PROMPT_VERSIONS if prompt_versions is None else prompt_versions),
        "detecting_party": detecting_party,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "s_g": s_g.to_dict(),
        "s_d": s_d.to_dict(),
        "s_g_stats": s_g.stats(),
        "s_d_stats": s_d.stats(),
        "drift_probe": probe,
        "loop_stats": stats,
    }
    path = snapshot_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_snapshot(path: str | Path) -> dict:
    """Load a snapshot JSON file, raising if it does not exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the GAN loop first (make gan)")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_snapshot_path(snapshot_dir: str | Path, ref: str) -> Path:
    """Accept ``final``, ``round0050``, ``50``, or a path to a snapshot file."""
    candidate = Path(ref)
    if candidate.suffix == ".json":
        return candidate
    name = snapshot_name(int(ref)) if ref.isdigit() else ref
    return Path(snapshot_dir) / f"{name}.json"


def list_snapshots(snapshot_dir: str | Path) -> list[Path]:
    """List all snapshots of a loop run, Phase-A milestones in round order, final last."""
    snapshot_dir = Path(snapshot_dir)
    rounds = sorted(snapshot_dir.glob("round*.json"))
    final = snapshot_dir / f"{FINAL_SNAPSHOT}.json"
    return rounds + ([final] if final.exists() else [])


def detector_memory(snapshot: dict) -> StrategyMemory:
    """Rebuild the Detector memory S_D from a snapshot."""
    return StrategyMemory.from_dict(snapshot["s_d"])

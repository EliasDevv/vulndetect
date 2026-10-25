"""Experiment config loading, config hashing and seed fixing."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config and check required keys."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("experiment", "seed"):
        if key not in cfg:
            raise ValueError(f"config {path} is missing required key {key!r}")
    return cfg


def config_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a config file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    """Fix all relevant RNGs (python, numpy, torch if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

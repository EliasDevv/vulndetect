"""GAN runner: evaluate one snapshot's detector memory S_D on the configured split."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vulndetect.common import (
    PROMPT_VERSION,
    config_sha256,
    load_config,
    make_prediction,
    parse_label,
    set_seed,
    write_run,
)
from vulndetect.common.model import CHAT_DATE_STRING
from vulndetect.data import DEFAULT_DATASET_DIR, load_split
from vulndetect.data.schema import LABEL_VULN
from vulndetect.eval import append_run, evaluate, format_report, logging_enabled, summary_metrics
from vulndetect.eval.metrics import DEFAULT_POLICY
from vulndetect.gan.prompts import PROMPT_VERSIONS, render_detector_prompt
from vulndetect.gan.snapshot import (
    detector_memory,
    list_snapshots,
    read_snapshot,
    resolve_snapshot_path,
)

logger = logging.getLogger(__name__)

APPROACH = "gan"


def snapshot_dir_for(cfg: dict, seed: int) -> Path:
    """Return the snapshot directory for a seed (``output.snapshots_from`` or ``loop_dir``)."""
    base = cfg["output"].get("snapshots_from") or cfg["output"]["loop_dir"]
    return Path(base) / f"seed{seed}" / "snapshots"


def eval_decoding(cfg: dict) -> dict:
    """Return the decoding settings for snapshot evaluation (``eval`` over ``model`` block)."""
    model_cfg, ev = cfg["model"], cfg.get("eval") or {}
    return {
        "temperature": float(ev.get("temperature", model_cfg.get("temperature", 0.0))),
        "top_p": ev.get("top_p", model_cfg.get("top_p")),
        "max_new_tokens": ev.get("max_new_tokens", model_cfg.get("max_new_tokens")),
    }


def run(
    config_path: str | Path,
    snapshot_ref: str = "final",
    limit: int | None = None,
    model=None,
) -> dict:
    """Evaluate one snapshot, write its run artifacts and return the metrics report."""
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    snapshot_path = resolve_snapshot_path(snapshot_dir_for(cfg, seed), snapshot_ref)
    snapshot = read_snapshot(snapshot_path)
    s_d = detector_memory(snapshot)
    strategies = s_d.render()
    logger.info(
        "snapshot %s: round=%s, %d strategy entries", snapshot["name"], snapshot["round"], len(s_d)
    )

    ds_cfg = cfg.get("dataset", {})
    records = load_split(ds_cfg.get("split", "test"), ds_cfg.get("dir", DEFAULT_DATASET_DIR))
    if limit is not None:
        records = records[:limit]
    logger.info("loaded %d records", len(records))

    if model is None:
        from vulndetect.common.model import load_model  # lazy import: pulls in torch

        model = load_model(cfg["model"])

    decoding = eval_decoding(cfg)
    policy = cfg.get("eval", {}).get("unparseable", DEFAULT_POLICY)
    prompts = [render_detector_prompt(rec, strategies) for rec in records]
    outputs = model.generate(
        prompts,
        temperature=decoding["temperature"],
        top_p=decoding["top_p"],
        max_new_tokens=decoding["max_new_tokens"],
    )
    preds = [
        make_prediction(rec, out, parse_label(out))
        for rec, out in zip(records, outputs, strict=True)
    ]

    out_dir = (
        Path(cfg["output"].get("predictions_dir", f"results/predictions/{APPROACH}"))
        / f"seed{seed}"
        / snapshot["name"]
    )
    report = evaluate(preds, policy=policy)
    report["parser_version"] = "d09-v2"
    n = len(preds) or 1
    manifest = {
        "approach": APPROACH,
        "model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "prompt_version": PROMPT_VERSION,
        "gan_prompt_versions": dict(PROMPT_VERSIONS),
        "dataset": {**ds_cfg, "n": len(records), "limit": limit},
        "unparseable_policy": policy,
        "decoding": {
            **decoding,
            "batch_size": cfg["model"].get("batch_size"),
            "chat_date_string": cfg["model"].get("chat_date_string", CHAT_DATE_STRING),
            "eval_override": "eval" in cfg,
            "loop_detector_temperature": cfg["model"].get("temperature", 0.0),
        },
        "gan": {
            "snapshot": snapshot["name"],
            "snapshot_path": str(snapshot_path),
            "snapshots_from": cfg["output"].get("snapshots_from"),
            "round": snapshot["round"],
            "phase": snapshot["phase"],
            "loop_git_commit": snapshot.get("git_commit"),
            "loop_config_sha256": snapshot.get("config_sha256"),
            "memory_policy": s_d.policy,
            "n_strategies": len(s_d),
            "strategies": list(s_d.entries),
            "drift_probe": snapshot.get("drift_probe"),
            "prompt_equals_baseline": not strategies,
            "vulnerable_prediction_share": round(
                sum(p["pred_label"] == LABEL_VULN for p in preds) / n, 4
            ),
        },
        "logged": limit is None and logging_enabled(cfg),
    }
    write_run(out_dir, preds, manifest)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if manifest["logged"]:
        append_run(
            approach=APPROACH,
            model=cfg["model"]["name"],
            seed=seed,
            params={
                "prompt_version": PROMPT_VERSION,
                "detector_block_version": PROMPT_VERSIONS["detector_block"],
                "snapshot": snapshot["name"],
                "round": snapshot["round"],
                "n_strategies": len(s_d),
                "memory_policy": s_d.policy,
                "unparseable_policy": policy,
                **manifest["decoding"],
            },
            metrics=summary_metrics(report),
            output=str(out_dir),
        )

    print(format_report(report))
    logger.info("wrote run artifacts to %s", out_dir)
    return report


def run_all_snapshots(
    config_path: str | Path, limit: int | None = None, model=None
) -> dict[str, dict]:
    """Evaluate every snapshot of the run (the round-count ablation)."""
    cfg = load_config(config_path)
    snapshots = list_snapshots(snapshot_dir_for(cfg, int(cfg["seed"])))
    if not snapshots:
        raise FileNotFoundError(
            f"no snapshots in {snapshot_dir_for(cfg, int(cfg['seed']))} — run the loop first"
        )
    reports: dict[str, dict] = {}
    for path in snapshots:
        logger.info("evaluating snapshot %s", path.stem)
        reports[path.stem] = run(config_path, path.stem, limit=limit, model=model)
    return reports

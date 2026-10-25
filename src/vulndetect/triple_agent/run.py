"""Triple-Agent runner: evaluate one snapshot's S_D with the Dual-Agent party (RQ3)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vulndetect.common import (
    PROMPT_VERSION,
    config_sha256,
    load_config,
    make_prediction,
    set_seed,
    write_run,
)
from vulndetect.common.model import CHAT_DATE_STRING
from vulndetect.data import DEFAULT_DATASET_DIR, load_split
from vulndetect.data.schema import LABEL_VULN
from vulndetect.dual_agent.prompt import AUTHORS_VALIDATOR_VERSION
from vulndetect.eval import append_run, evaluate, format_report, logging_enabled, summary_metrics
from vulndetect.eval.metrics import DEFAULT_POLICY
from vulndetect.gan.loop import DecodingOverride, MeteredModel, summarize_prompt_lengths
from vulndetect.gan.prompts import PROMPT_VERSIONS as GAN_PROMPT_VERSIONS
from vulndetect.gan.run import eval_decoding, snapshot_dir_for
from vulndetect.gan.snapshot import (
    detector_memory,
    list_snapshots,
    read_snapshot,
    resolve_snapshot_path,
)
from vulndetect.triple_agent.party import party_factory_from_config
from vulndetect.triple_agent.prompts import PROMPT_VERSIONS

logger = logging.getLogger(__name__)

APPROACH = "triple_agent"
PARSER_VERSION = "d09-v2"


def _lengths_by_pass(meter: MeteredModel, n: int) -> dict:
    """Split the meter's prompt-length entries into detector and validator passes."""
    entries = meter.drain("eval")
    meter.history = [("detector", e) for e in entries[:n]] + [("validator", e) for e in entries[n:]]
    summary = summarize_prompt_lengths([meter])
    return {"context_window": summary["context_window"], "by_pass": summary["by_role"]}


def run(
    config_path: str | Path,
    snapshot_ref: str = "final",
    limit: int | None = None,
    model=None,
) -> dict:
    """Evaluate one snapshot with the two-stage party and return the metrics report."""
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)
    factory = party_factory_from_config(cfg)

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
        from vulndetect.common.model import load_model

        model = load_model(cfg["model"])

    decoding = eval_decoding(cfg)
    policy = cfg.get("eval", {}).get("unparseable", DEFAULT_POLICY)
    eval_model = DecodingOverride(
        model,
        temperature=decoding["temperature"],
        top_p=decoding["top_p"],
        max_new_tokens=decoding["max_new_tokens"],
    )
    meter = MeteredModel(eval_model, counter=model)
    party = factory(meter)
    detections = party.detect_batch(records, strategies)
    lengths = _lengths_by_pass(meter, len(records))
    logger.info("detector + validator passes done (%d records)", len(records))

    preds = [
        make_prediction(rec, d.trace["validator_raw"], d.label)
        for rec, d in zip(records, detections, strict=True)
    ]
    stage1_preds = [
        make_prediction(rec, d.trace["detector_raw"], d.trace["detector_label"])
        for rec, d in zip(records, detections, strict=True)
    ]
    report = evaluate(preds, policy=policy)
    report["parser_version"] = PARSER_VERSION
    stats = party.stats()
    report["stage1"] = {
        **evaluate(stage1_preds, policy=policy),
        "parser_version": PARSER_VERSION,
        "agreement": stats,
    }

    out_dir = (
        Path(cfg["output"].get("predictions_dir", f"results/predictions/{APPROACH}"))
        / f"seed{seed}"
        / snapshot["name"]
    )
    n = len(preds) or 1
    manifest = {
        "approach": APPROACH,
        "model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "prompt_version": PROMPT_VERSION,
        "validator_prompt_version": AUTHORS_VALIDATOR_VERSION,
        "gan_prompt_versions": {**GAN_PROMPT_VERSIONS, **PROMPT_VERSIONS},
        "parser_version": PARSER_VERSION,
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
            "snapshot_detecting_party": snapshot.get("detecting_party"),
            "memory_policy": s_d.policy,
            "n_strategies": len(s_d),
            "strategies": list(s_d.entries),
            "drift_probe": snapshot.get("drift_probe"),
            "prompt_equals_dual_agent": not strategies,
            "vulnerable_prediction_share": round(
                sum(p["pred_label"] == LABEL_VULN for p in preds) / n, 4
            ),
        },
        "triple_agent": {
            **party.describe(),
            "stats": stats,
            "stage1_vulnerable_prediction_share": round(
                sum(p["pred_label"] == LABEL_VULN for p in stage1_preds) / n, 4
            ),
        },
        "prompt_lengths": lengths,
        "logged": limit is None and logging_enabled(cfg),
    }
    write_run(out_dir, preds, manifest)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    agent_lines = [
        json.dumps(
            {
                "id": rec["id"],
                "detector_label": d.trace["detector_label"],
                "detector_output": d.trace["detector_raw"],
                "validator_label": d.trace["validator_label"],
                "final_label": d.label,
                "flipped": d.trace["flipped"],
            },
            ensure_ascii=False,
        )
        for rec, d in zip(records, detections, strict=True)
    ]
    (out_dir / "agents.jsonl").write_text("\n".join(agent_lines) + "\n", encoding="utf-8")

    if manifest["logged"]:
        append_run(
            approach=APPROACH,
            model=cfg["model"]["name"],
            seed=seed,
            params={
                "prompt_version": PROMPT_VERSION,
                "validator_prompt_version": AUTHORS_VALIDATOR_VERSION,
                "detector_block_version": GAN_PROMPT_VERSIONS["detector_block"],
                "validator_block_version": PROMPT_VERSIONS["validator_block"],
                "memory_scope": party.memory_scope,
                "final_label_rule": party.final_label_rule,
                "snapshot": snapshot["name"],
                "round": snapshot["round"],
                "n_strategies": len(s_d),
                "memory_policy": s_d.policy,
                "unparseable_policy": policy,
                **manifest["decoding"],
            },
            metrics={
                **summary_metrics(report),
                "stage1_macro_f1": report["stage1"]["macro"]["f1"],
                "flip_rate": stats["flip_rate"],
            },
            output=str(out_dir),
        )

    print(format_report(report))
    logger.info("wrote run artifacts to %s", out_dir)
    return report


def run_all_snapshots(
    config_path: str | Path, limit: int | None = None, model=None
) -> dict[str, dict]:
    """Evaluate every snapshot of the run."""
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

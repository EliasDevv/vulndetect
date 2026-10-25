"""Dual-Agent runner (paper §III.E, Fig. 4): Detector pass, then Validator pass, then evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vulndetect.common import (
    DEFAULT_PARSER_VERSION,
    config_sha256,
    get_parser,
    load_config,
    make_prediction,
    set_seed,
    write_run,
)
from vulndetect.common.model import CHAT_DATE_STRING, TextGenerator
from vulndetect.data import DEFAULT_DATASET_DIR, load_split
from vulndetect.dual_agent.prompt import (
    AUTHORS_DETECTOR_VERSION,
    AUTHORS_VALIDATOR_VERSION,
    DUAL_AGENT_PROMPT_SETS,
    render_authors_detector,
    render_authors_validator,
)
from vulndetect.eval import append_run, evaluate, format_report, logging_enabled, summary_metrics
from vulndetect.eval.metrics import DEFAULT_POLICY

logger = logging.getLogger(__name__)

APPROACH = "dual_agent"

FINAL_LABEL_RULES = ("validator", "validator_then_detector")


def resolve_final_label(
    detector_label: str | None, validator_label: str | None, rule: str
) -> str | None:
    """Reduce the two agent labels to the final one according to ``rule``."""
    if rule not in FINAL_LABEL_RULES:
        raise ValueError(f"unknown final_label_rule {rule!r}; allowed: {FINAL_LABEL_RULES}")
    if rule == "validator_then_detector" and validator_label is None:
        return detector_label
    return validator_label


def load_detector_outputs(path: str | Path, records: list) -> dict:
    """Load detector texts of a finished run keyed by record id, requiring full coverage."""
    import hashlib

    path = Path(path)
    texts: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = "detector_output" if "detector_output" in row else "raw_output"
            texts[row["id"]] = row[key]
    missing = [r["id"] for r in records if r["id"] not in texts]
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} records have no detector text (first: {missing[0]!r}) — "
            "the source run must cover the whole split"
        )
    source_manifest = path.parent / "manifest.json"
    prompt_version = None
    if source_manifest.exists():
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        prompt_version = source.get("prompt_version")
    return {
        "texts": texts,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prompt_version": prompt_version,
    }


def run(
    config_path: str | Path,
    limit: int | None = None,
    model: TextGenerator | None = None,
) -> dict:
    """Execute one dual-agent run and return the metrics report."""
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    ds_cfg = cfg.get("dataset", {})
    records = load_split(ds_cfg.get("split", "test"), ds_cfg.get("dir", DEFAULT_DATASET_DIR))
    if limit is not None:
        records = records[:limit]
    logger.info("loaded %d records", len(records))

    da_cfg = cfg["dual_agent"]
    prompt_set = da_cfg["prompt_set"]
    final_label_rule = da_cfg["final_label_rule"]
    if prompt_set not in DUAL_AGENT_PROMPT_SETS:
        raise ValueError(f"unknown prompt_set {prompt_set!r}; allowed: {DUAL_AGENT_PROMPT_SETS}")
    if final_label_rule not in FINAL_LABEL_RULES:
        raise ValueError(
            f"unknown final_label_rule {final_label_rule!r}; allowed: {FINAL_LABEL_RULES}"
        )
    parser_version = cfg.get("parser_version", DEFAULT_PARSER_VERSION)
    parse = get_parser(parser_version)
    policy = cfg.get("eval", {}).get("unparseable", DEFAULT_POLICY)

    detector_source = da_cfg.get("detector_outputs_from")
    reused = load_detector_outputs(detector_source, records) if detector_source else None

    if model is None:
        from vulndetect.common.model import load_model

        model = load_model(cfg["model"])

    if reused is None:
        detector_prompts = [render_authors_detector(r) for r in records]
        detector_outputs = model.generate(detector_prompts)
        logger.info("detector pass done (%d records)", len(records))
    else:
        detector_outputs = [reused["texts"][r["id"]] for r in records]
        logger.info("detector pass SKIPPED: %d texts reused from %s", len(records), detector_source)
    detector_labels = [parse(out) for out in detector_outputs]

    validator_prompts = [
        render_authors_validator(rec, out)
        for rec, out in zip(records, detector_outputs, strict=True)
    ]
    validator_outputs = model.generate(validator_prompts)
    validator_labels = [parse(out) for out in validator_outputs]
    logger.info("validator pass done (%d records)", len(records))

    final_labels = [
        resolve_final_label(det, val, final_label_rule)
        for det, val in zip(detector_labels, validator_labels, strict=True)
    ]
    preds = [
        make_prediction(rec, out, label)
        for rec, out, label in zip(records, validator_outputs, final_labels, strict=True)
    ]

    out_dir = (
        Path(cfg["output"].get("predictions_dir", f"results/predictions/{APPROACH}"))
        / f"seed{seed}"
    )
    report = evaluate(preds, policy=policy)
    report["parser_version"] = parser_version

    flipped = [
        det is not None and val is not None and det != val
        for det, val in zip(detector_labels, validator_labels, strict=True)
    ]
    n = len(records)
    n_flipped = sum(flipped)
    n_agree = sum(
        det is not None and det == val
        for det, val in zip(detector_labels, validator_labels, strict=True)
    )
    prompt_version = AUTHORS_DETECTOR_VERSION
    if reused is not None and reused["prompt_version"]:
        prompt_version = reused["prompt_version"]
    validator_prompt_version = AUTHORS_VALIDATOR_VERSION
    temperature = cfg["model"].get("temperature", 0.0)
    manifest = {
        "approach": APPROACH,
        "model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "prompt_version": prompt_version,
        "validator_prompt_version": validator_prompt_version,
        "parser_version": parser_version,
        "unparseable_policy": policy,
        "dataset": {**ds_cfg, "n": n, "limit": limit},
        "decoding": {
            "temperature": temperature,
            "top_p": cfg["model"].get("top_p"),
            "do_sample": temperature > 0,
            "max_new_tokens": cfg["model"].get("max_new_tokens"),
            "batch_size": cfg["model"].get("batch_size"),
            "chat_date_string": cfg["model"].get("chat_date_string", CHAT_DATE_STRING),
        },
        "logged": limit is None and logging_enabled(cfg),
        "dual_agent": {
            "prompt_set": prompt_set,
            "final_label_rule": final_label_rule,
            "rounds": 1,
            "detector_outputs_from": (
                {k: reused[k] for k in ("path", "sha256", "prompt_version")} if reused else None
            ),
            "stats": {
                "n_detector_unparseable": sum(label is None for label in detector_labels),
                "n_validator_unparseable": sum(label is None for label in validator_labels),
                "n_flipped": n_flipped,
                "flip_rate": round(n_flipped / n, 4),
                "detector_validator_agreement": round(n_agree / n, 4),
            },
        },
    }
    write_run(out_dir, preds, manifest)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    agent_lines = [
        json.dumps(
            {
                "id": rec["id"],
                "detector_label": det,
                "detector_output": out,
                "validator_label": val,
                "final_label": fin,
                "flipped": flip,
            },
            ensure_ascii=False,
        )
        for rec, out, det, val, fin, flip in zip(
            records,
            detector_outputs,
            detector_labels,
            validator_labels,
            final_labels,
            flipped,
            strict=True,
        )
    ]
    (out_dir / "agents.jsonl").write_text("\n".join(agent_lines) + "\n", encoding="utf-8")

    if manifest["logged"]:
        append_run(
            approach=APPROACH,
            model=cfg["model"]["name"],
            seed=seed,
            params={
                "prompt_version": prompt_version,
                "validator_prompt_version": validator_prompt_version,
                "prompt_set": prompt_set,
                "parser_version": parser_version,
                "unparseable_policy": policy,
                "final_label_rule": final_label_rule,
                **manifest["decoding"],
            },
            metrics=summary_metrics(report),
            output=str(out_dir),
        )

    print(format_report(report))
    logger.info("wrote run artifacts to %s", out_dir)
    return report


def main() -> None:
    """Parse the CLI arguments and run the Dual-Agent evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Dual-Agent: Detector -> Validator (paper §III.E).")
    ap.add_argument("--config", default="configs/dual_agent.yaml", help="path to dual-agent YAML")
    ap.add_argument("--limit", type=int, help="only the first N records (smoke run, not logged)")
    args = ap.parse_args()
    run(args.config, limit=args.limit)


if __name__ == "__main__":
    main()

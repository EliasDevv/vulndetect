"""Baseline: direct prompting with the shared Fig. 1 prompt."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vulndetect.common import (
    DEFAULT_PARSER_VERSION,
    PROMPT_VERSION,
    config_sha256,
    get_parser,
    get_prompt_variant,
    load_config,
    make_prediction,
    set_seed,
    write_run,
)
from vulndetect.common.model import CHAT_DATE_STRING, TextGenerator
from vulndetect.data import DEFAULT_DATASET_DIR, load_split
from vulndetect.eval import append_run, evaluate, format_report, logging_enabled, summary_metrics
from vulndetect.eval.metrics import DEFAULT_POLICY

logger = logging.getLogger(__name__)

APPROACH = "baseline"


def run(
    config_path: str | Path,
    limit: int | None = None,
    model: TextGenerator | None = None,
) -> dict:
    """Execute one baseline run and return the metrics report; ``model`` is injectable."""
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    ds_cfg = cfg.get("dataset", {})
    records = load_split(ds_cfg.get("split", "test"), ds_cfg.get("dir", DEFAULT_DATASET_DIR))
    if limit is not None:
        records = records[:limit]
    logger.info("loaded %d records", len(records))

    if model is None:
        from vulndetect.common.model import load_model

        model = load_model(cfg["model"])

    variant = get_prompt_variant(cfg.get("prompt", {}).get("version", PROMPT_VERSION))
    parser_version = cfg.get("parser_version", DEFAULT_PARSER_VERSION)
    parse = get_parser(parser_version)
    policy = cfg.get("eval", {}).get("unparseable", DEFAULT_POLICY)

    prompts = [variant.render(r) for r in records]
    outputs = model.generate(prompts, system=variant.system)
    preds = [
        make_prediction(rec, out, parse(out)) for rec, out in zip(records, outputs, strict=True)
    ]

    out_dir = (
        Path(cfg["output"].get("predictions_dir", f"results/predictions/{APPROACH}"))
        / f"seed{seed}"
    )
    report = evaluate(preds, policy=policy)
    report["parser_version"] = parser_version
    temperature = cfg["model"].get("temperature", 0.0)
    manifest = {
        "approach": APPROACH,
        "model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "prompt_version": variant.version,
        "parser_version": parser_version,
        "unparseable_policy": policy,
        "dataset": {**ds_cfg, "n": len(records), "limit": limit},
        "decoding": {
            "temperature": temperature,
            "top_p": cfg["model"].get("top_p"),
            "do_sample": temperature > 0,
            "max_new_tokens": cfg["model"].get("max_new_tokens"),
            "batch_size": cfg["model"].get("batch_size"),
            "chat_date_string": cfg["model"].get("chat_date_string", CHAT_DATE_STRING),
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
                "prompt_version": variant.version,
                "parser_version": parser_version,
                "unparseable_policy": policy,
                **manifest["decoding"],
            },
            metrics=summary_metrics(report),
            output=str(out_dir),
        )

    print(format_report(report))
    logger.info("wrote run artifacts to %s", out_dir)
    return report


def main() -> None:
    """Parse CLI arguments and run the baseline experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Baseline: direct prompting on the test split.")
    ap.add_argument("--config", default="configs/baseline.yaml", help="path to baseline YAML")
    ap.add_argument("--limit", type=int, help="only the first N records (smoke run, not logged)")
    args = ap.parse_args()
    run(args.config, limit=args.limit)


if __name__ == "__main__":
    main()

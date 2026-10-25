"""SFT eval runner: score the fine-tuned adapter like every other approach."""

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
from vulndetect.sft.dataset import ALPACA_PROMPT_VERSION, to_alpaca_prompt
from vulndetect.sft.train import assert_seed_scoped_paths

logger = logging.getLogger(__name__)

APPROACH = "sft"

TRAINING_ECHO_KEYS = ("config_sha256", "batch_stats", "final_train_loss", "runtime_seconds")


def load_training_manifest(path: str | None) -> dict | None:
    """Load the manifest of the training run that produced the adapter, if present."""
    if not path or not Path(path).exists():
        logger.warning("no training manifest at %s — eval proceeds, provenance incomplete", path)
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(
    config_path: str | Path,
    limit: int | None = None,
    model: TextGenerator | None = None,
) -> dict:
    """Execute one SFT eval run and return the metrics report."""
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    adapter = cfg["model"].get("adapter")
    if not adapter:
        raise ValueError(
            "model.adapter missing in the SFT config — train first (`make sft-train`), "
            "or use the baseline config if you want the un-tuned model"
        )
    assert_seed_scoped_paths(
        seed,
        {
            "model.adapter": adapter,
            "output.training_manifest": cfg["output"].get("training_manifest"),
        },
    )

    ds_cfg = cfg.get("dataset", {})
    records = load_split(ds_cfg.get("split", "test"), ds_cfg.get("dir", DEFAULT_DATASET_DIR))
    if limit is not None:
        records = records[:limit]
    logger.info("loaded %d records", len(records))

    if model is None:
        from vulndetect.common.model import load_model

        model = load_model(cfg["model"])

    t_cfg = cfg.get("training", {})
    fmt = t_cfg.get("format", "chat")
    parser_version = cfg.get("parser_version", DEFAULT_PARSER_VERSION)
    parse = get_parser(parser_version)
    policy = cfg.get("eval", {}).get("unparseable", DEFAULT_POLICY)
    if fmt == "authors-alpaca":
        prompt_version = ALPACA_PROMPT_VERSION
        prompts = [to_alpaca_prompt(r) for r in records]
        outputs = model.generate(prompts, chat=False)
    else:
        variant = get_prompt_variant(cfg.get("prompt", {}).get("version", PROMPT_VERSION))
        prompt_version = variant.version
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

    tm_path = cfg["output"].get("training_manifest")
    tm = load_training_manifest(tm_path)
    temperature = cfg["model"].get("temperature", 0.0)
    manifest = {
        "approach": APPROACH,
        "model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "prompt_version": prompt_version,
        "parser_version": parser_version,
        "unparseable_policy": policy,
        "dataset": {**ds_cfg, "n": len(records), "limit": limit},
        "decoding": {
            "temperature": temperature,
            "top_p": cfg["model"].get("top_p"),
            "do_sample": temperature > 0,
            "max_new_tokens": cfg["model"].get("max_new_tokens"),
            "batch_size": cfg["model"].get("batch_size"),
            "chat_date_string": None
            if fmt == "authors-alpaca"
            else cfg["model"].get("chat_date_string", CHAT_DATE_STRING),
        },
        "logged": limit is None and logging_enabled(cfg),
        "sft": {
            "adapter": str(adapter),
            "format": fmt,
            "training_manifest": tm_path,
            "training": {k: tm.get(k) for k in TRAINING_ECHO_KEYS} if tm else None,
        },
    }
    write_run(out_dir, preds, manifest)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if manifest["logged"]:
        batch_stats = (tm or {}).get("batch_stats", {})
        append_run(
            approach=APPROACH,
            model=cfg["model"]["name"],
            seed=seed,
            params={
                "prompt_version": prompt_version,
                "parser_version": parser_version,
                "unparseable_policy": policy,
                "format": fmt,
                "adapter": str(adapter),
                "max_steps": t_cfg.get("max_steps"),
                "learning_rate": t_cfg.get("learning_rate"),
                "lora_r": t_cfg.get("lora_r"),
                "effective_epochs": batch_stats.get("effective_epochs"),
                **manifest["decoding"],
            },
            metrics=summary_metrics(report),
            output=str(out_dir),
        )

    print(format_report(report))
    logger.info("wrote run artifacts to %s", out_dir)
    return report


def main() -> None:
    """Parse the CLI arguments and evaluate the SFT adapter."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="SFT: evaluate the QLoRA adapter (paper §III.D).")
    ap.add_argument("--config", default="configs/sft.yaml", help="path to the SFT YAML")
    ap.add_argument("--limit", type=int, help="only the first N records (smoke run, not logged)")
    args = ap.parse_args()
    run(args.config, limit=args.limit)


if __name__ == "__main__":
    main()

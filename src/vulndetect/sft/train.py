"""QLoRA training (paper §III.D, Table I) with TRL SFTTrainer, peft and bitsandbytes."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from vulndetect.common import PROMPT_VERSION, config_sha256, load_config, set_seed
from vulndetect.data import DEFAULT_DATASET_DIR, load_split
from vulndetect.eval import append_run, logging_enabled
from vulndetect.sft.dataset import ALPACA_PROMPT_VERSION, to_alpaca_examples, to_examples

logger = logging.getLogger(__name__)

APPROACH = "sft_train"

TABLE1_KEYS = (
    "learning_rate",
    "optimizer",
    "max_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "weight_decay",
    "lr_scheduler_type",
)
D25_KEYS = (
    "format",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "max_grad_norm",
    "max_seq_length",
    "packing",
    "completion_only_loss",
    "gradient_checkpointing",
    "target_modules",
    "quantization",
)

FORMATS = ("chat", "authors-alpaca")


def validate_training_cfg(t_cfg: dict) -> None:
    """Validate that the training config has all required keys and consistent settings."""
    missing = [k for k in (*TABLE1_KEYS, *D25_KEYS) if k not in t_cfg]
    if missing:
        raise KeyError(f"training config is missing keys {missing} (paper Table I)")
    if int(t_cfg["max_steps"]) == -1 and "num_train_epochs" not in t_cfg:
        raise KeyError("max_steps: -1 requires num_train_epochs (sft_extended variant)")
    if t_cfg["format"] not in FORMATS:
        raise ValueError(f"training.format must be one of {FORMATS}, got {t_cfg['format']!r}")
    if t_cfg["format"] == "authors-alpaca" and t_cfg["completion_only_loss"]:
        raise ValueError(
            "training.format: authors-alpaca requires completion_only_loss: false — "
            "the authors' notebook trains loss-on-everything on a plain-text dataset "
            "(TRL cannot mask a completion in a bare text field)"
        )
    quant = t_cfg["quantization"]
    if quant.get("bits") not in (4, "none"):
        raise ValueError(f"quantization.bits must be 4 or 'none', got {quant.get('bits')!r}")
    if quant.get("bits") == "none" and t_cfg["optimizer"] == "adamw_8bit":
        raise ValueError(
            "quantization.bits: none but optimizer is still adamw_8bit — the 8-bit "
            "optimizer itself needs bitsandbytes/CUDA; change the optimizer too"
        )


def assert_seed_scoped_paths(seed: int, paths: dict[str, str | None]) -> None:
    """Raise if any given path lacks the ``seed<N>`` segment of the config seed."""
    tag = f"seed{seed}"
    bad = {name: str(p) for name, p in paths.items() if p and tag not in str(p)}
    if bad:
        raise ValueError(
            f"config seed is {seed} but these paths lack a '{tag}' segment: {bad} — "
            "seed-scoped adapter/manifest paths must match the config's seed"
        )


def effective_batch_stats(n: int, max_steps: int, per_device: int, grad_accum: int) -> dict:
    """Compute effective batch size, examples seen and effective epochs."""
    effective_batch = per_device * grad_accum
    if max_steps == -1:
        return {"effective_batch": effective_batch, "examples_seen": None, "effective_epochs": None}
    examples_seen = max_steps * effective_batch
    return {
        "effective_batch": effective_batch,
        "examples_seen": examples_seen,
        "effective_epochs": round(examples_seen / n, 4),
    }


def truncation_stats(examples: list[dict], tokenizer, max_length: int) -> dict:
    """Measure the share of examples whose tokenized text exceeds ``max_length``."""
    lengths = [
        len(tokenizer(ex["text"])["input_ids"])
        if "text" in ex
        else len(
            tokenizer.apply_chat_template(
                ex["prompt"] + ex["completion"], tokenize=True, add_generation_prompt=False
            )
        )
        for ex in examples
    ]
    n_over = sum(length > max_length for length in lengths)
    return {
        "max_length": max_length,
        "n_over": n_over,
        "share_over": round(n_over / len(lengths), 4) if lengths else None,
        "max_observed": max(lengths, default=None),
    }


def build_bnb_config(quant_cfg: dict):
    """Build a BitsAndBytesConfig from the quantization block (None if bits is 'none')."""
    if quant_cfg["bits"] == "none":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg["quant_type"],
        bnb_4bit_use_double_quant=bool(quant_cfg["double_quant"]),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["compute_dtype"]),
    )


def build_lora_config(t_cfg: dict):
    """Build the LoraConfig from the training config."""
    from peft import LoraConfig

    return LoraConfig(
        task_type="CAUSAL_LM",
        r=int(t_cfg["lora_r"]),
        lora_alpha=int(t_cfg["lora_alpha"]),
        lora_dropout=float(t_cfg["lora_dropout"]),
        bias="none",
        target_modules=list(t_cfg["target_modules"]),
    )


def _torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def train_dtype(t_cfg: dict):
    """Return the training dtype, i.e. the QLoRA compute dtype (default float16)."""
    return _torch_dtype(t_cfg["quantization"].get("compute_dtype", "float16"))


def mixed_precision_flags(t_cfg: dict) -> tuple[bool, bool]:
    """Return the trainer's ``(bf16, fp16)`` flags derived from the training dtype."""
    if not t_cfg.get("mixed_precision", True):
        return False, False
    dtype = train_dtype(t_cfg)
    return dtype is _torch_dtype("bfloat16"), dtype is _torch_dtype("float16")


def cast_trainable_to_fp32(model) -> dict[str, int]:
    """Cast trainable LoRA parameters to fp32 for fp16 training; return prior dtype counts."""
    import torch

    seen: dict[str, int] = {}
    for param in model.parameters():
        if not param.requires_grad:
            continue
        seen[str(param.dtype)] = seen.get(str(param.dtype), 0) + 1
        if param.dtype is not torch.float32:
            param.data = param.data.to(torch.float32)
    return seen


def build_sft_config(t_cfg: dict, output_dir: str | Path, seed: int):
    """Build the TRL SFTConfig from the training config."""
    from trl import SFTConfig

    max_steps = int(t_cfg["max_steps"])
    _bf16, _fp16 = mixed_precision_flags(t_cfg)
    kwargs = {}
    if max_steps == -1:
        kwargs["num_train_epochs"] = float(t_cfg["num_train_epochs"])
    return SFTConfig(
        output_dir=str(output_dir),
        # trl's default "chunked_nll" crashes on quantized models; "nll" is the same loss.
        loss_type="nll",
        max_steps=max_steps,
        per_device_train_batch_size=int(t_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t_cfg["gradient_accumulation_steps"]),
        learning_rate=float(t_cfg["learning_rate"]),
        optim=t_cfg["optimizer"],
        weight_decay=float(t_cfg["weight_decay"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_steps=int(t_cfg["warmup_steps"]),
        max_grad_norm=float(t_cfg["max_grad_norm"]),
        max_length=int(t_cfg["max_seq_length"]),
        packing=bool(t_cfg["packing"]),
        completion_only_loss=bool(t_cfg["completion_only_loss"]),
        gradient_checkpointing=bool(t_cfg["gradient_checkpointing"]),
        bf16=_bf16,
        fp16=_fp16,
        logging_steps=1,
        save_strategy="no",
        seed=seed,
        report_to="none",
        **kwargs,
    )


def train(config_path: str | Path, limit: int | None = None, dry_run: bool = False) -> dict:
    """Run one QLoRA training (or a CPU-only dry run) and return the training manifest."""
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    t_cfg = cfg["training"]
    validate_training_cfg(t_cfg)
    assert_seed_scoped_paths(
        seed,
        {
            "output.adapter_dir": cfg["output"]["adapter_dir"],
            "output.training_manifest": cfg["output"].get("training_manifest"),
        },
    )

    ds_cfg = cfg.get("dataset", {})
    records = load_split(t_cfg.get("split", "train"), ds_cfg.get("dir", DEFAULT_DATASET_DIR))
    if limit is not None:
        records = records[:limit]
    fmt = t_cfg["format"]
    examples = to_alpaca_examples(records) if fmt == "authors-alpaca" else to_examples(records)
    prompt_version = ALPACA_PROMPT_VERSION if fmt == "authors-alpaca" else PROMPT_VERSION
    train_seed = int(t_cfg.get("seed", seed))
    logger.info(
        "built %d %s-format training examples from the %s split",
        len(examples),
        fmt,
        t_cfg["split"],
    )

    batch_stats = effective_batch_stats(
        len(examples),
        int(t_cfg["max_steps"]),
        int(t_cfg["per_device_train_batch_size"]),
        int(t_cfg["gradient_accumulation_steps"]),
    )
    manifest = {
        "approach": APPROACH,
        "model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "seed": seed,
        "train_seed": train_seed,
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "prompt_version": prompt_version,
        "dataset": {
            **ds_cfg,
            "split": t_cfg.get("split", "train"),
            "n": len(examples),
            "limit": limit,
        },
        "training": {k: t_cfg[k] for k in (*TABLE1_KEYS, *D25_KEYS)},
        "batch_stats": batch_stats,
    }
    if dry_run:
        print(json.dumps({**manifest, "dry_run": True}, indent=2, ensure_ascii=False))
        if fmt == "authors-alpaca":
            print("\n--- first example text (truncated to 700 chars) ---")
            print(examples[0]["text"][:700])
        else:
            print("\n--- first example prompt (truncated to 500 chars) ---")
            print(examples[0]["prompt"][0]["content"][:500])
            print("--- completion ---")
            print(examples[0]["completion"][0]["content"])
        return manifest

    import torch

    if t_cfg["quantization"]["bits"] == 4 and not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 4-bit training needs CUDA (bitsandbytes NF4 + adamw_8bit) — "
            "run this on the GPU server; locally only --dry-run and pytest work"
        )

    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTTrainer

    base_name = cfg["model"]["name"]
    revision = cfg["model"].get("revision")
    adapter_dir = Path(cfg["output"]["adapter_dir"])
    work_dir = adapter_dir.parent / "trainer_work"

    tokenizer = AutoTokenizer.from_pretrained(base_name, revision=revision)
    if fmt == "authors-alpaca":
        examples = [{"text": ex["text"] + tokenizer.eos_token} for ex in examples]
    manifest["truncation"] = truncation_stats(examples, tokenizer, int(t_cfg["max_seq_length"]))
    logger.info("truncation stats: %s", manifest["truncation"])

    from transformers import AutoModelForCausalLM

    bnb = build_bnb_config(t_cfg["quantization"])
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        revision=revision,
        quantization_config=bnb,
        dtype=train_dtype(t_cfg),
        device_map="auto",
    )
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(t_cfg, work_dir, train_seed),
        train_dataset=Dataset.from_list(examples),
        peft_config=build_lora_config(t_cfg),
        processing_class=tokenizer,
    )
    _, fp16_on = mixed_precision_flags(t_cfg)
    if fp16_on:
        manifest["adapter_dtypes_before_cast"] = cast_trainable_to_fp32(trainer.model)
        logger.info(
            "cast trainable params to fp32 (was %s)", manifest["adapter_dtypes_before_cast"]
        )

    start = time.time()
    trainer.train()
    runtime_s = round(time.time() - start, 1)
    trainer.save_model(str(adapter_dir))
    logger.info("saved adapter to %s (%.1f s)", adapter_dir, runtime_s)

    import bitsandbytes
    import peft
    import transformers
    import trl

    log_history = list(trainer.state.log_history)
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    manifest.update(
        {
            "runtime_seconds": runtime_s,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "log_history": log_history,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "trl": trl.__version__,
                "peft": peft.__version__,
                "bitsandbytes": bitsandbytes.__version__,
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
    )
    manifest_path = Path(cfg["output"]["training_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy(manifest_path, adapter_dir / "training_manifest.json")

    if limit is None and logging_enabled(cfg):
        append_run(
            approach=APPROACH,
            model=base_name,
            seed=seed,
            params={
                "prompt_version": prompt_version,
                "format": fmt,
                "train_seed": train_seed,
                **{k: t_cfg[k] for k in TABLE1_KEYS},
                "effective_batch": batch_stats["effective_batch"],
                "effective_epochs": batch_stats["effective_epochs"],
            },
            metrics={
                "final_train_loss": manifest["final_train_loss"],
                "n_train": len(examples),
                "truncation_share": manifest["truncation"]["share_over"],
            },
            output=str(adapter_dir),
        )
    return manifest


def main() -> None:
    """Parse the CLI arguments and run the QLoRA training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="SFT/QLoRA training (paper §III.D, Table I).")
    ap.add_argument("--config", default="configs/sft.yaml", help="path to the SFT YAML")
    ap.add_argument("--limit", type=int, help="only the first N records (smoke run, not logged)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config + dataset and print the effective-epoch math (no GPU)",
    )
    args = ap.parse_args()
    train(args.config, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""Offline tests for SFT dataset formatting, training config handling and the eval runner."""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from vulndetect.common.predictions import read_predictions
from vulndetect.common.prompt import render_prompt
from vulndetect.data.build import to_records, write_jsonl
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
from vulndetect.sft.dataset import to_example, to_examples
from vulndetect.sft.run import run
from vulndetect.sft.train import (
    assert_seed_scoped_paths,
    effective_batch_stats,
    train,
    validate_training_cfg,
)

N_TEST = 8

TRAINING_CFG = {
    "split": "train",
    "format": "chat",
    "learning_rate": 2.0e-4,
    "optimizer": "adamw_8bit",
    "max_steps": 30,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "max_grad_norm": 1.0,
    "max_seq_length": 4096,
    "packing": False,
    "completion_only_loss": True,
    "gradient_checkpointing": True,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "quantization": {
        "bits": 4,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
    },
}


class OracleModel:
    """Fake model that returns the gold labels in prompt order."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.calls = 0

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        self.calls += 1
        assert len(prompts) == len(self.labels)
        return [f"{label}. Reasoning: because the test says so." for label in self.labels]


@pytest.fixture
def workspace(tmp_path, raw_df, base_cfg, monkeypatch):
    """A tmp cwd with built train and test splits and an SFT config."""
    monkeypatch.chdir(tmp_path)
    from vulndetect.data.build import build_dataset

    records = to_records(build_dataset(raw_df, base_cfg))
    test_records: list[dict] = []
    for cwe in sorted({r["cwe"] for r in records}):
        cwe_test = [r for r in records if r["split"] == "test" and r["cwe"] == cwe]
        vuln = [r for r in cwe_test if r["vul"] == 1]
        safe = [r for r in cwe_test if r["vul"] == 0]
        if vuln and safe and len(test_records) < N_TEST:
            test_records += [vuln[0], safe[0]]
    assert len(test_records) >= 4, "fixture needs at least two CWEs in the test split"
    train_records = [r for r in records if r["split"] == "train"]
    write_jsonl(test_records, tmp_path / "dataset" / "test.jsonl")
    write_jsonl(train_records, tmp_path / "dataset" / "train.jsonl")

    cfg = {
        "experiment": "sft",
        "seed": 42,
        "model": {
            "provider": "huggingface",
            "name": "meta-llama/Llama-3.2-3B-Instruct",
            "adapter": "models/sft/seed42/adapter",
            "temperature": 0.0,
            "max_new_tokens": 512,
            "batch_size": 8,
        },
        "dataset": {"name": "bigvul", "split": "test", "dir": "dataset"},
        "training": dict(TRAINING_CFG),
        "output": {
            "predictions_dir": "results/predictions/sft",
            "adapter_dir": "models/sft/seed42/adapter",
            "training_manifest": "results/training/sft/seed42/training_manifest.json",
        },
    }
    cfg_path = tmp_path / "sft.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path, test_records, train_records


def rewrite_cfg(cfg_path, mutate) -> None:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mutate(cfg)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def sample_record() -> dict:
    return {
        "id": "r1",
        "commit_message": "fix overflow",
        "func_before": "int f() { /*bug*/ return 0; }",
        "func_after": "int f() { return 0; }",
        "cve_id": "CVE-2020-0001",
        "cwe": "CWE-119",
        "project": "proj",
        "lang": "C",
        "label": LABEL_VULN,
    }


def test_to_example_golden():
    record = sample_record()
    ex = to_example(record)
    assert ex["prompt"] == [{"role": "user", "content": render_prompt(record)}]
    assert ex["completion"] == [{"role": "assistant", "content": LABEL_VULN}]
    assert set(ex) == {"prompt", "completion"}


def test_to_examples_order_and_labels():
    vuln = sample_record()
    safe = {**sample_record(), "id": "r2", "label": LABEL_SAFE}
    examples = to_examples([vuln, safe])
    assert [e["completion"][0]["content"] for e in examples] == [LABEL_VULN, LABEL_SAFE]


def test_to_alpaca_text_golden():
    from vulndetect.sft.dataset import to_alpaca_text

    record = sample_record()
    text = to_alpaca_text(record)
    assert text.startswith("### Instruction:\n")
    assert text.index("### Instruction:") < text.index("### Input:") < text.index("### Response:")
    assert "CVE/CWE information (if available)" in text
    assert "Answer with `Vulnerable` or `Not Vulnerable`" in text
    assert "- CWE ID (if available): CWE-119" in text
    assert "CVE-2020-0001" not in text
    assert "- Commit Message: fix overflow" in text
    assert "- Code Before Change: int f() { /*bug*/ return 0; }" in text
    assert text.endswith(f"### Response:\nClassification: {LABEL_VULN}")


def test_to_alpaca_prompt_is_text_with_empty_response():
    from vulndetect.sft.dataset import to_alpaca_prompt, to_alpaca_text

    record = sample_record()
    prompt = to_alpaca_prompt(record)
    assert prompt.endswith("### Response:\n")
    assert to_alpaca_text(record) == prompt + f"Classification: {LABEL_VULN}"


def test_to_alpaca_examples_shape():
    from vulndetect.sft.dataset import to_alpaca_examples

    examples = to_alpaca_examples([sample_record(), {**sample_record(), "label": LABEL_SAFE}])
    assert all(set(e) == {"text"} for e in examples)
    assert examples[1]["text"].endswith(f"Classification: {LABEL_SAFE}")


def test_validate_training_cfg_format():
    from vulndetect.sft.train import validate_training_cfg

    bad_format = {**TRAINING_CFG, "format": "chatml"}
    with pytest.raises(ValueError, match="training.format"):
        validate_training_cfg(bad_format)

    alpaca_masked = {**TRAINING_CFG, "format": "authors-alpaca", "completion_only_loss": True}
    with pytest.raises(ValueError, match="completion_only_loss"):
        validate_training_cfg(alpaca_masked)

    alpaca_ok = {**TRAINING_CFG, "format": "authors-alpaca", "completion_only_loss": False}
    validate_training_cfg(alpaca_ok)


def test_alpaca_dry_run_and_train_seed(workspace, capsys):
    cfg_path, _, train_records = workspace
    rewrite_cfg(
        cfg_path,
        lambda c: c["training"].update(
            {"format": "authors-alpaca", "completion_only_loss": False, "seed": 3407}
        ),
    )
    manifest = train(cfg_path, dry_run=True)
    assert manifest["prompt_version"] == "authors-alpaca-v1"
    assert manifest["seed"] == 42
    assert manifest["train_seed"] == 3407
    assert manifest["training"]["format"] == "authors-alpaca"
    out = capsys.readouterr().out
    assert "### Instruction:" in out


def test_alpaca_eval_uses_raw_prompts(workspace, tmp_path):
    from vulndetect.sft.dataset import to_alpaca_prompt

    cfg_path, test_records, _ = workspace
    rewrite_cfg(
        cfg_path,
        lambda c: c["training"].update({"format": "authors-alpaca", "completion_only_loss": False}),
    )

    class RecordingOracle(OracleModel):
        def __init__(self, labels):
            super().__init__(labels)
            self.kwargs: dict = {}
            self.prompts: list[str] = []

        def generate(self, prompts, **kwargs):
            self.kwargs = kwargs
            self.prompts = list(prompts)
            return super().generate(prompts)

    model = RecordingOracle([r["label"] for r in test_records])
    run(cfg_path, model=model)
    assert model.kwargs.get("chat") is False
    assert model.prompts == [to_alpaca_prompt(r) for r in test_records]

    manifest = json.loads(
        (tmp_path / "results" / "predictions" / "sft" / "seed42" / "manifest.json").read_text()
    )
    assert manifest["prompt_version"] == "authors-alpaca-v1"
    assert manifest["sft"]["format"] == "authors-alpaca"


def test_effective_batch_stats():
    stats = effective_batch_stats(4994, 30, 2, 4)
    assert stats == {
        "effective_batch": 8,
        "examples_seen": 240,
        "effective_epochs": pytest.approx(0.0481, abs=1e-4),
    }
    assert effective_batch_stats(4994, -1, 2, 4)["examples_seen"] is None


def test_validate_training_cfg():
    validate_training_cfg(dict(TRAINING_CFG))

    incomplete = dict(TRAINING_CFG)
    del incomplete["max_steps"]
    with pytest.raises(KeyError, match="max_steps"):
        validate_training_cfg(incomplete)

    no_cuda_optim = dict(TRAINING_CFG)
    no_cuda_optim["quantization"] = {**TRAINING_CFG["quantization"], "bits": "none"}
    with pytest.raises(ValueError, match="adamw_8bit"):
        validate_training_cfg(no_cuda_optim)

    extended = dict(TRAINING_CFG)
    extended["max_steps"] = -1
    with pytest.raises(KeyError, match="num_train_epochs"):
        validate_training_cfg(extended)
    extended["num_train_epochs"] = 1
    validate_training_cfg(extended)


def test_assert_seed_scoped_paths():
    assert_seed_scoped_paths(43, {"a": "models/sft/seed43/adapter", "b": None})
    with pytest.raises(ValueError, match="seed43"):
        assert_seed_scoped_paths(43, {"output.adapter_dir": "models/sft/seed42/adapter"})


def test_train_seed_mismatch_fails_before_anything(workspace):
    cfg_path, _, _ = workspace
    rewrite_cfg(cfg_path, lambda c: c.update(seed=43))
    with pytest.raises(ValueError, match="seed43"):
        train(cfg_path, dry_run=True)


def test_train_dry_run(workspace, capsys):
    cfg_path, _, train_records = workspace
    manifest = train(cfg_path, dry_run=True)
    assert manifest["approach"] == "sft_train"
    assert manifest["dataset"]["n"] == len(train_records)
    assert manifest["batch_stats"]["effective_batch"] == 8
    assert manifest["batch_stats"]["examples_seen"] == 240
    assert manifest["training"]["max_steps"] == 30
    assert not (cfg_path.parent / "models").exists()
    assert not (cfg_path.parent / "results" / "training").exists()
    out = capsys.readouterr().out
    assert "You are an expert in cybersecurity" in out


def test_build_lora_config():
    pytest.importorskip("peft")
    from vulndetect.sft.train import build_lora_config

    lora = build_lora_config(TRAINING_CFG)
    assert lora.r == 16
    assert lora.lora_alpha == 16
    assert lora.lora_dropout == pytest.approx(0.1)
    assert set(lora.target_modules) == set(TRAINING_CFG["target_modules"])
    assert lora.bias == "none"


def test_build_sft_config(tmp_path):
    pytest.importorskip("trl")
    from vulndetect.sft.train import build_sft_config

    cpu_cfg = {**TRAINING_CFG, "mixed_precision": False}
    args = build_sft_config(cpu_cfg, tmp_path / "work", seed=42)
    assert args.max_steps == 30
    assert args.learning_rate == pytest.approx(2.0e-4)
    assert args.optim == "adamw_8bit"
    assert args.weight_decay == pytest.approx(0.01)
    assert args.warmup_steps == 5
    assert args.max_length == 4096
    assert args.packing is False
    assert args.completion_only_loss is True
    assert args.per_device_train_batch_size == 2
    assert args.gradient_accumulation_steps == 4
    assert args.seed == 42

    assert args.bf16 is False and args.fp16 is False

    extended = {**cpu_cfg, "max_steps": -1, "num_train_epochs": 1}
    args = build_sft_config(extended, tmp_path / "work2", seed=42)
    assert args.max_steps == -1
    assert args.num_train_epochs == 1.0


def test_build_bnb_config():
    pytest.importorskip("transformers")
    import torch

    from vulndetect.sft.train import build_bnb_config

    bnb = build_bnb_config(TRAINING_CFG["quantization"])
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.bnb_4bit_use_double_quant is True
    assert bnb.bnb_4bit_compute_dtype == torch.bfloat16
    assert build_bnb_config({**TRAINING_CFG["quantization"], "bits": "none"}) is None


def test_sft_run_end_to_end(workspace, tmp_path):
    cfg_path, test_records, _ = workspace
    tm_path = tmp_path / "results" / "training" / "sft" / "seed42" / "training_manifest.json"
    tm_path.parent.mkdir(parents=True)
    tm_path.write_text(
        json.dumps(
            {
                "config_sha256": "abc",
                "batch_stats": {
                    "effective_batch": 8,
                    "examples_seen": 240,
                    "effective_epochs": 0.0481,
                },
                "final_train_loss": 0.42,
                "runtime_seconds": 123.4,
            }
        ),
        encoding="utf-8",
    )
    model = OracleModel([r["label"] for r in test_records])
    report = run(cfg_path, model=model)

    assert model.calls == 1
    assert report["macro"]["f1"] == pytest.approx(1.0)

    out_dir = tmp_path / "results" / "predictions" / "sft" / "seed42"
    preds = read_predictions(out_dir / "predictions.jsonl")
    assert all(p["pred_label"] == p["gold_label"] for p in preds)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["approach"] == "sft"
    assert manifest["prompt_version"] == "fig1-v3"
    assert manifest["decoding"]["batch_size"] == 8
    assert manifest["sft"]["adapter"] == "models/sft/seed42/adapter"
    assert manifest["sft"]["training"]["final_train_loss"] == 0.42
    assert manifest["sft"]["training"]["batch_stats"]["effective_epochs"] == 0.0481
    assert (out_dir / "metrics.json").exists()

    rows = list(csv.DictReader((tmp_path / "results" / "experiments.csv").open()))
    assert len(rows) == 1
    assert rows[0]["approach"] == "sft"
    for key in ("adapter", "max_steps", "effective_epochs"):
        assert key in rows[0]["params"]


def test_sft_run_without_training_manifest_warns_not_fails(workspace, tmp_path):
    cfg_path, test_records, _ = workspace
    model = OracleModel([r["label"] for r in test_records])
    run(cfg_path, model=model)
    manifest = json.loads(
        (tmp_path / "results" / "predictions" / "sft" / "seed42" / "manifest.json").read_text()
    )
    assert manifest["sft"]["training"] is None


def test_sft_run_missing_adapter_fails_loudly(workspace):
    cfg_path, test_records, _ = workspace
    rewrite_cfg(cfg_path, lambda c: c["model"].pop("adapter"))
    with pytest.raises(ValueError, match="adapter"):
        run(cfg_path, model=OracleModel([r["label"] for r in test_records]))


def test_sft_run_seed_mismatch_fails_loudly(workspace):
    cfg_path, test_records, _ = workspace
    rewrite_cfg(cfg_path, lambda c: c.update(seed=43))
    with pytest.raises(ValueError, match="seed43"):
        run(cfg_path, model=OracleModel([r["label"] for r in test_records]))


def test_sft_limit_smoke_run_not_logged(workspace, tmp_path):
    cfg_path, test_records, _ = workspace
    limit = 3
    model = OracleModel([r["label"] for r in test_records[:limit]])
    report = run(cfg_path, limit=limit, model=model)
    assert report["n"] == limit
    assert not (tmp_path / "results" / "experiments.csv").exists()


def test_repo_config_pins_table1():
    """Checks the Table I hyperparameters in configs/sft.yaml."""
    from pathlib import Path

    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sft.yaml").read_text(encoding="utf-8")
    )
    t = cfg["training"]
    assert t["learning_rate"] == pytest.approx(2.0e-4)
    assert t["optimizer"] == "adamw_8bit"
    assert t["max_steps"] == 30
    assert t["lora_r"] == 16
    assert t["lora_alpha"] == 16
    assert t["lora_dropout"] == pytest.approx(0.1)
    assert t["weight_decay"] == pytest.approx(0.01)
    assert t["lr_scheduler_type"] == "linear"
    assert t["quantization"]["bits"] == 4
    validate_training_cfg(t)


@pytest.mark.parametrize("seed", [43, 44])
def test_repo_seed_configs_match_sft(seed):
    """Checks that SFT seed configs differ from sft.yaml only in seed and seed-scoped paths."""
    from pathlib import Path

    configs = Path(__file__).parents[1] / "configs"
    base = yaml.safe_load((configs / "sft.yaml").read_text(encoding="utf-8"))
    variant = yaml.safe_load((configs / f"sft_seed{seed}.yaml").read_text(encoding="utf-8"))

    assert variant["seed"] == seed
    assert variant["training"] == base["training"]
    assert variant["dataset"] == base["dataset"]
    assert {k: v for k, v in variant["model"].items() if k != "adapter"} == {
        k: v for k, v in base["model"].items() if k != "adapter"
    }
    assert variant["output"]["predictions_dir"] == base["output"]["predictions_dir"]
    assert_seed_scoped_paths(
        seed,
        {
            "model.adapter": variant["model"]["adapter"],
            "output.adapter_dir": variant["output"]["adapter_dir"],
            "output.training_manifest": variant["output"]["training_manifest"],
        },
    )


def test_mixed_precision_flags_follow_the_compute_dtype():
    """Checks that the trainer precision flags follow the QLoRA compute dtype."""
    from vulndetect.sft.train import mixed_precision_flags

    fp16_cfg = {"quantization": {"compute_dtype": "float16"}}
    bf16_cfg = {"quantization": {"compute_dtype": "bfloat16"}}
    assert mixed_precision_flags(fp16_cfg) == (False, True)
    assert mixed_precision_flags(bf16_cfg) == (True, False)
    assert mixed_precision_flags({**fp16_cfg, "mixed_precision": False}) == (False, False)
    assert mixed_precision_flags({**bf16_cfg, "mixed_precision": False}) == (False, False)
    assert mixed_precision_flags({"quantization": {}}) == (False, True)


def test_cast_trainable_to_fp32_only_touches_trainable_params():
    """Checks that only trainable parameters are cast to float32."""
    import torch
    from torch import nn

    from vulndetect.sft.train import cast_trainable_to_fp32

    model = nn.Module()
    model.lora_a = nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
    model.lora_b = nn.Parameter(torch.zeros(2, 2, dtype=torch.float16))
    frozen = nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16), requires_grad=False)
    model.frozen = frozen

    seen = cast_trainable_to_fp32(model)

    assert seen == {"torch.bfloat16": 1, "torch.float16": 1}
    assert model.lora_a.dtype is torch.float32
    assert model.lora_b.dtype is torch.float32
    assert model.frozen.dtype is torch.bfloat16
    assert cast_trainable_to_fp32(model) == {"torch.float32": 2}

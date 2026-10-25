"""End-to-end tests of the baseline runner with an injected fake model."""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from vulndetect.baseline.run import run
from vulndetect.common.model import CHAT_DATE_STRING
from vulndetect.common.predictions import read_predictions
from vulndetect.data.build import to_records, write_jsonl
from vulndetect.data.schema import LABEL_VULN

N_TEST = 8


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
    """Temporary working directory with a built test split and baseline config."""
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
    write_jsonl(test_records, tmp_path / "dataset" / "test.jsonl")

    cfg = {
        "experiment": "baseline",
        "seed": 42,
        "model": {
            "provider": "huggingface",
            "name": "meta-llama/Llama-3.2-3B-Instruct",
            "temperature": 0.0,
            "max_new_tokens": 512,
            "batch_size": 8,
        },
        "dataset": {"name": "bigvul", "split": "test", "dir": "dataset"},
        "output": {"predictions_dir": "results/predictions/baseline"},
    }
    cfg_path = tmp_path / "baseline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path, test_records


def test_baseline_run_end_to_end(workspace, tmp_path):
    cfg_path, test_records = workspace
    model = OracleModel([r["label"] for r in test_records])
    report = run(cfg_path, model=model)

    assert model.calls == 1
    assert report["n"] == len(test_records)
    assert report["parse_failure_rate"] == 0.0
    assert report["macro"]["f1"] == pytest.approx(1.0)
    assert report["micro"]["accuracy"] == pytest.approx(1.0)

    out_dir = tmp_path / "results" / "predictions" / "baseline" / "seed42"
    preds = read_predictions(out_dir / "predictions.jsonl")
    assert [p["id"] for p in preds] == [r["id"] for r in test_records]
    assert all(p["pred_label"] == p["gold_label"] for p in preds)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["approach"] == "baseline"
    assert manifest["seed"] == 42
    assert manifest["prompt_version"] == "fig1-v3"
    assert manifest["decoding"]["batch_size"] == 8
    assert (out_dir / "metrics.json").exists()

    rows = list(csv.DictReader((tmp_path / "results" / "experiments.csv").open()))
    assert len(rows) == 1
    assert rows[0]["approach"] == "baseline"


def test_baseline_limit_smoke_run_not_logged(workspace, tmp_path):
    cfg_path, test_records = workspace
    limit = 3
    model = OracleModel([r["label"] for r in test_records[:limit]])
    report = run(cfg_path, limit=limit, model=model)
    assert report["n"] == limit
    assert (tmp_path / "results" / "predictions" / "baseline" / "seed42").exists()
    assert not (tmp_path / "results" / "experiments.csv").exists()


def test_baseline_val_style_run_not_logged(workspace, tmp_path):
    """Checks that a full run with log_experiments disabled writes no CSV row."""
    cfg_path, test_records = workspace
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["output"] = {
        "predictions_dir": "results/predictions/baseline_val",
        "log_experiments": False,
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    model = OracleModel([r["label"] for r in test_records])
    report = run(cfg_path, model=model)
    assert report["n"] == len(test_records)

    out_dir = tmp_path / "results" / "predictions" / "baseline_val" / "seed42"
    assert (out_dir / "predictions.jsonl").exists()
    assert (out_dir / "metrics.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["logged"] is False
    assert not (tmp_path / "results" / "experiments.csv").exists()


def test_baseline_authors_strict_wiring(workspace, tmp_path):
    """Checks the authors-strict-v1 prompt, system message and drop policy wiring."""
    from vulndetect.common.prompt import AUTHORS_STRICT_SYSTEM, render_authors_strict

    cfg_path, test_records = workspace
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["prompt"] = {"version": "authors-strict-v1"}
    cfg["parser_version"] = "authors-v1"
    cfg["eval"] = {"unparseable": "drop"}
    cfg["model"].update({"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 256})
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class RecordingOracle(OracleModel):
        def __init__(self, labels):
            super().__init__(labels)
            self.kwargs: dict = {}
            self.prompts: list[str] = []

        def generate(self, prompts, **kwargs):
            self.kwargs = kwargs
            self.prompts = list(prompts)
            return [f"Classification: {label}\nReasoning: test." for label in self.labels]

    model = RecordingOracle([r["label"] for r in test_records])
    report = run(cfg_path, model=model)

    assert model.kwargs.get("system") == AUTHORS_STRICT_SYSTEM
    assert model.prompts == [render_authors_strict(r) for r in test_records]
    assert report["macro"]["f1"] == pytest.approx(1.0)
    assert report["unparseable_policy"] == "drop"

    manifest = json.loads(
        (tmp_path / "results" / "predictions" / "baseline" / "seed42" / "manifest.json").read_text()
    )
    assert manifest["prompt_version"] == "authors-strict-v1"
    assert manifest["parser_version"] == "authors-v1"
    assert manifest["unparseable_policy"] == "drop"
    assert manifest["decoding"] == {
        "temperature": 0.6,
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 256,
        "batch_size": 8,
        "chat_date_string": CHAT_DATE_STRING,
    }
    metrics = json.loads(
        (tmp_path / "results" / "predictions" / "baseline" / "seed42" / "metrics.json").read_text()
    )
    assert metrics["parser_version"] == "authors-v1"


def test_baseline_mixed_and_unparseable_outputs(workspace, tmp_path):
    cfg_path, test_records = workspace

    class NoisyModel:
        def generate(self, prompts: list[str], **kwargs) -> list[str]:
            outs = []
            for i in range(len(prompts)):
                if i == 0:
                    outs.append("I cannot tell.")
                elif i % 2:
                    outs.append("Not Vulnerable — pure refactor.")
                else:
                    outs.append("Vulnerable: unbounded copy.")
            return outs

    report = run(cfg_path, model=NoisyModel())
    assert report["n_unparseable"] == 1
    preds = read_predictions(
        tmp_path / "results" / "predictions" / "baseline" / "seed42" / "predictions.jsonl"
    )
    assert preds[0]["pred_label"] is None
    assert {p["pred_label"] for p in preds[1:]} == {LABEL_VULN, "Not Vulnerable"}

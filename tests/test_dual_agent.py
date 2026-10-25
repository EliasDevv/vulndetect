"""Tests for Dual-Agent prompt assembly and the end-to-end runner with fake models."""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from vulndetect.common.parsing import parse_label
from vulndetect.common.predictions import read_predictions
from vulndetect.data.build import to_records, write_jsonl
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
from vulndetect.dual_agent.prompt import render_authors_detector, render_authors_validator
from vulndetect.dual_agent.run import resolve_final_label, run

N_TEST = 8

# Detector output containing literal braces.
BRACED_ASSESSMENT = "Vulnerable. The loop `while (i--) { buf[i] = 0; }` underflows."


class ScriptedDualModel:
    """Fake model returning scripted detector outputs, then validator outputs."""

    def __init__(self, detector_outputs: list[str], validator_outputs: list[str]):
        self.scripts = [detector_outputs, validator_outputs]
        self.seen: list[list[str]] = []

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        assert len(self.seen) < len(self.scripts), "more generate() calls than scripted passes"
        outputs = self.scripts[len(self.seen)]
        assert len(prompts) == len(outputs)
        self.seen.append(list(prompts))
        return list(outputs)


@pytest.fixture
def workspace(tmp_path, raw_df, base_cfg, monkeypatch):
    """Temporary working directory with a built test split and dual-agent config."""
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
        "experiment": "dual_agent",
        "seed": 42,
        "dual_agent": {
            "prompt_set": "authors-v1",
            "final_label_rule": "validator",
        },
        "model": {
            "provider": "huggingface",
            "name": "meta-llama/Llama-3.2-3B-Instruct",
            "temperature": 0.0,
            "max_new_tokens": 512,
            "batch_size": 8,
        },
        "dataset": {"name": "bigvul", "split": "test", "dir": "dataset"},
        "output": {"predictions_dir": "results/predictions/dual_agent"},
    }
    cfg_path = tmp_path / "dual_agent.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path, test_records


def rewrite_cfg(cfg_path, mutate) -> None:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mutate(cfg)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def wrong(label: str) -> str:
    return LABEL_SAFE if label == LABEL_VULN else LABEL_VULN


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
    }


def test_result_line_parses():
    """Checks that a "Result: ..." verdict line parses to the right label."""
    assert parse_label("Result: Vulnerable\nThe analyst was right.") == LABEL_VULN
    assert parse_label("Result: Not Vulnerable\nThe claim is speculative.") == LABEL_SAFE


def test_resolve_final_label_rules():
    assert resolve_final_label(LABEL_VULN, LABEL_SAFE, "validator") == LABEL_SAFE
    assert resolve_final_label(LABEL_VULN, None, "validator") is None
    assert resolve_final_label(LABEL_VULN, None, "validator_then_detector") == LABEL_VULN
    assert resolve_final_label(None, LABEL_SAFE, "validator_then_detector") == LABEL_SAFE
    with pytest.raises(ValueError, match="final_label_rule"):
        resolve_final_label(LABEL_VULN, LABEL_SAFE, "majority")


def test_dual_agent_run_end_to_end(workspace, tmp_path):
    cfg_path, test_records = workspace
    detector_outputs = [f"{wrong(r['label'])}. Reasoning: shallow guess." for r in test_records]
    validator_outputs = [
        f"Result: {r['label']}\nThe first analyst's reasoning was flawed." for r in test_records
    ]
    model = ScriptedDualModel(detector_outputs, validator_outputs)
    report = run(cfg_path, model=model)

    assert len(model.seen) == 2
    assert model.seen[0] == [render_authors_detector(r) for r in test_records]
    for prompt, rec, det in zip(model.seen[1], test_records, detector_outputs, strict=True):
        assert prompt == render_authors_validator(rec, det)
        assert rec["func_before"] in prompt
        assert det in prompt

    assert report["n"] == len(test_records)
    assert report["parse_failure_rate"] == 0.0
    assert report["macro"]["f1"] == pytest.approx(1.0)

    out_dir = tmp_path / "results" / "predictions" / "dual_agent" / "seed42"
    preds = read_predictions(out_dir / "predictions.jsonl")
    assert [p["id"] for p in preds] == [r["id"] for r in test_records]
    assert all(p["pred_label"] == p["gold_label"] for p in preds)
    assert [p["raw_output"] for p in preds] == validator_outputs

    agents = [
        json.loads(line)
        for line in (out_dir / "agents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [a["id"] for a in agents] == [r["id"] for r in test_records]
    for a, rec, det in zip(agents, test_records, detector_outputs, strict=True):
        assert a["detector_output"] == det
        assert a["detector_label"] == wrong(rec["label"])
        assert a["validator_label"] == rec["label"] == a["final_label"]
        assert a["flipped"] is True

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["approach"] == "dual_agent"
    assert manifest["prompt_version"] == "authors-detector-v1"
    assert manifest["decoding"]["batch_size"] == 8
    assert manifest["validator_prompt_version"] == "authors-validator-v1"
    da = manifest["dual_agent"]
    assert da["final_label_rule"] == "validator"
    assert da["rounds"] == 1
    assert da["stats"]["n_flipped"] == len(test_records)
    assert da["stats"]["flip_rate"] == 1.0
    assert da["stats"]["detector_validator_agreement"] == 0.0
    assert da["stats"]["n_detector_unparseable"] == 0
    assert da["stats"]["n_validator_unparseable"] == 0
    assert (out_dir / "metrics.json").exists()

    rows = list(csv.DictReader((tmp_path / "results" / "experiments.csv").open()))
    assert len(rows) == 1
    assert rows[0]["approach"] == "dual_agent"
    for key in ("validator_prompt_version", "prompt_set", "final_label_rule"):
        assert key in rows[0]["params"]


def test_detector_unparseable_still_validated(workspace, tmp_path):
    cfg_path, test_records = workspace
    detector_outputs = ["I cannot tell."] + [
        f"{r['label']}. Reasoning: ok." for r in test_records[1:]
    ]
    validator_outputs = [f"Result: {r['label']}\nConfirmed." for r in test_records]
    model = ScriptedDualModel(detector_outputs, validator_outputs)
    report = run(cfg_path, model=model)

    assert "I cannot tell." in model.seen[1][0]
    assert report["parse_failure_rate"] == 0.0
    out_dir = tmp_path / "results" / "predictions" / "dual_agent" / "seed42"
    first_agent = json.loads((out_dir / "agents.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_agent["detector_label"] is None
    assert first_agent["flipped"] is False
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["dual_agent"]["stats"]["n_detector_unparseable"] == 1


def test_validator_unparseable_final_label_rules(workspace, tmp_path):
    cfg_path, test_records = workspace
    detector_outputs = [f"{r['label']}. Reasoning: ok." for r in test_records]
    validator_outputs = ["Audit inconclusive."] + [
        f"Result: {r['label']}\nConfirmed." for r in test_records[1:]
    ]

    run(cfg_path, model=ScriptedDualModel(detector_outputs, validator_outputs))
    out_dir = tmp_path / "results" / "predictions" / "dual_agent" / "seed42"
    preds = read_predictions(out_dir / "predictions.jsonl")
    assert preds[0]["pred_label"] is None
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["dual_agent"]["stats"]["n_validator_unparseable"] == 1

    rewrite_cfg(
        cfg_path, lambda c: c["dual_agent"].update(final_label_rule="validator_then_detector")
    )
    run(cfg_path, model=ScriptedDualModel(detector_outputs, validator_outputs))
    preds = read_predictions(out_dir / "predictions.jsonl")
    assert preds[0]["pred_label"] == test_records[0]["label"]


def test_missing_or_bad_dual_agent_config_fails(workspace):
    cfg_path, test_records = workspace

    rewrite_cfg(cfg_path, lambda c: c.pop("dual_agent"))
    with pytest.raises(KeyError):
        run(cfg_path, model=ScriptedDualModel([], []))

    cfg_path2 = cfg_path.parent / "dual_agent2.yaml"
    cfg_path2.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    rewrite_cfg(
        cfg_path2,
        lambda c: c.update(dual_agent={"prompt_set": "authors-v1"}),
    )
    with pytest.raises(KeyError):
        run(cfg_path2, model=ScriptedDualModel([], []))

    rewrite_cfg(
        cfg_path2,
        lambda c: c.update(
            dual_agent={
                "prompt_set": "authors-v1",
                "final_label_rule": "majority",
            }
        ),
    )
    with pytest.raises(ValueError, match="final_label_rule"):
        run(cfg_path2, model=ScriptedDualModel([], []))

    rewrite_cfg(
        cfg_path2,
        lambda c: c.update(
            dual_agent={
                "prompt_set": "authors-v2",
                "final_label_rule": "validator",
            }
        ),
    )
    with pytest.raises(ValueError, match="prompt_set"):
        run(cfg_path2, model=ScriptedDualModel([], []))


def test_limit_smoke_run_not_logged(workspace, tmp_path):
    cfg_path, test_records = workspace
    limit = 3
    subset = test_records[:limit]
    model = ScriptedDualModel(
        [f"{r['label']}. Reasoning: ok." for r in subset],
        [f"Result: {r['label']}\nConfirmed." for r in subset],
    )
    report = run(cfg_path, limit=limit, model=model)
    assert report["n"] == limit
    assert (tmp_path / "results" / "predictions" / "dual_agent" / "seed42").exists()
    assert not (tmp_path / "results" / "experiments.csv").exists()


def test_val_style_run_not_logged(workspace, tmp_path):
    """Checks that a full run with log_experiments disabled writes no CSV row."""
    cfg_path, test_records = workspace
    rewrite_cfg(
        cfg_path,
        lambda c: c["output"].update(
            predictions_dir="results/predictions/dual_agent_val", log_experiments=False
        ),
    )
    model = ScriptedDualModel(
        [f"{r['label']}. Reasoning: ok." for r in test_records],
        [f"Result: {r['label']}\nConfirmed." for r in test_records],
    )
    report = run(cfg_path, model=model)
    assert report["n"] == len(test_records)

    out_dir = tmp_path / "results" / "predictions" / "dual_agent_val" / "seed42"
    assert (out_dir / "predictions.jsonl").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["logged"] is False
    assert not (tmp_path / "results" / "experiments.csv").exists()


def test_authors_templates_match_fixtures(fixtures_dir):
    """Checks that the transcribed templates equal the extracted fixtures."""
    from vulndetect.dual_agent.prompt import (
        AUTHORS_DETECTOR_TEMPLATE,
        AUTHORS_VALIDATOR_TEMPLATE,
    )

    assert AUTHORS_DETECTOR_TEMPLATE == (fixtures_dir / "authors_detector_v1.txt").read_text(
        encoding="utf-8"
    )
    assert AUTHORS_VALIDATOR_TEMPLATE == (fixtures_dir / "authors_validator_v1.txt").read_text(
        encoding="utf-8"
    )


def test_render_authors_prompts():
    record = sample_record()
    detector = render_authors_detector(record)
    assert detector.startswith("You are an expert in cybersecurity")
    assert "- Commit Message: fix overflow" in detector
    assert record["func_before"] in detector

    validator = render_authors_validator(record, BRACED_ASSESSMENT)
    assert validator.startswith("You are an audit and validation agent")
    assert record["func_before"] in validator
    assert BRACED_ASSESSMENT in validator
    assert validator.rstrip().endswith("Result: <Vulnerable or Not Vulnerable>")


def test_authors_prompt_set_end_to_end(workspace, tmp_path):
    cfg_path, test_records = workspace
    detector_outputs = [f"{r['label']}. Reasoning: ok." for r in test_records]
    validator_outputs = [
        f"Issues Found: None\nRevised Reasoning: solid.\nResult: {r['label']}" for r in test_records
    ]
    model = ScriptedDualModel(detector_outputs, validator_outputs)
    report = run(cfg_path, model=model)

    assert model.seen[0] == [render_authors_detector(r) for r in test_records]
    for prompt, det in zip(model.seen[1], detector_outputs, strict=True):
        assert prompt.startswith("You are an audit and validation agent")
        assert det in prompt
    assert report["macro"]["f1"] == pytest.approx(1.0)

    out_dir = tmp_path / "results" / "predictions" / "dual_agent" / "seed42"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["prompt_version"] == "authors-detector-v1"
    assert manifest["validator_prompt_version"] == "authors-validator-v1"
    assert manifest["dual_agent"]["prompt_set"] == "authors-v1"


def test_run_is_deterministic(workspace, tmp_path):
    cfg_path, test_records = workspace
    detector_outputs = [f"{wrong(r['label'])}. Reasoning: guess." for r in test_records]
    validator_outputs = [f"Result: {r['label']}\nCorrected." for r in test_records]
    out_dir = tmp_path / "results" / "predictions" / "dual_agent" / "seed42"

    artifacts = []
    for _ in range(2):
        run(cfg_path, model=ScriptedDualModel(detector_outputs, validator_outputs))
        artifacts.append(
            (
                (out_dir / "predictions.jsonl").read_bytes(),
                (out_dir / "agents.jsonl").read_bytes(),
            )
        )
    assert artifacts[0] == artifacts[1]


class ValidatorOnlyRecorder:
    """Fake model that records the prompts of each generate() call."""

    def __init__(self):
        self.passes: list[list[str]] = []

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        self.passes.append(list(prompts))
        text = "Issues Found: None\nRevised Reasoning: fine.\nResult: Not Vulnerable"
        return [text] * len(prompts)


def _agents_file(tmp_path, records, text_for):
    import json as _json

    path = tmp_path / "prev" / "seed42" / "agents.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(_json.dumps({"id": r["id"], "detector_output": text_for(r)}) + "\n")
    (path.parent / "manifest.json").write_text(
        _json.dumps({"prompt_version": "authors-strict-v1"}), encoding="utf-8"
    )
    return path


def test_detector_outputs_from_skips_pass_one_and_audits_the_frozen_texts(workspace, tmp_path):
    cfg_path, test_records = workspace
    src = _agents_file(tmp_path, test_records, lambda r: f"FROZEN DETECTOR TEXT for {r['id']}")
    rewrite_cfg(cfg_path, lambda c: c["dual_agent"].__setitem__("detector_outputs_from", str(src)))
    model = ValidatorOnlyRecorder()
    run(cfg_path, model=model)
    assert len(model.passes) == 1
    for rec, prompt in zip(test_records, model.passes[0], strict=True):
        assert f"FROZEN DETECTOR TEXT for {rec['id']}" in prompt
    out_dir = tmp_path / "results" / "predictions" / "dual_agent" / "seed42"
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prompt_version"] == "authors-strict-v1"
    assert manifest["dual_agent"]["detector_outputs_from"]["path"] == str(src)
    assert len(manifest["dual_agent"]["detector_outputs_from"]["sha256"]) == 64
    agents = [json.loads(line) for line in (out_dir / "agents.jsonl").open(encoding="utf-8")]
    assert all(a["detector_output"].startswith("FROZEN DETECTOR TEXT") for a in agents)


def test_detector_outputs_from_must_cover_every_record(workspace, tmp_path):
    cfg_path, test_records = workspace
    src = _agents_file(tmp_path, test_records[:-1], lambda r: "x")
    rewrite_cfg(cfg_path, lambda c: c["dual_agent"].__setitem__("detector_outputs_from", str(src)))
    with pytest.raises(ValueError, match="no detector text"):
        run(cfg_path, model=ValidatorOnlyRecorder())

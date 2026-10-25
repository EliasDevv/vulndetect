"""Guards for the label-definition ablation: one variable, val only, no logging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).parent
CODE = HERE.parents[1]
sys.path.insert(0, str(HERE))

import labeldef  # noqa: E402

from vulndetect.common.predictions import write_predictions  # noqa: E402
from vulndetect.data.build import write_jsonl  # noqa: E402
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN, RECORD_FIELDS  # noqa: E402


def test_template_is_fig1_plus_the_authors_definition():
    from vulndetect.common.prompt import AUTHORS_STRICT_TEMPLATE, PROMPT_TEMPLATE

    assert labeldef.TEMPLATE.startswith(PROMPT_TEMPLATE)  # Fig. 1 byte-identical
    assert labeldef.TEMPLATE[len(PROMPT_TEMPLATE) :] == "\n" + labeldef.DEFINITION
    assert labeldef.DEFINITION in AUTHORS_STRICT_TEMPLATE  # sourced, not self-authored
    assert "{" not in labeldef.DEFINITION and "}" not in labeldef.DEFINITION


def test_config_differs_from_baseline_val_only_in_name_and_output():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((CODE / "configs" / "baseline_val.yaml").read_text(encoding="utf-8"))
    for key in ("seed", "model", "parser_version", "eval", "dataset"):
        assert cfg[key] == base[key], key
    assert "prompt" not in cfg  # the prompt lives in labeldef.py, not in the registry
    assert cfg["dataset"]["split"] == "val"
    assert cfg["output"]["log_experiments"] is False
    assert cfg["output"]["predictions_dir"] != base["output"]["predictions_dir"]


def _record(i: int, cwe: str, vul: int) -> dict:
    before = f"int f{i}() {{ return {i}; }}"
    rec = {
        "id": f"c{i}:{i:012x}",
        "code": before,
        "func_before": before,
        "func_after": before if vul == 0 else before.replace("return", "return 1 +"),
        "commit_message": "" if i == 0 else "fix bounds check",
        "cve_id": "CVE-2020-0001",
        "cwe": cwe,
        "project": "proj",
        "lang": "C",
        "label": LABEL_VULN if vul else LABEL_SAFE,
        "vul": vul,
        "commit_id": f"c{i}",
        "split": "val",
    }
    return {k: rec[k] for k in RECORD_FIELDS}


class EchoModel:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        assert not kwargs, "must call generate like the fig1-v3 baseline (no system turn)"
        self.prompts += prompts
        return ["The change fixes a vulnerability. Therefore: Vulnerable."] * len(prompts)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [_record(0, "CWE-119", 1), _record(1, "CWE-119", 0)]
    records += [_record(2, "CWE-20", 1), _record(3, "CWE-20", 0)]
    write_jsonl(records, tmp_path / "dataset" / "val.jsonl")
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["dir"] = "dataset"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path, cfg


def test_run_writes_val_artifacts_and_never_logs(workspace, tmp_path):
    path, cfg = workspace
    model = EchoModel()
    report = labeldef.run(path, model=model)
    assert report["n"] == 4
    assert all(p.endswith("\n" + labeldef.DEFINITION) for p in model.prompts)
    assert "- Commit Message: N/A" in model.prompts[0]  # same fallback as the baseline
    out = tmp_path / cfg["output"]["predictions_dir"] / "seed42"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["prompt_version"] == labeldef.LABELDEF_PROMPT_VERSION
    assert manifest["logged"] is False and manifest["dataset"]["split"] == "val"
    assert (out / "predictions.jsonl").exists() and (out / "metrics.json").exists()
    assert not (tmp_path / "results" / "experiments.csv").exists()


def test_smoke_run_writes_to_its_own_directory(workspace, tmp_path):
    path, cfg = workspace
    labeldef.run(path, limit=2, model=EchoModel())
    assert (tmp_path / (cfg["output"]["predictions_dir"] + "_smoke") / "seed42").is_dir()
    assert not (tmp_path / cfg["output"]["predictions_dir"]).exists()


@pytest.mark.parametrize("change", [{"split": "test"}, {"log": True}])
def test_run_refuses_test_split_and_logging(workspace, change):
    path, cfg = workspace
    if "split" in change:
        cfg["dataset"]["split"] = change["split"]
    else:
        cfg["output"]["log_experiments"] = True
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        labeldef.run(path, model=EchoModel())


def test_analyze_counts_fix_described_false_negatives(tmp_path):
    def pred(i, gold, text):
        return {
            "id": f"p{i}",
            "cwe": "CWE-119",
            "gold_label": gold,
            "pred_label": None,
            "raw_output": text,
        }

    fixed_but_no = "The change fixes a vulnerability. Therefore it is Not Vulnerable."
    unchanged = "The code before and after is identical. Therefore: Not Vulnerable."
    base = [
        pred(0, LABEL_VULN, fixed_but_no),
        pred(1, LABEL_VULN, fixed_but_no),
        pred(2, LABEL_SAFE, unchanged),
        pred(3, LABEL_SAFE, unchanged),
    ]
    ablation = [
        pred(0, LABEL_VULN, "Vulnerable: it fixes a vulnerability."),
        base[1],
        base[2],
        base[3],
    ]
    for name, rows in (("a", ablation), ("b", base)):
        write_predictions(rows, tmp_path / name / "predictions.jsonl")
    res = labeldef.analyze(
        tmp_path / "a",
        references={labeldef.REFERENCE: str(tmp_path / "b")},
        out=tmp_path / "analysis.json",
    )
    base_d, abl_d = res["runs"][labeldef.REFERENCE], res["runs"][labeldef.THIS]
    assert base_d["fn_text_names_fix"] == 2 and abl_d["fn_text_names_fix"] == 1
    assert base_d["unchanged_statement"]["n_written"] == 2
    assert res["mcnemar_vs_baseline_val"][labeldef.THIS]["b"] == 1
    assert (tmp_path / "analysis.json").exists()

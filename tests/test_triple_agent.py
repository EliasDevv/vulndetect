"""Offline tests for the Triple-Agent prompts, party, loop, runner and configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from test_gan_loop import GAN_CFG, VERDICT_TEXTS, RoleFake, balanced_test_records, rewrite_cfg

from vulndetect.common.predictions import read_predictions
from vulndetect.common.prompt import render_prompt
from vulndetect.data.build import build_dataset, to_records, write_jsonl
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
from vulndetect.dual_agent.prompt import render_authors_detector, render_authors_validator
from vulndetect.gan import parsing as gp
from vulndetect.gan.loop import run_loop as gan_run_loop
from vulndetect.gan.prompts import DETECTOR_BLOCK_SEPARATOR, DETECTOR_HEADER, render_detector_prompt
from vulndetect.gan.prompts import PROMPT_VERSIONS as GAN_PROMPT_VERSIONS
from vulndetect.gan.strategies import StrategyMemory
from vulndetect.triple_agent import run as triple_run
from vulndetect.triple_agent.loop import LOOP_APPROACH, run_loop
from vulndetect.triple_agent.party import DualAgentParty, party_factory_from_config
from vulndetect.triple_agent.prompts import (
    MEMORY_SCOPES,
    PROMPT_VERSIONS,
    TRANSCRIPT_DETECTOR_SEP,
    TRANSCRIPT_VALIDATOR_SEP,
    render_transcript,
    render_validator_prompt,
)

SOURCE_FUNC = "int copy_in(char *src, int n) { char buf[8]; memcpy(buf, src, n); return 0; }"
DETECTOR_TEXT = "Vulnerable. The call `memcpy(buf, src, n)` with n > 8 overflows {buf}."

# Phrase that only occurs in the validator prompt; the fake dispatches on it.
VALIDATOR_MARKER = "You are an audit and validation agent"

VALIDATOR_TEXTS = {
    LABEL_VULN: (
        "Issues Found: None\n"
        "Revised Reasoning: the pre-change function copies without a bound.\n"
        "Result: Vulnerable"
    ),
    LABEL_SAFE: (
        "Issues Found: the commit message was taken as evidence.\n"
        "Revised Reasoning: the change is a rename.\n"
        "Result: Not Vulnerable"
    ),
    None: VERDICT_TEXTS[None],
}


def record(label: str = LABEL_VULN, ident: str = "c1:abc") -> dict:
    return {
        "id": ident,
        "code": SOURCE_FUNC,
        "func_before": SOURCE_FUNC,
        "func_after": SOURCE_FUNC.replace("n)", "8)"),
        "commit_message": "refactor helper {inline}",
        "cve_id": "CVE-2020-0001",
        "cwe": "CWE-119",
        "project": "proj",
        "lang": "C",
        "label": label,
        "vul": 1 if label == LABEL_VULN else 0,
        "commit_id": "c1",
        "split": "train",
    }


def memory(entries: list[str]) -> str:
    return StrategyMemory(role="detector", entries=list(entries)).render()


class TwoPassFake:
    """Fake model returning detector outputs on call 1 and validator outputs on call 2."""

    def __init__(self, detector_outputs: list[str], validator_outputs: list[str]):
        self.scripts = [detector_outputs, validator_outputs]
        self.seen: list[list[str]] = []

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        assert len(self.seen) < len(self.scripts), "more generate() calls than scripted passes"
        outputs = self.scripts[len(self.seen)]
        assert len(prompts) == len(outputs)
        self.seen.append(list(prompts))
        return list(outputs)


class TripleRoleFake(RoleFake):
    """RoleFake plus the validator stage, which may answer differently from the detector."""

    ROLE_MARKERS = {**RoleFake.ROLE_MARKERS, "validate": VALIDATOR_MARKER}

    def __init__(self, verdict=LABEL_VULN, *, validator_verdict=LABEL_VULN, **kwargs):
        super().__init__(verdict, **kwargs)
        self.validator_verdict = validator_verdict

    def _validate(self, prompt: str) -> str:
        return VALIDATOR_TEXTS[self.validator_verdict]


def disagreeing_fake() -> TripleRoleFake:
    """Returns a fake where the detector says Vulnerable and the validator Not Vulnerable."""
    return TripleRoleFake(verdict=LABEL_VULN, validator_verdict=LABEL_SAFE)


def test_stage1_with_empty_memory_is_the_authors_detector_prompt():
    rec = record()
    assert render_detector_prompt(rec, "") == render_authors_detector(rec) == render_prompt(rec)


@pytest.mark.parametrize("scope", MEMORY_SCOPES)
def test_stage2_with_empty_memory_is_the_authors_validator_prompt(scope):
    rec = record()
    prompt = render_validator_prompt(rec, DETECTOR_TEXT, "", scope=scope)
    assert prompt == render_authors_validator(rec, DETECTOR_TEXT)


def test_stage2_detector_only_scope_ignores_the_memory():
    rec = record()
    prompt = render_validator_prompt(
        rec, DETECTOR_TEXT, memory(["Check bounds."]), scope="detector_only"
    )
    assert prompt == render_authors_validator(rec, DETECTOR_TEXT)


def test_stage2_both_prepends_the_gan_block_and_keeps_the_validator_prompt_as_suffix():
    rec = record()
    strategies = memory(["Check bounds.", "Distrust the commit message."])
    prompt = render_validator_prompt(rec, DETECTOR_TEXT, strategies, scope="both")
    base = render_authors_validator(rec, DETECTOR_TEXT)
    assert prompt == DETECTOR_HEADER + strategies + DETECTOR_BLOCK_SEPARATOR + base
    assert prompt.endswith(base) and prompt.count(base) == 1
    block = prompt[: len(prompt) - len(base)]
    assert render_detector_prompt(rec, strategies).startswith(block)
    assert "1. Check bounds." in prompt and "{buf}" in prompt


def test_unknown_scope_fails():
    with pytest.raises(ValueError, match="memory_scope"):
        render_validator_prompt(record(), DETECTOR_TEXT, "x", scope="validator_only")


def test_transcript_keeps_both_stages_and_never_looks_like_a_marker():
    transcript = render_transcript(DETECTOR_TEXT, "Result: {Not Vulnerable}")
    assert transcript == (
        TRANSCRIPT_DETECTOR_SEP
        + DETECTOR_TEXT
        + TRANSCRIPT_VALIDATOR_SEP
        + "Result: {Not Vulnerable}"
    )
    assert gp.split_sections(transcript) == {}
    for needle in (*RoleFake.ROLE_MARKERS.values(), VALIDATOR_MARKER):
        assert needle not in transcript


def test_prompt_versions_are_disjoint_from_the_gan_versions():
    assert set(PROMPT_VERSIONS).isdisjoint(GAN_PROMPT_VERSIONS)
    assert PROMPT_VERSIONS["validator"] == "authors-validator-v1"
    assert all(v.startswith("triple-") for k, v in PROMPT_VERSIONS.items() if k != "validator")


def test_party_runs_two_passes_and_reports_the_validator_verdict():
    recs = [record(LABEL_VULN, "c1:a"), record(LABEL_SAFE, "c1:b")]
    fake = TwoPassFake(
        [VERDICT_TEXTS[LABEL_VULN]] * 2, [VALIDATOR_TEXTS[LABEL_VULN], VALIDATOR_TEXTS[LABEL_SAFE]]
    )
    party = DualAgentParty(fake, memory_scope="both", final_label_rule="validator")
    strategies = memory(["Check bounds."])
    dets = party.detect_batch(recs, strategies)

    assert fake.seen[0] == [render_detector_prompt(r, strategies) for r in recs]
    assert fake.seen[1] == [
        render_validator_prompt(r, VERDICT_TEXTS[LABEL_VULN], strategies, scope="both")
        for r in recs
    ]
    assert [d.label for d in dets] == [LABEL_VULN, LABEL_SAFE]
    assert dets[1].raw == render_transcript(VERDICT_TEXTS[LABEL_VULN], VALIDATOR_TEXTS[LABEL_SAFE])
    assert dets[1].trace == {
        "detector_label": LABEL_VULN,
        "detector_raw": VERDICT_TEXTS[LABEL_VULN],
        "validator_label": LABEL_SAFE,
        "validator_raw": VALIDATOR_TEXTS[LABEL_SAFE],
        "flipped": True,
    }
    assert dets[0].trace["flipped"] is False

    stats = party.stats()
    assert stats["n_detections"] == 2 and stats["n_flipped"] == 1 and stats["flip_rate"] == 0.5
    assert stats["n_agree"] == 1 and stats["detector_validator_agreement"] == 0.5
    assert stats["agreement_matrix"][LABEL_VULN] == {LABEL_VULN: 1, LABEL_SAFE: 1, "None": 0}
    assert stats["flips_by_detector_correctness"] == {
        "detector_right_validator_wrong": 0,
        "detector_wrong_validator_right": 1,
    }
    assert party.describe() == {
        "name": "dual_agent",
        "memory_scope": "both",
        "final_label_rule": "validator",
        "prompt_versions": PROMPT_VERSIONS,
    }


@pytest.mark.parametrize(
    ("rule", "expected"), [("validator", None), ("validator_then_detector", LABEL_VULN)]
)
def test_party_unparseable_validator_follows_the_final_label_rule(rule, expected):
    fake = TwoPassFake([VERDICT_TEXTS[LABEL_VULN]], [VERDICT_TEXTS[None]])
    party = DualAgentParty(fake, memory_scope="both", final_label_rule=rule)
    (det,) = party.detect_batch([record()], "")
    assert det.label == expected
    assert det.trace["validator_label"] is None and det.trace["flipped"] is False
    stats = party.stats()
    assert stats["n_validator_unparseable"] == 1 and stats["n_detector_unparseable"] == 0
    assert stats["agreement_matrix"][LABEL_VULN]["None"] == 1


def test_party_detect_is_a_batch_of_one_and_empty_batches_are_free():
    fake = TwoPassFake([VERDICT_TEXTS[LABEL_SAFE]], [VALIDATOR_TEXTS[LABEL_SAFE]])
    party = DualAgentParty(fake, memory_scope="detector_only", final_label_rule="validator")
    det = party.detect(record(), memory(["Check bounds."]))
    assert det.label == LABEL_SAFE and len(fake.seen) == 2
    assert fake.seen[1] == [render_authors_validator(record(), VERDICT_TEXTS[LABEL_SAFE])]
    assert party.detect_batch([], "") == [] and party.stats()["n_detections"] == 1


def test_party_rejects_unknown_values_and_the_factory_validates_before_any_model():
    with pytest.raises(ValueError, match="memory_scope"):
        DualAgentParty(object(), memory_scope="all", final_label_rule="validator")
    with pytest.raises(ValueError, match="final_label_rule"):
        DualAgentParty(object(), memory_scope="both", final_label_rule="majority")
    with pytest.raises(KeyError):
        party_factory_from_config({"gan": {}})
    with pytest.raises(KeyError):
        party_factory_from_config({"triple_agent": {"memory_scope": "both"}})
    with pytest.raises(ValueError, match="memory_scope"):
        party_factory_from_config(
            {"triple_agent": {"memory_scope": "x", "final_label_rule": "validator"}}
        )
    factory = party_factory_from_config(
        {"triple_agent": {"memory_scope": "both", "final_label_rule": "validator"}}
    )
    party = factory(object())
    assert isinstance(party, DualAgentParty) and party.memory_scope == "both"


TRIPLE_CFG = {
    **GAN_CFG,
    "experiment": "triple_agent_test",
    "triple_agent": {"memory_scope": "both", "final_label_rule": "validator"},
    "output": {
        "loop_dir": "results/triple_agent",
        "predictions_dir": "results/predictions/triple_agent",
        "log_experiments": True,
    },
}


@pytest.fixture
def records(raw_df, base_cfg) -> list[dict]:
    return to_records(build_dataset(raw_df, base_cfg))


@pytest.fixture
def ws(tmp_path, records, monkeypatch):
    """A tmp cwd with train and val splits and a triple-agent config, without a test split."""
    monkeypatch.chdir(tmp_path)
    write_jsonl([r for r in records if r["split"] == "train"], tmp_path / "dataset" / "train.jsonl")
    write_jsonl(balanced_test_records(records, "val"), tmp_path / "dataset" / "val.jsonl")
    cfg_path = tmp_path / "triple_agent.yaml"
    cfg_path.write_text(yaml.safe_dump(TRIPLE_CFG), encoding="utf-8")
    return cfg_path


@pytest.fixture
def ws_with_test(ws, records, tmp_path):
    write_jsonl(balanced_test_records(records, "test"), tmp_path / "dataset" / "test.jsonl")
    return ws


def read_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_triple_loop_uses_both_stages_and_learns_only_where_the_party_is_wrong(ws):
    fake = disagreeing_fake()
    manifest = run_loop(ws, generator_model=fake, detector_model=fake)

    assert manifest["approach"] == LOOP_APPROACH
    party = manifest["detecting_party"]
    assert party["name"] == "dual_agent" and party["memory_scope"] == "both"
    assert party["prompt_versions"] == PROMPT_VERSIONS
    assert manifest["gan_prompt_versions"] == {**GAN_PROMPT_VERSIONS, **PROMPT_VERSIONS}
    assert (
        manifest["phase_a"]["n_detector_wrong"] == 2
        and manifest["phase_a"]["n_detector_updates"] == 2
    )
    assert manifest["phase_a"]["n_generator_updates"] == 0
    assert manifest["final_memories"]["s_g"]["entries"] == []
    assert manifest["final_memories"]["s_d"]["entries"]
    # 2 Phase-A rounds + 2 Phase-B items + 3 snapshots x 4 probe records
    assert fake.calls["validate"] == fake.calls["detect"] == 2 + 2 + 3 * 4
    assert party["stats"]["n_detections"] == 16 and party["stats"]["n_flipped"] == 16
    assert party["stats"]["agreement_matrix"][LABEL_VULN][LABEL_SAFE] == 16

    rows = read_rows("results/triple_agent/seed42/rounds.jsonl")
    a_rows = [r for r in rows if r["phase"] == "A"]
    assert len(a_rows) == 2
    for row in a_rows:
        assert row["detector_label"] == LABEL_SAFE and row["correct"] is False
        assert row["detection_trace"]["detector_label"] == LABEL_VULN
        assert row["detection_trace"]["validator_label"] == LABEL_SAFE
        assert row["detection_trace"]["flipped"] is True
        assert TRANSCRIPT_DETECTOR_SEP in row["detector_raw"]
        assert TRANSCRIPT_VALIDATOR_SEP in row["detector_raw"]
        assert len(row["prompt_lengths"]["detection"]) == 2
    update_prompt = fake.prompts["detector_update"][0]
    assert TRANSCRIPT_VALIDATOR_SEP.strip() in update_prompt
    assert (
        VERDICT_TEXTS[LABEL_VULN] in update_prompt and VALIDATOR_TEXTS[LABEL_SAFE] in update_prompt
    )
    b_wrong = [r for r in rows if r["phase"] == "B" and not r["correct"]]
    assert b_wrong and all(TRANSCRIPT_VALIDATOR_SEP.strip() in p for p in fake.prompts["reflect"])

    generated = read_rows("results/triple_agent/seed42/generated.jsonl")
    assert all(
        g["detection_trace"]["detector_label"] == LABEL_VULN for g in generated if g["accepted"]
    )

    snap = json.loads(Path("results/triple_agent/seed42/snapshots/final.json").read_text())
    assert snap["detecting_party"] == {
        "name": "dual_agent",
        "memory_scope": "both",
        "final_label_rule": "validator",
        "prompt_versions": PROMPT_VERSIONS,
    }
    assert snap["prompt_versions"] == {**GAN_PROMPT_VERSIONS, **PROMPT_VERSIONS}
    probe = snap["drift_probe"]
    assert probe["vulnerable_share"] == 0.0 and probe["stage1"]["vulnerable_share"] == 1.0
    assert probe["stage1"]["n_unparseable"] == 0
    lengths = manifest["prompt_lengths"]
    assert set(lengths["by_role"]) == {"generator", "detection", "update", "reflector", "probe"}
    assert lengths["by_role"]["detection"]["n"] == 2 * (2 + 2)
    assert lengths["by_role"]["probe"]["n"] == 2 * 12


def test_triple_loop_fails_before_the_model_on_a_missing_party_block(ws):
    rewrite_cfg(ws, lambda cfg: cfg.pop("triple_agent"))
    with pytest.raises(KeyError):
        run_loop(ws, generator_model=disagreeing_fake(), detector_model=disagreeing_fake())


def loop_then_eval(ws, snapshot="final", *, limit=None):
    loop_fake = disagreeing_fake()
    run_loop(ws, generator_model=loop_fake, detector_model=loop_fake)
    eval_fake = disagreeing_fake()
    report = triple_run.run(ws, snapshot, limit=limit, model=eval_fake)
    return report, eval_fake


def load_test_records() -> list[dict]:
    return read_rows("dataset/test.jsonl")


def test_eval_of_round0_uses_byte_identical_dual_agent_prompts(ws_with_test):
    report, fake = loop_then_eval(ws_with_test, "round0000")
    recs = load_test_records()
    assert fake.prompts["detect"] == [render_authors_detector(r) for r in recs]
    assert fake.prompts["validate"] == [
        render_authors_validator(r, VERDICT_TEXTS[LABEL_VULN]) for r in recs
    ]

    out = Path("results/predictions/triple_agent/seed42/round0000")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["approach"] == "triple_agent" and manifest["logged"] is True
    assert manifest["prompt_version"] == "fig1-v3"
    assert manifest["validator_prompt_version"] == "authors-validator-v1"
    assert manifest["gan"]["prompt_equals_dual_agent"] is True
    assert manifest["gan"]["round"] == 0 and manifest["gan"]["n_strategies"] == 0
    assert manifest["gan"]["snapshot_detecting_party"]["name"] == "dual_agent"
    assert manifest["gan"]["vulnerable_prediction_share"] == 0.0
    assert manifest["triple_agent"]["memory_scope"] == "both"
    assert manifest["triple_agent"]["stats"]["flip_rate"] == 1.0
    assert manifest["triple_agent"]["stage1_vulnerable_prediction_share"] == 1.0
    assert report["n"] == len(recs)
    preds = read_predictions(out / "predictions.jsonl")
    assert len(preds) == len(recs)
    assert all(p["pred_label"] == LABEL_SAFE for p in preds)
    assert all(p["raw_output"] == VALIDATOR_TEXTS[LABEL_SAFE] for p in preds)


def test_eval_metrics_carry_the_stage1_block_and_the_agreement_matrix(ws_with_test):
    report, _ = loop_then_eval(ws_with_test, "final")
    recs = load_test_records()
    n = len(recs)
    n_vuln = sum(r["label"] == LABEL_VULN for r in recs)
    assert report["n"] == n == report["stage1"]["n"]
    assert report["micro"]["recall"] == 0.0 and report["stage1"]["micro"]["recall"] == 1.0
    agreement = report["stage1"]["agreement"]
    matrix = agreement["agreement_matrix"]
    assert sum(sum(row.values()) for row in matrix.values()) == n
    assert matrix[LABEL_VULN][LABEL_SAFE] == n
    assert agreement["flips_by_detector_correctness"] == {
        "detector_right_validator_wrong": n_vuln,
        "detector_wrong_validator_right": n - n_vuln,
    }
    metrics = json.loads(
        (Path("results/predictions/triple_agent/seed42/final") / "metrics.json").read_text()
    )
    assert metrics["parser_version"] == "d09-v2" == metrics["stage1"]["parser_version"]
    assert metrics["stage1"]["macro"]["recall"] == 1.0


def test_eval_of_the_final_snapshot_prepends_the_block_on_both_stages(ws_with_test):
    _, fake = loop_then_eval(ws_with_test, "final")
    manifest = json.loads(
        (Path("results/predictions/triple_agent/seed42/final") / "manifest.json").read_text()
    )
    strategies = manifest["gan"]["strategies"]
    assert strategies, "the loop should have filled S_D"
    assert manifest["gan"]["prompt_equals_dual_agent"] is False
    recs = load_test_records()
    for prompt, rec in zip(fake.prompts["detect"], recs, strict=True):
        assert prompt.startswith(DETECTOR_HEADER) and prompt.endswith(render_prompt(rec))
        assert strategies[0] in prompt
    for prompt, rec in zip(fake.prompts["validate"], recs, strict=True):
        assert prompt.startswith(DETECTOR_HEADER)
        assert prompt.endswith(render_authors_validator(rec, VERDICT_TEXTS[LABEL_VULN]))
    lengths = manifest["prompt_lengths"]
    assert (
        set(lengths["by_pass"]) == {"detector", "validator"} and lengths["context_window"] is None
    )
    assert lengths["by_pass"]["detector"]["max_chars"] == max(
        len(p) for p in fake.prompts["detect"]
    )
    assert lengths["by_pass"]["validator"]["max_chars"] == max(
        len(p) for p in fake.prompts["validate"]
    )
    assert lengths["by_pass"]["validator"]["n"] == len(recs)
    assert lengths["by_pass"]["validator"]["n_over_context"] == 0


def test_eval_detector_only_scope_leaves_the_validator_prompt_untouched(ws_with_test):
    rewrite_cfg(
        ws_with_test, lambda cfg: cfg["triple_agent"].update({"memory_scope": "detector_only"})
    )
    _, fake = loop_then_eval(ws_with_test, "final")
    recs = load_test_records()
    assert fake.prompts["validate"] == [
        render_authors_validator(r, VERDICT_TEXTS[LABEL_VULN]) for r in recs
    ]
    assert all(p.startswith(DETECTOR_HEADER) for p in fake.prompts["detect"])


def test_eval_writes_agents_jsonl_and_one_experiments_row_per_snapshot(ws_with_test):
    fake = disagreeing_fake()
    run_loop(ws_with_test, generator_model=fake, detector_model=fake)
    reports = triple_run.run_all_snapshots(ws_with_test, model=disagreeing_fake())
    assert set(reports) == {"round0000", "round0002", "final"}

    rows = list(csv.DictReader(Path("results/experiments.csv").open(encoding="utf-8")))
    assert len(rows) == 3 and {row["approach"] for row in rows} == {"triple_agent"}
    params = [json.loads(row["params"]) for row in rows]
    assert [p["snapshot"] for p in params] == ["round0000", "round0002", "final"]
    assert [p["round"] for p in params] == [0, 2, None]
    assert all(p["memory_scope"] == "both" and p["final_label_rule"] == "validator" for p in params)
    assert all(p["validator_block_version"] == "triple-validator-block-v1" for p in params)
    assert all(p["prompt_version"] == "fig1-v3" for p in params)
    metrics = [json.loads(row["metrics"]) for row in rows]
    assert all("stage1_macro_f1" in m and m["flip_rate"] == 1.0 for m in metrics)
    assert all(row["output"].startswith("results/predictions/triple_agent/seed42/") for row in rows)

    agents = read_rows("results/predictions/triple_agent/seed42/final/agents.jsonl")
    recs = load_test_records()
    assert [a["id"] for a in agents] == [r["id"] for r in recs]
    assert set(agents[0]) == {
        "id",
        "detector_label",
        "detector_output",
        "validator_label",
        "final_label",
        "flipped",
    }
    assert all(
        a["detector_label"] == LABEL_VULN and a["final_label"] == LABEL_SAFE and a["flipped"]
        for a in agents
    )


def test_eval_limit_and_opt_out_stay_out_of_the_log(ws_with_test):
    report, _ = loop_then_eval(ws_with_test, "final", limit=4)
    assert report["n"] == 4 and not Path("results/experiments.csv").exists()
    rewrite_cfg(ws_with_test, lambda cfg: cfg["output"].update({"log_experiments": False}))
    triple_run.run(ws_with_test, "final", model=disagreeing_fake())
    assert not Path("results/experiments.csv").exists()


def test_eval_reads_the_gan_snapshots_for_the_transfer_control(ws_with_test):
    """Checks that the party can evaluate snapshots written by the GAN loop."""
    gan_cfg = Path("gan.yaml")
    gan_cfg.write_text(yaml.safe_dump(GAN_CFG), encoding="utf-8")
    fake = RoleFake(verdict=LABEL_SAFE)
    gan_run_loop(gan_cfg, generator_model=fake, detector_model=fake)
    rewrite_cfg(
        ws_with_test,
        lambda cfg: cfg["output"].update(
            {"snapshots_from": "results/gan", "log_experiments": False}
        ),
    )
    triple_run.run(ws_with_test, "final", model=disagreeing_fake())
    manifest = json.loads(
        (Path("results/predictions/triple_agent/seed42/final") / "manifest.json").read_text()
    )
    assert manifest["gan"]["snapshots_from"] == "results/gan"
    assert manifest["gan"]["snapshot_detecting_party"]["name"] == "single_detector"
    assert manifest["gan"]["strategies"]


def test_eval_without_a_loop_run_points_at_the_loop(ws_with_test):
    with pytest.raises(FileNotFoundError, match="run the GAN loop first"):
        triple_run.run(ws_with_test, "final", model=disagreeing_fake())


class KwargsTripleFake(TripleRoleFake):
    """TripleRoleFake that also records the decoding overrides it was called with."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs: list[dict] = []

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        self.kwargs.append(dict(kwargs))
        return super().generate(prompts, **kwargs)


def test_eval_decoding_follows_the_eval_block_not_the_loop(ws_with_test):
    """Checks that snapshot evaluation uses the eval decoding block when present."""
    loop_fake = disagreeing_fake()
    run_loop(ws_with_test, generator_model=loop_fake, detector_model=loop_fake)

    fake = KwargsTripleFake(verdict=LABEL_VULN, validator_verdict=LABEL_SAFE)
    triple_run.run(ws_with_test, "final", model=fake)
    assert fake.kwargs and all(k["temperature"] == 0.0 and k["top_p"] is None for k in fake.kwargs)
    manifest = json.loads(
        (Path("results/predictions/triple_agent/seed42/final") / "manifest.json").read_text()
    )
    assert manifest["decoding"]["eval_override"] is False

    rewrite_cfg(ws_with_test, lambda cfg: cfg.update({"eval": {"temperature": 0.6, "top_p": 0.9}}))
    fake = KwargsTripleFake(verdict=LABEL_VULN, validator_verdict=LABEL_SAFE)
    triple_run.run(ws_with_test, "final", model=fake)
    assert fake.kwargs and all(k["temperature"] == 0.6 and k["top_p"] == 0.9 for k in fake.kwargs)
    manifest = json.loads(
        (Path("results/predictions/triple_agent/seed42/final") / "manifest.json").read_text()
    )
    assert manifest["decoding"]["temperature"] == 0.6 and manifest["decoding"]["top_p"] == 0.9
    assert manifest["decoding"]["eval_override"] is True
    assert manifest["decoding"]["loop_detector_temperature"] == 0.0


CONFIG_DIR = Path(__file__).parents[1] / "configs"
TRIPLE_LADDER = ("triple_agent", "triple_agent_val", "triple_agent_smoke")


def load_cfg(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", TRIPLE_LADDER)
def test_shipped_configs_provide_every_key_the_loop_and_the_party_require(name):
    cfg = load_cfg(name)
    assert set(cfg["triple_agent"]) == {"memory_scope", "final_label_rule"}
    party_factory_from_config(cfg)
    gan = cfg["gan"]
    for key in ("rounds", "snapshot_rounds", "max_new_strategies_per_update"):
        assert key in gan
    assert set(gan["memory"]) >= {"policy", "max_entries", "max_entry_words", "dedup"}
    assert set(gan["generation"]) >= {"validate", "temperature", "top_p", "max_new_tokens"}
    assert set(gan["phase_b"]) >= {"enabled", "n_items"}
    assert set(gan["drift_probe"]) >= {"enabled", "split", "n"}
    assert gan["drift_probe"]["split"] == "val" and gan["phase_b"]["enabled"] is True
    assert max(gan["snapshot_rounds"]) <= gan["rounds"] and 0 in gan["snapshot_rounds"]
    for key in ("loop_dir", "predictions_dir"):
        assert key in cfg["output"]
    assert cfg["dataset"]["train_split"] == "train"
    assert cfg["model"]["temperature"] == 0.0 and cfg["model"]["revision"]


def test_triple_agent_eval_decoding_matches_the_shared_baseline_decoding():
    """Checks that the eval decoding matches the baseline while the loop stays greedy."""
    base = load_cfg("baseline")["model"]
    for name in TRIPLE_LADDER:
        cfg = load_cfg(name)
        assert (cfg["eval"]["temperature"], cfg["eval"]["top_p"]) == (
            base["temperature"],
            base["top_p"],
        )
        assert cfg["eval"]["unparseable"] == load_cfg("baseline")["eval"]["unparseable"]
        assert cfg["model"]["temperature"] == 0.0


def test_official_config_evaluates_on_test_logs_and_reads_the_val_snapshots():
    cfg, val, smoke = (
        load_cfg("triple_agent"),
        load_cfg("triple_agent_val"),
        load_cfg("triple_agent_smoke"),
    )
    assert cfg["experiment"] == "triple_agent" and cfg["dataset"]["split"] == "test"
    assert cfg["output"].get("log_experiments", True) is True
    assert cfg["output"]["predictions_dir"] == "results/predictions/triple_agent"
    assert cfg["output"]["snapshots_from"] == val["output"]["loop_dir"]
    assert "snapshots_from" not in val["output"] and "snapshots_from" not in smoke["output"]
    assert cfg["triple_agent"] == {"memory_scope": "both", "final_label_rule": "validator"}


@pytest.mark.parametrize(
    "stage", ["triple_agent_val", "triple_agent_smoke", "triple_agent_val_ganmem"]
)
def test_diagnostic_configs_never_touch_test_and_never_log(stage):
    cfg, official = load_cfg(stage), load_cfg("triple_agent")
    assert cfg["experiment"] == stage
    assert cfg["dataset"]["split"] == "val"
    assert cfg["output"]["log_experiments"] is False
    assert cfg["output"]["predictions_dir"] != official["output"]["predictions_dir"]
    assert cfg["output"].get("loop_dir") != official["output"]["loop_dir"]


def test_val_is_a_true_dress_rehearsal_and_the_loop_is_the_gan_loop():
    base, val, smoke = (
        load_cfg("triple_agent"),
        load_cfg("triple_agent_val"),
        load_cfg("triple_agent_smoke"),
    )
    assert val["seed"] == base["seed"] and val["model"] == base["model"]
    assert val["gan"] == base["gan"] and val["triple_agent"] == base["triple_agent"]
    assert set(val) == set(base)
    assert smoke["model"] == base["model"] and smoke["triple_agent"] == base["triple_agent"]
    gan = load_cfg("gan")
    assert base["gan"] == gan["gan"] and base["model"] == gan["model"]
    ganmem = load_cfg("triple_agent_val_ganmem")
    assert ganmem["output"]["snapshots_from"] == load_cfg("gan_val")["output"]["loop_dir"]
    assert ganmem["triple_agent"] == base["triple_agent"] and ganmem["model"] == base["model"]


TRIPLE_EXTRA_SEEDS = (43, 44)


@pytest.mark.parametrize("seed", TRIPLE_EXTRA_SEEDS)
def test_triple_seed_variants_differ_from_seed42_only_in_seed_and_experiment(seed):
    """Checks that seed configs differ from seed 42 only in seed and experiment name."""
    for base_name, name in (
        ("triple_agent_val", f"triple_agent_val_seed{seed}"),
        ("triple_agent", f"triple_agent_seed{seed}"),
    ):
        base, variant = load_cfg(base_name), load_cfg(name)
        assert variant["seed"] == seed
        assert variant["experiment"] == name
        if base_name == "triple_agent_val":
            assert variant["gan"]["drift_probe"].pop("seed") == base["seed"] == 42
            assert "seed" not in base["gan"]["drift_probe"]
        rest = lambda cfg: {k: v for k, v in cfg.items() if k not in ("seed", "experiment")}  # noqa: E731
        assert rest(variant) == rest(base), name


@pytest.mark.parametrize("seed", TRIPLE_EXTRA_SEEDS)
def test_triple_seed_test_config_reads_that_seeds_loop_snapshots(seed):
    """Checks that each seed's test config reads the snapshots of its loop config."""
    cfg, loop = load_cfg(f"triple_agent_seed{seed}"), load_cfg(f"triple_agent_val_seed{seed}")
    assert cfg["output"]["snapshots_from"] == loop["output"]["loop_dir"]
    expected = Path(loop["output"]["loop_dir"]) / f"seed{seed}" / "snapshots"
    assert triple_run.snapshot_dir_for(cfg, seed) == expected
    assert cfg["dataset"]["split"] == "test" and cfg["output"]["log_experiments"] is True
    assert loop["dataset"]["split"] == "val" and loop["output"]["log_experiments"] is False


@pytest.mark.parametrize("seed", TRIPLE_EXTRA_SEEDS)
def test_triple_seed_configs_keep_the_shared_eval_decoding(seed):
    """Checks that seed configs keep the shared eval decoding and a greedy loop."""
    base = load_cfg("baseline")
    for name in (f"triple_agent_val_seed{seed}", f"triple_agent_seed{seed}"):
        cfg = load_cfg(name)
        assert (cfg["eval"]["temperature"], cfg["eval"]["top_p"]) == (
            base["model"]["temperature"],
            base["model"]["top_p"],
        )
        assert cfg["eval"]["unparseable"] == base["eval"]["unparseable"]
        assert cfg["model"]["temperature"] == 0.0

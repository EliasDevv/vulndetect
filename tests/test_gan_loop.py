"""Tests for the GAN loop and snapshot runner using offline fake models."""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

from vulndetect.common.predictions import read_predictions
from vulndetect.common.prompt import render_prompt
from vulndetect.data.build import build_dataset, to_records, write_jsonl
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
from vulndetect.gan import run as gan_run
from vulndetect.gan.loop import (
    MeteredModel,
    RoundLog,
    SplitHygieneError,
    assert_train_only,
    build_phase_b_pool,
    build_probe_pool,
    build_source_pool,
    run_loop,
    run_phase_a,
    run_phase_b,
    summarize_prompt_lengths,
)
from vulndetect.gan.parsing import (
    M_COMMIT_MESSAGE,
    M_DECEPTION_NOTES,
    M_EVIDENCE,
    M_LESSON,
    M_MODIFIED_FUNCTION,
    M_NEW_STRATEGIES,
    M_WENT_WRONG,
    marker,
)
from vulndetect.gan.roles import SingleDetector, detect
from vulndetect.gan.strategies import StrategyMemory

_SOURCE_FUNC_RE = re.compile(
    r"- Function:\n(?P<func>.*?)\n+Reply in exactly this format", re.DOTALL
)

VERDICT_TEXTS = {
    LABEL_VULN: "Step 1-4 analysed. Therefore, the answer is Vulnerable.",
    LABEL_SAFE: "Step 1-4 analysed. Therefore, the answer is Not Vulnerable.",
    None: "I would need the surrounding call sites before saying anything here.",
}


class RoleFake:
    """Fake model that serves every loop role, dispatching on the prompt wording."""

    ROLE_MARKERS = {
        "generate": "acting as an adversarial code\ngenerator",
        "generator_update": "maintaining a written playbook",
        "detector_update": "maintaining a\nwritten list of detection strategies",
        "reflect": "acting as a reflector",
        "summarize": "Condense the following list",
    }

    def __init__(self, verdict: str | None = LABEL_VULN, *, generation="ok", updates="ok"):
        self.verdict = verdict
        self.generation = generation
        self.updates = updates
        self.calls: Counter[str] = Counter()
        self.prompts: dict[str, list[str]] = defaultdict(list)
        self._n = 0

    def role_of(self, prompt: str) -> str:
        for role, needle in self.ROLE_MARKERS.items():
            if needle in prompt:
                return role
        return "detect"

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        outputs = []
        for prompt in prompts:
            role = self.role_of(prompt)
            self.calls[role] += 1
            self.prompts[role].append(prompt)
            outputs.append(getattr(self, f"_{role}")(prompt))
        return outputs

    def _detect(self, prompt: str) -> str:
        return VERDICT_TEXTS[self.verdict]

    def _generate(self, prompt: str) -> str:
        if self.generation == "malformed":
            return "Sorry, I will not write vulnerable code."
        source = _SOURCE_FUNC_RE.search(prompt).group("func")
        func = source if self.generation == "unchanged" else self._mutate(source)
        return (
            f"{marker(M_COMMIT_MESSAGE)}\nrefactor bounds handling\n"
            f"{marker(M_MODIFIED_FUNCTION)}\n{func}\n"
            f"{marker(M_DECEPTION_NOTES)}\nThe length check now runs one element late.\n"
        )

    @staticmethod
    def _mutate(func: str) -> str:
        if "return" in func:
            return func.replace("return", "int extra = 1; return", 1)
        return func.replace("{", "{ int extra = 1;", 1)

    def _strategy_update(self) -> str:
        if self.updates == "malformed":
            return "I have nothing to add."
        if self.updates == "markerless":
            self._n += 1
            return f"- Lesson number {self._n} about buffer sizes.\n"
        self._n += 1
        return f"{marker(M_NEW_STRATEGIES)}\n- Lesson number {self._n} about buffer sizes.\n"

    def _generator_update(self, prompt: str) -> str:
        return self._strategy_update()

    def _detector_update(self, prompt: str) -> str:
        return self._strategy_update()

    def _summarize(self, prompt: str) -> str:
        return f"{marker(M_NEW_STRATEGIES)}\n- Merged lesson.\n"

    def _reflect(self, prompt: str) -> str:
        return (
            f"{marker(M_WENT_WRONG)}\nYou assumed the commit message was decisive.\n"
            f"{marker(M_EVIDENCE)}\nThe pre-change function copies without a bound.\n"
            f"{marker(M_LESSON)}\nJudge the pre-change function, not the commit message.\n"
        )


class DetectorFake:
    """Records the prompts of one batched pass and answers with a fixed verdict."""

    def __init__(self, verdict: str | None = LABEL_VULN):
        self.verdict = verdict
        self.prompts: list[str] = []

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        self.prompts += list(prompts)
        return [VERDICT_TEXTS[self.verdict]] * len(prompts)


@pytest.fixture
def records(raw_df, base_cfg) -> list[dict]:
    return to_records(build_dataset(raw_df, base_cfg))


@pytest.fixture
def train_records(records) -> list[dict]:
    return [r for r in records if r["split"] == "train"]


def balanced_test_records(records: list[dict], split: str, per_cwe: int = 1) -> list[dict]:
    """Select one vulnerable and one safe record per CWE from a split."""
    out: list[dict] = []
    for cwe in sorted({r["cwe"] for r in records}):
        pool = [r for r in records if r["split"] == split and r["cwe"] == cwe]
        vuln = [r for r in pool if r["label"] == LABEL_VULN][:per_cwe]
        safe = [r for r in pool if r["label"] == LABEL_SAFE][:per_cwe]
        if vuln and safe:
            out += vuln + safe
    assert len(out) >= 4, "fixture needs at least two usable CWEs"
    return out


GAN_CFG = {
    "experiment": "gan_test",
    "seed": 42,
    "gan": {
        "rounds": 2,
        "snapshot_rounds": [0, 2],
        "max_new_strategies_per_update": 2,
        "memory": {"policy": "fifo", "max_entries": 12, "max_entry_words": 30, "dedup": True},
        "generation": {
            "validate": "structural",
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 1024,
        },
        "phase_b": {"enabled": True, "n_items": 2},
        "drift_probe": {"enabled": True, "split": "val", "n": 4},
    },
    "model": {
        "provider": "huggingface",
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 512,
        "batch_size": 2,
    },
    "dataset": {"name": "bigvul", "split": "test", "train_split": "train", "dir": "dataset"},
    "output": {
        "loop_dir": "results/gan",
        "predictions_dir": "results/predictions/gan",
        "log_experiments": True,
    },
}


@pytest.fixture
def workspace(tmp_path, records, monkeypatch):
    """A temporary working directory with train and val splits but no test split."""
    monkeypatch.chdir(tmp_path)
    write_jsonl([r for r in records if r["split"] == "train"], tmp_path / "dataset" / "train.jsonl")
    write_jsonl(balanced_test_records(records, "val"), tmp_path / "dataset" / "val.jsonl")
    cfg_path = tmp_path / "gan.yaml"
    cfg_path.write_text(yaml.safe_dump(GAN_CFG), encoding="utf-8")
    return cfg_path


@pytest.fixture
def workspace_with_test(workspace, records, tmp_path):
    """The same workspace plus a test split, for the runner tests."""
    write_jsonl(balanced_test_records(records, "test"), tmp_path / "dataset" / "test.jsonl")
    return workspace


def rewrite_cfg(cfg_path, mutate) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    mutate(cfg)
    Path(cfg_path).write_text(yaml.safe_dump(cfg), encoding="utf-8")


def phase_a(tmp_path, fake, sources, *, rounds=3, validate="structural", s_g=None, s_d=None):
    """run_phase_a with fresh memories and a throwaway log."""
    s_g = s_g or StrategyMemory(role="generator")
    s_d = s_d or StrategyMemory(role="detector")
    log = RoundLog(tmp_path / "rounds.jsonl")
    generated_log = RoundLog(tmp_path / "generated.jsonl")
    try:
        stats = run_phase_a(
            fake,
            fake,
            sources,
            s_g,
            s_d,
            rounds=rounds,
            max_new=2,
            max_words=30,
            validate=validate,
            log=log,
            generated_log=generated_log,
        )
    finally:
        log.close()
        generated_log.close()
    rows = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    return stats, s_g, s_d, rows


def test_correct_detector_updates_only_the_generator(tmp_path, train_records):
    """A correct detector verdict updates only the generator memory."""
    fake = RoleFake(verdict=LABEL_VULN)
    stats, s_g, s_d, rows = phase_a(tmp_path, fake, train_records, rounds=3)

    assert stats["rounds_completed"] == 3
    assert stats["n_detector_correct"] == 3 and stats["n_detector_wrong"] == 0
    assert stats["n_generator_updates"] == 3 and stats["n_detector_updates"] == 0
    assert len(s_g) == 3 and len(s_d) == 0
    assert s_d.render() == ""
    assert {row["updated"] for row in rows} == {"generator"}
    assert fake.calls["generator_update"] == 3 and fake.calls["detector_update"] == 0


def test_wrong_detector_updates_only_the_detector(tmp_path, train_records):
    """A wrong detector verdict updates only the detector memory."""
    fake = RoleFake(verdict=LABEL_SAFE)
    stats, s_g, s_d, rows = phase_a(tmp_path, fake, train_records, rounds=3)

    assert stats["n_detector_wrong"] == 3 and stats["n_detector_correct"] == 0
    assert stats["n_detector_updates"] == 3 and stats["n_generator_updates"] == 0
    assert len(s_d) == 3 and len(s_g) == 0
    assert {row["updated"] for row in rows} == {"detector"}
    assert rows[-1]["cum_false_negative_rate"] == 1.0


def test_unparseable_verdict_counts_as_wrong(tmp_path, train_records):
    fake = RoleFake(verdict=None)
    stats, s_g, s_d, _ = phase_a(tmp_path, fake, train_records, rounds=2)
    assert stats["n_detector_unparseable"] == 2
    assert stats["n_detector_wrong"] == 2 and len(s_d) == 2 and len(s_g) == 0


def test_exactly_one_memory_changes_per_round(tmp_path, train_records):
    """Each Phase-A round changes exactly one of the two memories."""
    for verdict in (LABEL_VULN, LABEL_SAFE):
        stats, s_g, s_d, rows = phase_a(
            tmp_path, RoleFake(verdict=verdict), train_records, rounds=3
        )
        for row in rows:
            grew, other = ("s_g", "s_d") if row["updated"] == "generator" else ("s_d", "s_g")
            assert row[grew]["n_entries"] >= 1
            assert row[other]["n_entries"] == 0
        assert len(s_g) + len(s_d) == 3


def test_memories_start_null_and_only_the_winner_grows(tmp_path, train_records):
    """Both memories start empty and only the winning side grows."""
    fake = RoleFake(verdict=LABEL_VULN)
    stats, s_g, s_d, _ = phase_a(tmp_path, fake, train_records, rounds=1)
    assert stats["rounds_completed"] == 1
    first_detect = fake.prompts["detect"][0]
    assert first_detect.startswith("You are an expert in cybersecurity")
    assert "detection strategies from earlier cases" not in first_detect
    assert s_g.to_dict()["entries"] and not s_d.to_dict()["entries"]


def test_malformed_generation_neither_consumes_a_round_nor_touches_memories(
    tmp_path, train_records
):
    fake = RoleFake(generation="malformed")
    sources = train_records[:3]
    stats, s_g, s_d, rows = phase_a(tmp_path, fake, sources, rounds=2)

    assert stats["rounds_completed"] == 0
    assert stats["n_generation_rejected"] == 3
    assert stats["rejections_by_reason"] == {"missing_sections": 3}
    assert len(s_g) == 0 and len(s_d) == 0
    assert all(row["accepted"] is False for row in rows)
    assert {row["rejection_reason"] for row in rows} == {"missing_sections"}
    assert fake.calls["detect"] == 0


def test_unchanged_generation_is_rejected_by_the_structural_gate(tmp_path, train_records):
    stats, _, _, _ = phase_a(
        tmp_path, RoleFake(generation="unchanged"), train_records[:2], rounds=2
    )
    assert stats["rejections_by_reason"] == {"unchanged": 2}
    assert stats["rounds_completed"] == 0


def test_validate_none_accepts_what_the_gate_would_reject(tmp_path, train_records):
    stats, s_g, _, _ = phase_a(
        tmp_path, RoleFake(generation="unchanged"), train_records, rounds=2, validate="none"
    )
    assert stats["rounds_completed"] == 2 and stats["n_generation_rejected"] == 0
    assert len(s_g) == 2


def test_unknown_validate_mode_fails_loudly(tmp_path, train_records):
    with pytest.raises(ValueError, match="unknown validate mode"):
        phase_a(tmp_path, RoleFake(), train_records, rounds=1, validate="compile")


def test_malformed_update_is_counted_and_adds_nothing(tmp_path, train_records):
    stats, s_g, s_d, rows = phase_a(
        tmp_path, RoleFake(verdict=LABEL_SAFE, updates="malformed"), train_records, rounds=2
    )
    assert stats["n_malformed_updates"] == 2 and stats["n_marker_missing"] == 0
    assert stats["n_detector_updates"] == 2 and len(s_d) == 0
    assert all(row["update_parsed"] is False for row in rows)


def test_phase_a_persists_generated_items_and_raw_outputs(tmp_path, train_records):
    """Phase A writes generated items and raw model outputs to the logs."""
    stats, s_g, s_d, rows = phase_a(tmp_path, RoleFake(verdict=LABEL_SAFE), train_records, rounds=2)
    items = [json.loads(line) for line in (tmp_path / "generated.jsonl").read_text().splitlines()]
    assert [i["round"] for i in items] == [1, 2] and all(i["accepted"] for i in items)
    for item in items:
        assert item["function"] and item["commit_message"] and item["deception_notes"]
        assert item["generator_raw"] and item["detector_raw"]
        assert item["detector_label"] == LABEL_SAFE and item["correct"] is False
    accepted = [r for r in rows if r["accepted"]]
    assert all(r["detector_raw"] and r["update_raw"] for r in accepted)


def test_rejected_generations_are_persisted_with_their_raw_output(tmp_path, train_records):
    phase_a(tmp_path, RoleFake(generation="malformed"), train_records, rounds=1)
    items = [json.loads(line) for line in (tmp_path / "generated.jsonl").read_text().splitlines()]
    assert items and all(not i["accepted"] and i["rejection_reason"] for i in items)
    assert all(i["generator_raw"] and "function" not in i for i in items)


def test_markerless_update_is_taken_via_the_fallback_and_counted(tmp_path, train_records):
    """A markerless update is parsed via the fallback and counted."""
    stats, s_g, s_d, rows = phase_a(
        tmp_path, RoleFake(verdict=LABEL_SAFE, updates="markerless"), train_records, rounds=2
    )
    assert stats["n_malformed_updates"] == 0 and stats["n_marker_missing"] == 2
    assert stats["n_detector_updates"] == 2 and len(s_d) == 2
    assert all(row["update_parsed"] and row["update_marker_present"] is False for row in rows)


def phase_b(tmp_path, fake, items, *, s_d=None):
    s_d = s_d or StrategyMemory(role="detector")
    log = RoundLog(tmp_path / "rounds_b.jsonl")
    try:
        stats = run_phase_b(fake, fake, items, s_d, max_new=2, max_words=30, log=log)
    finally:
        log.close()
    rows = [json.loads(line) for line in (tmp_path / "rounds_b.jsonl").read_text().splitlines()]
    return stats, s_d, rows


def test_phase_b_reflects_only_on_errors(tmp_path, records):
    items = balanced_test_records(records, "train")
    fake = RoleFake(verdict=LABEL_VULN)
    stats, s_d, rows = phase_b(tmp_path, fake, items)

    n_safe = sum(r["label"] == LABEL_SAFE for r in items)
    assert stats["n_wrong"] == n_safe and stats["n_reflections"] == n_safe
    assert stats["wrong_by_gold"][LABEL_SAFE] == n_safe
    assert stats["wrong_by_gold"][LABEL_VULN] == 0
    assert stats["n_detector_updates"] == n_safe and len(s_d) == n_safe
    assert fake.calls["reflect"] == n_safe
    reflected = [row for row in rows if row["reflected"]]
    assert all(row["gold_label"] == LABEL_SAFE for row in reflected)
    assert all(row["reflection_structured"] for row in reflected)
    assert stats["vulnerable_prediction_share"] == 1.0


def test_phase_b_markerless_update_is_taken_via_the_fallback(tmp_path, records):
    """A markerless Phase-B update is parsed via the fallback."""
    items = balanced_test_records(records, "train")
    fake = RoleFake(verdict=LABEL_VULN, updates="markerless")
    stats, s_d, rows = phase_b(tmp_path, fake, items)
    assert stats["n_wrong"] > 0
    assert stats["n_malformed_updates"] == 0 and stats["n_marker_missing"] == stats["n_wrong"]
    assert len(s_d) == stats["n_wrong"]
    assert all(
        row["update_marker_present"] is False for row in rows if row["updated"] == "detector"
    )
    assert all(row["detector_raw"] for row in rows)
    updated = [row for row in rows if row["updated"] == "detector"]
    assert all(row["reflection_raw"] and row["update_raw"] for row in updated)


def test_phase_b_makes_no_update_when_the_detector_is_right(tmp_path, records):
    items = [r for r in records if r["split"] == "train" and r["label"] == LABEL_VULN][:3]
    fake = RoleFake(verdict=LABEL_VULN)
    stats, s_d, rows = phase_b(tmp_path, fake, items)
    assert stats["n_correct"] == 3 and stats["n_reflections"] == 0
    assert len(s_d) == 0 and fake.calls["reflect"] == 0
    assert all(row["updated"] == "none" for row in rows)


def test_unstructured_reflection_is_still_used_but_counted(tmp_path, records):
    class Unstructured(RoleFake):
        def _reflect(self, prompt):
            return "You over-trusted the commit message."

    items = [r for r in records if r["split"] == "train" and r["label"] == LABEL_SAFE][:2]
    stats, s_d, rows = phase_b(tmp_path, Unstructured(verdict=LABEL_VULN), items)
    assert stats["n_reflections"] == 2 and stats["n_reflections_unstructured"] == 2
    assert len(s_d) == 2
    assert all(row["reflection_structured"] is False for row in rows)


def test_assert_train_only_rejects_other_splits():
    assert_train_only([{"split": "train"}], where="unit")
    with pytest.raises(SplitHygieneError, match="may only see split='train'"):
        assert_train_only([{"split": "train"}, {"split": "test"}], where="unit")


def test_phase_a_refuses_test_records(tmp_path, records):
    test_records = [r for r in records if r["split"] == "test"][:2]
    with pytest.raises(SplitHygieneError):
        phase_a(tmp_path, RoleFake(), test_records, rounds=1)


def test_phase_b_refuses_test_records(tmp_path, records):
    test_records = [r for r in records if r["split"] == "test"][:2]
    with pytest.raises(SplitHygieneError):
        phase_b(tmp_path, RoleFake(), test_records)


def test_drift_probe_split_must_be_val(workspace):
    with pytest.raises(SplitHygieneError, match="drift probe split"):
        build_probe_pool("test", 4, 42, "dataset")
    with pytest.raises(SplitHygieneError, match="drift probe split"):
        build_probe_pool("train", 4, 42, "dataset")


def test_loop_rejects_a_non_train_train_split(workspace):
    rewrite_cfg(workspace, lambda cfg: cfg["dataset"].update({"train_split": "val"}))
    with pytest.raises(SplitHygieneError, match="dataset.train_split"):
        run_loop(workspace, generator_model=RoleFake(), detector_model=RoleFake())


def test_pools_are_deterministic_and_disjoint(train_records):
    phase_b_items = build_phase_b_pool(train_records, 6, seed=42)
    again = build_phase_b_pool(train_records, 6, seed=42)
    assert [r["id"] for r in phase_b_items] == [r["id"] for r in again]
    assert len(phase_b_items) == 6
    assert sum(r["label"] == LABEL_VULN for r in phase_b_items) == 3
    assert len({r["cwe"] for r in phase_b_items}) > 1

    sources = build_source_pool(train_records, {r["id"] for r in phase_b_items}, seed=42)
    assert all(r["label"] == LABEL_SAFE for r in sources)
    assert not ({r["id"] for r in sources} & {r["id"] for r in phase_b_items})
    assert [r["id"] for r in sources] == [
        r["id"] for r in build_source_pool(train_records, {r["id"] for r in phase_b_items}, 42)
    ]
    other = build_source_pool(train_records, set(), seed=43)
    assert [r["id"] for r in sources] != [r["id"] for r in other]


def test_probe_pool_is_balanced_and_stable(workspace):
    pool = build_probe_pool("val", 4, 42, "dataset")
    assert len(pool) == 4
    assert sum(r["label"] == LABEL_VULN for r in pool) == 2
    assert [r["id"] for r in pool] == [r["id"] for r in build_probe_pool("val", 4, 42, "dataset")]


def test_snapshot_dir_for_prefers_snapshots_from():
    cfg = {"output": {"loop_dir": "results/gan"}}
    assert gan_run.snapshot_dir_for(cfg, 42) == Path("results/gan/seed42/snapshots")
    cfg["output"]["snapshots_from"] = "results/gan_val"
    assert gan_run.snapshot_dir_for(cfg, 42) == Path("results/gan_val/seed42/snapshots")


def test_run_loop_writes_snapshots_log_and_manifest_without_any_test_split(workspace):
    """The loop writes snapshots, log and manifest without any test split present."""
    fake = RoleFake(verdict=LABEL_SAFE)
    manifest = run_loop(workspace, generator_model=fake, detector_model=fake)

    out = Path("results/gan/seed42")
    assert sorted(p.name for p in (out / "snapshots").glob("*.json")) == [
        "final.json",
        "round0000.json",
        "round0002.json",
    ]
    round0 = json.loads((out / "snapshots" / "round0000.json").read_text())
    assert round0["s_d"]["entries"] == [] and round0["phase"] == "init"
    assert round0["drift_probe"]["n"] == 4
    assert round0["drift_probe"]["vulnerable_share"] == 0.0
    final = json.loads((out / "snapshots" / "final.json").read_text())
    assert final["phase"] == "B" and final["round"] is None
    assert len(final["s_d"]["entries"]) > len(round0["s_d"]["entries"])
    assert final["loop_stats"]["phase_a"]["rounds_completed"] == 2

    rows = [json.loads(line) for line in (out / "rounds.jsonl").read_text().splitlines()]
    assert sum(row["phase"] == "A" and row["accepted"] for row in rows) == 2
    assert sum(row["phase"] == "B" for row in rows) == 2
    items = [json.loads(line) for line in (out / "generated.jsonl").read_text().splitlines()]
    assert sum(i["accepted"] for i in items) == 2
    assert manifest["generated_items_path"].endswith("generated.jsonl")

    saved = json.loads((out / "loop_manifest.json").read_text())
    assert saved == manifest
    assert manifest["approach"] == "gan_loop"
    assert manifest["phase_a"]["rounds_completed"] == 2
    assert manifest["phase_b"]["n_items"] == 2
    assert manifest["decoding"]["generator"]["temperature"] == 0.8
    assert manifest["decoding"]["detector"]["temperature"] == 0.0
    assert len(manifest["source_order_sha256"]) == 64
    assert manifest["gan_prompt_versions"]["detector_block"].startswith("gan-")
    assert manifest["final_memories"]["s_d"]["entries"]
    assert manifest["snapshots"] == ["round0000.json", "round0002.json", "final.json"]


def test_run_loop_item_order_is_reproducible_for_the_same_seed(workspace):
    first = run_loop(workspace, generator_model=RoleFake(), detector_model=RoleFake())
    ids_first = [
        json.loads(line).get("source_id")
        for line in Path("results/gan/seed42/rounds.jsonl").read_text().splitlines()
    ]
    second = run_loop(workspace, generator_model=RoleFake(), detector_model=RoleFake())
    ids_second = [
        json.loads(line).get("source_id")
        for line in Path("results/gan/seed42/rounds.jsonl").read_text().splitlines()
    ]
    assert first["source_order_sha256"] == second["source_order_sha256"]
    assert ids_first == ids_second


def test_run_loop_can_skip_phase_b_for_the_w_o_sr_ablation(workspace):
    rewrite_cfg(workspace, lambda cfg: cfg["gan"]["phase_b"].update({"enabled": False}))
    manifest = run_loop(workspace, generator_model=RoleFake(), detector_model=RoleFake())
    assert manifest["phase_b"] is None
    final = json.loads(Path("results/gan/seed42/snapshots/final.json").read_text())
    assert final["phase"] == "A"


def test_run_loop_stops_when_the_source_pool_is_exhausted(workspace):
    rewrite_cfg(workspace, lambda cfg: cfg["gan"].update({"rounds": 10_000}))
    manifest = run_loop(
        workspace,
        generator_model=RoleFake(generation="malformed"),
        detector_model=RoleFake(generation="malformed"),
    )
    assert manifest["phase_a"]["rounds_completed"] == 0
    assert manifest["phase_a"]["n_generation_rejected"] == manifest["dataset"]["n_source_pool"]


def loop_then_eval(workspace, snapshot="final", *, loop_fake=None, limit=None):
    loop_fake = loop_fake or RoleFake(verdict=LABEL_SAFE)
    run_loop(workspace, generator_model=loop_fake, detector_model=loop_fake)
    detector = DetectorFake(verdict=LABEL_VULN)
    report = gan_run.run(workspace, snapshot, limit=limit, model=detector)
    return report, detector


def test_eval_of_the_round0_snapshot_uses_byte_identical_baseline_prompts(workspace_with_test):
    """Evaluating the round-0 snapshot uses exactly the baseline prompts."""
    report, detector = loop_then_eval(workspace_with_test, "round0000")
    test_records = [
        json.loads(line) for line in Path("dataset/test.jsonl").read_text().splitlines()
    ]
    assert detector.prompts == [render_prompt(r) for r in test_records]

    out = Path("results/predictions/gan/seed42/round0000")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["gan"]["prompt_equals_baseline"] is True
    assert manifest["gan"]["round"] == 0 and manifest["gan"]["n_strategies"] == 0
    assert manifest["gan"]["vulnerable_prediction_share"] == 1.0
    assert manifest["logged"] is True
    assert report["n"] == len(test_records)
    assert len(read_predictions(out / "predictions.jsonl")) == len(test_records)
    assert json.loads((out / "metrics.json").read_text())["n"] == len(test_records)


def test_eval_of_the_final_snapshot_prepends_the_strategy_block(workspace_with_test):
    report, detector = loop_then_eval(workspace_with_test, "final")
    test_records = [
        json.loads(line) for line in Path("dataset/test.jsonl").read_text().splitlines()
    ]
    manifest = json.loads(
        (Path("results/predictions/gan/seed42/final") / "manifest.json").read_text()
    )
    strategies = manifest["gan"]["strategies"]
    assert strategies, "the loop should have filled S_D"
    assert manifest["gan"]["prompt_equals_baseline"] is False
    for prompt, rec in zip(detector.prompts, test_records, strict=True):
        assert prompt.endswith(render_prompt(rec))
        assert strategies[0] in prompt
    assert manifest["gan"]["drift_probe"]["n"] == 4
    assert manifest["gan"]["memory_policy"] == "fifo"


def test_eval_appends_one_experiments_row_per_snapshot(workspace_with_test):
    fake = RoleFake(verdict=LABEL_SAFE)
    run_loop(workspace_with_test, generator_model=fake, detector_model=fake)
    reports = gan_run.run_all_snapshots(workspace_with_test, model=DetectorFake())

    assert set(reports) == {"round0000", "round0002", "final"}
    rows = list(csv.DictReader(Path("results/experiments.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    assert {row["approach"] for row in rows} == {"gan"}
    params = [json.loads(row["params"]) for row in rows]
    assert [p["snapshot"] for p in params] == ["round0000", "round0002", "final"]
    assert [p["round"] for p in params] == [0, 2, None]
    assert all(p["prompt_version"] == "fig1-v3" for p in params)
    assert all(p["memory_policy"] == "fifo" for p in params)
    assert all(row["output"].startswith("results/predictions/gan/seed42/") for row in rows)


class KwargsFake(DetectorFake):
    """DetectorFake that also records the decoding overrides it was called with."""

    def __init__(self, verdict: str | None = LABEL_VULN):
        super().__init__(verdict)
        self.kwargs: list[dict] = []

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        self.kwargs.append(dict(kwargs))
        return super().generate(prompts, **kwargs)


def test_eval_decoding_follows_the_eval_block_not_the_loop(workspace_with_test):
    """Snapshot evaluation uses the eval decoding block while the loop stays greedy."""
    fake = RoleFake(verdict=LABEL_SAFE)
    run_loop(workspace_with_test, generator_model=fake, detector_model=fake)

    det = KwargsFake()
    gan_run.run(workspace_with_test, "final", model=det)
    assert det.kwargs and all(k["temperature"] == 0.0 and k["top_p"] is None for k in det.kwargs)
    manifest = json.loads(Path("results/predictions/gan/seed42/final/manifest.json").read_text())
    assert manifest["decoding"]["eval_override"] is False

    rewrite_cfg(
        workspace_with_test, lambda cfg: cfg.update({"eval": {"temperature": 0.6, "top_p": 0.9}})
    )
    det = KwargsFake()
    gan_run.run(workspace_with_test, "final", model=det)
    assert det.kwargs and all(k["temperature"] == 0.6 and k["top_p"] == 0.9 for k in det.kwargs)
    manifest = json.loads(Path("results/predictions/gan/seed42/final/manifest.json").read_text())
    assert manifest["decoding"]["temperature"] == 0.6 and manifest["decoding"]["top_p"] == 0.9
    assert manifest["decoding"]["eval_override"] is True
    assert manifest["decoding"]["loop_detector_temperature"] == 0.0


def test_loop_with_zero_rounds_skips_phase_a_and_still_runs_phase_b(workspace):
    """With zero rounds the loop skips Phase A and still runs Phase B."""
    rewrite_cfg(workspace, lambda cfg: cfg["gan"].update({"rounds": 0, "snapshot_rounds": [0]}))
    fake = RoleFake(verdict=LABEL_VULN)
    manifest = run_loop(workspace, generator_model=fake, detector_model=fake)
    out = Path("results/gan/seed42")
    assert sorted(p.name for p in (out / "snapshots").glob("*.json")) == [
        "final.json",
        "round0000.json",
    ]
    a, b = manifest["phase_a"], manifest["phase_b"]
    assert (
        a["rounds_completed"] == 0 and a["sources_consumed"] == 0 and a["n_detector_updates"] == 0
    )
    assert fake.calls["generate"] == 0
    assert b["n_items"] > 0 and b["n_detector_updates"] > 0
    final = json.loads((out / "snapshots" / "final.json").read_text())
    assert (
        final["phase"] == "B" and len(final["s_d"]["entries"]) > 0 and final["s_g"]["entries"] == []
    )


def test_eval_does_not_log_when_the_config_opts_out(workspace_with_test):
    rewrite_cfg(workspace_with_test, lambda cfg: cfg["output"].update({"log_experiments": False}))
    loop_then_eval(workspace_with_test, "final")
    assert not Path("results/experiments.csv").exists()


def test_eval_limit_keeps_a_smoke_run_out_of_the_log(workspace_with_test):
    report, _ = loop_then_eval(workspace_with_test, "final", limit=4)
    assert report["n"] == 4
    assert not Path("results/experiments.csv").exists()
    manifest = json.loads(
        (Path("results/predictions/gan/seed42/final") / "manifest.json").read_text()
    )
    assert manifest["logged"] is False and manifest["dataset"]["limit"] == 4


def test_eval_without_a_loop_run_points_at_the_loop(workspace_with_test):
    with pytest.raises(FileNotFoundError, match="run the GAN loop first"):
        gan_run.run(workspace_with_test, "final", model=DetectorFake())


class CountingFake:
    """A fake that can tokenize (whitespace words) and has a tiny context window."""

    context_window = 3

    def __init__(self):
        self.prompts: list[str] = []

    def count_tokens(self, prompt: str, *, system=None, chat=True) -> int:
        return len(prompt.split())

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        self.prompts += list(prompts)
        return ["ok"] * len(prompts)


def test_single_detector_is_the_module_level_detect(train_records):
    fake = DetectorFake(verdict=LABEL_VULN)
    party = SingleDetector(fake)
    rec = train_records[0]
    got, ref = party.detect(rec, ""), detect(fake, rec, "")
    assert (got.label, got.raw, got.trace) == (ref.label, ref.raw, None)
    assert party.describe() == {"name": "single_detector", "prompt_versions": {}}
    assert party.stats() == {"n_detections": 1}
    assert len(party.detect_batch(train_records[:3], "")) == 3
    assert party.stats() == {"n_detections": 4}


def test_metered_model_counts_tokens_and_warns_over_the_window(caplog):
    inner = CountingFake()
    meter = MeteredModel(inner)
    with caplog.at_level(logging.WARNING, logger="vulndetect.gan.loop"):
        out = meter.generate(["a b", "a b c d"])
    assert out == ["ok", "ok"] and inner.prompts == ["a b", "a b c d"]
    entries = meter.drain("detection")
    assert [e["tokens"] for e in entries] == [2, 4]
    assert [e["over_context"] for e in entries] == [False, True]
    assert entries[1]["chars"] == len("a b c d") and entries[1]["output_chars"] == 2
    assert "exceeds the context window" in caplog.text
    assert meter.drain("detection") == []
    assert summarize_prompt_lengths([meter]) == {
        "context_window": 3,
        "by_role": {
            "detection": {
                "n": 2,
                "max_chars": 7,
                "max_tokens": 4,
                "p99_tokens": 4,
                "n_over_context": 1,
            }
        },
    }


def test_metered_model_without_a_tokenizer_records_characters_only():
    meter = MeteredModel(DetectorFake())
    meter.generate(["abc"])
    (entry,) = meter.drain("probe")
    assert entry == {
        "chars": 3,
        "tokens": None,
        "output_chars": len(VERDICT_TEXTS[LABEL_VULN]),
        "over_context": False,
    }
    assert summarize_prompt_lengths([meter, DetectorFake()]) == {
        "context_window": None,
        "by_role": {
            "probe": {
                "n": 1,
                "max_chars": 3,
                "max_tokens": None,
                "p99_tokens": None,
                "n_over_context": 0,
            }
        },
    }


def test_run_loop_names_the_single_party_and_meters_every_role(workspace):
    fake = RoleFake(verdict=LABEL_SAFE)
    manifest = run_loop(workspace, generator_model=fake, detector_model=fake)
    assert manifest["approach"] == "gan_loop"
    # 2 Phase-A rounds + 2 Phase-B items + 3 snapshots x 4 probe records
    assert manifest["detecting_party"] == {
        "name": "single_detector",
        "prompt_versions": {},
        "stats": {"n_detections": 2 + 2 + 3 * 4},
    }

    rows = [
        json.loads(line)
        for line in Path("results/gan/seed42/rounds.jsonl").read_text().splitlines()
    ]
    assert rows and all(row["detection_trace"] is None for row in rows)
    for row in (r for r in rows if r["phase"] == "A"):
        assert set(row["prompt_lengths"]) == {"generator", "detection", "update"}
        (entry,) = row["prompt_lengths"]["detection"]
        assert entry["chars"] > 0 and entry["tokens"] is None and entry["over_context"] is False
    b_rows = [r for r in rows if r["phase"] == "B"]
    wrong = [r for r in b_rows if not r["correct"]]
    right = [r for r in b_rows if r["correct"]]
    assert wrong and right
    assert all(set(r["prompt_lengths"]) == {"detection", "reflector", "update"} for r in wrong)
    assert all(set(r["prompt_lengths"]) == {"detection"} for r in right)

    lengths = manifest["prompt_lengths"]
    assert lengths["context_window"] is None
    assert set(lengths["by_role"]) == {"generator", "detection", "update", "reflector", "probe"}
    assert lengths["by_role"]["probe"]["n"] == 12 and lengths["by_role"]["detection"]["n"] == 4
    assert all(
        v["n_over_context"] == 0 and v["max_tokens"] is None for v in lengths["by_role"].values()
    )

    generated = [
        json.loads(line)
        for line in Path("results/gan/seed42/generated.jsonl").read_text().splitlines()
    ]
    assert all(g["detection_trace"] is None for g in generated if g["accepted"])
    snap = json.loads(Path("results/gan/seed42/snapshots/final.json").read_text())
    assert snap["detecting_party"] == {"name": "single_detector", "prompt_versions": {}}
    assert snap["drift_probe"]["n"] == 4 and "stage1" not in snap["drift_probe"]


def test_probe_pool_can_be_pinned_across_run_seeds(workspace):
    """``drift_probe.seed`` pins the probe pool independently of the run seed."""

    def loop(seed: int, probe_seed: int | None = None) -> dict:
        cfg = yaml.safe_load(yaml.safe_dump(GAN_CFG))  # deep copy
        cfg["seed"] = seed
        if probe_seed is not None:
            cfg["gan"]["drift_probe"]["seed"] = probe_seed
        path = Path(f"gan_{seed}_{probe_seed}.yaml")
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        fake = RoleFake(verdict=LABEL_SAFE)
        return run_loop(path, generator_model=fake, detector_model=fake)

    reference, pinned, unpinned = loop(42), loop(43, probe_seed=42), loop(44)
    assert pinned["probe_ids_sha256"] == reference["probe_ids_sha256"]
    assert reference["gan"]["drift_probe"]["seed_realised"] == 42
    assert pinned["gan"]["drift_probe"]["seed_realised"] == 42
    assert unpinned["gan"]["drift_probe"]["seed_realised"] == 44


def test_eval_scores_with_the_configured_unparseable_policy(workspace_with_test):
    """Snapshot evaluation scores with the configured unparseable policy."""
    cfg = yaml.safe_load(workspace_with_test.read_text(encoding="utf-8"))
    cfg["eval"] = {"unparseable": "drop"}
    workspace_with_test.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    report, _ = loop_then_eval(workspace_with_test, "final")
    manifest = json.loads(
        (Path("results/predictions/gan/seed42/final") / "manifest.json").read_text()
    )
    assert report["unparseable_policy"] == "drop" and report["alt"]["policy"] == "wrong"
    assert manifest["unparseable_policy"] == "drop"

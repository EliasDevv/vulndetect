"""Unit tests for GAN prompts, parsing, strategy memories, snapshots and configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vulndetect.common.prompt import render_prompt
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
from vulndetect.gan import parsing as gp
from vulndetect.gan.prompts import (
    DETECTOR_HEADER,
    PROMPT_VERSIONS,
    UPDATE_KINDS,
    cwe_label,
    render_detector_prompt,
    render_detector_update_prompt,
    render_generator_prompt,
    render_generator_update_prompt,
    render_reflector_prompt,
    render_summarize_prompt,
)
from vulndetect.gan.snapshot import (
    detector_memory,
    list_snapshots,
    read_snapshot,
    resolve_snapshot_path,
    snapshot_name,
    write_snapshot,
)
from vulndetect.gan.strategies import (
    CAPACITY_POLICIES,
    StrategyMemory,
    dedup_key,
    new_memories,
    normalise_entry,
)

# Literal braces check that prompts are built by concatenation, not str.format.
SOURCE_FUNC = "int copy_in(char *src, int n) { char buf[8]; memcpy(buf, src, n); return 0; }"
GEN_FUNC = "int copy_in(char *src, int n) { char buf[8]; memcpy(buf, src, n + 1); return 0; }"
DETECTOR_TEXT = "Vulnerable. The call `memcpy(buf, src, n)` with n > 8 overflows {buf}."


def record(label: str = LABEL_VULN) -> dict:
    return {
        "id": "c1:abc",
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


def memory(entries: list[str], role: str = "detector") -> StrategyMemory:
    return StrategyMemory(role=role, entries=list(entries))


def test_empty_memory_is_byte_identical_to_the_baseline_prompt():
    """An empty memory yields exactly the baseline prompt."""
    rec = record()
    assert render_detector_prompt(rec, memory([]).render()) == render_prompt(rec)


def test_strategy_block_is_prepended_and_fig1_stays_the_suffix():
    rec = record()
    strategies = memory(["Check every memcpy length against the buffer size."]).render()
    prompt = render_detector_prompt(rec, strategies)
    assert prompt.startswith(DETECTOR_HEADER)
    assert prompt.endswith(render_prompt(rec))
    assert "1. Check every memcpy length against the buffer size." in prompt
    assert prompt.count(render_prompt(rec)) == 1


def test_strategies_are_numbered_in_order():
    block = memory(["First lesson.", "Second lesson.", "Third lesson."]).render()
    assert block == "1. First lesson.\n2. Second lesson.\n3. Third lesson."


def test_generator_prompt_carries_code_metadata_and_format():
    prompt = render_generator_prompt(record(), "")
    assert SOURCE_FUNC in prompt
    assert "CWE-119 (Improper Restriction of Operations" in prompt
    assert "- Project: proj" in prompt and "- Programming Language: C" in prompt
    for name in gp.GENERATOR_SECTIONS:
        assert gp.marker(name) in prompt
    assert "Strategies you compiled in earlier rounds" not in prompt


def test_generator_prompt_includes_playbook_when_non_empty():
    prompt = render_generator_prompt(record(), memory(["Hide the flaw in a helper."]).render())
    assert "Strategies you compiled in earlier rounds" in prompt
    assert "1. Hide the flaw in a helper." in prompt


def test_generator_update_prompt_contains_both_functions_and_the_assessment():
    prompt = render_generator_update_prompt(
        record(),
        GEN_FUNC,
        "I shortened the bound check.",
        DETECTOR_TEXT,
        "",
        max_new=2,
        max_words=30,
    )
    assert SOURCE_FUNC in prompt and GEN_FUNC in prompt
    assert DETECTOR_TEXT in prompt and "I shortened the bound check." in prompt
    assert "at most 2 new playbook entries" in prompt
    assert "at most 30 words" in prompt
    assert gp.marker(gp.M_NEW_STRATEGIES) in prompt


def test_cwe_label_falls_back_to_the_bare_id():
    assert cwe_label("CWE-20") == "CWE-20 (Improper Input Validation)"
    assert cwe_label("CWE-9999") == "CWE-9999"


@pytest.mark.parametrize("kind", UPDATE_KINDS)
def test_detector_update_embeds_the_task_byte_identically(kind):
    rec = record()
    prompt = render_detector_update_prompt(
        rec, DETECTOR_TEXT, "evidence text", kind, "", max_new=1, max_words=25
    )
    assert render_prompt(rec) in prompt
    assert DETECTOR_TEXT in prompt and "evidence text" in prompt
    assert "at most 1 new strategy entries" in prompt
    assert gp.marker(gp.M_NEW_STRATEGIES) in prompt


def test_detector_update_variants_differ_in_the_evidence_framing():
    rec = record()
    adversary = render_detector_update_prompt(
        rec, DETECTOR_TEXT, "notes", "adversary", "", max_new=1, max_words=25
    )
    reflection = render_detector_update_prompt(
        rec, DETECTOR_TEXT, "notes", "reflection", "", max_new=1, max_words=25
    )
    assert "deliberately modified" in adversary and "deliberately modified" not in reflection
    assert "reflection below" in reflection and "reflection below" not in adversary
    assert "How the flaw was hidden" in adversary
    assert "Reflection on your mistake" in reflection


def test_detector_update_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown update kind"):
        render_detector_update_prompt(
            record(), DETECTOR_TEXT, "x", "gradient", "", max_new=1, max_words=10
        )


def test_reflector_prompt_states_both_labels_and_the_format():
    rec = record(LABEL_SAFE)
    prompt = render_reflector_prompt(rec, DETECTOR_TEXT, LABEL_VULN, max_words=30)
    assert f'answered "{LABEL_VULN}"' in prompt
    assert f'the correct answer is "{LABEL_SAFE}"' in prompt
    assert render_prompt(rec) in prompt
    for name in gp.REFLECTOR_SECTIONS:
        assert gp.marker(name) in prompt


def test_reflector_prompt_handles_an_unparseable_verdict():
    prompt = render_reflector_prompt(record(), "waffle", None, max_words=30)
    assert 'answered "no clear verdict"' in prompt


def test_summarize_prompt_lists_the_entries():
    prompt = render_summarize_prompt(
        ["A lesson.", "Another lesson."], "detector", max_entries=1, max_words=20
    )
    assert "1. A lesson." in prompt and "2. Another lesson." in prompt
    assert "at most 1 entries" in prompt
    assert gp.marker(gp.M_NEW_STRATEGIES) in prompt


def test_prompt_versions_are_all_distinct_and_tagged():
    assert len(set(PROMPT_VERSIONS.values())) == len(PROMPT_VERSIONS)
    assert all(v.startswith("gan-") for v in PROMPT_VERSIONS.values())


def generator_output(func: str = GEN_FUNC, message: str = "refactor bounds handling") -> str:
    return (
        f"{gp.marker(gp.M_COMMIT_MESSAGE)}\n{message}\n"
        f"{gp.marker(gp.M_MODIFIED_FUNCTION)}\n{func}\n"
        f"{gp.marker(gp.M_DECEPTION_NOTES)}\nThe length check is gone.\n"
    )


def test_split_sections_and_require_sections():
    sections = gp.split_sections(generator_output())
    assert set(gp.GENERATOR_SECTIONS) <= set(sections)
    assert sections[gp.M_MODIFIED_FUNCTION] == GEN_FUNC
    assert gp.require_sections(generator_output(), gp.GENERATOR_SECTIONS) is not None
    assert gp.require_sections("no markers here", gp.GENERATOR_SECTIONS) is None


def test_split_sections_keeps_the_first_of_a_repeated_marker():
    text = generator_output() + generator_output(func="int other() { return 1; }")
    assert gp.split_sections(text)[gp.M_MODIFIED_FUNCTION] == GEN_FUNC


@pytest.mark.parametrize(
    "body, expected",
    [
        ("- One thing.\n- Two things.", ["One thing.", "Two things."]),
        ("1. One thing.\n2) Two things.", ["One thing.", "Two things."]),
        ("NONE", []),
        ("  none  ", []),
        ("", []),
    ],
)
def test_parse_strategy_entries(body, expected):
    text = f"{gp.marker(gp.M_NEW_STRATEGIES)}\n{body}\n"
    assert gp.parse_strategy_entries(text, max_entries=5) == expected


def test_parse_strategy_update_without_marker_takes_the_bare_list():
    """A bare list without the marker line is still parsed."""
    parse = gp.parse_strategy_update("- just a list, no marker", max_entries=5)
    assert parse.entries == ["just a list, no marker"] and parse.marker_present is False
    assert gp.parse_strategy_entries("- a\n- b\n- c", max_entries=2) == ["a", "b"]


def test_parse_strategy_update_prose_without_list_is_malformed():
    parse = gp.parse_strategy_update("I have nothing to add.", max_entries=5)
    assert parse.entries is None and parse.marker_present is False
    assert gp.parse_strategy_entries("", max_entries=5) is None


def test_parse_strategy_update_bare_none_without_marker_is_nothing_learned():
    assert gp.parse_strategy_update("NONE", max_entries=5) == gp.StrategyParse([], False)


def test_parse_strategy_update_with_marker_reports_it():
    text = f"{gp.marker(gp.M_NEW_STRATEGIES)}\n- One thing.\n"
    assert gp.parse_strategy_update(text, max_entries=5) == gp.StrategyParse(["One thing."], True)


_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        (
            "gan_update_markerless_numbered.txt",
            [
                "Verify input validation: Always check if user-provided input is validated and "
                "sanitized before using it to prevent potential buffer overflow vulnerabilities.",
                "Consider indirect attacks: Be aware that a seemingly cosmetic change may "
                "introduce a vulnerability that can be exploited indirectly, such as through a "
                "buffer overflow attack on a related function.",
            ],
        ),
        (
            "gan_update_markerless_bullets.txt",
            [
                "When assessing the impact of a code change, verify that the change being "
                "evaluated is explicitly stated in the commit message, not just a related issue.",
                "Always consider the broader context of the commit message, including any "
                "disclaimers or caveats, to ensure accurate interpretation of the change's intent.",
            ],
        ),
    ],
)
def test_parse_strategy_update_real_smoke_outputs(fixture, expected):
    """Real markerless model update answers are parsed into the expected entries."""
    parse = gp.parse_strategy_update((_FIXTURES / fixture).read_text(), max_entries=2)
    assert parse.marker_present is False
    assert parse.entries == expected


def test_parse_strategy_entries_truncates_beyond_the_cap():
    body = "\n".join(f"- entry {i}" for i in range(5))
    entries = gp.parse_strategy_entries(f"{gp.marker(gp.M_NEW_STRATEGIES)}\n{body}", max_entries=2)
    assert entries == ["entry 0", "entry 1"]


def test_extract_function_name_skips_keywords():
    assert gp.extract_function_name("if (x) { copy_in(a, b); }") == "copy_in"
    assert gp.extract_function_name(SOURCE_FUNC) == "copy_in"
    assert gp.extract_function_name("no call here") is None


@pytest.mark.parametrize(
    "func, reason",
    [
        (GEN_FUNC, None),
        ("", "empty_function"),
        ("   ", "empty_function"),
        ("```c\nint copy_in() { return 0; }\n```", "markdown_fence"),
        ("int copy_in(char *s) { if (s) {", "unbalanced_braces"),
        ("int copy_in(char *s) return 0;", "unbalanced_braces"),
        ("int other(char *s) { return 0; }", "function_name_lost"),
        (SOURCE_FUNC, "unchanged"),
        ("int copy_in() { " + "int x; " * 60 + "}", "length_out_of_range"),
        ("int copy_in() {}", "length_out_of_range"),
    ],
)
def test_validate_generated_function(func, reason):
    assert gp.validate_generated_function(func, SOURCE_FUNC) == reason
    assert reason is None or reason in gp.REJECTION_REASONS


def test_memory_starts_empty_and_renders_nothing():
    s_g, s_d = new_memories(
        {"policy": "fifo", "max_entries": 12, "max_entry_words": 30, "dedup": True}
    )
    assert len(s_g) == 0 and len(s_d) == 0
    assert s_g.render() == "" and s_d.render() == ""
    assert s_g.role == "generator" and s_d.role == "detector"


def test_add_normalises_deduplicates_and_counts():
    mem = memory([])
    report = mem.add(["- Check bounds.", "Check   bounds.", "Verify the length."])
    assert mem.entries == ["Check bounds.", "Verify the length."]
    assert len(report.added) == 2 and report.duplicates == 1
    report2 = mem.add(["check bounds"])
    assert report2.duplicates == 1 and len(mem) == 2


def test_add_truncates_over_long_entries():
    mem = StrategyMemory(role="detector", max_entry_words=5)
    report = mem.add(["one two three four five six seven"])
    assert mem.entries == ["one two three four five ..."]
    assert report.truncated == 1


def test_dedup_can_be_disabled():
    mem = StrategyMemory(role="detector", dedup=False)
    mem.add(["Same lesson.", "Same lesson."])
    assert len(mem) == 2


def test_fifo_keeps_the_newest_and_head_the_oldest():
    entries = [f"Lesson {i}." for i in range(5)]
    fifo = StrategyMemory(role="detector", policy="fifo", max_entries=3)
    report = fifo.add(entries)
    assert fifo.entries == ["Lesson 2.", "Lesson 3.", "Lesson 4."]
    assert report.evicted == 2

    head = StrategyMemory(role="detector", policy="head", max_entries=3)
    head.add(entries)
    assert head.entries == ["Lesson 0.", "Lesson 1.", "Lesson 2."]


def test_unbounded_never_evicts():
    mem = StrategyMemory(role="detector", policy="unbounded", max_entries=2)
    mem.add([f"Lesson {i}." for i in range(6)])
    assert len(mem) == 6


class SummarizeFake:
    """Returns a condensed list; records the prompts it was given."""

    def __init__(self, entries=("Merged lesson A.", "Merged lesson B.")):
        self.entries = list(entries)
        self.prompts: list[str] = []

    def generate(self, prompts, **kwargs):
        self.prompts += list(prompts)
        body = "\n".join(f"- {e}" for e in self.entries)
        return [f"{gp.marker(gp.M_NEW_STRATEGIES)}\n{body}\n"]


def test_summarize_policy_condenses_via_the_model():
    fake = SummarizeFake()
    mem = StrategyMemory(role="detector", policy="summarize", max_entries=2)
    report = mem.add([f"Lesson {i}." for i in range(4)], model=fake)
    assert mem.entries == ["Merged lesson A.", "Merged lesson B."]
    assert report.summarized is True
    assert "Condense the following list of 4 detector strategies" in fake.prompts[0]


def test_summarize_policy_without_a_model_fails_loudly():
    mem = StrategyMemory(role="detector", policy="summarize", max_entries=1)
    with pytest.raises(ValueError, match="needs a model"):
        mem.add(["A.", "B."])


def test_summarize_falls_back_to_fifo_when_unparseable():
    class Empty:
        def generate(self, prompts, **kwargs):
            return ["I cannot summarise."]

    mem = StrategyMemory(role="detector", policy="summarize", max_entries=2)
    mem.add(["A one.", "B two.", "C three."], model=Empty())
    assert mem.entries == ["B two.", "C three."]


def test_memory_round_trip_and_validation():
    mem = memory(["Lesson one.", "Lesson two."])
    restored = StrategyMemory.from_dict(json.loads(json.dumps(mem.to_dict())))
    assert restored.to_dict() == mem.to_dict()
    assert restored.render() == mem.render()
    with pytest.raises(ValueError, match="unknown memory role"):
        StrategyMemory(role="reflector")
    with pytest.raises(ValueError, match="unknown capacity policy"):
        StrategyMemory(role="detector", policy="lru")
    assert set(CAPACITY_POLICIES) == {"fifo", "head", "summarize", "unbounded"}


def test_stats_and_helpers():
    mem = memory(["Lesson one.", "Lesson two."])
    stats = mem.stats()
    assert stats["n_entries"] == 2 and stats["n_chars"] > 0 and stats["approx_tokens"] > 0
    assert dedup_key("Check the Bounds!") == "check the bounds"
    assert normalise_entry("  - a  b  ", max_words=10) == "a b"


def test_snapshot_round_trip_carries_provenance(tmp_path):
    s_g, s_d = memory(["Hide it."], role="generator"), memory(["Find it."])
    path = write_snapshot(
        tmp_path / "snapshots",
        round_no=50,
        phase="A",
        s_g=s_g,
        s_d=s_d,
        seed=42,
        config_path="configs/gan.yaml",
        config_sha256="deadbeef",
        probe={"vulnerable_share": 0.9},
    )
    assert path.name == "round0050.json"
    snap = read_snapshot(path)
    assert snap["round"] == 50 and snap["phase"] == "A" and snap["seed"] == 42
    assert snap["config_sha256"] == "deadbeef"
    assert snap["prompt_versions"] == PROMPT_VERSIONS
    assert "git_commit" in snap
    assert snap["drift_probe"]["vulnerable_share"] == 0.9
    restored = detector_memory(snap)
    assert restored.entries == ["Find it."] and restored.role == "detector"
    assert restored.render() == s_d.render()


def test_snapshot_names_and_resolution(tmp_path):
    assert snapshot_name(0) == "round0000"
    assert snapshot_name(None) == "final"
    assert resolve_snapshot_path(tmp_path, "final") == tmp_path / "final.json"
    assert resolve_snapshot_path(tmp_path, "50") == tmp_path / "round0050.json"
    assert resolve_snapshot_path(tmp_path, "round0010") == tmp_path / "round0010.json"
    explicit = tmp_path / "elsewhere" / "round0010.json"
    assert resolve_snapshot_path(tmp_path, str(explicit)) == explicit


def test_unknown_phase_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown phase"):
        write_snapshot(
            tmp_path,
            round_no=1,
            phase="C",
            s_g=memory([], role="generator"),
            s_d=memory([]),
            seed=42,
            config_path="c.yaml",
            config_sha256="x",
        )


def test_list_snapshots_orders_rounds_then_final(tmp_path):
    for round_no in (0, 10, 200, None):
        write_snapshot(
            tmp_path,
            round_no=round_no,
            phase="A" if round_no is not None else "B",
            s_g=memory([], role="generator"),
            s_d=memory([]),
            seed=42,
            config_path="c.yaml",
            config_sha256="x",
        )
    assert [p.stem for p in list_snapshots(tmp_path)] == [
        "round0000",
        "round0010",
        "round0200",
        "final",
    ]


def test_read_snapshot_missing_file_points_at_the_loop(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the GAN loop first"):
        read_snapshot(tmp_path / "final.json")


CONFIG_DIR = Path(__file__).parents[1] / "configs"


def load_cfg(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def test_official_gan_config_evaluates_on_test_and_logs():
    cfg = load_cfg("gan")
    assert cfg["experiment"] == "gan"
    assert cfg["dataset"]["split"] == "test"
    assert cfg["dataset"]["train_split"] == "train"
    assert cfg["output"].get("log_experiments", True) is True
    assert 0 in cfg["gan"]["snapshot_rounds"]
    assert cfg["gan"]["rounds"] in cfg["gan"]["snapshot_rounds"]
    assert cfg["gan"]["phase_b"]["enabled"] is True
    assert cfg["gan"]["drift_probe"]["split"] == "val"
    assert cfg["model"]["temperature"] == 0.0
    assert cfg["gan"]["generation"]["temperature"] > 0


@pytest.mark.parametrize("stage", ["gan_val", "gan_smoke", "gan_val_phaseb_only"])
def test_diagnostic_gan_configs_never_touch_test_and_never_log(stage):
    cfg = load_cfg(stage)
    assert cfg["experiment"] == stage
    assert cfg["dataset"]["split"] == "val"
    assert cfg["output"]["log_experiments"] is False
    assert cfg["output"]["loop_dir"] != load_cfg("gan")["output"]["loop_dir"]
    assert cfg["output"]["predictions_dir"] != load_cfg("gan")["output"]["predictions_dir"]


@pytest.mark.parametrize("name", ["gan", "gan_val", "gan_smoke", "gan_val_phaseb_only"])
def test_shipped_configs_provide_every_key_the_loop_requires(name):
    """Shipped GAN configs contain every key the loop reads."""
    cfg = load_cfg(name)
    gan = cfg["gan"]
    for key in ("rounds", "snapshot_rounds", "max_new_strategies_per_update"):
        assert key in gan
    assert set(gan["memory"]) >= {"policy", "max_entries", "max_entry_words", "dedup"}
    assert set(gan["generation"]) >= {"validate", "temperature", "top_p", "max_new_tokens"}
    assert set(gan["phase_b"]) >= {"enabled", "n_items"}
    assert set(gan["drift_probe"]) >= {"enabled", "split", "n"}
    assert gan["memory"]["policy"] in CAPACITY_POLICIES
    assert gan["generation"]["validate"] in ("structural", "none")
    assert max(gan["snapshot_rounds"]) <= gan["rounds"]
    s_g, s_d = new_memories(gan["memory"])
    assert len(s_g) == 0 and len(s_d) == 0
    for key in ("loop_dir", "predictions_dir"):
        assert key in cfg["output"]
    assert cfg["dataset"]["train_split"] == "train"


def test_official_eval_reads_the_val_stage_snapshots():
    """The official GAN eval reads the snapshots written by the val stage."""
    base, variant = load_cfg("gan"), load_cfg("gan_val")
    assert base["output"]["snapshots_from"] == variant["output"]["loop_dir"]
    assert "snapshots_from" not in variant["output"]
    assert "snapshots_from" not in load_cfg("gan_smoke")["output"]


def test_gan_eval_decoding_matches_the_shared_baseline_decoding():
    """GAN eval decoding matches the baseline while the loop detector stays greedy."""
    base = load_cfg("baseline")["model"]
    for name in (
        "gan",
        "gan_val",
        "gan_smoke",
        "gan_val_phaseb_only",
        "gan_val_seed43",
        "gan_val_seed44",
        "gan_seed43",
        "gan_seed44",
    ):
        cfg = load_cfg(name)
        assert (cfg["eval"]["temperature"], cfg["eval"]["top_p"]) == (
            base["temperature"],
            base["top_p"],
        )
        assert cfg["eval"]["unparseable"] == load_cfg("baseline")["eval"]["unparseable"]
        assert cfg["model"]["temperature"] == 0.0


def test_phase_b_only_ablation_differs_from_gan_val_only_in_rounds_and_paths():
    """The Phase-B-only ablation matches gan_val except for rounds and output paths."""
    base, abl = load_cfg("gan_val"), load_cfg("gan_val_phaseb_only")
    assert abl["gan"]["rounds"] == 0 and abl["gan"]["snapshot_rounds"] == [0]
    for key in ("memory", "generation", "phase_b", "drift_probe", "max_new_strategies_per_update"):
        assert abl["gan"][key] == base["gan"][key]
    assert abl["model"] == base["model"] and abl["eval"] == base["eval"]
    assert abl["dataset"] == base["dataset"] and abl["seed"] == base["seed"]
    assert abl["output"]["loop_dir"] != base["output"]["loop_dir"]
    assert "snapshots_from" not in abl["output"]


def test_gan_val_is_a_true_dress_rehearsal_of_the_official_run():
    base, variant = load_cfg("gan"), load_cfg("gan_val")
    assert variant["seed"] == base["seed"]
    assert variant["model"] == base["model"]
    assert variant["gan"] == base["gan"]
    assert set(variant) == set(base)


GAN_EXTRA_SEEDS = (43, 44)


@pytest.mark.parametrize("seed", GAN_EXTRA_SEEDS)
def test_seed_variants_differ_from_seed42_only_in_seed_and_experiment(seed):
    """Extra-seed configs differ from seed 42 only in seed and experiment name."""
    for base_name, name in (("gan_val", f"gan_val_seed{seed}"), ("gan", f"gan_seed{seed}")):
        base, variant = load_cfg(base_name), load_cfg(name)
        assert variant["seed"] == seed
        assert variant["experiment"] == name
        if base_name == "gan_val":
            # The drift probe seed is pinned to 42, so it is removed before comparing.
            assert variant["gan"]["drift_probe"].pop("seed") == base["seed"] == 42
            assert "seed" not in base["gan"]["drift_probe"]
        rest = lambda cfg: {k: v for k, v in cfg.items() if k not in ("seed", "experiment")}  # noqa: E731
        assert rest(variant) == rest(base), name


@pytest.mark.parametrize("seed", GAN_EXTRA_SEEDS)
def test_seed_test_config_reads_that_seeds_loop_snapshots(seed):
    """Each seed test config evaluates the snapshots of its own val loop."""
    from vulndetect.gan.run import snapshot_dir_for

    cfg, loop = load_cfg(f"gan_seed{seed}"), load_cfg(f"gan_val_seed{seed}")
    assert cfg["output"]["snapshots_from"] == loop["output"]["loop_dir"]
    expected = Path(loop["output"]["loop_dir"]) / f"seed{seed}" / "snapshots"
    assert snapshot_dir_for(cfg, seed) == expected
    assert cfg["dataset"]["split"] == "test" and cfg["output"]["log_experiments"] is True
    assert loop["dataset"]["split"] == "val" and loop["output"]["log_experiments"] is False

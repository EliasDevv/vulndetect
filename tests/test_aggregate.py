"""Offline tests for vulndetect.eval.aggregate on synthetic run directories."""

from __future__ import annotations

import json

import pytest

from vulndetect.common.predictions import write_predictions
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
from vulndetect.eval.aggregate import (
    AggregationError,
    aggregate,
    aggregate_approach,
    check_consistency,
    discover_runs,
    load_reference,
    render_latex,
    render_markdown,
    split_approach,
    write_outputs,
)
from vulndetect.eval.metrics import evaluate

CWES = ("CWE-119", "CWE-20")


def make_preds(flip_first: bool = False, none_first: bool = False) -> list[dict]:
    """Build four predictions per CWE (two vulnerable, two safe), correct unless flipped."""
    preds = []
    for cwe in CWES:
        for i, gold in enumerate([LABEL_VULN, LABEL_VULN, LABEL_SAFE, LABEL_SAFE]):
            pred = gold
            if flip_first and i == 0:
                pred = LABEL_SAFE if gold == LABEL_VULN else LABEL_VULN
            if none_first and i == 0:
                pred = None
            preds.append(
                {
                    "id": f"{cwe}-{i}",
                    "cwe": cwe,
                    "gold_label": gold,
                    "pred_label": pred,
                    "raw_output": pred if pred is not None else "no verdict given",
                }
            )
    return preds


def write_run_dir(
    root,
    approach: str,
    seed: int,
    preds: list[dict],
    policy: str = "drop",
    manifest_overrides: dict | None = None,
    with_predictions: bool = True,
    metrics_overrides: dict | None = None,
    snapshot: str | None = None,
) -> None:
    run_dir = root / approach / f"seed{seed}"
    if snapshot is not None:
        run_dir = run_dir / snapshot
    run_dir.mkdir(parents=True)
    manifest = {
        "approach": approach,
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "seed": seed,
        "prompt_version": "authors-strict-v1",
        "parser_version": "authors-v1",
        "unparseable_policy": policy,
        "dataset": {"name": "bigvul", "split": "test", "n": len(preds), "limit": None},
        "decoding": {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 256, "batch_size": 2},
        **({"gan": {"snapshot": snapshot}} if snapshot is not None else {}),
        **(manifest_overrides or {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = {**evaluate(preds, policy=policy), **(metrics_overrides or {})}
    (run_dir / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    if with_predictions:
        write_predictions(preds, run_dir / "predictions.jsonl")


@pytest.fixture
def runs_root(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds())
    write_run_dir(root, "a", 43, make_preds())
    write_run_dir(root, "b", 42, make_preds(flip_first=True))
    write_run_dir(root, "b", 43, make_preds())
    return root


def test_discover_runs_orders_seeds(runs_root):
    runs = discover_runs(runs_root, "a")
    assert [r["seed"] for r in runs] == [42, 43]


def test_discover_missing_approach_fails(runs_root):
    with pytest.raises(AggregationError, match="does not exist"):
        discover_runs(runs_root, "nope")


def test_discover_seed_mismatch_fails(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds(), manifest_overrides={"seed": 99})
    with pytest.raises(AggregationError, match="directory seed"):
        discover_runs(root, "a")


def test_consistency_rejects_mixed_provenance(runs_root):
    runs = discover_runs(runs_root, "a")
    runs[1]["manifest"]["prompt_version"] = "fig1-v2"
    with pytest.raises(AggregationError, match="prompt_version"):
        check_consistency("a", runs, split="test")


def test_consistency_rejects_limit_and_wrong_split(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds())
    runs = discover_runs(root, "a")
    with pytest.raises(AggregationError, match="expected 'val'"):
        check_consistency("a", runs, split="val")
    runs[0]["manifest"]["dataset"]["limit"] = 10
    with pytest.raises(AggregationError, match="smoke"):
        check_consistency("a", runs, split="test")


def test_aggregate_rejects_mixed_policies(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds(), policy="drop")
    write_run_dir(root, "b", 42, make_preds(), policy="wrong")
    with pytest.raises(AggregationError, match="policies"):
        aggregate(root, ("a", "b"), split="test")


def test_aggregate_approach_mean_std(runs_root):
    agg = aggregate_approach(discover_runs(runs_root, "b"))
    # seed 42: one flip per CWE of 4 -> per-CWE acc 0.75; seed 43: 1.0
    cwe = agg["per_cwe"]["CWE-119"]
    assert cwe["accuracy"]["mean"] == pytest.approx(0.875)
    assert cwe["accuracy"]["values"] == [0.75, 1.0]
    assert cwe["accuracy"]["std"] == pytest.approx(0.1768, abs=1e-4)
    assert agg["macro"]["accuracy"]["mean"] == pytest.approx(0.875)
    assert agg["seeds"] == [42, 43]
    assert agg["alt"]["policy"] == "wrong"
    assert set(agg["secondary"]["macro"]) == {"mcc", "youden_j", "pred_positive_rate"}
    assert agg["secondary"]["macro"]["mcc"]["values"][1] == pytest.approx(1.0)
    assert agg["secondary"]["macro"]["pred_positive_rate"]["values"][1] == pytest.approx(0.5)
    assert "mcc" not in agg["macro"]


def test_aggregate_single_seed_has_no_std(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds())
    agg = aggregate_approach(discover_runs(root, "a"))
    assert agg["macro"]["f1"]["std"] is None


def test_full_aggregate_ttest_and_mcnemar(runs_root):
    comparison = aggregate(runs_root, ("a", "b"), split="test")
    assert comparison["policy"] == "drop"
    assert set(comparison["approaches"]) == {"a", "b"}

    (ttest,) = comparison["ttests"]
    assert {ttest["a"], ttest["b"]} == {"a", "b"}
    assert ttest["mean_diff"] != 0.0
    assert 0.0 <= ttest["p"] <= 1.0

    by_seed = {m["seed"]: m for m in comparison["mcnemar"]}
    assert by_seed[42]["n_a_correct_b_wrong"] == 2
    assert by_seed[42]["n_a_wrong_b_correct"] == 0
    assert by_seed[43]["n_a_correct_b_wrong"] == 0 and by_seed[43]["p"] == 1.0
    assert by_seed[42]["a"] == "a" and by_seed[42]["b"] == "b"


def test_mcnemar_skips_missing_predictions(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds())
    write_run_dir(root, "b", 42, make_preds(), with_predictions=False)
    comparison = aggregate(root, ("a", "b"), split="test")
    (entry,) = comparison["mcnemar"]
    assert "rsync" in entry["skipped"]


def test_aggregate_rejects_different_sizes(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds())
    write_run_dir(root, "b", 42, make_preds()[:4], manifest_overrides=None)
    with pytest.raises(AggregationError, match="different dataset sizes"):
        aggregate(root, ("a", "b"), split="test")


def test_write_outputs_and_rendering(runs_root, tmp_path):
    comparison = aggregate(runs_root, ("a", "b"), split="test")
    paths = write_outputs(comparison, tmp_path / "metrics")
    assert all(path.exists() for path in paths.values())

    md = paths["md"].read_text(encoding="utf-8")
    assert "All (Avg) overview" in md
    assert "| a |" in md and "| b |" in md
    assert "Paired t-tests" in md and "McNemar" in md
    assert "±" in md

    tex = paths["tex"].read_text(encoding="utf-8")
    assert tex.startswith("% Auto-generated")
    assert "\\begin{tabular}" in tex and "\\bottomrule" in tex

    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["split"] == "test"
    assert loaded["approaches"]["a"]["macro"]["f1"]["mean"] == pytest.approx(1.0)


def test_reference_csv_merges_into_markdown(runs_root, tmp_path):
    ref_csv = tmp_path / "paper_reference.csv"
    ref_csv.write_text(
        "approach,metric,value\nbaseline,f1,0.67\nbaseline,accuracy,0.65\n", encoding="utf-8"
    )
    reference = load_reference(ref_csv)
    assert reference == {"baseline": {"f1": 0.67, "accuracy": 0.65}}
    comparison = aggregate(runs_root, ("a", "b"), split="test")
    md = render_markdown(comparison, reference)
    assert "baseline (paper)" in md
    assert "0.670" in md


def test_latex_single_seed_has_no_pm(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds())
    comparison = aggregate(root, ("a",), split="test")
    tex = render_latex(comparison)
    assert "$\\pm$" not in tex


def test_mcnemar_keeps_approach_names_distinct_from_counts(tmp_path):
    """Checks that McNemar's discordant counts do not overwrite the approach names."""
    root = tmp_path / "predictions"
    write_run_dir(root, "rag", 42, make_preds())
    write_run_dir(root, "sft", 42, make_preds())
    comparison = aggregate(root, ("rag", "sft"), split="test")

    (entry,) = comparison["mcnemar"]
    assert entry["a"] == "rag"
    assert entry["b"] == "sft"
    assert isinstance(entry["n_a_correct_b_wrong"], int)
    assert isinstance(entry["n_a_wrong_b_correct"], int)

    row = [ln for ln in render_markdown(comparison).splitlines() if ln.startswith("| rag | sft |")]
    assert row, "McNemar row for rag/sft missing from the rendered table"


def _preds_only_d09_can_read() -> list[dict]:
    """Build predictions in fix/mitigate wording that the authors-v1 parser cannot read."""
    preds = []
    for cwe in CWES:
        for i, gold in enumerate([LABEL_VULN, LABEL_VULN, LABEL_SAFE, LABEL_SAFE]):
            raw = (
                "Therefore the change fixes a vulnerability."
                if gold == LABEL_VULN
                else "Therefore the change does not fix a vulnerability."
            )
            preds.append(
                {
                    "id": f"{cwe}-{i}",
                    "cwe": cwe,
                    "gold_label": gold,
                    "pred_label": None,
                    "raw_output": raw,
                }
            )
    return preds


def test_mcnemar_and_provenance_follow_the_parser_in_metrics_json(tmp_path):
    """Checks that McNemar and provenance use the parser recorded in metrics.json."""
    from vulndetect.common.parsing import PARSERS

    root = tmp_path / "predictions"
    for approach in ("a", "b"):
        preds = _preds_only_d09_can_read()
        rescored = [{**p, "pred_label": PARSERS["d09-v2"](p["raw_output"])} for p in preds]
        assert all(p["pred_label"] == p["gold_label"] for p in rescored)
        write_run_dir(root, approach, 42, preds)
        report = {**evaluate(rescored, policy="drop"), "parser_version": "d09-v2"}
        (root / approach / "seed42" / "metrics.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    comparison = aggregate(root, ("a", "b"), split="test")
    assert comparison["parser_version"] == "d09-v2"
    for approach in ("a", "b"):
        prov = comparison["provenance"][approach]
        assert prov["parser_version"] == "d09-v2"
        assert prov["manifest_parser_version"] == "authors-v1"
    (m,) = comparison["mcnemar"]
    assert m["parser_version"] == "d09-v2"
    assert m["policy"] == "drop"
    assert (m["n"], m["n_excluded"]) == (8, 0)
    assert (m["n_a_correct_b_wrong"], m["n_a_wrong_b_correct"]) == (0, 0)
    assert comparison["approaches"]["a"]["macro"]["f1"]["mean"] == pytest.approx(1.0)


def test_aggregate_rejects_mixed_parsers(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds())
    write_run_dir(root, "b", 42, make_preds(), metrics_overrides={"parser_version": "d09-v2"})
    with pytest.raises(AggregationError, match="mixed parsers"):
        aggregate(root, ("a", "b"), split="test")


def test_mcnemar_excludes_unparseable_pairs_under_drop(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds(none_first=True))
    write_run_dir(root, "b", 42, make_preds(flip_first=True))
    (m,) = aggregate(root, ("a", "b"), split="test")["mcnemar"]
    assert (m["n"], m["n_excluded"]) == (6, 2)
    assert (m["n_a_correct_b_wrong"], m["n_a_wrong_b_correct"]) == (0, 0)


@pytest.fixture
def gan_root(tmp_path):
    """Snapshot runs with the full ladder on seed 42 and round0000/final on seeds 43 and 44."""
    root = tmp_path / "predictions"
    write_run_dir(root, "baseline", 42, make_preds(flip_first=True))
    for snap in ("round0000", "round0050", "final"):
        write_run_dir(root, "gan", 42, make_preds(flip_first=snap == "round0000"), snapshot=snap)
    for seed in (43, 44):
        write_run_dir(root, "gan", seed, make_preds(flip_first=True), snapshot="round0000")
        write_run_dir(root, "gan", seed, make_preds(), snapshot="final")
    return root


def test_split_approach():
    assert split_approach("baseline") == ("baseline", None)
    assert split_approach("gan/final") == ("gan", "final")
    for bad in ("gan/", "/final", "gan/seed42/final"):
        with pytest.raises(AggregationError, match="bad approach"):
            split_approach(bad)


def test_discover_snapshot_runs_across_seeds(gan_root):
    runs = discover_runs(gan_root, "gan/final")
    assert [r["seed"] for r in runs] == [42, 43, 44]
    assert all(r["dir"].name == "final" for r in runs)


def test_discover_skips_seeds_without_the_snapshot(gan_root, caplog):
    runs = discover_runs(gan_root, "gan/round0050")
    assert [r["seed"] for r in runs] == [42]
    assert "seed43" in caplog.text and "seed44" in caplog.text


def test_discover_unknown_snapshot_fails(gan_root):
    with pytest.raises(AggregationError, match="no seed\\*/round9999/ runs"):
        discover_runs(gan_root, "gan/round9999")


def test_discover_rejects_manifest_naming_another_snapshot(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(
        root,
        "gan",
        42,
        make_preds(),
        snapshot="final",
        manifest_overrides={"gan": {"snapshot": "round0000"}},
    )
    with pytest.raises(AggregationError, match="manifest names snapshot"):
        discover_runs(root, "gan/final")


def test_aggregate_gan_snapshots_next_to_a_flat_approach(gan_root):
    comparison = aggregate(gan_root, ("baseline", "gan/round0000", "gan/final"), "test")
    assert comparison["approaches"]["gan/final"]["seeds"] == [42, 43, 44]
    assert comparison["approaches"]["baseline"]["seeds"] == [42]
    pairs = {(t["a"], t["b"]) for t in comparison["ttests"]}
    assert ("gan/final", "gan/round0000") in pairs and ("baseline", "gan/final") in pairs
    within = [
        m for m in comparison["mcnemar"] if {m["a"], m["b"]} == {"gan/round0000", "gan/final"}
    ]
    assert sorted(m["seed"] for m in within) == [42, 43, 44]
    across = [m for m in comparison["mcnemar"] if {m["a"], m["b"]} == {"baseline", "gan/final"}]
    assert [m["seed"] for m in across] == [42]
    assert "gan/final" in render_markdown(comparison) and "gan/final" in render_latex(comparison)


def test_mcnemar_rows_carry_sensitivity_and_direction(tmp_path):
    root = tmp_path / "predictions"
    write_run_dir(root, "a", 42, make_preds())
    write_run_dir(root, "b", 42, make_preds(none_first=True))
    comparison = aggregate(root, ("a", "b"), split="test")
    (row,) = comparison["mcnemar"]
    assert row["policy"] == "drop" and row["n_excluded"] == 2 and row["direction"] == "tie"
    sens = row["sensitivity"]
    assert sens["policy"] == "wrong" and sens["n"] == 8
    assert (sens["n_a_correct_b_wrong"], sens["n_a_wrong_b_correct"]) == (2, 0)
    assert sens["direction"] == "a>b"


def test_seed_verdict_takes_p_max_and_requires_one_direction(tmp_path):
    from vulndetect.eval.aggregate import seed_verdicts

    root = tmp_path / "predictions"
    for seed in (42, 43, 44):
        write_run_dir(root, "a", seed, make_preds())
        write_run_dir(root, "b", seed, make_preds(flip_first=True))
    comparison = aggregate(root, ("a", "b"), split="test")
    (v,) = comparison["verdicts"]["pairs"]
    ps = [row["p"] for row in comparison["mcnemar"]]
    assert v["seeds"] == [42, 43, 44]
    assert v["p_max"] == max(ps) and v["direction"] == "a>b"
    assert v["p_by_seed"] == {42: ps[0], 43: ps[1], 44: ps[2]}
    assert v["p_max"] == pytest.approx(0.5) and v["significant"] is False

    rows = [
        {
            "a": "x",
            "b": "y",
            "seed": s,
            "policy": "drop",
            "p": p,
            "direction": d,
            "sensitivity": {"policy": "wrong", "p": p, "direction": d},
        }
        for s, p, d in ((42, 0.001, "a>b"), (43, 0.02, "a>b"), (44, 0.03, "a>b"))
    ]
    (v,) = seed_verdicts(rows)["pairs"]
    assert v["p_max"] == 0.03 and v["significant"] is True
    rows[2]["direction"] = "b>a"
    (v,) = seed_verdicts(rows)["pairs"]
    assert v["direction"] == "mixed" and v["significant"] is False
    rows[2]["direction"] = "a>b"
    rows[1]["p"] = 0.2
    (v,) = seed_verdicts(rows)["pairs"]
    assert v["p_max"] == 0.2 and v["significant"] is False


def test_family_holm_adjusts_p_max_together(tmp_path):
    from vulndetect.eval.aggregate import AggregationError, parse_families, seed_verdicts

    families = parse_families(["g:base, g:dual", "t:g"])
    assert families == [[("g", "base"), ("g", "dual")], [("t", "g")]]
    with pytest.raises(AggregationError, match="a:b"):
        parse_families(["g-base"])

    def row(a, b, p, p_alt):
        return {
            "a": a,
            "b": b,
            "seed": 42,
            "policy": "drop",
            "p": p,
            "direction": "a>b",
            "sensitivity": {"policy": "wrong", "p": p_alt, "direction": "a>b"},
        }

    rows = [row("g", "base", 0.01, 0.03), row("g", "dual", 0.04, 0.02), row("t", "g", 0.2, 0.2)]
    out = seed_verdicts(rows, [[("g", "base"), ("dual", "g")]])
    by_pair = {(v["a"], v["b"]): v for v in out["pairs"]}
    # Holm over (0.01, 0.04): 0.01*2 = 0.02, then max(0.02, 0.04) = 0.04
    assert by_pair[("g", "base")]["p_holm"] == pytest.approx(0.02)
    assert by_pair[("g", "dual")]["p_holm"] == pytest.approx(0.04)
    assert by_pair[("g", "dual")]["sensitivity"]["p_holm"] == pytest.approx(0.04)
    assert by_pair[("g", "base")]["sensitivity"]["p_holm"] == pytest.approx(0.04)
    assert "p_holm" not in by_pair[("t", "g")]
    assert out["families"] == [{"index": 0, "pairs": ["g:base", "dual:g"], "size": 2}]
    with pytest.raises(AggregationError, match="no scored McNemar rows"):
        seed_verdicts(rows, [[("g", "nope")]])


def test_family_flag_reaches_the_rendered_output(runs_root, tmp_path):
    comparison = aggregate(runs_root, ("a", "b"), split="test", families=[[("a", "b")]])
    (v,) = comparison["verdicts"]["pairs"]
    assert v["family"] == 0 and v["p_holm"] == v["p_max"]
    md = render_markdown(comparison)
    assert "Seed rule" in md and "p_holm" in md and "sensitivity" in md

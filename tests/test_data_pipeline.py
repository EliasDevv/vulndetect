"""Tests for the canonical Big-Vul data pipeline on synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from vulndetect.data.build import build_dataset, compute_stats, to_records
from vulndetect.data.filter import clean, filter_cwe, filter_lang
from vulndetect.data.load import load_raw
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN, RECORD_FIELDS, TARGET_CWES
from vulndetect.data.split import SPLIT_NAMES, split_by_commit


def test_load_raw_selects_columns(raw_csv):
    df = load_raw(raw_csv)
    for col in ("CWE ID", "func_before", "func_after", "vul", "commit_id"):
        assert col in df.columns
    assert df["vul"].isin([0, 1]).all()


def test_load_raw_missing_file():
    with pytest.raises(FileNotFoundError):
        load_raw("does/not/exist.csv")


def test_filter_cwe_exact(raw_df):
    out = filter_cwe(raw_df)
    assert set(out["cwe"].unique()) == set(TARGET_CWES)
    assert "CWE-119;CWE-20" not in set(out["CWE ID"])
    assert "CWE-79" not in set(out["CWE ID"])
    assert "" not in set(out["CWE ID"])


def test_filter_lang_drops_python(raw_df):
    out = filter_lang(filter_cwe(raw_df), ["C", "C++"])
    assert out["lang"].str.lower().isin({"c", "c++"}).all()
    assert "edge_lang" not in set(out["commit_id"])


def test_clean_drops_empty_and_dupes(raw_df):
    out = clean(filter_lang(filter_cwe(raw_df), ["C", "C++"]))
    assert (out["func_before"].str.strip() != "").all()
    assert not out.duplicated(subset=["commit_id", "func_before"]).any()


def _build(raw_df, cfg):
    return build_dataset(raw_df, cfg)


def test_no_commit_overlap(raw_df, base_cfg):
    df = _build(raw_df, base_cfg)
    sets = {s: set(df[df["split"] == s]["commit_id"]) for s in SPLIT_NAMES}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_split_covers_all_rows(raw_df, base_cfg):
    df = _build(raw_df, base_cfg)
    assert set(df["split"].unique()) <= set(SPLIT_NAMES)
    assert df["split"].notna().all()


def test_split_ratios(raw_df, base_cfg):
    pool = clean(filter_lang(filter_cwe(raw_df), ["C", "C++"]))
    df = split_by_commit(pool, base_cfg["split"]["ratios"], seed=base_cfg["seed"])
    n = len(df)
    for s, target in base_cfg["split"]["ratios"].items():
        frac = (df["split"] == s).sum() / n
        assert abs(frac - target) < 0.12, f"{s}: {frac:.2f} vs {target}"


def test_final_splits_balanced_per_cwe(raw_df, base_cfg):
    df = _build(raw_df, base_cfg)
    for split in SPLIT_NAMES:
        sub = df[df["split"] == split]
        for cwe in TARGET_CWES:
            c = sub[sub["cwe"] == cwe]
            n_vuln = int((c["vul"] == 1).sum())
            n_safe = int((c["vul"] == 0).sum())
            assert n_vuln == n_safe, f"{split}/{cwe}: {n_vuln} vuln vs {n_safe} safe"


def test_target_size_caps(raw_df):
    cfg = {
        "seed": 1,
        "target_cwes": list(TARGET_CWES),
        "lang_filter": ["C", "C++"],
        "split": {
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "target_sizes": {"train": 10, "val": 4, "test": 4},
        },
    }
    df = build_dataset(raw_df, cfg)
    assert (df["split"] == "train").sum() <= 10
    assert (df["split"] == "val").sum() <= 4
    assert (df["split"] == "test").sum() <= 4


def test_record_schema(raw_df, base_cfg):
    df = _build(raw_df, base_cfg)
    records = to_records(df)
    assert records
    for rec in records[:50]:
        assert tuple(rec.keys()) == RECORD_FIELDS
        assert rec["label"] in (LABEL_VULN, LABEL_SAFE)
        assert rec["cwe"] in TARGET_CWES
        assert isinstance(rec["vul"], int)
        assert rec["split"] in SPLIT_NAMES
        assert (rec["label"] == LABEL_VULN) == (rec["vul"] == 1)
        assert rec["code"] == rec["func_before"]


def test_cve_id_preserved(raw_df, base_cfg):
    from conftest import KNOWN_CVE, KNOWN_CVE_COMMIT

    from vulndetect.data.schema import NA, make_id

    records = to_records(_build(raw_df, base_cfg))

    expected = {
        make_id(str(r["commit_id"]), str(r["func_before"])): (str(r["CVE ID"]) or NA)
        for r in raw_df.to_dict("records")
    }
    assert records
    for rec in records:
        assert rec["cve_id"] == expected[rec["id"]], rec["id"]

    by_commit = {r["commit_id"]: r for r in records if r["vul"] == 1}
    assert by_commit[KNOWN_CVE_COMMIT]["cve_id"] == KNOWN_CVE
    assert all(r["cve_id"].startswith("CVE-") for r in records if r["vul"] == 1)
    assert all(r["cve_id"] == NA for r in records if r["vul"] == 0)


def test_ids_unique(raw_df, base_cfg):
    records = to_records(_build(raw_df, base_cfg))
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids))


def test_compute_stats_balance(raw_df, base_cfg):
    df = _build(raw_df, base_cfg)
    stats = compute_stats(df, tuple(TARGET_CWES))
    assert stats["n_total"] == len(df)
    total = sum(stats["per_split"][s]["n"] for s in SPLIT_NAMES)
    assert total == len(df)


def test_run_end_to_end(raw_csv, tmp_path, monkeypatch):
    import json

    import yaml

    from vulndetect.data.build import run
    from vulndetect.data.schema import NA

    cfg = {
        "experiment": "data_build",
        "seed": 42,
        "source": {"name": "bigvul", "raw_path": raw_csv},
        "target_cwes": list(TARGET_CWES),
        "lang_filter": ["C", "C++"],
        "balance": "per_cwe",
        "split": {
            "by": "commit_id",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "target_sizes": {"train": 100000, "val": 100000, "test": 100000},
        },
        "output": {
            "dir": str(tmp_path / "processed"),
            "stats_path": str(tmp_path / "stats.json"),
        },
    }
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    stats = run(cfg_path)

    for split in SPLIT_NAMES:
        path = tmp_path / "processed" / f"{split}.jsonl"
        assert path.exists(), split
        with path.open(encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        assert recs, split
        for rec in recs:
            assert tuple(rec.keys()) == RECORD_FIELDS
            assert rec["split"] == split
            assert rec["label"] in (LABEL_VULN, LABEL_SAFE)
            assert rec["cve_id"].startswith("CVE-") or rec["cve_id"] == NA
        assert any(r["cve_id"].startswith("CVE-") for r in recs), split

    manifest = json.loads((tmp_path / "processed" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["source_sha256"]) == 64

    written = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert written["seed"] == 42
    assert written["n_total"] == stats["n_total"] > 0

    assert (tmp_path / "results" / "experiments.csv").exists()


def test_determinism_same_seed(raw_df, base_cfg):
    a = to_records(_build(raw_df, base_cfg))
    b = to_records(_build(raw_df, base_cfg))
    by_split_a = {s: sorted(r["id"] for r in a if r["split"] == s) for s in SPLIT_NAMES}
    by_split_b = {s: sorted(r["id"] for r in b if r["split"] == s) for s in SPLIT_NAMES}
    assert by_split_a == by_split_b


def test_different_seed_changes_assignment(raw_df, base_cfg):
    import copy

    cfg2 = copy.deepcopy(base_cfg)
    cfg2["seed"] = 7
    a = {r["id"]: r["split"] for r in to_records(_build(raw_df, base_cfg))}
    b = {r["id"]: r["split"] for r in to_records(_build(raw_df, cfg2))}
    assert a != b


def test_filter_lang_keeps_bigvul_cpp_spelling():
    """Checks that the language filter keeps Big-Vul's "cpp" spelling of C++."""
    df = pd.DataFrame({"lang": ["C", "cpp", "C++", "Python"], "commit_id": list("abcd")})
    assert filter_lang(df, ["C", "C++", "cpp"])["lang"].tolist() == ["C", "cpp", "C++"]
    assert filter_lang(df, ["C", "C++"])["lang"].tolist() == ["C", "C++"]


def _freeze(raw_df, cfg):
    """Build the legacy dataset and the reference that freezes it."""
    from vulndetect.data.build import assign_and_balance, prepare_pool
    from vulndetect.data.reference import build_reference

    balanced, commit_split = assign_and_balance(prepare_pool(raw_df, cfg), cfg)
    return balanced, build_reference(balanced, commit_split, {"origin": "unit test"})


def _new_commit_rows(raw_df, commit: str, cwe: str, n_vul: int, n_safe: int, lang: str = "cpp"):
    """Rows of a commit the reference has never seen (same columns as raw_df)."""
    template = raw_df.iloc[0].to_dict()
    rows = []
    for i in range(n_vul + n_safe):
        vul = 1 if i < n_vul else 0
        rows.append(
            {
                **template,
                "CWE ID": cwe,
                "commit_id": commit,
                "func_before": f"int {commit}_{i}() {{ return {i}; }}",
                "func_after": f"int {commit}_{i}() {{ return {i + 100}; }}"
                if vul
                else f"int {commit}_{i}() {{ return {i}; }}",
                "lang": lang,
                "vul": vul,
                "CVE ID": "CVE-2021-0001" if vul else "",
            }
        )
    return pd.DataFrame(rows)


def test_reference_round_trip_reproduces_the_legacy_build(raw_df, base_cfg):
    """Checks that without new commits the anchored build equals the legacy build."""
    from vulndetect.data.build import build_dataset_with_reference

    legacy, ref = _freeze(raw_df, base_cfg)
    anchored, stats = build_dataset_with_reference(raw_df, base_cfg, ref)
    assert to_records(anchored) == to_records(legacy)
    assert stats["n_new_commits"] == 0
    assert all(s["n_added"] == 0 for s in stats["per_split"].values())


def test_reference_extension_keeps_old_rows_and_appends_new_commit_pairs(raw_df, base_cfg):
    """Checks that a new commit is assigned by hash while existing rows keep their order."""
    import copy

    from vulndetect.data.build import build_dataset_with_reference
    from vulndetect.data.reference import hash_split

    legacy, ref = _freeze(raw_df, base_cfg)
    raw2 = pd.concat([raw_df, _new_commit_rows(raw_df, "newc", TARGET_CWES[1], 2, 3)])
    cfg2 = copy.deepcopy(base_cfg)
    cfg2["lang_filter"] = ["C", "C++", "cpp"]
    anchored, stats = build_dataset_with_reference(raw2, cfg2, ref)

    old, new = to_records(legacy), to_records(anchored)
    target = hash_split("newc")
    for split in SPLIT_NAMES:
        old_s = [r for r in old if r["split"] == split]
        new_s = [r for r in new if r["split"] == split]
        assert new_s[: len(old_s)] == old_s
        added = new_s[len(old_s) :]
        if split == target:
            assert len(added) == 4
            assert {r["commit_id"] for r in added} == {"newc"}
            assert sum(r["vul"] for r in added) == 2
            assert [r["vul"] for r in added] == [1, 0, 1, 0]
        else:
            assert added == []
    assert stats["n_new_commits"] == 1
    assert stats["per_split"][target]["added_pairs_per_cwe"] == {TARGET_CWES[1]: 2}


def test_reference_extension_respects_frozen_caps(raw_df, base_cfg):
    """Top-up stops at the split cap; the reference rows are never dropped."""
    import copy

    from vulndetect.data.build import build_dataset_with_reference
    from vulndetect.data.reference import hash_split

    legacy, ref = _freeze(raw_df, base_cfg)
    target = hash_split("newc")
    n_old = int((legacy["split"] == target).sum())
    raw2 = pd.concat([raw_df, _new_commit_rows(raw_df, "newc", TARGET_CWES[1], 2, 3)])
    for room, expected_added in ((0, 0), (1, 0), (2, 2), (3, 2), (4, 4)):
        cfg2 = copy.deepcopy(base_cfg)
        cfg2["lang_filter"] = ["C", "C++", "cpp"]
        cfg2["split"]["target_sizes"][target] = n_old + room
        anchored, stats = build_dataset_with_reference(raw2, cfg2, ref)
        assert int((anchored["split"] == target).sum()) == n_old + expected_added, room
        assert stats["per_split"][target]["n_added"] == expected_added


def test_reference_missing_instance_fails_loudly(raw_df, base_cfg):
    from vulndetect.data.build import build_dataset_with_reference

    _, ref = _freeze(raw_df, base_cfg)
    commit = ref["selected"]["train"][0].split(":")[0]
    raw2 = raw_df[raw_df["commit_id"] != commit]
    with pytest.raises(ValueError, match="missing from the pool"):
        build_dataset_with_reference(raw2, base_cfg, ref)


def test_hash_split_is_a_pure_80_10_10_function():
    import collections

    from vulndetect.data.reference import hash_split

    assert hash_split("abc") == hash_split("abc")
    shares = collections.Counter(hash_split(f"commit{i}") for i in range(4000))
    assert 0.76 < shares["train"] / 4000 < 0.84
    assert 0.07 < shares["val"] / 4000 < 0.13
    assert 0.07 < shares["test"] / 4000 < 0.13


def test_cli_write_reference_then_anchored_build_is_byte_identical(raw_csv, tmp_path, monkeypatch):
    """Checks that freezing a reference and rebuilding from it yields identical bytes."""
    import copy
    import json

    import yaml

    from vulndetect.data.build import run

    cfg = {
        "experiment": "data_build",
        "seed": 42,
        "source": {"name": "bigvul", "raw_path": raw_csv},
        "target_cwes": list(TARGET_CWES),
        "lang_filter": ["C", "C++"],
        "balance": "per_cwe",
        "split": {
            "by": "commit_id",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "target_sizes": {"train": 100000, "val": 100000, "test": 100000},
        },
        "output": {"dir": str(tmp_path / "legacy"), "stats_path": str(tmp_path / "s1.json")},
    }
    monkeypatch.chdir(tmp_path)
    p1 = tmp_path / "legacy.yaml"
    p1.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    ref_path = tmp_path / "ref.json"
    run(p1, write_reference_path=ref_path)
    assert ref_path.exists()
    legacy_manifest = json.loads((tmp_path / "legacy" / "manifest.json").read_text())
    assert legacy_manifest["split_reference"] is None

    cfg2 = copy.deepcopy(cfg)
    cfg2["split"]["reference"] = str(ref_path)
    cfg2["output"] = {"dir": str(tmp_path / "anchored"), "stats_path": str(tmp_path / "s2.json")}
    p2 = tmp_path / "anchored.yaml"
    p2.write_text(yaml.safe_dump(cfg2), encoding="utf-8")
    run(p2)
    for split in SPLIT_NAMES:
        assert (tmp_path / "anchored" / f"{split}.jsonl").read_bytes() == (
            tmp_path / "legacy" / f"{split}.jsonl"
        ).read_bytes(), split
    manifest = json.loads((tmp_path / "anchored" / "manifest.json").read_text())
    assert manifest["split_reference"]["n_new_commits"] == 0
    assert manifest["split_hashes"] == legacy_manifest["split_hashes"]
    with pytest.raises(ValueError, match="WITHOUT split.reference"):
        run(p2, write_reference_path=tmp_path / "ref2.json")

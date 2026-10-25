"""Tests for vulndetect.common: label parser, prompt rendering, prediction I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vulndetect.common import (
    PROMPT_VERSION,
    make_prediction,
    parse_label,
    read_predictions,
    render_prompt,
    write_predictions,
    write_run,
)
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN, NA, Record


def make_record(**overrides) -> Record:
    rec: Record = {
        "id": "deadbeef:0123456789ab",
        "code": "int f() { return 1; }",
        "func_before": "int f() { return 1; }",
        "func_after": "int f() { return 2; }",
        "commit_message": "fix overflow",
        "cve_id": "CVE-2020-1234",
        "cwe": "CWE-119",
        "project": "proj",
        "lang": "C",
        "label": LABEL_VULN,
        "vul": 1,
        "commit_id": "deadbeef",
        "split": "test",
    }
    rec.update(overrides)  # type: ignore[typeddict-item]
    return rec


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Vulnerable", LABEL_VULN),
        ("Not Vulnerable", LABEL_SAFE),
        ("not vulnerable", LABEL_SAFE),
        ("NOT VULNERABLE", LABEL_SAFE),
        ("**Vulnerable**. The buffer is copied without bounds checks.", LABEL_VULN),
        ("Answer: Not Vulnerable. The change only renames a variable.", LABEL_SAFE),
        ("The code change is vulnerable because strcpy is unbounded.", LABEL_VULN),
        ("This is not a vulnerable change; it refactors logging.", LABEL_SAFE),
        ("Non-vulnerable: the patch adds a bounds check.", LABEL_SAFE),
        ("not-vulnerable", LABEL_SAFE),
        ("Vulnerable. Although parts look not vulnerable, the copy overflows.", LABEL_VULN),
        ("After analysis: the function is not vulnerable, i.e. no vulnerable path.", LABEL_SAFE),
        ("The code is invulnerable to injection.", None),
        ("I cannot determine the security impact of this change.", None),
        ("", None),
        # Semantic verdicts about whether the change fixes a vulnerability.
        ("Therefore, the code change fixes a vulnerability.", LABEL_VULN),
        ("The modification fixes an existing issue in the bounds check.", LABEL_VULN),
        ("The change addresses the vulnerability described in the CVE.", LABEL_VULN),
        ("In conclusion, the code change does not fix or mitigate a vulnerability.", LABEL_SAFE),
        ("Overall, the change is neutral from a security perspective.", LABEL_SAFE),
        ("Therefore, the code remains vulnerable to potential security risks.", LABEL_SAFE),
        ("It does not fix the flaw although it addresses the vulnerability.", LABEL_SAFE),
        (
            "First I analyze whether the change is vulnerable in any way. "
            "In conclusion, the code change does not fix or mitigate a vulnerability.",
            LABEL_SAFE,
        ),
        ("Without more information, it is difficult to provide a definitive answer.", None),
    ],
)
def test_parse_label(raw, expected):
    assert parse_label(raw) == expected


_FIXTURES = Path(__file__).parent / "fixtures" / "gpu_smoke_predictions.jsonl"

# Expected labels are the model's own verdicts, not the gold labels.
_SMOKE_EXPECTED = {
    "c76fac666ea038753294f2ac94d310f8adece9ce:bf546291e0d7": LABEL_SAFE,
    "c13e1da62b5f5f0e6fe8c1f769a5a28415415244:324e5a02fcca": LABEL_SAFE,
    "268068f25673242d1d5130d96202d3288c91b700:230d4a678c42": LABEL_SAFE,
    "7f0126ff011142c8619b10a6e64d04d1745c503a:1169d76388e6": LABEL_VULN,
    "d2b1d124d4055c2fcbe5126863487c52fd58cca1:aa619f3570fc": LABEL_SAFE,
    "f6e9d0d9955e85bdd7540b251cd50d598dacc5e6:8f32c26474ec": LABEL_SAFE,
    "5b65968b6c64fa02e74ca6b965bf5998b911e826:5080b1d770a6": LABEL_SAFE,
    "f6e9d0d9955e85bdd7540b251cd50d598dacc5e6:aaa9c68bae43": None,
    "c7e8e8a8f7a70b343ca1e0f90a31e35ab2d16de1:e906a13ba559": LABEL_VULN,
    "f6e9d0d9955e85bdd7540b251cd50d598dacc5e6:6e4f42407ec3": LABEL_SAFE,
}


def test_parse_label_on_real_gpu_smoke_outputs():
    rows = [json.loads(line) for line in _FIXTURES.read_text().splitlines()]
    assert len(rows) == len(_SMOKE_EXPECTED) == 10
    got = {row["id"]: parse_label(row["raw_output"]) for row in rows}
    assert got == _SMOKE_EXPECTED


def test_render_prompt_golden():
    prompt = render_prompt(make_record())
    assert PROMPT_VERSION == "fig1-v3"
    assert prompt.startswith("You are an expert in cybersecurity")
    assert "1) Analyze the Commit Message" in prompt
    assert (
        "4) Assess the Overall Security Impact\n"
        "   Does the modification fix an existing issue, introduce a new vulnerability, "
        "or is it neutral?"
    ) in prompt
    assert prompt.rstrip().endswith("Vulnerable or Not Vulnerable and provide reasoning.")
    assert "- Commit Message: fix overflow" in prompt
    assert "- Code Before Change: int f() { return 1; }" in prompt
    assert "- Code After Change: int f() { return 2; }" in prompt
    assert "- CVE ID (if available): CVE-2020-1234" in prompt
    assert "- CWE ID (if available): CWE-119" in prompt
    assert "- Project: proj" in prompt
    assert "- Programming Language: C" in prompt
    for placeholder in ("{commit_message}", "{func_before}", "{cve_id}", "{cwe_id}"):
        assert placeholder not in prompt


def test_render_prompt_missing_optional_fields_become_na():
    prompt = render_prompt(make_record(cve_id="", cwe="", commit_message=""))
    assert NA == "N/A"
    assert f"- CVE ID (if available): {NA}\n" in prompt
    assert f"- CWE ID (if available): {NA}\n" in prompt
    assert f"- Commit Message: {NA}\n" in prompt
    assert "None" not in prompt


def test_render_prompt_braces_in_code_are_literal():
    rec = make_record(func_before="void g() { int a[2] = {0}; }")
    assert "int a[2] = {0};" in render_prompt(rec)


def test_prompt_variant_registry():
    from vulndetect.common.prompt import PROMPT_VARIANTS, get_prompt_variant

    fig1 = get_prompt_variant("fig1-v3")
    assert fig1.system is None
    assert fig1.render is render_prompt

    strict = get_prompt_variant("authors-strict-v1")
    assert strict.system == "You are a cybersecurity and software vulnerability detection expert."
    assert set(PROMPT_VARIANTS) == {"fig1-v3", "authors-strict-v1"}

    with pytest.raises(KeyError, match="unknown prompt version"):
        get_prompt_variant("fig1-v2")


def test_fig1_v3_is_the_authors_detector_prompt_byte_exact(fixtures_dir):
    """Checks that fig1-v3 and the dual-agent detector equal the authors' prompt byte for byte."""
    from vulndetect.common.prompt import PROMPT_TEMPLATE
    from vulndetect.dual_agent.prompt import AUTHORS_DETECTOR_TEMPLATE

    fixture = (fixtures_dir / "authors_detector_v1.txt").read_text(encoding="utf-8")
    assert PROMPT_TEMPLATE == fixture
    assert AUTHORS_DETECTOR_TEMPLATE == PROMPT_TEMPLATE
    assert "\n" not in PROMPT_TEMPLATE.split("\n\n")[0]


def test_authors_strict_template_matches_fixture(fixtures_dir):
    """The transcription equals the fixture machine-extracted from the notebook."""
    from vulndetect.common.prompt import AUTHORS_STRICT_TEMPLATE

    fixture = (fixtures_dir / "authors_strict_v1_user.txt").read_text(encoding="utf-8")
    assert AUTHORS_STRICT_TEMPLATE == fixture


def test_render_authors_strict():
    from vulndetect.common.prompt import render_authors_strict

    prompt = render_authors_strict(make_record(func_before="void g() { int a[2] = {0}; }"))
    assert prompt.startswith("\nYou are an expert in cybersecurity")
    assert "Response Format (Strictly Follow This Format):" in prompt
    assert "Classification: <Vulnerable/Not Vulnerable>" in prompt
    assert "Answer with `Vulnerable` or `Not Vulnerable`" in prompt
    assert "int a[2] = {0};" in prompt
    assert "- CVE ID (if available): CVE-2020-1234" in prompt
    assert "- CWE ID (if available): N/A" in render_authors_strict(make_record(cwe=""))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Classification: Vulnerable\nReasoning: ...", LABEL_VULN),
        ("Classification: Not Vulnerable\nReasoning: ...", LABEL_SAFE),
        ("not_vulnerable", LABEL_SAFE),
        ("not-vulnerable", LABEL_SAFE),
        ("NOT   VULNERABLE", LABEL_SAFE),
        ("vulnerable ... later: not vulnerable", LABEL_VULN),
        ("not vulnerable ... later: vulnerable", LABEL_SAFE),
        ("the code is invulnerable", None),
        ("no verdict here", None),
        ("", None),
        ("I cannot determine this.", None),
        ("The change fixes the vulnerability.", None),
    ],
)
def test_parse_label_authors(raw, expected):
    from vulndetect.common.parsing import parse_label_authors

    assert parse_label_authors(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Based on the provided information, I would classify the code change as "
            "Not Vulnerable.\n\nHere's my reasoning:\n1. The commit message does not "
            "mention security terms.\n2. The change fixes a formatting issue.",
            LABEL_SAFE,
        ),
        ("vulnerable ... later: not vulnerable", LABEL_VULN),
        ("not vulnerable ... later: vulnerable", LABEL_SAFE),
        ("In summary, the code change mitigates a vulnerability.", LABEL_VULN),
        ("The code change does not fix or mitigate a vulnerability.", LABEL_SAFE),
        ("I cannot determine whether this change is security relevant.", None),
        ("no verdict here", None),
        ("", None),
    ],
)
def test_parse_label_hybrid(raw, expected):
    from vulndetect.common.parsing import parse_label_hybrid

    assert parse_label_hybrid(raw) == expected


def test_hybrid_never_loses_an_authors_verdict():
    """Checks that the hybrid parser never changes a verdict authors-v1 already made."""
    from vulndetect.common.parsing import parse_label_authors, parse_label_hybrid

    cases = [
        "Classification: Vulnerable\nReasoning: ...",
        "Classification: Not Vulnerable\nReasoning: ...",
        "vulnerable ... later: not vulnerable",
        "not vulnerable ... later: vulnerable",
        "It is Not Vulnerable. However, the code fixes a vulnerability elsewhere.",
        "no verdict here",
        "",
    ]
    for raw in cases:
        authors = parse_label_authors(raw)
        if authors is not None:
            assert parse_label_hybrid(raw) == authors, raw


def test_get_parser():
    from vulndetect.common.parsing import get_parser, parse_label_authors, parse_label_hybrid

    assert get_parser("d09-v2") is parse_label
    assert get_parser("authors-v1") is parse_label_authors
    assert get_parser("hybrid-v1") is parse_label_hybrid
    with pytest.raises(KeyError, match="unknown parser version"):
        get_parser("authors-v2")


def test_make_prediction_and_roundtrip(tmp_path):
    rec = make_record()
    preds = [
        make_prediction(rec, "Vulnerable because ...", LABEL_VULN),
        make_prediction(make_record(id="c2:ab", label=LABEL_SAFE, vul=0), "garbage", None),
    ]
    path = tmp_path / "preds.jsonl"
    write_predictions(preds, path)
    loaded = read_predictions(path)
    assert loaded == preds
    assert loaded[0]["gold_label"] == LABEL_VULN
    assert loaded[1]["pred_label"] is None


def test_make_prediction_rejects_unknown_label():
    with pytest.raises(ValueError):
        make_prediction(make_record(), "raw", "Maybe")


def test_write_run_writes_predictions_and_manifest(tmp_path):
    rec = make_record()
    preds = [make_prediction(rec, "Vulnerable", LABEL_VULN)]
    pred_path, manifest_path = write_run(
        tmp_path / "run", preds, {"approach": "baseline", "prompt_version": PROMPT_VERSION}
    )
    assert pred_path.exists() and manifest_path.exists()
    assert read_predictions(pred_path) == preds
    assert "baseline" in manifest_path.read_text()

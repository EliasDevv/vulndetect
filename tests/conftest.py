"""Shared pytest fixtures providing a synthetic Big-Vul-shaped dataset."""

from __future__ import annotations

import copy
import zlib

import pandas as pd
import pytest

from vulndetect.data.schema import TARGET_CWES

KNOWN_CVE = "CVE-2017-7586"
KNOWN_CVE_COMMIT = "c_known_cve"

COMMITS_PER_CWE = 20


def _row(cwe: str, commit: str, func_before: str, vul: int, lang: str = "C") -> dict:
    fixed = func_before.replace("/*bug*/", "/*fixed*/") if vul == 1 else func_before
    return {
        "CWE ID": cwe,
        "codeLink": f"http://example/{commit}",
        "commit_id": commit,
        "commit_message": "fix overflow" if vul == 1 else "refactor helper",
        "func_after": fixed,
        "func_before": func_before,
        "lang": lang,
        "project": f"proj{commit[:2]}",
        "vul": vul,
        "CVE ID": f"CVE-2020-{1000 + zlib.crc32(commit.encode()) % 9000}" if vul == 1 else "",
    }


def make_raw_df(commits_per_cwe: int = COMMITS_PER_CWE) -> pd.DataFrame:
    """A synthetic raw frame with native Big-Vul columns + edge-case rows."""
    rows: list[dict] = []
    for ci, cwe in enumerate(TARGET_CWES):
        for k in range(commits_per_cwe):
            commit = f"c{ci}_{k}"
            rows.append(
                _row(cwe, commit, f"int f_{ci}_{k}_v() {{ /*bug*/ return {ci * 100 + k}; }}", 1)
            )
            rows.append(_row(cwe, commit, f"int f_{ci}_{k}_s() {{ return {ci * 100 + k}; }}", 0))
        for k in range(commits_per_cwe // 2):
            commit = f"c{ci}_safe{k}"
            for j in range(3):
                rows.append(_row(cwe, commit, f"int g_{ci}_{k}_{j}() {{ return {j}; }}", 0))

    known = _row(TARGET_CWES[0], KNOWN_CVE_COMMIT, "int known_cve_v() { /*bug*/ return 7; }", 1)
    known["CVE ID"] = KNOWN_CVE
    rows.append(known)
    rows.append(_row(TARGET_CWES[0], KNOWN_CVE_COMMIT, "int known_cve_s() { return 7; }", 0))

    # Edge-case rows that filtering and cleaning must drop.
    rows.append(_row(TARGET_CWES[0], "c0_0", "int f_0_0_v() { /*bug*/ return 0; }", 1))
    rows.append(_row("", "edge_blank", "int blank() { return 1; }", 1))
    rows.append(_row("CWE-119;CWE-20", "edge_multi", "int multi() { return 1; }", 1))
    rows.append(_row("CWE-79", "edge_other", "int other() { return 1; }", 1))
    rows.append(_row(TARGET_CWES[0], "edge_lang", "def py(): return 1", 1, lang="Python"))
    rows.append(_row(TARGET_CWES[0], "edge_empty", "   ", 1))

    return pd.DataFrame(rows)


@pytest.fixture
def fixtures_dir():
    """Path to the tests/fixtures directory."""
    from pathlib import Path

    return Path(__file__).parent / "fixtures"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture
def raw_csv(tmp_path) -> str:
    path = tmp_path / "bigvul.csv"
    make_raw_df().to_csv(path, index=False)
    return str(path)


@pytest.fixture
def base_cfg() -> dict:
    return copy.deepcopy(
        {
            "seed": 42,
            "target_cwes": list(TARGET_CWES),
            "lang_filter": ["C", "C++"],
            "balance": "per_cwe",
            "split": {
                "by": "commit_id",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "target_sizes": {"train": 100000, "val": 100000, "test": 100000},
            },
            "source": {"name": "bigvul", "raw_path": "unused-in-unit-tests"},
            "output": {"dir": "unused", "stats_path": "unused"},
        }
    )

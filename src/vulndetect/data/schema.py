"""Canonical dataset schema, label strings and target CWEs."""

from __future__ import annotations

import hashlib
from typing import TypedDict

LABEL_VULN = "Vulnerable"
LABEL_SAFE = "Not Vulnerable"
LABELS = (LABEL_VULN, LABEL_SAFE)

TARGET_CWES = ("CWE-119", "CWE-20", "CWE-399", "CWE-264", "CWE-200")

RAW_COLUMNS = (
    "CWE ID",
    "codeLink",
    "commit_id",
    "commit_message",
    "func_after",
    "func_before",
    "lang",
    "project",
    "vul",
)

NA = "N/A"

RECORD_FIELDS = (
    "id",
    "code",
    "func_before",
    "func_after",
    "commit_message",
    "cve_id",
    "cwe",
    "project",
    "lang",
    "label",
    "vul",
    "commit_id",
    "split",
)


class Record(TypedDict):
    """One labeled instance, written as a single JSONL line."""

    id: str
    code: str
    func_before: str
    func_after: str
    commit_message: str
    cve_id: str
    cwe: str
    project: str
    lang: str
    label: str
    vul: int
    commit_id: str
    split: str


def func_hash(func_before: str) -> str:
    """First 12 hex chars of the sha1 of the function body (stable across runs)."""
    return hashlib.sha1(func_before.encode("utf-8")).hexdigest()[:12]


def make_id(commit_id: str, func_before: str) -> str:
    """Stable per-instance id: commit + function hash."""
    return f"{commit_id}:{func_hash(func_before)}"


def label_from_vul(vul: int) -> str:
    """Map the native `vul` flag to the canonical label string."""
    return LABEL_VULN if int(vul) == 1 else LABEL_SAFE

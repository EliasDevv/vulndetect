"""The shared classification prompt (paper Fig. 1), the authors' strict variant and the variant registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vulndetect.data.schema import NA, Record

PROMPT_VERSION = "fig1-v3"

PROMPT_TEMPLATE = """You are an expert in cybersecurity and software vulnerability detection. Given a code snippet with modifications, analyze whether the change fixes or mitigates a vulnerability. Use the commit message, function before and after changes, and CVE information (if available) to determine vulnerability.

1) Analyze the Commit Message
   Does the commit message mention security or vulnerability related terms like buffer overflow, fix, CVE, or patch?
2) Compare Function Before and After Changes
   What specific modifications were made to the code? Do the changes address known security issues, or do they introduce new risks?
3) Check for CVE/CWE References
   If a CVE is mentioned, what vulnerability does it relate to? If a CWE is provided, does the change align with known vulnerability patterns?
4) Assess the Overall Security Impact
   Does the modification fix an existing issue, introduce a new vulnerability, or is it neutral?

Here is the data:
- Commit Message: {commit_message}
- Code Before Change: {func_before}
- Code After Change: {func_after}
- CVE ID (if available): {cve_id}
- CWE ID (if available): {cwe_id}
- Project: {project}
- Programming Language: {lang}

Question:
Does this code change fix or mitigate a vulnerability? Answer with Vulnerable or Not Vulnerable and provide reasoning."""


def render_prompt(record: Record) -> str:
    """Fill the Fig. 1 template from a canonical record (missing fields -> N/A)."""
    return PROMPT_TEMPLATE.format(
        commit_message=record["commit_message"] or NA,
        func_before=record["func_before"],
        func_after=record["func_after"],
        cve_id=record.get("cve_id") or NA,
        cwe_id=record.get("cwe") or NA,
        project=record["project"] or NA,
        lang=record["lang"] or NA,
    )


AUTHORS_STRICT_SYSTEM = "You are a cybersecurity and software vulnerability detection expert."

AUTHORS_STRICT_TEMPLATE = """
You are an expert in cybersecurity and software vulnerability detection. Given a code snippet with modifications, analyze whether the change fixes or mitigates a vulnerability. Use the commit message, function before and after changes, and CVE information (if available) to determine vulnerability.

1) Analyze the Commit Message
   Does the commit message mention security or vulnerability related terms like buffer overflow, fix, CVE, or patch?
2) Compare Function Before and After Changes
   What specific modifications were made to the code? Do the changes address known security issues, or do they introduce new risks?
3) Check for CVE/CWE References
   If a CVE is mentioned, what vulnerability does it relate to? If a CWE is provided, does the change align with known vulnerability patterns?
4) Assess the Overall Security Impact
   Does the modification fix an existing issue, introduce a new vulnerability, or is it neutral?

Here is the data:
- Commit Message: {commit_message}
- Code Before Change: {func_before}
- Code After Change: {func_after}
- CVE ID (if available): {cve_id}
- CWE ID (if available): {cwe_id}
- Project: {project}
- Programming Language: {lang}

Question:
Does this code change fix or mitigate a vulnerability? Answer with `Vulnerable` or `Not Vulnerable` and provide reasoning.

Response Format (Strictly Follow This Format):
  1. Classification: Reply with exactly `Vulnerable` or `Not Vulnerable` (do not use any variations). Vulnerable if the code change is related to a known vulnerability (i.e., the change either fixes or introduces a security issue). Not Vulnerable when the code change is not associated with a vulnerability (i.e., it is a regular code update or enhancement without security implications).
  2. Reasoning: Provide a clear explanation of why the classification was made.

DO NOT return any additional text. DO NOT repeat the input data. ONLY output the Classification and Reasoning:
Your Response Should Look Like:
Classification: <Vulnerable/Not Vulnerable>
Reasoning: <Your explanation here>

"""


def fill_record_fields(template: str, record: Record, **extra: str) -> str:
    """Fill an authors' template with the record's data fields plus optional extras (missing fields -> N/A)."""
    return template.format(
        commit_message=record["commit_message"] or NA,
        func_before=record["func_before"],
        func_after=record["func_after"],
        cve_id=record.get("cve_id") or NA,
        cwe_id=record.get("cwe") or NA,
        project=record["project"] or NA,
        lang=record["lang"] or NA,
        **extra,
    )


def render_authors_strict(record: Record) -> str:
    """Fill the authors' strict prompt (user turn of authors-strict-v1)."""
    return fill_record_fields(AUTHORS_STRICT_TEMPLATE, record)


@dataclass(frozen=True)
class PromptVariant:
    """One selectable prompt version: manifest name, system turn, user renderer."""

    version: str
    system: str | None
    render: Callable[[Record], str]


PROMPT_VARIANTS: dict[str, PromptVariant] = {
    "fig1-v3": PromptVariant(version="fig1-v3", system=None, render=render_prompt),
    "authors-strict-v1": PromptVariant(
        version="authors-strict-v1",
        system=AUTHORS_STRICT_SYSTEM,
        render=render_authors_strict,
    ),
}


def get_prompt_variant(version: str) -> PromptVariant:
    """Resolve a config ``prompt.version`` value; unknown names fail loudly."""
    try:
        return PROMPT_VARIANTS[version]
    except KeyError:
        raise KeyError(
            f"unknown prompt version {version!r} — known: {sorted(PROMPT_VARIANTS)}"
        ) from None

"""Prompts for the Dual-Agent Detector and validation agent (§III.E), taken from the authors."""

from __future__ import annotations

from vulndetect.common.prompt import fill_record_fields
from vulndetect.data.schema import Record

DUAL_AGENT_PROMPT_SETS = ("authors-v1",)

AUTHORS_DETECTOR_VERSION = "authors-detector-v1"
AUTHORS_VALIDATOR_VERSION = "authors-validator-v1"

AUTHORS_DETECTOR_TEMPLATE = """You are an expert in cybersecurity and software vulnerability detection. Given a code snippet with modifications, analyze whether the change fixes or mitigates a vulnerability. Use the commit message, function before and after changes, and CVE information (if available) to determine vulnerability.

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

AUTHORS_VALIDATOR_TEMPLATE = """You are an audit and validation agent for software vulnerability detection. A first agent has already analyzed a code change and produced a verdict with reasoning. Your task is not to redo the analysis from scratch, but to critically review the first agent's output, correct it where it is wrong, and issue the final verdict.

Review the reasoning against the following checks:

1) Evidence Grounding
   Is every claim supported by the data actually provided? Flag any conclusion drawn from the commit message alone when the code before and after changes do not show the corresponding fix. A commit message claiming a fix is not evidence that the fix appears in the snippet under review.
2) Unsupported Assumptions and Speculation
   Did the agent invent context that is not present, such as functions, callers, buffer sizes, or execution paths outside the snippet? Mark any conclusion that depends on such assumptions as speculative.
3) Logical Consistency
   Does the stated reasoning actually lead to the stated verdict? Flag cases where the analysis describes a neutral or unrelated change but the verdict claims a security fix, or the reverse.
4) CVE/CWE Alignment
   If a CVE or CWE is referenced, does the described code change match that weakness pattern? A mismatch between the referenced weakness and the observed modification is an error.
5) Revision
   State whether the first agent's verdict should be kept or overturned, and give the corrected reasoning.

Here is the data:
- Commit Message: {commit_message}
- Code Before Change: {func_before}
- Code After Change: {func_after}
- CVE ID (if available): {cve_id}
- CWE ID (if available): {cwe_id}
- Project: {project}
- Programming Language: {lang}

Here is the first agent's output:
{agent1_response}

Respond in this format:
Issues Found: <list the logical inconsistencies, unsupported assumptions, or reasoning errors, or state None>
Revised Reasoning: <the corrected analysis, grounded only in the data above>
Result: <Vulnerable or Not Vulnerable>"""


def render_authors_detector(record: Record) -> str:
    """Render the authors' detector prompt for a record."""
    return fill_record_fields(AUTHORS_DETECTOR_TEMPLATE, record)


def render_authors_validator(record: Record, detector_output: str) -> str:
    """Render the authors' validator prompt from the record and the detector's output."""
    return fill_record_fields(AUTHORS_VALIDATOR_TEMPLATE, record, agent1_response=detector_output)

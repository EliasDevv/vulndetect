"""Build SFT training examples in chat or authors' alpaca serialization."""

from __future__ import annotations

from vulndetect.common.prompt import render_prompt
from vulndetect.data.schema import NA, Record

ALPACA_PROMPT_VERSION = "authors-alpaca-v1"

SFT_INSTRUCTION = (
    "You are an expert in cybersecurity and software vulnerability detection. "
    "Given a code snippet with modifications, analyze whether the change fixes or "
    "mitigates a vulnerability. Use the commit message, function before and after "
    "changes, and CVE/CWE information (if available) to determine vulnerability. "
    "Answer with `Vulnerable` or `Not Vulnerable` and provide reasoning."
)

ALPACA_TEMPLATE = """### Instruction:
{}

### Input:
{}

### Response:
{}"""

ALPACA_INPUT_TEMPLATE = """- CWE ID (if available): {cwe_id}
- Commit Message: {commit_message}
- Code Before Change: {func_before}
- Code After Change: {func_after}
- Project: {project}
- Programming Language: {lang}"""


def to_example(record: Record) -> dict:
    """Build one TRL prompt-completion example from the inference prompt and label."""
    return {
        "prompt": [{"role": "user", "content": render_prompt(record)}],
        "completion": [{"role": "assistant", "content": record["label"]}],
    }


def to_examples(records: list[Record]) -> list[dict]:
    """Convert all records to prompt-completion examples, in input order."""
    return [to_example(record) for record in records]


def _alpaca_input(record: Record) -> str:
    return ALPACA_INPUT_TEMPLATE.format(
        cwe_id=record.get("cwe") or NA,
        commit_message=record["commit_message"] or NA,
        func_before=record["func_before"],
        func_after=record["func_after"],
        project=record["project"] or NA,
        lang=record["lang"] or NA,
    )


def to_alpaca_text(record: Record) -> str:
    """Render one alpaca training text without the EOS token."""
    output = f"Classification: {record['label']}"
    return ALPACA_TEMPLATE.format(SFT_INSTRUCTION, _alpaca_input(record), output)


def to_alpaca_examples(records: list[Record]) -> list[dict]:
    """Convert all records to ``{"text": ...}`` rows."""
    return [{"text": to_alpaca_text(record)} for record in records]


def to_alpaca_prompt(record: Record) -> str:
    """Render the alpaca inference prompt with an empty response."""
    return ALPACA_TEMPLATE.format(SFT_INSTRUCTION, _alpaca_input(record), "")

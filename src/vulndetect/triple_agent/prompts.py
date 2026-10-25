"""Prompt composition for the Triple-Agent detecting party (RQ3)."""

from __future__ import annotations

from vulndetect.data.schema import Record
from vulndetect.dual_agent.prompt import AUTHORS_VALIDATOR_VERSION, render_authors_validator
from vulndetect.gan.prompts import DETECTOR_BLOCK_SEPARATOR, DETECTOR_HEADER

VALIDATOR_BLOCK_VERSION = "triple-validator-block-v1"
TRANSCRIPT_VERSION = "triple-transcript-v1"

MEMORY_SCOPES = ("both", "detector_only")

PROMPT_VERSIONS = {
    "validator": AUTHORS_VALIDATOR_VERSION,
    "validator_block": VALIDATOR_BLOCK_VERSION,
    "transcript": TRANSCRIPT_VERSION,
}

TRANSCRIPT_DETECTOR_SEP = "--- Detector (first agent) ---\n"
TRANSCRIPT_VALIDATOR_SEP = "\n\n--- Validator (second agent, final verdict) ---\n"


def render_validator_prompt(
    record: Record, detector_output: str, strategies: str, *, scope: str
) -> str:
    """Render the stage-2 prompt: optional strategy block plus the authors' validator prompt."""
    if scope not in MEMORY_SCOPES:
        raise ValueError(f"unknown memory_scope {scope!r}; allowed: {MEMORY_SCOPES}")
    base = render_authors_validator(record, detector_output)
    if not strategies or scope == "detector_only":
        return base
    return DETECTOR_HEADER + strategies + DETECTOR_BLOCK_SEPARATOR + base


def render_transcript(detector_raw: str, validator_raw: str) -> str:
    """Concatenate both stage outputs into one labelled transcript."""
    return TRANSCRIPT_DETECTOR_SEP + detector_raw + TRANSCRIPT_VALIDATOR_SEP + validator_raw

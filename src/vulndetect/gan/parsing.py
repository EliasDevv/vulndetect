"""Parsing of GAN role outputs: marker sections, strategy entries and the generation gate."""

from __future__ import annotations

import re
from dataclasses import dataclass

M_COMMIT_MESSAGE = "COMMIT MESSAGE"
M_MODIFIED_FUNCTION = "MODIFIED FUNCTION"
M_DECEPTION_NOTES = "DECEPTION NOTES"
M_NEW_STRATEGIES = "NEW STRATEGIES"
M_WENT_WRONG = "WHERE THE ANALYSIS WENT WRONG"
M_EVIDENCE = "EVIDENCE FOR THE CORRECT ANSWER"
M_LESSON = "LESSON"

GENERATOR_SECTIONS = (M_COMMIT_MESSAGE, M_MODIFIED_FUNCTION, M_DECEPTION_NOTES)
REFLECTOR_SECTIONS = (M_WENT_WRONG, M_EVIDENCE, M_LESSON)


def marker(name: str) -> str:
    """Return the literal marker line as it appears in prompts and outputs."""
    return f"=== {name} ==="


_MARKER_RE = re.compile(r"^[ \t]*=== *(?P<name>[A-Z][A-Z ]*[A-Z]) *===[ \t]*$", re.MULTILINE)

# Matches list entries like "- foo", "* foo", "1. foo", "2) foo".
_ENTRY_RE = re.compile(r"^[ \t]*(?:[-*•]|\d+[.)])[ \t]+(?P<entry>\S.*?)[ \t]*$", re.MULTILINE)

_NONE_RE = re.compile(r"^\W*NONE\W*$", re.IGNORECASE | re.MULTILINE)


def split_sections(text: str) -> dict[str, str]:
    """Map every ``=== NAME ===`` marker in ``text`` to its body; the first occurrence wins."""
    sections: dict[str, str] = {}
    matches = list(_MARKER_RE.finditer(text or ""))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.setdefault(m.group("name"), text[m.end() : end].strip("\n").strip())
    return sections


def require_sections(text: str, names: tuple[str, ...]) -> dict[str, str] | None:
    """Return the sections keyed by ``names``, or None if any of them is missing."""
    sections = split_sections(text)
    if any(name not in sections for name in names):
        return None
    return {name: sections[name] for name in names}


@dataclass(frozen=True)
class StrategyParse:
    """Result of :func:`parse_strategy_update` (``entries`` is None when malformed)."""

    entries: list[str] | None
    marker_present: bool


def _list_entries(body: str, max_entries: int) -> list[str]:
    entries = [m.group("entry") for m in _ENTRY_RE.finditer(body)]
    return [e for e in entries if not _NONE_RE.match(e)][:max_entries]


def parse_strategy_update(text: str, max_entries: int) -> StrategyParse:
    """Parse the strategy entries of an update answer, with or without the marker line."""
    sections = split_sections(text)
    if M_NEW_STRATEGIES in sections:
        body = sections[M_NEW_STRATEGIES].strip()
        if not body or _NONE_RE.search(body):
            return StrategyParse([], True)
        return StrategyParse(_list_entries(body, max_entries), True)
    body = (text or "").strip()
    if _NONE_RE.search(body):
        return StrategyParse([], False)
    entries = _list_entries(body, max_entries)
    return StrategyParse(entries or None, False)


def parse_strategy_entries(text: str, max_entries: int) -> list[str] | None:
    """Return only the entries of :func:`parse_strategy_update` (None if malformed)."""
    return parse_strategy_update(text, max_entries).entries


REJECTION_REASONS = (
    "missing_sections",
    "empty_function",
    "markdown_fence",
    "unbalanced_braces",
    "function_name_lost",
    "unchanged",
    "length_out_of_range",
)

MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 2.5

_C_KEYWORDS = frozenset(
    "if while for switch return sizeof do else case defined static inline "
    "typedef struct union enum const void unsigned signed".split()
)
_CALLABLE_RE = re.compile(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
_FENCE_RE = re.compile(r"^[ \t]*```", re.MULTILINE)


def extract_function_name(code: str) -> str | None:
    """Return the first identifier followed by ``(`` that is not a C keyword."""
    for m in _CALLABLE_RE.finditer(code or ""):
        if m.group("name") not in _C_KEYWORDS:
            return m.group("name")
    return None


def validate_generated_function(func: str, source_func: str) -> str | None:
    """Return a rejection reason from :data:`REJECTION_REASONS`, or None if the function passes."""
    if not func or not func.strip():
        return "empty_function"
    if _FENCE_RE.search(func):
        return "markdown_fence"
    if func.count("{") == 0 or func.count("{") != func.count("}"):
        return "unbalanced_braces"
    name = extract_function_name(source_func)
    if name is not None and name not in func:
        return "function_name_lost"
    if func.strip() == (source_func or "").strip():
        return "unchanged"
    ratio = len(func) / max(len(source_func or ""), 1)
    if not (MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO):
        return "length_out_of_range"
    return None

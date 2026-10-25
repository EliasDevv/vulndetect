"""Parse free-text model output into a canonical label (or None)."""

from __future__ import annotations

import re

from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN

# Explicit answer words; the lookbehinds exclude "remains/still vulnerable".
_ANSWER_RE = re.compile(
    r"\b(?P<safe>(?:not|non)[\s\-]*(?:a\s+|an\s+)?vulnerable)\b"
    r"|(?<!remain\s)(?<!remains\s)(?<!remained\s)(?<!still\s)\b(?P<vuln>vulnerable)\b",
    re.IGNORECASE,
)

_REFUSAL_RE = re.compile(
    r"\bdifficult to (?:provide|give) a definitive answer\b"
    r"|\b(?:cannot|can\s?not|can't|unable to|impossible to|difficult to|hard to)"
    r"\s+(?:\w+\s+){0,2}?determine\b",
    re.IGNORECASE,
)

_VULN_CUE_RE = re.compile(
    r"\b(?:fixes|fixed|mitigates|mitigated|addresses|addressed|resolves|resolved"
    r"|patches|patched)\s+(?:\w+\s+){0,3}?"
    r"(?:vulnerabilit(?:y|ies)|existing issue|security (?:issue|flaw|risk)|CVE-\d)",
    re.IGNORECASE,
)

_SAFE_CUE_RE = re.compile(
    r"\b(?:does|do|did|would|will|may|might)\s+not\s+(?:\w+\s+){0,2}?"
    r"(?:fix|mitigate|address|resolve|patch)\b"
    r"|\b(?:doesn't|don't|didn't)\s+(?:\w+\s+){0,2}?(?:fix|mitigate|address|resolve|patch)\b"
    r"|\bno security (?:issues?|risks?|impact)\b"
    r"|\b(?:remains?|remained|still)\s+vulnerable\b"
    r"|\bis\s+neutral\b",
    re.IGNORECASE,
)

_CONCLUSION_RE = re.compile(
    r"\b(?:in conclusion|in summary|to summarize|to conclude|therefore|thus|overall)\b",
    re.IGNORECASE,
)

_TAIL_CHARS = 600

# Sentinel: a window matched a refusal, so the result is None.
_REFUSED = object()

_SENTENCE_BREAKS = ".!?\n"


def _windows(text: str) -> list[str]:
    """Search windows, narrowest first: conclusion, tail, full text."""
    windows = []
    markers = list(_CONCLUSION_RE.finditer(text))
    if markers:
        windows.append(text[markers[-1].start() :])
    if len(text) > _TAIL_CHARS:
        start = len(text) - _TAIL_CHARS
        boundary = max(text.rfind(ch, 0, start) for ch in _SENTENCE_BREAKS)
        windows.append(text[boundary + 1 :] if boundary != -1 else text)
    windows.append(text)
    return windows


def _sentence_around(window: str, pos: int) -> str:
    start = max(window.rfind(ch, 0, pos) for ch in _SENTENCE_BREAKS) + 1
    ends = [e for e in (window.find(ch, pos) for ch in _SENTENCE_BREAKS) if e != -1]
    return window[start : min(ends)] if ends else window[start:]


def _scan(window: str) -> str | object | None:
    m = _ANSWER_RE.search(window)
    if m is not None:
        return LABEL_SAFE if m.group("safe") else LABEL_VULN
    if _REFUSAL_RE.search(window):
        return _REFUSED
    safe = _SAFE_CUE_RE.search(window)
    vuln = _VULN_CUE_RE.search(window)
    if safe and (vuln is None or safe.start() <= vuln.start()):
        return LABEL_SAFE
    if vuln:
        if _SAFE_CUE_RE.search(_sentence_around(window, vuln.start())):
            return LABEL_SAFE
        return LABEL_VULN
    return None


def parse_label(raw_output: str) -> str | None:
    """Return LABEL_VULN, LABEL_SAFE, or None; searches conclusion, tail, then full text."""
    text = raw_output or ""
    for window in _windows(text):
        verdict = _scan(window)
        if verdict is _REFUSED:
            return None
        if verdict is not None:
            return verdict
    return None


_AUTHORS_NOT_VULN = re.compile(r"\bnot[\s_-]*vulnerable\b", re.IGNORECASE)
_AUTHORS_VULN = re.compile(r"\bvulnerable\b", re.IGNORECASE)


def parse_label_authors(raw_output: str) -> str | None:
    """The authors' string-matching rule (rag.py), mapped onto our labels."""
    text = raw_output or ""
    neg = _AUTHORS_NOT_VULN.search(text)
    pos = _AUTHORS_VULN.search(text)
    if neg and (pos is None or neg.start() <= pos.start()):
        return LABEL_SAFE
    if pos:
        return LABEL_VULN
    return None


def parse_label_hybrid(raw_output: str) -> str | None:
    """The authors' rule where it decides; our semantic cues only where it does not."""
    verdict = parse_label_authors(raw_output)
    return verdict if verdict is not None else parse_label(raw_output)


PARSERS = {
    "d09-v2": parse_label,
    "authors-v1": parse_label_authors,
    "hybrid-v1": parse_label_hybrid,
}

DEFAULT_PARSER_VERSION = "d09-v2"


def get_parser(version: str):
    """Resolve a config ``parser_version`` value; unknown names fail loudly."""
    try:
        return PARSERS[version]
    except KeyError:
        raise KeyError(f"unknown parser version {version!r} — known: {sorted(PARSERS)}") from None

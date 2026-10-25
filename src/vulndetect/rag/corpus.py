"""Render one retrieval document per CWE entry."""

from __future__ import annotations

from vulndetect.rag.catalog import CweEntry


def chunk_prefix(entry: CweEntry) -> str:
    """Return the per-chunk identity line ``"CWE-nnn: Name\\n"``."""
    return f"{entry.cwe_id}: {entry.name}\n"


def render_body(entry: CweEntry) -> str:
    """Render the document body with one section per selected field ("" if the entry is empty)."""
    parts: list[str] = []
    if entry.description:
        parts.append(f"Description: {entry.description}")
    if entry.extended_description:
        parts.append(f"Extended Description: {entry.extended_description}")
    if entry.consequences:
        lines = "\n".join(f"- {c}" for c in entry.consequences)
        parts.append(f"Common Consequences:\n{lines}")
    if entry.mitigations:
        lines = "\n".join(f"- {m}" for m in entry.mitigations)
        parts.append(f"Potential Mitigations:\n{lines}")
    if entry.demonstrative_examples:
        lines = "\n".join(f"- {e}" for e in entry.demonstrative_examples)
        parts.append(f"Demonstrative Examples:\n{lines}")
    if entry.observed_examples:
        lines = "\n".join(f"- {o.reference}: {o.description}" for o in entry.observed_examples)
        parts.append(f"Observed Examples (CVEs):\n{lines}")
    if entry.members:
        lines = "\n".join(f"- {m}" for m in entry.members)
        parts.append(f"Members:\n{lines}")
    if not parts:
        return ""
    return "\n\n".join([f"Abstraction: {entry.abstraction}.", *parts])

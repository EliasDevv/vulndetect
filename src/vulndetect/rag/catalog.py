"""Parse the MITRE CWE XML catalog (weaknesses and categories) into plain records."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_NS = {"cwe": "http://cwe.mitre.org/cwe-7", "xhtml": "http://www.w3.org/1999/xhtml"}


@dataclass(frozen=True)
class ObservedExample:
    """One Observed_Example: a real-world CVE that exhibits the weakness."""

    reference: str
    description: str


@dataclass(frozen=True)
class CweEntry:
    """One catalog entry (weakness or category) with the selected text fields."""

    cwe_id: str
    name: str
    abstraction: str
    status: str
    description: str
    extended_description: str
    consequences: tuple[str, ...]
    mitigations: tuple[str, ...]
    demonstrative_examples: tuple[str, ...]
    observed_examples: tuple[ObservedExample, ...]
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class CweCatalog:
    """The parsed catalog: entries plus the snapshot identity from the XML root."""

    version: str
    date: str
    entries: tuple[CweEntry, ...]
    n_deprecated_filtered: int
    n_weaknesses: int
    n_categories: int


def _text(el: ET.Element | None) -> str:
    """Return all text of an element, whitespace-normalized ('' if absent)."""
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _consequence(el: ET.Element) -> str:
    scopes = [_text(s) for s in el.findall("cwe:Scope", _NS)]
    impacts = [_text(i) for i in el.findall("cwe:Impact", _NS)]
    note = _text(el.find("cwe:Note", _NS))
    parts = []
    if scopes:
        parts.append(f"Scope: {', '.join(scopes)}.")
    if impacts:
        parts.append(f"Impact: {', '.join(impacts)}.")
    if note:
        parts.append(note)
    return " ".join(parts)


def _mitigation(el: ET.Element) -> str:
    phases = [_text(p) for p in el.findall("cwe:Phase", _NS)]
    strategy = _text(el.find("cwe:Strategy", _NS))
    description = _text(el.find("cwe:Description", _NS))
    notes = _text(el.find("cwe:Effectiveness_Notes", _NS))
    parts = []
    if phases:
        parts.append(f"Phase: {', '.join(phases)}.")
    if strategy:
        parts.append(f"Strategy: {strategy}.")
    if description:
        parts.append(description)
    if notes:
        parts.append(notes)
    return " ".join(parts)


def _demonstrative_example(el: ET.Element) -> str:
    """Render one demonstrative example as prose, tagging code blocks with nature and language."""
    parts = []
    for child in el:
        tag = child.tag.split("}")[1]
        if tag == "Example_Code":
            nature = child.get("Nature", "Example")
            language = child.get("Language", "unknown language")
            parts.append(f"[{nature} code, {language}] {_text(child)}")
        elif tag in ("Intro_Text", "Body_Text"):
            parts.append(_text(child))
    return " ".join(p for p in parts if p)


def _observed_example(el: ET.Element) -> ObservedExample | None:
    reference = _text(el.find("cwe:Reference", _NS))
    if not reference:
        return None
    return ObservedExample(reference=reference, description=_text(el.find("cwe:Description", _NS)))


def _entry(w: ET.Element) -> CweEntry:
    def findall(parent_tag: str, child_tag: str) -> list[ET.Element]:
        parent = w.find(f"cwe:{parent_tag}", _NS)
        return parent.findall(f"cwe:{child_tag}", _NS) if parent is not None else []

    observed = [_observed_example(el) for el in findall("Observed_Examples", "Observed_Example")]
    return CweEntry(
        cwe_id=f"CWE-{w.get('ID')}",
        name=w.get("Name", ""),
        abstraction=w.get("Abstraction", ""),
        status=w.get("Status", ""),
        description=_text(w.find("cwe:Description", _NS)),
        extended_description=_text(w.find("cwe:Extended_Description", _NS)),
        consequences=tuple(
            c for el in findall("Common_Consequences", "Consequence") if (c := _consequence(el))
        ),
        mitigations=tuple(
            m for el in findall("Potential_Mitigations", "Mitigation") if (m := _mitigation(el))
        ),
        demonstrative_examples=tuple(
            d
            for el in findall("Demonstrative_Examples", "Demonstrative_Example")
            if (d := _demonstrative_example(el))
        ),
        observed_examples=tuple(o for o in observed if o is not None),
    )


def _category_entry(el: ET.Element, names: dict[str, str]) -> CweEntry:
    """Convert one <Category> into a CweEntry, resolving member ids to names via ``names``."""
    members: list[str] = []
    relationships = el.find("cwe:Relationships", _NS)
    if relationships is not None:
        for member in relationships.findall("cwe:Has_Member", _NS):
            member_id = f"CWE-{member.get('CWE_ID')}"
            name = names.get(member_id)
            members.append(f"{member_id}: {name}" if name else member_id)
    return CweEntry(
        cwe_id=f"CWE-{el.get('ID')}",
        name=el.get("Name", ""),
        abstraction="Category",
        status=el.get("Status", ""),
        description=_text(el.find("cwe:Summary", _NS)),
        extended_description="",
        consequences=(),
        mitigations=(),
        demonstrative_examples=(),
        observed_examples=(),
        members=tuple(dict.fromkeys(members)),
    )


def load_catalog(path: str | Path) -> CweCatalog:
    """Parse the CWE XML snapshot into weaknesses then categories, skipping deprecated entries."""
    root = ET.parse(Path(path)).getroot()
    weaknesses = root.find("cwe:Weaknesses", _NS)
    if weaknesses is None:
        raise ValueError(f"{path}: no <Weaknesses> element — not a CWE catalog XML?")
    kept: list[CweEntry] = []
    n_deprecated = 0
    for w in weaknesses.findall("cwe:Weakness", _NS):
        if w.get("Status") == "Deprecated":
            n_deprecated += 1
            continue
        kept.append(_entry(w))
    n_weaknesses = len(kept)

    categories = root.find("cwe:Categories", _NS)
    raw_categories: list[ET.Element] = []
    for c in categories.findall("cwe:Category", _NS) if categories is not None else []:
        if c.get("Status") == "Deprecated":
            n_deprecated += 1
            continue
        raw_categories.append(c)

    names = {e.cwe_id: e.name for e in kept}
    names.update({f"CWE-{c.get('ID')}": c.get("Name", "") for c in raw_categories})
    kept.extend(_category_entry(c, names) for c in raw_categories)

    return CweCatalog(
        version=root.get("Version", "unknown"),
        date=root.get("Date", "unknown"),
        entries=tuple(kept),
        n_deprecated_filtered=n_deprecated,
        n_weaknesses=n_weaknesses,
        n_categories=len(raw_categories),
    )

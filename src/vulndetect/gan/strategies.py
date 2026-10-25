"""The append-only strategy memories S_G and S_D with pluggable capacity policies."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from vulndetect.gan.parsing import parse_strategy_entries
from vulndetect.gan.prompts import render_summarize_prompt

logger = logging.getLogger(__name__)

CAPACITY_POLICIES = ("fifo", "head", "summarize", "unbounded")

ROLES = ("generator", "detector")

_WS_RE = re.compile(r"\s+")
_LEADING_MARKER_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")


def normalise_entry(text: str, max_words: int) -> str:
    """One-line, marker-free, word-capped entry text."""
    entry = _WS_RE.sub(" ", (text or "").strip())
    entry = _LEADING_MARKER_RE.sub("", entry).strip()
    words = entry.split(" ")
    if len(words) > max_words:
        entry = " ".join(words[:max_words]) + " ..."
    return entry


def dedup_key(entry: str) -> str:
    """Loose identity of an entry: lowercase words only (punctuation ignored)."""
    return _WS_RE.sub(" ", _NON_WORD_RE.sub(" ", entry.lower())).strip()


@dataclass
class AddReport:
    """Summary of what one update did to the memory."""

    added: list[str] = field(default_factory=list)
    duplicates: int = 0
    truncated: int = 0
    evicted: int = 0
    summarized: bool = False

    def to_dict(self) -> dict:
        """Return the report as a dict with the number of added entries."""
        return {
            "n_added": len(self.added),
            "duplicates": self.duplicates,
            "truncated": self.truncated,
            "evicted": self.evicted,
            "summarized": self.summarized,
        }


@dataclass
class StrategyMemory:
    """An ordered list of imperative strategy sentences (S_G or S_D), initially empty."""

    role: str
    entries: list[str] = field(default_factory=list)
    policy: str = "fifo"
    max_entries: int = 12
    max_entry_words: int = 30
    dedup: bool = True

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown memory role {self.role!r}; allowed: {ROLES}")
        if self.policy not in CAPACITY_POLICIES:
            raise ValueError(
                f"unknown capacity policy {self.policy!r}; allowed: {CAPACITY_POLICIES}"
            )
        if self.max_entries < 1 or self.max_entry_words < 1:
            raise ValueError("max_entries and max_entry_words must be >= 1")

    def __len__(self) -> int:
        return len(self.entries)

    def render(self) -> str:
        """Render the numbered block inserted into prompts (empty string if no entries)."""
        if not self.entries:
            return ""
        return "\n".join(f"{i}. {e}" for i, e in enumerate(self.entries, start=1))

    def stats(self) -> dict:
        """Return entry count, character count and approximate tokens of the memory."""
        text = self.render()
        return {
            "n_entries": len(self.entries),
            "n_chars": len(text),
            "approx_tokens": round(len(text) / 4),
        }

    def add(self, new_entries: list[str], *, model=None) -> AddReport:
        """Normalise, deduplicate and append ``new_entries``, then apply the capacity policy."""
        report = AddReport()
        for raw in new_entries:
            full = normalise_entry(raw, max_words=10**9)
            entry = normalise_entry(raw, self.max_entry_words)
            if not entry:
                continue
            if entry != full:
                report.truncated += 1
            if self.dedup and dedup_key(entry) in {dedup_key(e) for e in self.entries}:
                report.duplicates += 1
                continue
            if self.dedup and dedup_key(entry) in {dedup_key(e) for e in report.added}:
                report.duplicates += 1
                continue
            report.added.append(entry)
        self.entries.extend(report.added)
        self._apply_policy(report, model)
        return report

    def _apply_policy(self, report: AddReport, model) -> None:
        before = len(self.entries)
        if self.policy == "unbounded" or before <= self.max_entries:
            return
        if self.policy == "fifo":
            self.entries = self.entries[before - self.max_entries :]
        elif self.policy == "head":
            self.entries = self.entries[: self.max_entries]
        elif self.policy == "summarize":
            self._summarize(model)
            report.summarized = True
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries :]
        report.evicted = before - len(self.entries)

    def _summarize(self, model) -> None:
        if model is None:
            raise ValueError("memory policy 'summarize' needs a model — pass model=... to add()")
        prompt = render_summarize_prompt(
            self.entries, self.role, max_entries=self.max_entries, max_words=self.max_entry_words
        )
        raw = model.generate([prompt])[0]
        condensed = parse_strategy_entries(raw, self.max_entries)
        if not condensed:
            logger.warning("summarize returned no parseable entries; falling back to fifo")
            self.entries = self.entries[-self.max_entries :]
            return
        self.entries = [normalise_entry(e, self.max_entry_words) for e in condensed]

    def to_dict(self) -> dict:
        """Serialise the memory and its settings to a dict."""
        return {
            "role": self.role,
            "policy": self.policy,
            "max_entries": self.max_entries,
            "max_entry_words": self.max_entry_words,
            "dedup": self.dedup,
            "entries": list(self.entries),
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategyMemory:
        """Rebuild a memory from the dict produced by :meth:`to_dict`."""
        return cls(
            role=data["role"],
            entries=list(data.get("entries", [])),
            policy=data.get("policy", "fifo"),
            max_entries=int(data.get("max_entries", 12)),
            max_entry_words=int(data.get("max_entry_words", 30)),
            dedup=bool(data.get("dedup", True)),
        )


def new_memories(memory_cfg: dict) -> tuple[StrategyMemory, StrategyMemory]:
    """Build both empty memories from the config's ``gan.memory`` block."""
    kwargs = {
        "policy": memory_cfg["policy"],
        "max_entries": int(memory_cfg["max_entries"]),
        "max_entry_words": int(memory_cfg["max_entry_words"]),
        "dedup": bool(memory_cfg["dedup"]),
    }
    return StrategyMemory(role="generator", **kwargs), StrategyMemory(role="detector", **kwargs)

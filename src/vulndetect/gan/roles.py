"""The three LLM-GAN roles: Generator M_G, Detector M_D and Reflector M_R."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vulndetect.common.parsing import parse_label
from vulndetect.data.schema import LABEL_VULN, Record
from vulndetect.gan.parsing import (
    GENERATOR_SECTIONS,
    M_COMMIT_MESSAGE,
    M_DECEPTION_NOTES,
    M_MODIFIED_FUNCTION,
    REFLECTOR_SECTIONS,
    parse_strategy_update,
    require_sections,
    validate_generated_function,
)
from vulndetect.gan.prompts import (
    render_detector_prompt,
    render_detector_update_prompt,
    render_generator_prompt,
    render_generator_update_prompt,
    render_reflector_prompt,
)

VALIDATION_MODES = ("structural", "none")

SYNTHETIC_ID_PREFIX = "gen"


@dataclass(frozen=True)
class Detection:
    """The Detector's verdict and raw explanation, plus an optional multi-agent trace."""

    label: str | None
    raw: str
    trace: dict | None = None


@dataclass(frozen=True)
class GeneratedItem:
    """A generated vulnerable item x' with its commit message and deception notes (eq. 1)."""

    record: Record
    function: str
    commit_message: str
    deception_notes: str
    source_id: str
    raw: str


@dataclass(frozen=True)
class Rejection:
    """A generation that did not pass the structural gate."""

    reason: str
    raw: str


@dataclass(frozen=True)
class StrategyUpdate:
    """The new strategy entries a role derived, plus its raw output (eqs. 3, 4, 7)."""

    entries: list[str]
    raw: str
    parsed: bool
    marker_present: bool


@dataclass(frozen=True)
class Reflection:
    """Eq. (6): the Reflector's feedback r_x."""

    text: str
    raw: str
    structured: bool


def synthetic_record(source: Record, function: str, commit_message: str, round_no: int) -> Record:
    """Assemble the generated item x' as a canonical Vulnerable train record."""
    return {
        "id": f"{SYNTHETIC_ID_PREFIX}:{source['id']}:r{round_no}",
        "code": function,
        "func_before": function,
        "func_after": source["func_before"],
        "commit_message": commit_message,
        "cve_id": source["cve_id"],
        "cwe": source["cwe"],
        "project": source["project"],
        "lang": source["lang"],
        "label": LABEL_VULN,
        "vul": 1,
        "commit_id": source["commit_id"],
        "split": "train",
    }


def generate_fake(
    model,
    source: Record,
    strategies: str,
    round_no: int,
    *,
    validate: str = "structural",
) -> GeneratedItem | Rejection:
    """Generate a vulnerable variant of a clean function (eq. 1), or return a :class:`Rejection`."""
    if validate not in VALIDATION_MODES:
        raise ValueError(f"unknown validate mode {validate!r}; allowed: {VALIDATION_MODES}")
    raw = model.generate([render_generator_prompt(source, strategies)])[0]
    sections = require_sections(raw, GENERATOR_SECTIONS)
    if sections is None:
        return Rejection("missing_sections", raw)
    function = sections[M_MODIFIED_FUNCTION].strip()
    if validate == "structural":
        reason = validate_generated_function(function, source["func_before"])
        if reason is not None:
            return Rejection(reason, raw)
    elif not function:
        return Rejection("empty_function", raw)
    commit_message = " ".join(sections[M_COMMIT_MESSAGE].split())
    return GeneratedItem(
        record=synthetic_record(source, function, commit_message, round_no),
        function=function,
        commit_message=commit_message,
        deception_notes=sections[M_DECEPTION_NOTES],
        source_id=source["id"],
        raw=raw,
    )


def detect(model, record: Record, strategies: str) -> Detection:
    """Run the Detector on one record with the given strategies (eqs. 2, 5)."""
    raw = model.generate([render_detector_prompt(record, strategies)])[0]
    return Detection(parse_label(raw), raw)


def detect_batch(model, records: list[Record], strategies: str) -> list[Detection]:
    """Run the Detector on a batch of records with a fixed memory (used by the drift probe)."""
    prompts = [render_detector_prompt(r, strategies) for r in records]
    return [Detection(parse_label(raw), raw) for raw in model.generate(prompts)]


class DetectingParty(Protocol):
    """Interface of the detecting side of the game (single Detector or a multi-agent party)."""

    name: str
    prompt_versions: dict[str, str]

    def detect(self, record: Record, strategies: str) -> Detection:
        """Classify one record with the given strategies."""
        ...

    def detect_batch(self, records: list[Record], strategies: str) -> list[Detection]:
        """Classify a batch of records with the given strategies."""
        ...

    def describe(self) -> dict:
        """Return a JSON-serialisable description of the party."""
        ...

    def stats(self) -> dict:
        """Return the running counters of the party."""
        ...


class SingleDetector:
    """The RQ2 detecting party wrapping :func:`detect` and :func:`detect_batch`."""

    name = "single_detector"
    prompt_versions: dict[str, str] = {}

    def __init__(self, model):
        self.model = model
        self._n = 0

    def detect(self, record: Record, strategies: str) -> Detection:
        """Run the Detector on one record and count the call."""
        self._n += 1
        return detect(self.model, record, strategies)

    def detect_batch(self, records: list[Record], strategies: str) -> list[Detection]:
        """Run the Detector on a batch of records and count the calls."""
        self._n += len(records)
        return detect_batch(self.model, records, strategies)

    def describe(self) -> dict:
        """Return the party name and its prompt versions."""
        return {"name": self.name, "prompt_versions": dict(self.prompt_versions)}

    def stats(self) -> dict:
        """Return the number of detections so far."""
        return {"n_detections": self._n}


def update_generator_strategies(
    model,
    source: Record,
    item: GeneratedItem,
    detection: Detection,
    strategies: str,
    *,
    max_new: int,
    max_words: int,
) -> StrategyUpdate:
    """Let the Generator derive new S_G entries after the Detector caught its item (eq. 3)."""
    prompt = render_generator_update_prompt(
        source,
        item.function,
        item.deception_notes,
        detection.raw,
        strategies,
        max_new=max_new,
        max_words=max_words,
    )
    raw = model.generate([prompt])[0]
    parse = parse_strategy_update(raw, max_new)
    return StrategyUpdate(parse.entries or [], raw, parse.entries is not None, parse.marker_present)


def update_detector_strategies(
    model,
    record: Record,
    detector_output: str,
    evidence: str,
    kind: str,
    strategies: str,
    *,
    max_new: int,
    max_words: int,
) -> StrategyUpdate:
    """Let the Detector derive new S_D entries after a wrong verdict (eqs. 4, 7)."""
    prompt = render_detector_update_prompt(
        record,
        detector_output,
        evidence,
        kind,
        strategies,
        max_new=max_new,
        max_words=max_words,
    )
    raw = model.generate([prompt])[0]
    parse = parse_strategy_update(raw, max_new)
    return StrategyUpdate(parse.entries or [], raw, parse.entries is not None, parse.marker_present)


def reflect(model, record: Record, detection: Detection, *, max_words: int) -> Reflection:
    """Let the Reflector explain the Detector's mistake on a real training item (eq. 6)."""
    prompt = render_reflector_prompt(record, detection.raw, detection.label, max_words=max_words)
    raw = model.generate([prompt])[0]
    sections = require_sections(raw, REFLECTOR_SECTIONS)
    if sections is None:
        return Reflection(raw.strip(), raw, False)
    text = "\n".join(f"{name}: {sections[name]}" for name in REFLECTOR_SECTIONS)
    return Reflection(text, raw, True)

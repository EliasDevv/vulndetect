"""The Triple-Agent detecting party: Detector -> Validator on one model."""

from __future__ import annotations

import copy
from collections.abc import Callable

from vulndetect.common.parsing import parse_label
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN, Record
from vulndetect.dual_agent.run import FINAL_LABEL_RULES, resolve_final_label
from vulndetect.gan.prompts import render_detector_prompt
from vulndetect.gan.roles import Detection
from vulndetect.triple_agent.prompts import (
    MEMORY_SCOPES,
    PROMPT_VERSIONS,
    render_transcript,
    render_validator_prompt,
)

PARTY_NAME = "dual_agent"

MATRIX_KEYS = (LABEL_VULN, LABEL_SAFE, "None")


def _key(label: str | None) -> str:
    return "None" if label is None else label


class DualAgentParty:
    """Detector -> Validator chain acting as one detecting party."""

    name = PARTY_NAME
    prompt_versions: dict[str, str] = dict(PROMPT_VERSIONS)

    def __init__(self, model, *, memory_scope: str, final_label_rule: str):
        if memory_scope not in MEMORY_SCOPES:
            raise ValueError(f"unknown memory_scope {memory_scope!r}; allowed: {MEMORY_SCOPES}")
        if final_label_rule not in FINAL_LABEL_RULES:
            raise ValueError(
                f"unknown final_label_rule {final_label_rule!r}; allowed: {FINAL_LABEL_RULES}"
            )
        self.model = model
        self.memory_scope = memory_scope
        self.final_label_rule = final_label_rule
        self._n = 0
        self._n_detector_unparseable = 0
        self._n_validator_unparseable = 0
        self._n_flipped = 0
        self._n_agree = 0
        self._matrix = {d: {v: 0 for v in MATRIX_KEYS} for d in MATRIX_KEYS}
        self._flips = {"detector_right_validator_wrong": 0, "detector_wrong_validator_right": 0}

    def detect(self, record: Record, strategies: str) -> Detection:
        """Run the Detector and Validator on one record."""
        return self.detect_batch([record], strategies)[0]

    def detect_batch(self, records: list[Record], strategies: str) -> list[Detection]:
        """Run the detector and validator as two batched passes over all records."""
        if not records:
            return []
        detector_raw = self.model.generate([render_detector_prompt(r, strategies) for r in records])
        validator_raw = self.model.generate(
            [
                render_validator_prompt(r, out, strategies, scope=self.memory_scope)
                for r, out in zip(records, detector_raw, strict=True)
            ]
        )
        detections = []
        for record, d_raw, v_raw in zip(records, detector_raw, validator_raw, strict=True):
            det, val = parse_label(d_raw), parse_label(v_raw)
            final = resolve_final_label(det, val, self.final_label_rule)
            flipped = det is not None and val is not None and det != val
            self._count(record, det, val, flipped)
            detections.append(
                Detection(
                    final,
                    render_transcript(d_raw, v_raw),
                    {
                        "detector_label": det,
                        "detector_raw": d_raw,
                        "validator_label": val,
                        "validator_raw": v_raw,
                        "flipped": flipped,
                    },
                )
            )
        return detections

    def _count(self, record: Record, det: str | None, val: str | None, flipped: bool) -> None:
        self._n += 1
        self._n_detector_unparseable += det is None
        self._n_validator_unparseable += val is None
        self._n_flipped += flipped
        self._n_agree += det is not None and det == val
        self._matrix[_key(det)][_key(val)] += 1
        if flipped:
            side = (
                "detector_right_validator_wrong"
                if det == record["label"]
                else ("detector_wrong_validator_right")
            )
            self._flips[side] += 1

    def describe(self) -> dict:
        """Return the party configuration and its prompt versions."""
        return {
            "name": self.name,
            "memory_scope": self.memory_scope,
            "final_label_rule": self.final_label_rule,
            "prompt_versions": dict(self.prompt_versions),
        }

    def stats(self) -> dict:
        """Return detection counters, flip and agreement rates and the agreement matrix."""
        n = self._n
        return {
            "n_detections": n,
            "n_detector_unparseable": self._n_detector_unparseable,
            "n_validator_unparseable": self._n_validator_unparseable,
            "n_flipped": self._n_flipped,
            "flip_rate": round(self._n_flipped / n, 4) if n else None,
            "n_agree": self._n_agree,
            "detector_validator_agreement": round(self._n_agree / n, 4) if n else None,
            "agreement_matrix": copy.deepcopy(self._matrix),
            "flips_by_detector_correctness": dict(self._flips),
        }


def party_factory_from_config(cfg: dict) -> Callable[[object], DualAgentParty]:
    """Validate ``cfg["triple_agent"]`` and return a factory that builds the party for a model."""
    block = cfg["triple_agent"]
    scope, rule = block["memory_scope"], block["final_label_rule"]
    if scope not in MEMORY_SCOPES:
        raise ValueError(f"unknown triple_agent.memory_scope {scope!r}; allowed: {MEMORY_SCOPES}")
    if rule not in FINAL_LABEL_RULES:
        raise ValueError(
            f"unknown triple_agent.final_label_rule {rule!r}; allowed: {FINAL_LABEL_RULES}"
        )
    return lambda model: DualAgentParty(model, memory_scope=scope, final_label_rule=rule)

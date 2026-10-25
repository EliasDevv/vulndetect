"""The LLM-GAN loop: Phase A (inter-adversary) and Phase B (self-reflection) on the train split."""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import logging
import math
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from vulndetect.common import config_sha256, load_config, set_seed
from vulndetect.common.prompt import PROMPT_VERSION
from vulndetect.data import DEFAULT_DATASET_DIR, load_split
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN, Record
from vulndetect.gan.prompts import PROMPT_VERSIONS
from vulndetect.gan.roles import (
    VALIDATION_MODES,
    GeneratedItem,
    SingleDetector,
    generate_fake,
    reflect,
    update_detector_strategies,
    update_generator_strategies,
)
from vulndetect.gan.snapshot import list_snapshots, write_snapshot
from vulndetect.gan.strategies import StrategyMemory, new_memories

logger = logging.getLogger(__name__)

TRAIN_SPLIT = "train"
PROBE_SPLITS = ("val",)


class SplitHygieneError(RuntimeError):
    """Raised when loop code is handed records from a split it must not see."""


def assert_train_only(records: list[Record], *, where: str) -> None:
    """Raise if any record does not come from the train split."""
    offenders = {r.get("split") for r in records} - {TRAIN_SPLIT}
    if offenders:
        raise SplitHygieneError(
            f"{where}: the GAN loop may only see split={TRAIN_SPLIT!r}, "
            f"got {sorted(str(o) for o in offenders)} — test data must reach "
            f"vulndetect.gan.run only"
        )


def assert_probe_split(records: list[Record], split: str) -> None:
    """Raise unless the drift probe split and its records are validation data."""
    if split not in PROBE_SPLITS:
        raise SplitHygieneError(f"drift probe split must be one of {PROBE_SPLITS}, got {split!r}")
    offenders = {r.get("split") for r in records} - {split}
    if offenders:
        raise SplitHygieneError(f"drift probe records are not all from {split!r}: {offenders}")


def _round_robin(groups: dict[str, list[Record]]) -> Iterator[Record]:
    """Cycle the (sorted) group keys, taking one record at a time."""
    keys = sorted(groups)
    queues = {k: list(groups[k]) for k in keys}
    i = 0
    while any(queues.values()):
        key = keys[i % len(keys)]
        i += 1
        if queues[key]:
            yield queues[key].pop(0)


def build_phase_b_pool(records: list[Record], n_items: int, seed: int) -> list[Record]:
    """Select ``n_items`` seeded, label-balanced and CWE-stratified train records for Phase B."""
    rng = random.Random(seed + 1)
    pool = list(records)
    rng.shuffle(pool)
    queues = {}
    for label in (LABEL_VULN, LABEL_SAFE):
        per_cwe: dict[str, list[Record]] = defaultdict(list)
        for r in pool:
            if r["label"] == label:
                per_cwe[r["cwe"]].append(r)
        queues[label] = _round_robin(per_cwe)

    out: list[Record] = []
    exhausted: set[str] = set()
    label_cycle = itertools.cycle((LABEL_VULN, LABEL_SAFE))
    while len(out) < n_items and len(exhausted) < len(queues):
        label = next(label_cycle)
        if label in exhausted:
            continue
        item = next(queues[label], None)
        if item is None:
            exhausted.add(label)
            continue
        out.append(item)
    return out


def build_source_pool(records: list[Record], exclude_ids: set[str], seed: int) -> list[Record]:
    """Order the Not-Vulnerable train records used as Phase-A sources (seeded, CWE round-robin)."""
    rng = random.Random(seed)
    safe = [r for r in records if r["label"] == LABEL_SAFE and r["id"] not in exclude_ids]
    rng.shuffle(safe)
    per_cwe: dict[str, list[Record]] = defaultdict(list)
    for r in safe:
        per_cwe[r["cwe"]].append(r)
    return list(_round_robin(per_cwe))


def build_probe_pool(
    split: str, n: int, seed: int, dataset_dir: str | Path = DEFAULT_DATASET_DIR
) -> list[Record]:
    """Select a fixed, label-balanced validation subset for the drift probe."""
    if split not in PROBE_SPLITS:
        raise SplitHygieneError(f"drift probe split must be one of {PROBE_SPLITS}, got {split!r}")
    records = load_split(split, dataset_dir)
    assert_probe_split(records, split)
    rng = random.Random(seed + 2)
    vuln = [r for r in records if r["label"] == LABEL_VULN]
    safe = [r for r in records if r["label"] == LABEL_SAFE]
    rng.shuffle(vuln)
    rng.shuffle(safe)
    half = max(n // 2, 1)
    return sorted(vuln[:half] + safe[:half], key=lambda r: r["id"])


def source_order_sha256(records: list[Record]) -> str:
    """Return a SHA-256 fingerprint of the record id order."""
    return hashlib.sha256(",".join(r["id"] for r in records).encode("utf-8")).hexdigest()


class RoundLog:
    """Append-only JSONL log with one line per round or step, flushed immediately."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, row: dict) -> None:
        """Append one row as a JSON line and flush it."""
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying log file."""
        self._fh.close()


class DecodingOverride:
    """A TextGenerator that runs the wrapped model with fixed, overridden decoding settings."""

    def __init__(self, model, *, temperature: float, top_p: float | None, max_new_tokens: int):
        self.model = model
        self.temperature = float(temperature)
        self.top_p = top_p
        self.max_new_tokens = int(max_new_tokens)

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate completions with the overridden decoding settings."""
        return self.model.generate(
            prompts,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )


class MeteredModel:
    """A TextGenerator wrapper that records the character and token length of every prompt."""

    def __init__(self, model, *, counter=None):
        self.model = model
        self._counter = model if counter is None else counter
        self._pending: list[dict] = []
        self.history: list[tuple[str, dict]] = []

    @property
    def context_window(self) -> int | None:
        """Return the context window of the token counter, or None if unknown."""
        window = getattr(self._counter, "context_window", None)
        return int(window) if window else None

    def _tokens(self, prompt: str, kwargs: dict) -> int | None:
        count = getattr(self._counter, "count_tokens", None)
        if count is None:
            return None
        return int(count(prompt, system=kwargs.get("system"), chat=kwargs.get("chat", True)))

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate completions and record prompt length, output length and overflow."""
        outputs = self.model.generate(prompts, **kwargs)
        window = self.context_window
        for prompt, out in zip(prompts, outputs, strict=True):
            tokens = self._tokens(prompt, kwargs)
            over = bool(window and tokens is not None and tokens > window)
            if over:
                logger.warning(
                    "prompt of %d tokens exceeds the context window (%d)", tokens, window
                )
            self._pending.append(
                {
                    "chars": len(prompt),
                    "tokens": tokens,
                    "output_chars": len(out),
                    "over_context": over,
                }
            )
        return outputs

    def drain(self, role: str) -> list[dict]:
        """Return the entries since the last drain and archive them under ``role``."""
        entries, self._pending = self._pending, []
        self.history.extend((role, e) for e in entries)
        return entries


def _drain(model, role: str, lengths: dict) -> None:
    """Record the metered prompt lengths of the last role call (no-op for plain models)."""
    drain = getattr(model, "drain", None)
    if drain is None:
        return
    entries = drain(role)
    if entries:
        lengths[role] = entries


def _p99(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]


def summarize_prompt_lengths(models: list) -> dict:
    """Summarise prompt lengths per role over every metered model of the run."""
    by_role: dict[str, list[dict]] = defaultdict(list)
    window = None
    for model in models:
        history = getattr(model, "history", None)
        if history is None:
            continue
        window = window or getattr(model, "context_window", None)
        for role, entry in history:
            by_role[role].append(entry)
    summary = {}
    for role, entries in sorted(by_role.items()):
        tokens = [e["tokens"] for e in entries if e["tokens"] is not None]
        summary[role] = {
            "n": len(entries),
            "max_chars": max(e["chars"] for e in entries),
            "max_tokens": max(tokens) if tokens else None,
            "p99_tokens": _p99(tokens),
            "n_over_context": sum(e["over_context"] for e in entries),
        }
    return {"context_window": window, "by_role": summary}


def run_drift_probe(
    detector_model, records: list[Record], s_d: StrategyMemory, *, party=None
) -> dict:
    """Measure the share of Vulnerable predictions and accuracy on the fixed probe subset."""
    party = party or SingleDetector(detector_model)
    detections = party.detect_batch(records, s_d.render())
    _drain(detector_model, "probe", {})
    n = len(records)

    def _share(labels: list[str | None]) -> dict:
        n_vuln = sum(label == LABEL_VULN for label in labels)
        n_correct = sum(label == r["label"] for label, r in zip(labels, records, strict=True))
        return {
            "n": n,
            "vulnerable_share": round(n_vuln / n, 4) if n else None,
            "accuracy": round(n_correct / n, 4) if n else None,
            "n_unparseable": sum(label is None for label in labels),
        }

    probe = {**_share([d.label for d in detections]), "n_strategies": len(s_d)}
    traces = [d.trace for d in detections]
    if traces and all(t is not None and "detector_label" in t for t in traces):
        probe["stage1"] = _share([t["detector_label"] for t in traces])
    return probe


def run_phase_a(
    generator_model,
    detector_model,
    sources: list[Record],
    s_g: StrategyMemory,
    s_d: StrategyMemory,
    *,
    rounds: int,
    max_new: int,
    max_words: int,
    validate: str,
    log: RoundLog,
    generated_log: RoundLog | None = None,
    on_milestone=None,
    party=None,
) -> dict:
    """Run Phase A (eqs. 1 to 4), updating S_G or S_D each round, and return its statistics."""
    if validate not in VALIDATION_MODES:
        raise ValueError(f"unknown validate mode {validate!r}; allowed: {VALIDATION_MODES}")
    assert_train_only(sources, where="phase A sources")
    party = party or SingleDetector(detector_model)

    stats = {
        "rounds_completed": 0,
        "n_generation_rejected": 0,
        "rejections_by_reason": defaultdict(int),
        "n_detector_correct": 0,
        "n_detector_wrong": 0,
        "n_detector_unparseable": 0,
        "n_generator_updates": 0,
        "n_detector_updates": 0,
        "n_malformed_updates": 0,
        "n_marker_missing": 0,
        "n_entries_added_generator": 0,
        "n_entries_added_detector": 0,
        "sources_consumed": 0,
        "source_ids": [],
    }
    if on_milestone:
        on_milestone(0, "init")

    for source in sources:
        if stats["rounds_completed"] >= rounds:
            break
        stats["sources_consumed"] += 1
        lengths: dict = {}
        item = generate_fake(
            generator_model, source, s_g.render(), stats["rounds_completed"] + 1, validate=validate
        )
        _drain(generator_model, "generator", lengths)
        if not isinstance(item, GeneratedItem):
            stats["n_generation_rejected"] += 1
            stats["rejections_by_reason"][item.reason] += 1
            log.write(
                {
                    "phase": "A",
                    "round": stats["rounds_completed"],
                    "source_id": source["id"],
                    "cwe": source["cwe"],
                    "accepted": False,
                    "rejection_reason": item.reason,
                    "prompt_lengths": lengths or None,
                }
            )
            if generated_log is not None:
                generated_log.write(
                    {
                        "round": stats["rounds_completed"],
                        "source_id": source["id"],
                        "cwe": source["cwe"],
                        "accepted": False,
                        "rejection_reason": item.reason,
                        "generator_raw": item.raw,
                    }
                )
            continue

        round_no = stats["rounds_completed"] + 1
        assert_train_only([item.record], where="generated item")
        detection = party.detect(item.record, s_d.render())
        _drain(detector_model, "detection", lengths)
        correct = detection.label == LABEL_VULN
        if detection.label is None:
            stats["n_detector_unparseable"] += 1
        if generated_log is not None:
            generated_log.write(
                {
                    "round": round_no,
                    "source_id": source["id"],
                    "item_id": item.record["id"],
                    "cwe": source["cwe"],
                    "accepted": True,
                    "function": item.function,
                    "commit_message": item.commit_message,
                    "deception_notes": item.deception_notes,
                    "generator_raw": item.raw,
                    "detector_label": detection.label,
                    "detector_raw": detection.raw,
                    "detection_trace": detection.trace,
                    "correct": correct,
                }
            )

        if correct:
            stats["n_detector_correct"] += 1
            update = update_generator_strategies(
                generator_model,
                source,
                item,
                detection,
                s_g.render(),
                max_new=max_new,
                max_words=max_words,
            )
            _drain(generator_model, "update", lengths)
            report = s_g.add(update.entries, model=generator_model)
            _drain(generator_model, "summarize", lengths)
            stats["n_generator_updates"] += 1
            stats["n_entries_added_generator"] += len(report.added)
            updated = "generator"
        else:
            stats["n_detector_wrong"] += 1
            update = update_detector_strategies(
                detector_model,
                item.record,
                detection.raw,
                item.deception_notes,
                "adversary",
                s_d.render(),
                max_new=max_new,
                max_words=max_words,
            )
            _drain(detector_model, "update", lengths)
            report = s_d.add(update.entries, model=detector_model)
            _drain(detector_model, "summarize", lengths)
            stats["n_detector_updates"] += 1
            stats["n_entries_added_detector"] += len(report.added)
            updated = "detector"
        if not update.parsed:
            stats["n_malformed_updates"] += 1
        elif not update.marker_present:
            stats["n_marker_missing"] += 1

        stats["rounds_completed"] = round_no
        stats["source_ids"].append(source["id"])
        log.write(
            {
                "phase": "A",
                "round": round_no,
                "source_id": source["id"],
                "item_id": item.record["id"],
                "cwe": source["cwe"],
                "accepted": True,
                "rejection_reason": None,
                "detector_label": detection.label,
                "detector_raw": detection.raw,
                "detection_trace": detection.trace,
                "correct": correct,
                "updated": updated,
                "update_parsed": update.parsed,
                "update_marker_present": update.marker_present,
                "update_raw": update.raw,
                "add_report": report.to_dict(),
                "s_g": s_g.stats(),
                "s_d": s_d.stats(),
                "cum_false_negative_rate": round(stats["n_detector_wrong"] / round_no, 4),
                "prompt_lengths": lengths or None,
            }
        )
        if on_milestone:
            on_milestone(round_no, "A")

    if stats["rounds_completed"] < rounds:
        logger.warning(
            "Phase A ran out of source records after %d of %d rounds (%d rejected)",
            stats["rounds_completed"],
            rounds,
            stats["n_generation_rejected"],
        )
    stats["rejections_by_reason"] = dict(stats["rejections_by_reason"])
    return stats


def run_phase_b(
    detector_model,
    reflector_model,
    items: list[Record],
    s_d: StrategyMemory,
    *,
    max_new: int,
    max_words: int,
    log: RoundLog,
    party=None,
) -> dict:
    """Run Phase B (eqs. 5 to 7), reflecting on wrong verdicts to update S_D; return statistics."""
    assert_train_only(items, where="phase B items")
    party = party or SingleDetector(detector_model)
    stats = {
        "n_items": len(items),
        "n_correct": 0,
        "n_wrong": 0,
        "n_unparseable": 0,
        "n_reflections": 0,
        "n_reflections_unstructured": 0,
        "n_detector_updates": 0,
        "n_malformed_updates": 0,
        "n_marker_missing": 0,
        "n_entries_added": 0,
        "n_vulnerable_predictions": 0,
        "wrong_by_gold": {LABEL_VULN: 0, LABEL_SAFE: 0},
    }
    for step, record in enumerate(items, start=1):
        lengths: dict = {}
        detection = party.detect(record, s_d.render())
        _drain(detector_model, "detection", lengths)
        correct = detection.label == record["label"]
        if detection.label is None:
            stats["n_unparseable"] += 1
        if detection.label == LABEL_VULN:
            stats["n_vulnerable_predictions"] += 1

        row = {
            "phase": "B",
            "step": step,
            "id": record["id"],
            "cwe": record["cwe"],
            "gold_label": record["label"],
            "detector_label": detection.label,
            "detector_raw": detection.raw,
            "detection_trace": detection.trace,
            "correct": correct,
            "reflected": False,
            "updated": "none",
        }
        if correct:
            stats["n_correct"] += 1
        else:
            stats["n_wrong"] += 1
            stats["wrong_by_gold"][record["label"]] += 1
            reflection = reflect(reflector_model, record, detection, max_words=max_words)
            _drain(reflector_model, "reflector", lengths)
            stats["n_reflections"] += 1
            if not reflection.structured:
                stats["n_reflections_unstructured"] += 1
            update = update_detector_strategies(
                detector_model,
                record,
                detection.raw,
                reflection.text,
                "reflection",
                s_d.render(),
                max_new=max_new,
                max_words=max_words,
            )
            _drain(detector_model, "update", lengths)
            report = s_d.add(update.entries, model=detector_model)
            _drain(detector_model, "summarize", lengths)
            stats["n_detector_updates"] += 1
            stats["n_entries_added"] += len(report.added)
            if not update.parsed:
                stats["n_malformed_updates"] += 1
            elif not update.marker_present:
                stats["n_marker_missing"] += 1
            row.update(
                {
                    "reflected": True,
                    "reflection_structured": reflection.structured,
                    "updated": "detector",
                    "update_parsed": update.parsed,
                    "update_marker_present": update.marker_present,
                    "reflection_raw": reflection.raw,
                    "update_raw": update.raw,
                    "add_report": report.to_dict(),
                }
            )
        row["s_d"] = s_d.stats()
        row["prompt_lengths"] = lengths or None
        log.write(row)

    n = len(items) or 1
    stats["vulnerable_prediction_share"] = round(stats["n_vulnerable_predictions"] / n, 4)
    stats["accuracy"] = round(stats["n_correct"] / n, 4)
    return stats


def run_loop(
    config_path: str | Path,
    *,
    generator_model=None,
    detector_model=None,
    party_factory=None,
    approach: str = "gan_loop",
) -> dict:
    """Run Phase A and Phase B for one config and write snapshots, round logs and the manifest."""
    started = dt.datetime.now()
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    gan_cfg = cfg["gan"]
    rounds = int(gan_cfg["rounds"])
    snapshot_rounds = sorted({int(r) for r in gan_cfg["snapshot_rounds"]})
    max_new = int(gan_cfg["max_new_strategies_per_update"])
    mem_cfg = gan_cfg["memory"]
    gen_cfg = gan_cfg["generation"]
    pb_cfg = gan_cfg["phase_b"]
    probe_cfg = gan_cfg["drift_probe"]
    validate = gen_cfg["validate"]
    if validate not in VALIDATION_MODES:
        raise ValueError(f"unknown gan.generation.validate {validate!r}: {VALIDATION_MODES}")
    max_words = int(mem_cfg["max_entry_words"])

    ds_cfg = cfg.get("dataset", {})
    dataset_dir = ds_cfg.get("dir", DEFAULT_DATASET_DIR)
    train_split = ds_cfg.get("train_split", TRAIN_SPLIT)
    if train_split != TRAIN_SPLIT:
        raise SplitHygieneError(f"dataset.train_split must be {TRAIN_SPLIT!r}, got {train_split!r}")
    records = load_split(train_split, dataset_dir)
    assert_train_only(records, where="train split")
    logger.info("loaded %d train records", len(records))

    phase_b_enabled = bool(pb_cfg["enabled"])
    phase_b_items = (
        build_phase_b_pool(records, int(pb_cfg["n_items"]), seed) if phase_b_enabled else []
    )
    sources = build_source_pool(records, {r["id"] for r in phase_b_items}, seed)
    logger.info("phase A source pool: %d, phase B items: %d", len(sources), len(phase_b_items))

    probe_enabled = bool(probe_cfg["enabled"])
    probe_seed = int(probe_cfg.get("seed", seed))
    probe_split = probe_cfg["split"]
    probe_pool = (
        build_probe_pool(probe_split, int(probe_cfg["n"]), probe_seed, dataset_dir)
        if probe_enabled
        else []
    )

    loaded = None
    if detector_model is None or generator_model is None:
        from vulndetect.common.model import load_model  # lazy import: pulls in torch

        loaded = load_model(cfg["model"])
    if detector_model is None:
        detector_model = loaded
    if generator_model is None:
        generator_model = DecodingOverride(
            loaded,
            temperature=float(gen_cfg["temperature"]),
            top_p=gen_cfg.get("top_p"),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
        )
    detector_model = MeteredModel(detector_model)
    generator_model = MeteredModel(
        generator_model, counter=loaded if loaded is not None else generator_model
    )
    party = (party_factory or SingleDetector)(detector_model)
    versions = {**PROMPT_VERSIONS, **party.prompt_versions}

    s_g, s_d = new_memories(mem_cfg)
    out_dir = Path(cfg["output"]["loop_dir"]) / f"seed{seed}"
    snapshot_dir = out_dir / "snapshots"
    cfg_sha = config_sha256(config_path)
    log = RoundLog(out_dir / "rounds.jsonl")
    generated_log = RoundLog(out_dir / "generated.jsonl")

    def snapshot(round_no: int | None, phase: str, stats: dict | None = None) -> None:
        probe = (
            run_drift_probe(detector_model, probe_pool, s_d, party=party) if probe_pool else None
        )
        path = write_snapshot(
            snapshot_dir,
            round_no=round_no,
            phase=phase,
            s_g=s_g,
            s_d=s_d,
            seed=seed,
            config_path=config_path,
            config_sha256=cfg_sha,
            probe=probe,
            stats=stats,
            prompt_versions=versions,
            detecting_party=party.describe(),
        )
        logger.info("snapshot %s (S_D: %d entries)", path.name, len(s_d))

    def on_milestone(round_no: int, phase: str) -> None:
        if round_no in snapshot_rounds:
            snapshot(round_no, phase)

    try:
        stats_a = run_phase_a(
            generator_model,
            detector_model,
            sources,
            s_g,
            s_d,
            rounds=rounds,
            max_new=max_new,
            max_words=max_words,
            validate=validate,
            log=log,
            generated_log=generated_log,
            on_milestone=on_milestone,
            party=party,
        )
        stats_b = None
        if phase_b_enabled:
            stats_b = run_phase_b(
                detector_model,
                detector_model,  # the Reflector uses the Detector model
                phase_b_items,
                s_d,
                max_new=max_new,
                max_words=max_words,
                log=log,
                party=party,
            )
        snapshot(None, "B" if phase_b_enabled else "A", {"phase_a": stats_a, "phase_b": stats_b})
    finally:
        log.close()
        generated_log.close()

    manifest = {
        "approach": approach,
        "model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": cfg_sha,
        "prompt_version": PROMPT_VERSION,
        "gan_prompt_versions": versions,
        "detecting_party": {**party.describe(), "stats": party.stats()},
        "prompt_lengths": summarize_prompt_lengths([generator_model, detector_model]),
        "dataset": {
            **ds_cfg,
            "train_split": train_split,
            "n_train": len(records),
            "n_source_pool": len(sources),
        },
        "decoding": {
            "detector": {
                "temperature": cfg["model"].get("temperature", 0.0),
                "max_new_tokens": cfg["model"].get("max_new_tokens"),
                "batch_size": cfg["model"].get("batch_size"),
            },
            "generator": {
                "temperature": gen_cfg["temperature"],
                "top_p": gen_cfg.get("top_p"),
                "max_new_tokens": gen_cfg["max_new_tokens"],
            },
        },
        "gan": {
            "rounds": rounds,
            "snapshot_rounds": snapshot_rounds,
            "max_new_strategies_per_update": max_new,
            "memory": dict(mem_cfg),
            "generation": dict(gen_cfg),
            "phase_b": dict(pb_cfg),
            "drift_probe": {
                **dict(probe_cfg),
                "seed_realised": probe_seed,
                "n_realised": len(probe_pool),
            },
        },
        "source_order_sha256": source_order_sha256(sources),
        "probe_ids_sha256": source_order_sha256(probe_pool),
        "phase_a": stats_a,
        "phase_b": stats_b,
        "final_memories": {
            "s_g": s_g.to_dict(),
            "s_d": s_d.to_dict(),
            "s_g_stats": s_g.stats(),
            "s_d_stats": s_d.stats(),
        },
        "snapshots": [p.name for p in list_snapshots(snapshot_dir)],
        "generated_items_path": str(out_dir / "generated.jsonl"),
        "duration_seconds": round((dt.datetime.now() - started).total_seconds(), 1),
    }
    (out_dir / "loop_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("loop done: %s", out_dir)
    return manifest

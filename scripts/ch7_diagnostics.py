"""Zero-GPU diagnostics for the Results chapter (thesis Chapter 7).

Reads finished artefacts and writes ``results/metrics/ch7_diagnostics.json``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from scipy import stats as sps

from vulndetect.common.parsing import get_parser
from vulndetect.common.predictions import read_predictions
from vulndetect.data.schema import LABEL_VULN
from vulndetect.eval.metrics import evaluate

SEEDS = (42, 43, 44)
RUNS = Path("results/predictions")
LOOPS = Path("results/gan_val")
TRIPLE_LOOPS = Path("results/triple_agent_val")
OUT = Path("results/metrics")
QUEUE_STATUS = Path("results/logs/gpu-server/queue_status.tsv")

# Inference switched from bf16 to fp16 on this date; earlier runs use another precision regime.
FP16_SINCE = dt.date(2026, 8, 28)

# Superseded experiments.csv rows, excluded from every figure derived from the CSV.
SUPERSEDED_ROWS = {
    ("2026-09-08T05:39:28", "rag", "42"),
}

# Model calls per evaluated test instance.
CALLS_PER_INSTANCE = {
    "baseline": 1,
    "rag": 1,
    "sft": 1,
    "dual_agent": 2,
    "gan": 1,
    "triple_agent": 2,
}


def load_run(run_dir: Path) -> dict:
    """Load manifest, metrics and predictions re-parsed with the effective parser."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    parser_version = metrics.get("parser_version") or manifest["parser_version"]
    parse = get_parser(parser_version)
    preds = read_predictions(run_dir / "predictions.jsonl")
    preds = [{**p, "pred_label": parse(p["raw_output"])} for p in preds]
    return {
        "dir": str(run_dir),
        "manifest": manifest,
        "metrics": metrics,
        "parser_version": parser_version,
        "policy": metrics.get("unparseable_policy") or manifest.get("unparseable_policy"),
        "preds": preds,
        "finished": dt.datetime.fromtimestamp((run_dir / "predictions.jsonl").stat().st_mtime),
    }


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def round0_check(baseline_preds: list[dict], round0_preds: list[dict]) -> dict:
    """Verdict-for-verdict comparison of a round-zero snapshot with the baseline."""
    base = {p["id"]: p for p in baseline_preds}
    snap = {p["id"]: p for p in round0_preds}
    if set(base) != set(snap):
        raise ValueError("round0 and baseline cover different instances")
    label_diff = [i for i in base if base[i]["pred_label"] != snap[i]["pred_label"]]
    raw_diff = [i for i in base if base[i]["raw_output"] != snap[i]["raw_output"]]
    return {
        "n": len(base),
        "n_label_differs": len(label_diff),
        "n_raw_output_differs": len(raw_diff),
        "identical": not label_diff and not raw_diff,
    }


def dual_agent_rules(agents: list[dict], preds: list[dict], parse, policy: str) -> dict:
    """Post-hoc re-scoring of a finished Dual-Agent run under alternative decision rules."""
    by_id = {a["id"]: a for a in agents}
    if set(by_id) != {p["id"] for p in preds}:
        raise ValueError("agents.jsonl and predictions.jsonl cover different instances")
    variants: dict[str, list[dict]] = {
        "validator": [],
        "validator_then_detector": [],
        "detector": [],
    }
    n_fallback = 0
    for p in preds:
        det = parse(by_id[p["id"]]["detector_output"])
        val = parse(p["raw_output"])
        fallback = det if val is None else val
        n_fallback += val is None
        for name, label in (
            ("validator", val),
            ("validator_then_detector", fallback),
            ("detector", det),
        ):
            variants[name].append({**p, "pred_label": label})
    out = {"n": len(preds), "n_validator_unparseable": n_fallback, "post_hoc": True}
    for name, rows in variants.items():
        rep = evaluate(rows, policy=policy)
        out[name] = {
            "macro_f1": rep["macro"]["f1"],
            "macro_accuracy": rep["macro"]["accuracy"],
            "pred_positive_rate": rep["macro"]["pred_positive_rate"],
            "parse_failure_rate": rep["parse_failure_rate"],
        }
    both = [
        (a["pred_label"], v["pred_label"])
        for a, v in zip(variants["detector"], variants["validator"], strict=True)
        if a["pred_label"] is not None and v["pred_label"] is not None
    ]
    # share of changed verdicts among instances where both verdicts are readable
    out["n_both_readable"] = len(both)
    out["n_validator_changed_detector"] = sum(a != v for a, v in both)
    return out


def pass_through_curve(rounds: list[dict], milestones=(10, 50, 100, 200)) -> dict:
    """Cumulative share of admitted phase-A cases judged Not Vulnerable."""
    admitted = [r for r in rounds if r.get("phase") == "A" and r.get("accepted")]
    admitted.sort(key=lambda r: r["round"])
    passed = 0
    curve = {}
    for k, r in enumerate(admitted, start=1):
        passed += r["detector_label"] is not None and r["detector_label"] != LABEL_VULN
        if r["round"] in milestones:
            curve[str(r["round"])] = passed / k
    return {
        "n_admitted": len(admitted),
        "n_passed": passed,
        "final_rate": passed / len(admitted) if admitted else None,
        "at_round": curve,
    }


def loop_stats(manifest: dict, rounds: list[dict], snapshots: dict[str, dict]) -> dict:
    """Gate, pass-through, phase B, probe drift and prompt lengths of one loop run."""
    pa, pb = manifest["phase_a"], manifest["phase_b"]
    lengths = manifest["prompt_lengths"]
    return {
        "seed": manifest["seed"],
        "duration_hours": manifest.get("duration_seconds", 0) / 3600,
        "gate": {
            "sources_consumed": pa["sources_consumed"],
            "rounds_completed": pa["rounds_completed"],
            "n_rejected": pa["n_generation_rejected"],
            "rejected_share": pa["n_generation_rejected"] / pa["sources_consumed"],
            "by_reason": pa["rejections_by_reason"],
        },
        "phase_a": {
            "n_detector_correct": pa["n_detector_correct"],
            "n_detector_wrong": pa["n_detector_wrong"],
            "n_detector_unparseable": pa["n_detector_unparseable"],
        },
        "pass_through": pass_through_curve(rounds),
        "phase_b": {
            "n_items": pb["n_items"],
            "accuracy": pb["accuracy"],
            "vulnerable_prediction_share": pb["vulnerable_prediction_share"],
            "wrong_by_gold": pb["wrong_by_gold"],
            "n_unparseable": pb["n_unparseable"],
        },
        "probe": {name: snap.get("drift_probe") for name, snap in sorted(snapshots.items())},
        "prompt_tokens": {
            "context_window": lengths["context_window"],
            "max_by_role": {role: v["max_tokens"] for role, v in lengths["by_role"].items()},
            "n_over_context": sum(v["n_over_context"] for v in lengths["by_role"].values()),
        },
    }


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for k successes in n trials."""
    if not 0 <= k <= n or n == 0:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    alpha = 1 - level
    lower = 0.0 if k == 0 else float(sps.beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def hand_check(rows: list[dict]) -> dict:
    """Count the hand-check categories of the rated sample."""
    cats = ("intended", "other", "broken", "none")
    verdicts = [r["verdict"].strip() for r in rows]
    unknown = sorted({v for v in verdicts if v not in cats})
    if unknown:
        raise ValueError(f"unrated or unknown verdicts: {unknown}")
    counts = Counter(verdicts)
    n = len(verdicts)
    k = counts["intended"]
    lo, hi = clopper_pearson(k, n)
    k_any = counts["intended"] + counts["other"]
    lo_any, hi_any = clopper_pearson(k_any, n)
    n_agree = sum(1 for r in rows if r.get("claude_suggestion", "").strip() == r["verdict"].strip())
    return {
        "n": n,
        "counts": {c: counts[c] for c in cats},
        "intended_share": k / n,
        "intended_ci95": [lo, hi],
        "any_weakness_share": k_any / n,
        "any_weakness_ci95": [lo_any, hi_any],
        "n_author_confirmed_preliminary": n_agree,
        "detector_vulnerable_by_category": {
            c: sum(
                1 for r in rows if r["verdict"].strip() == c and r["detector_label"] == LABEL_VULN
            )
            for c in cats
        },
    }


def verdict_change(a: list[dict], b: list[dict]) -> dict:
    """Share of instances whose parsed verdict differs, on instances both parsed."""
    la = {p["id"]: p["pred_label"] for p in a}
    lb = {p["id"]: p["pred_label"] for p in b}
    if set(la) != set(lb):
        raise ValueError("runs cover different instances")
    both = [i for i in la if la[i] is not None and lb[i] is not None]
    changed = sum(la[i] != lb[i] for i in both)
    return {"changed": changed, "n": len(both), "share": changed / len(both) if both else None}


def regime(finished: dt.datetime, dtype: str | None = None) -> str:
    """Precision regime of a run by its finish date, cross-checked with dtype if known."""
    by_date = "fp16" if finished.date() >= FP16_SINCE else "bf16"
    if dtype is not None and dtype.replace("float", "fp") != by_date:
        return f"{by_date} by date, config says {dtype} — check"
    return by_date


_BACKTICK = re.compile(r"`([^`]+)`")
_IDENT_SHAPES = (
    re.compile(r"\b[A-Za-z]\w*_\w+\b"),  # snake_case / SCREAMING_CASE with an underscore
    re.compile(r"\b[a-z]+[A-Z]\w*\b"),  # camelCase
    re.compile(r"\b[A-Z][a-z0-9]+[A-Z]\w*\b"),  # PascalCase with an inner capital
    re.compile(r"\b\w+(?:->|::)\w+\b"),  # member / scope access
    re.compile(r"\b\w+\(\)"),  # call syntax
)
_CODE_TOKEN = re.compile(r"[A-Za-z_]\w*|->|::|\d+|[^\s\w]")
_FUNC_NAME = re.compile(r"([A-Za-z_]\w*)\s*\(")


def identifier_candidates(entries: list[str]) -> list[str]:
    """Identifier-shaped spans in memory entries."""
    found: set[str] = set()
    for text in entries:
        found.update(s.strip() for s in _BACKTICK.findall(text))
        plain = text.replace("**", "")
        for pattern in _IDENT_SHAPES:
            found.update(pattern.findall(plain))
    return sorted(found)


def code_tokens(text: str) -> list[str]:
    """Split code text into identifier, number and operator tokens."""
    return _CODE_TOKEN.findall(text)


def ngrams(tokens: list[str], n: int = 3) -> set[tuple[str, ...]]:
    """Return the set of token n-grams of length n."""
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def function_name(func: str) -> str | None:
    """Name of a C/C++ function from its definition text (first ``name(`` before ``{``)."""
    head = func.split("{", 1)[0]
    names = _FUNC_NAME.findall(head)
    return names[-1] if names else None


def leakage(memories: dict[str, list[str]], test_records: list[dict], n: int = 3) -> dict:
    """Match candidates in the final memories against test function names and code n-grams."""
    names = {function_name(r["func_before"]) for r in test_records} - {None}
    idents = set()
    grams: set[tuple[str, ...]] = set()
    for r in test_records:
        toks = code_tokens(r["func_before"])
        idents.update(t for t in toks if re.match(r"[A-Za-z_]\w*$", t))
        grams |= ngrams(toks, n)
    out = {
        "n_test_functions": len(test_records),
        "n_test_function_names": len(names),
        "memories": {},
    }
    for label, entries in sorted(memories.items()):
        cands = identifier_candidates(entries)
        name_hits = sorted(c for c in cands if c.rstrip("()") in names)
        ident_hits = sorted(c for c in cands if c.rstrip("()") in idents)
        all_grams = {
            " ".join(g)
            for e in entries
            for g in ngrams(code_tokens(e.replace("**", "")), n) & grams
        }
        # identifier-like: at least one token with an underscore or a capital letter
        ident_grams = {
            g
            for g in all_grams
            if any(
                re.match(r"[A-Za-z_]\w*$", t) and ("_" in t or not t.islower()) for t in g.split()
            )
        }
        out["memories"][label] = {
            "n_entries": len(entries),
            "candidates": cands,
            "hits_function_name": name_hits,
            "hits_test_identifier": ident_hits,
            "hits_code_3gram_all": sorted(all_grams),
            "hits_code_3gram_identifier_like": sorted(ident_grams),
        }
    return out


def queue_durations(rows: list[dict]) -> dict[str, list[float]]:
    """Wall-clock hours per queue target from queue_status.tsv (exit 0 only)."""
    fmt = "%Y-%m-%d %H:%M:%S"
    out: dict[str, list[float]] = {}
    for r in rows:
        if r["exit"].strip() != "0":
            continue
        hours = dt.datetime.strptime(r["finished"], fmt) - dt.datetime.strptime(r["started"], fmt)
        out.setdefault(r["target"], []).append(hours.total_seconds() / 3600)
    return out


def csv_gaps(rows: list[dict]) -> list[dict]:
    """Hours between consecutive experiments.csv rows, superseded rows excluded."""
    kept = [r for r in rows if (r["timestamp"], r["approach"], r["seed"]) not in SUPERSEDED_ROWS]
    kept.sort(key=lambda r: r["timestamp"])
    out = []
    for prev, cur in zip(kept, kept[1:], strict=False):
        gap = dt.datetime.fromisoformat(cur["timestamp"]) - dt.datetime.fromisoformat(
            prev["timestamp"]
        )
        out.append(
            {
                "approach": cur["approach"],
                "seed": cur["seed"],
                "output": cur["output"],
                "hours_since_previous_row": gap.total_seconds() / 3600,
            }
        )
    return out


def sorted_agents(dual_dir: Path, baseline_dir: Path) -> list[dict]:
    """Return agents.jsonl rows in the order of the baseline's predictions.jsonl."""
    by_id = {a["id"]: a for a in read_jsonl(dual_dir / "agents.jsonl")}
    return [by_id[p["id"]] for p in read_predictions(baseline_dir / "predictions.jsonl")]


def _seed_dir(approach: str, seed: int, snapshot: str | None = None) -> Path:
    """Return the run directory of an approach and seed, optionally for a snapshot."""
    d = RUNS / approach / f"seed{seed}"
    return d / snapshot if snapshot else d


def _loop_stats_of(base: Path) -> dict:
    """Compute the loop statistics of one loop directory."""
    manifest = json.loads((base / "loop_manifest.json").read_text(encoding="utf-8"))
    snaps = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((base / "snapshots").glob("*.json"))
    }
    return loop_stats(manifest, read_jsonl(base / "rounds.jsonl"), snaps)


def _final_memories(loops: Path) -> dict[str, list[str]]:
    """Collect the final S_D and S_G entries of every seed from the loop manifests."""
    memories = {}
    for s in SEEDS:
        final = json.loads((loops / f"seed{s}" / "loop_manifest.json").read_text(encoding="utf-8"))[
            "final_memories"
        ]
        memories[f"s_d/seed{s}"] = final["s_d"]["entries"]
        memories[f"s_g/seed{s}"] = final["s_g"]["entries"]
    return memories


def build(use_tokenizer: bool = True) -> dict:
    """Compute all Chapter 7 diagnostics and return them as one dict."""
    out: dict = {"generated": dt.datetime.now().isoformat(timespec="seconds")}

    # round-zero consistency check on test
    out["round0_check"] = {
        str(s): round0_check(
            load_run(_seed_dir("baseline", s))["preds"],
            load_run(_seed_dir("gan", s, "round0000"))["preds"],
        )
        for s in SEEDS
    }

    # Dual-Agent post-hoc rules
    dual = {}
    for s in SEEDS:
        run = load_run(_seed_dir("dual_agent", s))
        agents = read_jsonl(_seed_dir("dual_agent", s) / "agents.jsonl")
        dual[str(s)] = dual_agent_rules(
            agents, run["preds"], get_parser(run["parser_version"]), run["policy"]
        )
    out["dual_agent_rules"] = dual

    # the Detector pass of the Dual-Agent system is compared with the baseline run
    out["dual_detector_vs_baseline"] = {
        str(s): round0_check(
            load_run(_seed_dir("baseline", s))["preds"],
            [
                {**p, "pred_label": None, "raw_output": a["detector_output"]}
                for p, a in zip(
                    load_run(_seed_dir("baseline", s))["preds"],
                    sorted_agents(_seed_dir("dual_agent", s), _seed_dir("baseline", s)),
                    strict=True,
                )
            ],
        )["n_raw_output_differs"]
        for s in SEEDS
    }

    # loop statistics
    loops = {str(s): _loop_stats_of(LOOPS / f"seed{s}") for s in SEEDS}
    out["loop_stats"] = loops

    # Triple-Agent (RQ3)
    triple: dict = {
        "round0_check": {
            str(s): round0_check(
                load_run(_seed_dir("dual_agent", s))["preds"],
                load_run(_seed_dir("triple_agent", s, "round0000"))["preds"],
            )
            for s in SEEDS
        },
        "stages": {},
        "loop_stats": {str(s): _loop_stats_of(TRIPLE_LOOPS / f"seed{s}") for s in SEEDS},
    }
    for s in SEEDS:
        for snap in ("round0000", "final"):
            d = _seed_dir("triple_agent", s, snap)
            run = load_run(d)
            triple["stages"][f"{snap}/seed{s}"] = dual_agent_rules(
                read_jsonl(d / "agents.jsonl"),
                run["preds"],
                get_parser(run["parser_version"]),
                run["policy"],
            )
    out["triple_agent"] = triple

    # hand check (seed 42)
    with (LOOPS / "seed42" / "inspection_scores.csv").open(encoding="utf-8", newline="") as fh:
        out["hand_check"] = hand_check(list(csv.DictReader(fh)))

    # verdict-change shares, with the precision regime of every run
    runs = {
        "greedy_round0000": _seed_dir("gan_val_greedy", 42, "round0000"),
        "greedy_final": _seed_dir("gan_val_greedy", 42, "final"),
        "greedy_round0000_other_date": _seed_dir("gan_val_greedy_date", 42, "round0000"),
        "sampled_round0000": _seed_dir("gan_val", 42, "round0000"),
        "sampled_final": _seed_dir("gan_val", 42, "final"),
        "sampled_baseline_pinned_date": _seed_dir("baseline_val", 42),
        "sampled_baseline_other_date": _seed_dir("baseline_val_date", 42),
    }
    loaded = {k: load_run(v) for k, v in runs.items() if (v / "predictions.jsonl").exists()}
    prov = {
        k: {
            "dir": r["dir"],
            "finished": r["finished"].isoformat(timespec="minutes"),
            "regime": regime(r["finished"]),
            "prompt_version": r["manifest"].get("prompt_version"),
            "decoding": r["manifest"].get("decoding"),
        }
        for k, r in loaded.items()
    }
    d87 = {"runs": prov}
    pairs = {
        "greedy_memory_effect": ("greedy_round0000", "greedy_final"),
        "greedy_date_effect": ("greedy_round0000", "greedy_round0000_other_date"),
        "sampled_memory_effect": ("sampled_round0000", "sampled_final"),
        "sampled_date_effect": ("sampled_baseline_pinned_date", "sampled_baseline_other_date"),
    }
    for name, (a, b) in pairs.items():
        if a in loaded and b in loaded:
            entry = verdict_change(loaded[a]["preds"], loaded[b]["preds"])
            regimes = {prov[a]["regime"], prov[b]["regime"]}
            entry["regime"] = regimes.pop() if len(regimes) == 1 else "mixed — other regime"
            same_prompt = prov[a]["prompt_version"] == prov[b]["prompt_version"]
            entry["same_prompt_version"] = same_prompt
            d87[name] = entry
    g_mem, g_date = d87.get("greedy_memory_effect"), d87.get("greedy_date_effect")
    if g_mem and g_date:
        d87["criterion_d_holds"] = g_mem["share"] > g_date["share"]
    out["d87d"] = d87

    # leakage in final memories
    test_records = read_jsonl(Path("data/processed/bigvul/test.jsonl"))
    memories = _final_memories(LOOPS)
    out["leakage"] = leakage(memories, test_records)
    out["final_memories"] = memories
    triple["final_memories"] = _final_memories(TRIPLE_LOOPS)
    triple["leakage"] = leakage(triple["final_memories"], test_records)

    # greedy val evaluations of every seed-42 snapshot (diagnostic only)
    out["greedy_val_diagnostic"] = {
        snap: {
            k: json.loads(
                (_seed_dir("gan_val_greedy", 42, snap) / "metrics.json").read_text(encoding="utf-8")
            )["macro"][k]
            for k in ("f1", "accuracy", "pred_positive_rate")
        }
        for snap in ("round0000", "round0010", "round0050", "round0200", "final")
    }

    # self-reflection stage alone on val, seed 42, beside the full loop
    phaseb = {
        "round0000": _seed_dir("gan_val", 42, "round0000"),
        "final": _seed_dir("gan_val", 42, "final"),
        "phaseb_only_final": _seed_dir("gan_val_phaseb_only", 42, "final"),
    }
    if all((d / "metrics.json").exists() for d in phaseb.values()):
        decodings = {
            json.dumps(
                json.loads((d / "manifest.json").read_text(encoding="utf-8")).get("decoding"),
                sort_keys=True,
            )
            for d in phaseb.values()
        }
        if len(decodings) != 1:
            raise ValueError("phase-B ablation runs differ in evaluation decoding")
        out["phaseb_only_val"] = {
            name: {
                k: json.loads((d / "metrics.json").read_text(encoding="utf-8"))["macro"][k]
                for k in ("f1", "accuracy", "pred_positive_rate")
            }
            for name, d in phaseb.items()
        }

    # cost
    cost: dict = {"calls_per_instance": CALLS_PER_INSTANCE}
    if QUEUE_STATUS.exists():
        with QUEUE_STATUS.open(encoding="utf-8", newline="") as fh:
            durations = queue_durations(list(csv.DictReader(fh, delimiter="\t")))
        cost["queue_hours"] = durations
        cost["queue_source"] = str(QUEUE_STATUS)
    with Path("results/experiments.csv").open(encoding="utf-8", newline="") as fh:
        cost["csv_gaps_plausibility"] = csv_gaps(list(csv.DictReader(fh)))
    cost["sft_training_seconds"] = {
        str(s): json.loads(
            (Path("results/training/sft") / f"seed{s}" / "training_manifest.json").read_text(
                encoding="utf-8"
            )
        ).get("runtime_seconds")
        for s in SEEDS
    }
    cost["gan_loop_hours"] = {s: v["duration_hours"] for s, v in loops.items()}
    cost["triple_loop_hours"] = {s: v["duration_hours"] for s, v in triple["loop_stats"].items()}
    if use_tokenizer:
        cost["output_tokens_per_instance"] = output_tokens()
    out["cost"] = cost
    return out


def output_tokens() -> dict:
    """Mean generated tokens per test instance (all model calls of the approach), seed-averaged."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    def count(texts: list[str]) -> float:
        return statistics.fmean(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)

    res: dict = {}
    for approach, snapshot in (
        ("baseline", None),
        ("rag", None),
        ("sft", None),
        ("dual_agent", None),
        ("gan", "final"),
        ("triple_agent", "final"),
    ):
        per_seed = []
        for s in SEEDS:
            d = _seed_dir(approach, s, snapshot)
            texts = [p["raw_output"] for p in read_predictions(d / "predictions.jsonl")]
            total = count(texts)
            if approach in ("dual_agent", "triple_agent"):  # two calls: detector + validator
                total += count([a["detector_output"] for a in read_jsonl(d / "agents.jsonl")])
            per_seed.append(total)
        key = f"{approach}/{snapshot}" if snapshot else approach
        res[key] = {"mean": statistics.fmean(per_seed), "per_seed": per_seed}
    return res


def main() -> None:
    """Parse the CLI arguments, build the diagnostics and write them as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT / "ch7_diagnostics.json"))
    ap.add_argument("--no-tokenizer", action="store_true", help="skip output-token counts")
    args = ap.parse_args()
    result = build(use_tokenizer=not args.no_tokenizer)
    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

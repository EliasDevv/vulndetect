"""Zero-GPU diagnostics for the Discussion chapter (thesis Chapter 8).

Reads finished artefacts and writes ``results/metrics/ch8_diagnostics.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
from pathlib import Path

from vulndetect.common.parsing import _VULN_CUE_RE, get_parser
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
from vulndetect.eval.metrics import evaluate
from vulndetect.eval.stats import mcnemar

_SPEC = importlib.util.spec_from_file_location(
    "ch7_diagnostics", Path(__file__).resolve().parent / "ch7_diagnostics.py"
)
ch7 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ch7)

SEEDS = ch7.SEEDS
RUNS = ch7.RUNS
OUT = ch7.OUT
POLICY = "drop"

RQ1_PAIRS = (
    ("rag", "sft"),
    ("sft", "dual_agent"),
    ("baseline", "dual_agent"),
    ("baseline", "rag"),
    ("baseline", "sft"),
    ("dual_agent", "rag"),
)
RQ2_PAIRS = (("baseline", "gan/final"), ("dual_agent", "gan/final"))
RQ3_PAIRS = (("gan/final", "triple_agent/final"),)

# Validation-split controls: (key, reference run, control run, what differs).
VAL_CONTROLS = (
    ("date", "baseline_val", "baseline_val_date", "chat-template date"),
    ("label_definition", "baseline_val", "baseline_labeldef_val", "explicit label definition"),
    ("rag_rerun", "rag_val", "rag_val_dec06", "second run, same configuration"),
    ("rag_random", "rag_val", "rag_val_random_dec06", "random catalogue chunks as context"),
    ("rag_none", "rag_val", "rag_val_nocontext_dec06", "no retrieved context"),
    ("validator_greedy", "dual_agent_val", "dual_agent_val_greedyval", "Validator greedy"),
    (
        "strict_detector",
        "baseline_val_strict",
        "dual_agent_val_strictdet",
        "Validator on the stricter detector's texts",
    ),
)


def _index(preds: list[dict]) -> dict[str, dict]:
    """Index predictions by id, raising on duplicate ids."""
    out = {p["id"]: p for p in preds}
    if len(out) != len(preds):
        raise ValueError("duplicate prediction id")
    return out


def _macro(preds: list[dict]) -> dict:
    """Return macro F1, accuracy, MCC and predicted-positive rate under the drop policy."""
    m = evaluate(preds, policy=POLICY)["macro"]
    return {k: m[k] for k in ("f1", "accuracy", "mcc", "pred_positive_rate")}


def flip_profile(a: list[dict], b: list[dict]) -> dict:
    """How the verdicts of run b differ from run a, on instances both parsed."""
    ia, ib = _index(a), _index(b)
    if set(ia) != set(ib):
        raise ValueError("runs cover different instances")
    both = [i for i in ia if ia[i]["pred_label"] is not None and ib[i]["pred_label"] is not None]
    to_vuln = [
        i for i in both if ia[i]["pred_label"] == LABEL_SAFE and ib[i]["pred_label"] == LABEL_VULN
    ]
    to_safe = [
        i for i in both if ia[i]["pred_label"] == LABEL_VULN and ib[i]["pred_label"] == LABEL_SAFE
    ]
    right_v = sum(ia[i]["gold_label"] == LABEL_VULN for i in to_vuln)
    right_s = sum(ia[i]["gold_label"] == LABEL_SAFE for i in to_safe)
    changed = len(to_vuln) + len(to_safe)
    ma, mb = _macro(a), _macro(b)
    return {
        "n": len(both),
        "changed": changed,
        "share": changed / len(both) if both else None,
        "to_vuln": len(to_vuln),
        "to_vuln_correct": right_v,
        "to_safe": len(to_safe),
        "to_safe_correct": right_s,
        "net_correct": (right_v + right_s) - (changed - right_v - right_s),
        "delta": {k: mb[k] - ma[k] for k in ma},
    }


def agreement(agents: list[dict], preds: list[dict], parse) -> dict:
    """Detector x Validator cross-table and the correctness of the Validator's overrides."""
    by_id = {r["id"]: r for r in agents}
    if set(by_id) != {p["id"] for p in preds}:
        raise ValueError("agents and predictions cover different instances")
    keys = (LABEL_VULN, LABEL_SAFE, None)
    table = {str(d): {str(v): 0 for v in keys} for d in keys}
    overrides = {
        "n": 0,
        "right": 0,
        "to_vuln": 0,
        "to_vuln_right": 0,
        "to_safe": 0,
        "to_safe_right": 0,
    }
    for p in preds:
        d = parse(by_id[p["id"]]["detector_output"])
        v = parse(p["raw_output"])
        table[str(d)][str(v)] += 1
        if d is not None and v is not None and d != v:
            right = v == p["gold_label"]
            overrides["n"] += 1
            overrides["right"] += right
            side = "to_vuln" if v == LABEL_VULN else "to_safe"
            overrides[side] += 1
            overrides[f"{side}_right"] += right

    def share_vuln(d: str) -> float | None:
        row = table[d]
        n = row[LABEL_VULN] + row[LABEL_SAFE]
        return row[LABEL_VULN] / n if n else None

    return {
        "table": table,
        "p_validator_vuln_given_detector_vuln": share_vuln(LABEL_VULN),
        "p_validator_vuln_given_detector_safe": share_vuln(LABEL_SAFE),
        "overrides": overrides,
        "override_right_share": overrides["right"] / overrides["n"] if overrides["n"] else None,
    }


def unchanged(record: dict) -> bool:
    """True when the fixing commit left the function text identical (exact string match)."""
    return record["func_before"] == record["func_after"]


def identity_shortcut(records: list[dict], runs: dict[str, list[list[dict]]]) -> dict:
    """Score the before/after identity as a label proxy and each approach on both sides."""
    same = {r["id"]: unchanged(r) for r in records}
    counts = {lab: {"unchanged": 0, "changed": 0} for lab in (LABEL_VULN, LABEL_SAFE)}
    for r in records:
        counts[r["label"]]["unchanged" if same[r["id"]] else "changed"] += 1
    rule = [
        {
            "id": r["id"],
            "cwe": r["cwe"],
            "gold_label": r["label"],
            "pred_label": LABEL_SAFE if same[r["id"]] else LABEL_VULN,
            "raw_output": "",
        }
        for r in records
    ]
    out = {"n": len(records), "counts": counts, "rule": _macro(rule), "approaches": {}}
    for name, per_seed in runs.items():
        side_acc: dict[str, list[float]] = {"unchanged": [], "changed": []}
        for preds in per_seed:
            for side, flag in (("unchanged", True), ("changed", False)):
                sub = [p for p in preds if same[p["id"]] is flag and p["pred_label"] is not None]
                if sub:
                    side_acc[side].append(
                        sum(p["pred_label"] == p["gold_label"] for p in sub) / len(sub)
                    )
        out["approaches"][name] = {
            side: statistics.fmean(v) if v else None for side, v in side_acc.items()
        }
    return out


def fix_described_fn(base: list[dict], other: list[dict], cue: re.Pattern = _VULN_CUE_RE) -> dict:
    """Baseline false negatives whose text names a fix, and how another approach does on them."""
    ib, io = _index(base), _index(other)
    if set(ib) != set(io):
        raise ValueError("runs cover different instances")
    fn = [i for i in ib if ib[i]["gold_label"] == LABEL_VULN and ib[i]["pred_label"] == LABEL_SAFE]
    tn = [i for i in ib if ib[i]["gold_label"] == LABEL_SAFE and ib[i]["pred_label"] == LABEL_SAFE]
    described = [i for i in fn if cue.search(ib[i]["raw_output"] or "")]
    rest = [i for i in fn if i not in set(described)]

    def right(ids: list[str]) -> float | None:
        return (
            sum(io[i]["pred_label"] == io[i]["gold_label"] for i in ids) / len(ids) if ids else None
        )

    return {
        "n_fn": len(fn),
        "n_fix_described": len(described),
        "other_right_on_fix_described": right(described),
        "other_right_on_other_fn": right(rest),
        "n_tn": len(tn),
        "other_right_on_tn": right(tn),
    }


def robustness(comparison: dict, pairs: tuple[tuple[str, str], ...]) -> list[dict]:
    """Verdict of each pair under the primary seed rule and every sensitivity analysis."""
    verdicts = {(v["a"], v["b"]): v for v in comparison["verdicts"]["pairs"]}
    ttests = {(t["a"], t["b"]): t for t in comparison["ttests"]}
    approaches = comparison["approaches"]
    alpha = comparison["verdicts"]["alpha"]

    def node(name: str) -> dict:
        return approaches[name]  # snapshot approaches are keyed "gan/final"

    def winner(a: str, b: str, direction: str) -> str:
        return {"a>b": a, "b>a": b}.get(direction, direction)

    rows = []
    for x, y in pairs:
        key = (x, y) if (x, y) in verdicts else (y, x)
        a, b = key
        v, t = verdicts[key], ttests[key]
        f1_a = node(a)["macro"]["f1"]["values"]
        f1_b = node(b)["macro"]["f1"]["values"]
        mcc_a = node(a)["secondary"]["macro"]["mcc"]["mean"]
        mcc_b = node(b)["secondary"]["macro"]["mcc"]["mean"]
        sens = v["sensitivity"]
        rows.append(
            {
                "a": a,
                "b": b,
                "display": [x, y],
                "primary": {
                    "p_max": v["p_max"],
                    "direction": winner(a, b, v["direction"]),
                    "significant": v["significant"],
                    "p_holm": v.get("p_holm"),
                    "holm_significant": v.get("p_holm") is not None and v["p_holm"] < alpha,
                },
                "wrong": {
                    "p_max": sens["p_max"],
                    "significant": sens["significant"],
                    "p_holm": sens.get("p_holm"),
                },
                "ttest": {
                    "p": t["p"],
                    "significant": t["p"] < alpha,
                    "direction": a if t["mean_diff"] > 0 else b,
                },
                "mcc_order": a if mcc_a > mcc_b else b,
                "mcc": {a: mcc_a, b: mcc_b},
                "f1_seeds_favouring": {
                    a: sum(fa > fb for fa, fb in zip(f1_a, f1_b, strict=True)),
                    b: sum(fb > fa for fa, fb in zip(f1_a, f1_b, strict=True)),
                },
            }
        )
    return rows


def control_row(ref: list[dict], run: list[dict]) -> dict:
    """One validation control against its reference run: metrics, flips, McNemar (drop)."""
    prof = flip_profile(ref, run)
    return {
        "metrics": _macro(run),
        "reference": _macro(ref),
        "changed_share": prof["share"],
        "delta": prof["delta"],
        "mcnemar_p": mcnemar(ref, run, policy=POLICY)["p"],
    }


def _preds(approach: str, seed: int, snapshot: str | None = None) -> list[dict]:
    """Load the predictions of an approach and seed, optionally for a snapshot."""
    return ch7.load_run(ch7._seed_dir(approach, seed, snapshot))["preds"]


def _single(run: str) -> list[dict]:
    """Load the seed-42 predictions of a single named run."""
    return ch7.load_run(RUNS / run / "seed42")["preds"]


def build() -> dict:
    """Compute all Chapter 8 diagnostics and return them as one dict."""
    out: dict = {"generated": ch7.dt.datetime.now().isoformat(timespec="seconds")}
    comparison = json.loads((OUT / "comparison_test.json").read_text(encoding="utf-8"))
    out["robustness"] = {
        "rq1": robustness(comparison, RQ1_PAIRS),
        "rq2a": robustness(comparison, RQ2_PAIRS),
        "rq3": robustness(comparison, RQ3_PAIRS),
    }

    runs = {
        name: {s: _preds(name, s) for s in SEEDS}
        for name in ("baseline", "rag", "sft", "dual_agent")
    }
    runs["gan/final"] = {s: _preds("gan", s, "final") for s in SEEDS}

    # noise floor on test: the same approach under another seed (sampling stream)
    out["flips"] = {
        "baseline_between_seeds": {
            f"{a}-{b}": flip_profile(runs["baseline"][a], runs["baseline"][b])
            for a, b in ((42, 43), (42, 44), (43, 44))
        },
        "gan_round0_to_final": {
            str(s): flip_profile(runs["baseline"][s], runs["gan/final"][s]) for s in SEEDS
        },
    }
    # the Dual-Agent Validator against its own Detector
    parse = get_parser("d09-v2")
    out["dual_agent_agreement"] = {
        str(s): agreement(
            ch7.read_jsonl(ch7._seed_dir("dual_agent", s) / "agents.jsonl"),
            runs["dual_agent"][s],
            parse,
        )
        for s in SEEDS
    }
    out["flips"]["detector_to_validator"] = {
        str(s): flip_profile(runs["baseline"][s], runs["dual_agent"][s]) for s in SEEDS
    }

    records = ch7.read_jsonl(Path("data/processed/bigvul/test.jsonl"))
    out["identity_shortcut"] = identity_shortcut(
        records, {name: list(per_seed.values()) for name, per_seed in runs.items()}
    )
    out["fix_described_fn"] = {
        name: {str(s): fix_described_fn(runs["baseline"][s], runs[name][s]) for s in SEEDS}
        for name in ("sft", "rag", "dual_agent", "gan/final")
    }

    out["val_controls"] = {}
    for key, ref, run, what in VAL_CONTROLS:
        row = control_row(_single(ref), _single(run))
        out["val_controls"][key] = {"reference_run": ref, "control_run": run, "varies": what, **row}
    return out


NAMES = {
    "baseline": "Baseline",
    "rag": "RAG",
    "sft": "SFT",
    "dual_agent": "Dual-Agent",
    "gan/final": "GAN final",
    "triple_agent/final": "Triple-Agent",
}


def main() -> None:
    """Parse the CLI arguments, build the diagnostics and write them as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT / "ch8_diagnostics.json"))
    args = ap.parse_args()
    result = build()
    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Post-hoc analysis of finished runs split by whether func_before == func_after."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from vulndetect.data import load_split
from vulndetect.eval.aggregate import load_run, load_scored_predictions

TRIVIAL = (
    ("always Vulnerable", 0.667, 0.500, 0.500, 0.000, 0.000, 1.000),
    ("func_before==after", 1.000, 1.000, 1.000, 1.000, 1.000, 1.000),
)


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews correlation coefficient."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else 0.0


def score(rows: list[dict], changed: dict[str, bool]) -> dict:
    tp = tn = fp = fn = 0
    hits = {True: [0, 0], False: [0, 0]}  # code changed -> [correct, total]
    for row in rows:
        pred = row["pred_label"]
        if pred is None:
            continue
        gold = row["gold_label"]
        if pred == "Vulnerable":
            tp, fp = (tp + 1, fp) if gold == "Vulnerable" else (tp, fp + 1)
        else:
            fn, tn = (fn + 1, tn) if gold == "Vulnerable" else (fn, tn + 1)
        bucket = hits[changed[row["id"]]]
        bucket[1] += 1
        bucket[0] += pred == gold
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "acc": (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0,
        "bal_acc": (rec + tnr) / 2,
        "mcc": mcc(tp, tn, fp, fn),
        "acc_unchanged": hits[False][0] / hits[False][1] if hits[False][1] else float("nan"),
        "acc_changed": hits[True][0] / hits[True][1] if hits[True][1] else float("nan"),
        "pred_positive_rate": (tp + fp) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0,
        "n_unparseable": sum(r["pred_label"] is None for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--approaches", default="baseline,rag,dual_agent,sft")
    ap.add_argument("--runs-root", default="results/predictions")
    ap.add_argument("--dataset-dir", default="data/processed/bigvul")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = load_split(args.split, args.dataset_dir)
    changed = {r["id"]: r["func_before"] != r["func_after"] for r in recs}
    n_changed = sum(changed.values())
    print(f"{args.split}: n={len(recs)}, code changed in {n_changed} ({n_changed / len(recs):.1%})")

    agree = sum((r["func_before"] != r["func_after"]) == (r["label"] == "Vulnerable") for r in recs)
    print(
        f"shortcut 'code changed <-> Vulnerable' agrees with gold: {agree}/{len(recs)} "
        f"({agree / len(recs):.1%})\n"
    )

    header = f"{'approach':<18}{'F1':>7}{'acc':>7}{'bal.acc':>9}{'MCC':>7}{'pred+':>8}"
    header += f"{'acc|unchanged':>15}{'acc|changed':>13}{'skill':>8}"
    print(header)
    print("-" * len(header))
    suffix = "" if args.split == "test" else f"_{args.split}"
    for name in args.approaches.split(","):
        path = Path(args.runs_root) / f"{name}{suffix}" / f"seed{args.seed}" / "predictions.jsonl"
        if not path.exists():
            print(f"{name:<18}  predictions.jsonl missing ({path})")
            continue
        run = load_run(path.parent)
        s = score(load_scored_predictions(run), changed)
        # skill: accuracy on changed instances above a null model with the same positive rate
        skill = s["acc_changed"] - s["pred_positive_rate"]
        print(
            f"{name:<18}{s['f1']:>7.3f}{s['acc']:>7.3f}{s['bal_acc']:>9.3f}{s['mcc']:>7.3f}"
            f"{s['pred_positive_rate']:>8.1%}{s['acc_unchanged']:>15.3f}{s['acc_changed']:>13.3f}"
            f"{skill:>+8.3f}   [parser {run['parser_version']}]"
        )
    print("-" * len(header))
    for label, f1, acc, bal, m, unch, ch in TRIVIAL:
        print(
            f"{label:<18}{f1:>7.3f}{acc:>7.3f}{bal:>9.3f}{m:>7.3f}{'':>8}{unch:>15.3f}{ch:>13.3f}"
            f"{'':>8}"
        )


if __name__ == "__main__":
    main()

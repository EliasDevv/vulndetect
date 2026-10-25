"""Post-hoc analysis of what the models write and whether it tracks the gold label."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from vulndetect.eval.aggregate import load_run, load_scored_predictions

CVE = re.compile(r"CVE-\d{4}-\d+", re.I)
FIXES = re.compile(
    r"\b(?:fix(?:es|ed)?|mitigat(?:es|ed)|address(?:es|ed)|patch(?:es|ed)|resolv(?:es|ed))\b"
    r"[^.]{0,60}?\b(?:vulnerabilit|security|CVE|issue|flaw)",
    re.I,
)
RELATED = re.compile(r"\brelated to\b[^.]{0,60}?\b(?:vulnerabilit|CVE|security)", re.I)
BARE_LABEL = re.compile(r"^\s*Classification:\s*(?:Not\s+)?Vulnerable\s*$", re.I)


def share(rows: list[dict], rx: re.Pattern) -> float:
    return sum(bool(rx.search(r["raw_output"] or "")) for r in rows) / len(rows) if rows else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--approaches", default="baseline,rag,dual_agent,sft")
    ap.add_argument("--runs-root", default="results/predictions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    suffix = "" if args.split == "test" else f"_{args.split}"
    print(
        f"{'approach':<14}{'median len':>11}{'bare label':>12}   "
        f"{'FN: CVE':>9}{'fixes':>7}{'related':>9}   {'TN: CVE':>9}{'fixes':>7}   {'delta':>7}"
    )
    print("-" * 100)
    for name in args.approaches.split(","):
        path = Path(args.runs_root) / f"{name}{suffix}" / f"seed{args.seed}" / "predictions.jsonl"
        if not path.exists():
            print(f"{name:<14}  predictions.jsonl missing ({path})")
            continue
        rows = load_scored_predictions(load_run(path.parent))
        lens = sorted(len(r["raw_output"] or "") for r in rows)
        bare = sum(bool(BARE_LABEL.match((r["raw_output"] or "").strip())) for r in rows) / len(
            rows
        )
        fn = [
            r
            for r in rows
            if r["gold_label"] == "Vulnerable" and r["pred_label"] == "Not Vulnerable"
        ]
        tn = [
            r
            for r in rows
            if r["gold_label"] == "Not Vulnerable" and r["pred_label"] == "Not Vulnerable"
        ]
        # delta: share of "fixes" language in false negatives minus true negatives
        delta = share(fn, FIXES) - share(tn, FIXES)
        print(
            f"{name:<14}{lens[len(lens) // 2]:>11}{bare:>11.1%}   "
            f"{share(fn, CVE):>9.1%}{share(fn, FIXES):>7.1%}{share(fn, RELATED):>9.1%}   "
            f"{share(tn, CVE):>9.1%}{share(tn, FIXES):>7.1%}   {delta:>+7.1%}"
        )
    print("\nFN = gold Vulnerable, predicted Not Vulnerable.  TN = both Not Vulnerable.")
    print("delta near 0 => the reasoning text does not distinguish the two classes.")


if __name__ == "__main__":
    main()

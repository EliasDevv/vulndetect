"""CLI: score a predictions.jsonl and print the per-CWE report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vulndetect.common.parsing import PARSERS, get_parser
from vulndetect.common.predictions import read_predictions

from .metrics import DEFAULT_POLICY, POLICIES, evaluate, format_report


def main() -> None:
    """Parse CLI arguments, score a predictions file and print the report."""
    ap = argparse.ArgumentParser(description="Score a canonical predictions file.")
    ap.add_argument("predictions", help="path to a predictions.jsonl")
    ap.add_argument("--out", help="also write the full metrics report as JSON here")
    ap.add_argument("--policy", choices=POLICIES, default=DEFAULT_POLICY)
    ap.add_argument("--parser", choices=sorted(PARSERS), help="re-parse raw_output before scoring")
    args = ap.parse_args()

    preds = read_predictions(args.predictions)
    if args.parser:
        parse = get_parser(args.parser)
        preds = [{**p, "pred_label": parse(p["raw_output"])} for p in preds]

    report = evaluate(preds, policy=args.policy)
    if args.parser:
        report["parser_version"] = args.parser
    print(format_report(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

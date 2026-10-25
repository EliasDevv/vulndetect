"""Draw a seeded sample of accepted generations and write inspection files for a hand check.

Usage (from code/):
    uv run python scripts/sample_generated.py --run gan_val --seed 42
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", default="gan_val", help="results/<run>/seed<N>/")
    ap.add_argument("--seed", type=int, default=42, help="loop seed (directory)")
    ap.add_argument("--n", type=int, default=30, help="sample size")
    ap.add_argument("--sample-seed", type=int, default=42, help="RNG seed of the draw")
    ap.add_argument("--train", default="data/processed/bigvul/train.jsonl")
    ap.add_argument(
        "--overwrite-scores",
        action="store_true",
        help="also rewrite an existing inspection_scores.csv",
    )
    args = ap.parse_args()

    run_dir = Path("results") / args.run / f"seed{args.seed}"
    rows = _read_jsonl(run_dir / "generated.jsonl")
    accepted = [r for r in rows if r.get("accepted")]
    if len(accepted) < args.n:
        raise SystemExit(f"only {len(accepted)} accepted items, need {args.n}")

    train = {r["id"]: r for r in _read_jsonl(Path(args.train))}

    rng = random.Random(args.sample_seed)
    sample = sorted(rng.sample(accepted, args.n), key=lambda r: r["round"])

    out_jsonl = run_dir / "inspection_sample.jsonl"
    out_md = run_dir / "inspection_sheet.md"
    out_csv = run_dir / "inspection_scores.csv"

    md = [
        f"# Hand check — {args.run}/seed{args.seed}",
        "",
        f"Sample: {args.n} of {len(accepted)} accepted items (of {len(rows)} generations), "
        f"`random.Random({args.sample_seed}).sample`, sorted by round.",
        "",
        "Verdicts go into `inspection_scores.csv`: intended | other | broken | none.",
        "",
    ]
    with out_jsonl.open("w") as fh:
        for k, r in enumerate(sample, 1):
            src = train[r["source_id"]]
            assert src["label"] == "Not Vulnerable", r["source_id"]
            clean = src["code"]
            r = dict(r, source_function=clean, project=src["project"], cve_id=src["cve_id"])
            fh.write(json.dumps(r) + "\n")
            diff = difflib.unified_diff(
                clean.splitlines(),
                r["function"].splitlines(),
                fromfile="clean (source)",
                tofile="generated",
                lineterm="",
                n=3,
            )
            md += [
                f"## {k:02d} — round {r['round']}, {r['cwe']}, {src['project']}",
                "",
                f"- item_id: `{r['item_id']}`",
                f"- source CVE (inherited): {src['cve_id']}",
                f"- detector said: **{r['detector_label']}** (correct={r['correct']})",
                f"- commit message: {r['commit_message']!r}",
                "",
                "**Deception notes (generator):**",
                "",
                "> " + (r["deception_notes"] or "").replace("\n", "\n> "),
                "",
                "```diff",
                *diff,
                "```",
                "",
            ]
    out_md.write_text("\n".join(md))

    if out_csv.exists() and not args.overwrite_scores:
        print(f"kept existing {out_csv}")
    else:
        with out_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["k", "round", "item_id", "cwe", "detector_label", "verdict", "note"])
            for k, r in enumerate(sample, 1):
                w.writerow([k, r["round"], r["item_id"], r["cwe"], r["detector_label"], "", ""])
        print(f"wrote {out_csv}")
    print(f"wrote {out_jsonl}\nwrote {out_md}")


if __name__ == "__main__":
    main()

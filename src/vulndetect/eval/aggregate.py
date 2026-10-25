"""Aggregate finished runs over seeds into the paper-style comparison with significance tests."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
from itertools import combinations
from pathlib import Path

from vulndetect.common.parsing import get_parser
from vulndetect.common.predictions import read_predictions

from .metrics import DEFAULT_POLICY, METRIC_NAMES, POLICIES, SECONDARY_METRIC_NAMES
from .stats import ALPHA, holm, mcnemar, paired_ttest

logger = logging.getLogger(__name__)

DEFAULT_RUNS_ROOT = Path("results/predictions")
DEFAULT_OUT_DIR = Path("results/metrics")
DEFAULT_APPROACHES = ("baseline", "rag", "sft", "dual_agent")

_SEED_DIR = re.compile(r"^seed(\d+)$")

_PROVENANCE_KEYS = ("model", "model_revision", "prompt_version", "decoding")


class AggregationError(ValueError):
    """Raised when runs are missing or must not be aggregated together."""


def _effective_scoring(manifest: dict, metrics: dict) -> dict:
    """Return the parser and policy the numbers in metrics.json were produced with."""
    parser = metrics.get("parser_version") or manifest.get("parser_version")
    policy = (
        metrics.get("unparseable_policy") or manifest.get("unparseable_policy") or DEFAULT_POLICY
    )
    out = {"parser_version": parser, "unparseable_policy": policy}
    manifest_parser = manifest.get("parser_version")
    if manifest_parser is not None and manifest_parser != parser:
        out["manifest_parser_version"] = manifest_parser
    return out


def load_run(run_dir: str | Path) -> dict:
    """Load one run directory into a dict with seed, manifest, metrics and scoring info."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    if not manifest_path.exists() or not metrics_path.exists():
        raise AggregationError(f"{run_dir}: manifest.json/metrics.json missing — broken run?")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "seed": int(manifest["seed"]),
        "dir": run_dir,
        "manifest": manifest,
        "metrics": metrics,
        **_effective_scoring(manifest, metrics),
    }


def load_scored_predictions(run: dict) -> list[dict]:
    """Load a run's predictions.jsonl, re-labelled with the run's effective parser."""
    parse = get_parser(run["parser_version"])
    preds = read_predictions(Path(run["dir"]) / "predictions.jsonl")
    return [{**p, "pred_label": parse(p["raw_output"])} for p in preds]


def split_approach(approach: str) -> tuple[str, str | None]:
    """``"gan/final"`` -> ``("gan", "final")``; ``"baseline"`` -> ``("baseline", None)``."""
    name, sep, snapshot = approach.partition("/")
    if not name or (sep and (not snapshot or "/" in snapshot)):
        raise AggregationError(
            f"bad approach {approach!r}: expected <approach> or <approach>/<snapshot>"
        )
    return name, snapshot or None


def discover_runs(runs_root: str | Path, approach: str) -> list[dict]:
    """Find all seed runs of one approach (or approach snapshot)."""
    name, snapshot = split_approach(approach)
    approach_dir = Path(runs_root) / name
    if not approach_dir.is_dir():
        raise AggregationError(f"no runs for {approach!r}: {approach_dir} does not exist")
    runs = []
    for seed_dir in sorted(approach_dir.iterdir()):
        m = _SEED_DIR.match(seed_dir.name)
        if not m:
            continue
        run_dir = seed_dir / snapshot if snapshot else seed_dir
        if snapshot and not run_dir.is_dir():
            logger.warning("%s: %s has no snapshot %r — seed skipped", approach, seed_dir, snapshot)
            continue
        run = load_run(run_dir)
        if run["seed"] != int(m.group(1)):
            raise AggregationError(
                f"{run_dir}: manifest seed {run['seed']} != directory seed {m.group(1)}"
            )
        recorded = (run["manifest"].get("gan") or {}).get("snapshot")
        if snapshot and recorded is not None and recorded != snapshot:
            raise AggregationError(
                f"{run_dir}: manifest names snapshot {recorded!r}, directory says {snapshot!r}"
            )
        runs.append(run)
    if not runs:
        where = f"seed*/{snapshot}/" if snapshot else "seed*/"
        raise AggregationError(f"no {where} runs under {approach_dir}")
    return runs


def _provenance(run: dict) -> dict:
    manifest = run["manifest"]
    prov = {k: manifest.get(k) for k in _PROVENANCE_KEYS}
    prov["parser_version"] = run["parser_version"]
    prov["unparseable_policy"] = run["unparseable_policy"]
    if "manifest_parser_version" in run:
        prov["manifest_parser_version"] = run["manifest_parser_version"]
    ds = manifest.get("dataset", {})
    prov["split"] = ds.get("split")
    prov["n"] = ds.get("n")
    prov["limit"] = ds.get("limit")
    return prov


def check_consistency(approach: str, runs: list[dict], split: str | None) -> dict:
    """Check that all seeds of an approach share one provenance and none are limited runs."""
    reference = _provenance(runs[0])
    for run in runs[1:]:
        prov = _provenance(run)
        if prov != reference:
            diffs = {k: (reference[k], prov[k]) for k in reference if reference[k] != prov[k]}
            raise AggregationError(
                f"{approach}: seed {run['seed']} provenance differs from seed "
                f"{runs[0]['seed']}: {diffs} — do not average mixed configurations"
            )
    if reference["limit"] is not None:
        raise AggregationError(f"{approach}: refusing to aggregate --limit (smoke) runs")
    if split is not None and reference["split"] != split:
        raise AggregationError(
            f"{approach}: runs are on split {reference['split']!r}, expected {split!r}"
        )
    return reference


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "values": list(values),
    }


def _aggregate_rows(rows: list[dict]) -> dict:
    """mean±std of the four metric names over one row (per CWE, macro, micro)."""
    return {m: _mean_std([row[m] for row in rows]) for m in METRIC_NAMES}


def aggregate_approach(runs: list[dict]) -> dict:
    """Aggregate one approach's metrics reports over seeds."""
    reports = [run["metrics"] for run in runs]
    cwes = sorted(reports[0]["per_cwe"])
    for run, report in zip(runs, reports, strict=True):
        if sorted(report["per_cwe"]) != cwes:
            raise AggregationError(
                f"seed {run['seed']}: per-CWE rows {sorted(report['per_cwe'])} != {cwes}"
            )
    agg = {
        "seeds": [run["seed"] for run in runs],
        "n": reports[0]["n"],
        "unparseable_policy": reports[0].get("unparseable_policy", "wrong"),
        "parse_failure_rate": _mean_std([r["parse_failure_rate"] for r in reports]),
        "per_cwe": {
            cwe: {
                "n": reports[0]["per_cwe"][cwe]["n"],
                **_aggregate_rows([r["per_cwe"][cwe] for r in reports]),
            }
            for cwe in cwes
        },
        "macro": _aggregate_rows([r["macro"] for r in reports]),
        "micro": _aggregate_rows([r["micro"] for r in reports]),
    }
    for block in ("macro", "micro"):
        if all(all(m in r[block] for m in SECONDARY_METRIC_NAMES) for r in reports):
            agg.setdefault("secondary", {})[block] = {
                m: _mean_std([r[block][m] for r in reports]) for m in SECONDARY_METRIC_NAMES
            }
    alts = [r.get("alt") for r in reports]
    if all(alt is not None for alt in alts):
        agg["alt"] = {
            "policy": alts[0]["policy"],
            "macro": _aggregate_rows([alt["macro"] for alt in alts]),
            "micro": _aggregate_rows([alt["micro"] for alt in alts]),
        }
    return agg


def pairwise_ttests(aggregates: dict[str, dict]) -> list[dict]:
    """Paired t-test over per-CWE F1 means for every approach pair."""
    results = []
    for a, b in combinations(sorted(aggregates), 2):
        cwes_a, cwes_b = sorted(aggregates[a]["per_cwe"]), sorted(aggregates[b]["per_cwe"])
        if cwes_a != cwes_b:
            raise AggregationError(f"{a} vs {b}: per-CWE rows differ ({cwes_a} vs {cwes_b})")
        f1s_a = [aggregates[a]["per_cwe"][c]["f1"]["mean"] for c in cwes_a]
        f1s_b = [aggregates[b]["per_cwe"][c]["f1"]["mean"] for c in cwes_a]
        results.append({"a": a, "b": b, "cwes": cwes_a, **paired_ttest(f1s_a, f1s_b)})
    return results


def pairwise_mcnemar(runs_by_approach: dict[str, list[dict]]) -> list[dict]:
    """Exact McNemar per seed-matched run pair; pairs without predictions are skipped."""
    results = []
    for a, b in combinations(sorted(runs_by_approach), 2):
        runs_a = {run["seed"]: run for run in runs_by_approach[a]}
        runs_b = {run["seed"]: run for run in runs_by_approach[b]}
        for seed in sorted(set(runs_a) & set(runs_b)):
            run_a, run_b = runs_a[seed], runs_b[seed]
            path_a = Path(run_a["dir"]) / "predictions.jsonl"
            path_b = Path(run_b["dir"]) / "predictions.jsonl"
            if not path_a.exists() or not path_b.exists():
                results.append(
                    {
                        "a": a,
                        "b": b,
                        "seed": seed,
                        "skipped": "predictions.jsonl missing (rsync from server)",
                    }
                )
                continue
            scoring = {(run["parser_version"], run["unparseable_policy"]) for run in (run_a, run_b)}
            if len(scoring) > 1:
                raise AggregationError(
                    f"{a} vs {b} seed {seed}: parser/policy differ {sorted(scoring)} — "
                    "McNemar must score both runs alike"
                )
            preds_a, preds_b = load_scored_predictions(run_a), load_scored_predictions(run_b)
            primary = run_a["unparseable_policy"]
            other = next(pol for pol in POLICIES if pol != primary)
            stat = mcnemar(preds_a, preds_b, policy=primary)
            alt = mcnemar(preds_a, preds_b, policy=other)
            results.append(
                {
                    "a": a,
                    "b": b,
                    "seed": seed,
                    "parser_version": run_a["parser_version"],
                    "policy": stat["policy"],
                    "n": stat["n"],
                    "n_excluded": stat["n_excluded"],
                    "n_a_correct_b_wrong": stat["b"],
                    "n_a_wrong_b_correct": stat["c"],
                    "p": stat["p"],
                    "direction": _direction(stat["b"], stat["c"]),
                    "sensitivity": {
                        "policy": alt["policy"],
                        "n": alt["n"],
                        "n_a_correct_b_wrong": alt["b"],
                        "n_a_wrong_b_correct": alt["c"],
                        "p": alt["p"],
                        "direction": _direction(alt["b"], alt["c"]),
                    },
                }
            )
    return results


def _direction(n_a_correct_b_wrong: int, n_a_wrong_b_correct: int) -> str:
    """Which side the discordant pairs favour: "a>b", "b>a" or "tie"."""
    if n_a_correct_b_wrong > n_a_wrong_b_correct:
        return "a>b"
    if n_a_wrong_b_correct > n_a_correct_b_wrong:
        return "b>a"
    return "tie"


def parse_families(specs: list[str] | None) -> list[list[tuple[str, str]]]:
    """``["x:y,x:z", "u:v"]`` -> ``[[("x","y"),("x","z")], [("u","v")]]``."""
    families = []
    for spec in specs or []:
        pairs = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if item.count(":") != 1:
                raise AggregationError(f"family pair must be 'a:b', got {item!r}")
            a, b = (part.strip() for part in item.split(":"))
            pairs.append((a, b))
        if pairs:
            families.append(pairs)
    return families


def seed_verdicts(
    mcnemar_rows: list[dict], families: list[list[tuple[str, str]]] | None = None
) -> dict:
    """Combine per-seed McNemar results per approach pair, with Holm adjustment per family."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in mcnemar_rows:
        if "skipped" in row:
            continue
        grouped.setdefault((row["a"], row["b"]), []).append(row)

    def verdict(rows: list[dict], key: str | None) -> dict:
        picked = [row[key] if key else row for row in rows]
        p_max = max(r["p"] for r in picked)
        directions = {r["direction"] for r in picked}
        direction = directions.pop() if len(directions) == 1 else "mixed"
        return {
            "p_max": p_max,
            "direction": direction,
            "significant": p_max < ALPHA and direction in ("a>b", "b>a"),
        }

    verdicts = {}
    for (a, b), rows in sorted(grouped.items()):
        entry = {
            "a": a,
            "b": b,
            "seeds": [row["seed"] for row in rows],
            "policy": rows[0]["policy"],
            "p_by_seed": {row["seed"]: row["p"] for row in rows},
            **verdict(rows, None),
            "sensitivity": {
                "policy": rows[0]["sensitivity"]["policy"],
                "p_by_seed": {row["seed"]: row["sensitivity"]["p"] for row in rows},
                **verdict(rows, "sensitivity"),
            },
        }
        verdicts[(a, b)] = entry

    def lookup(a: str, b: str) -> dict:
        if (a, b) in verdicts:
            return verdicts[(a, b)]
        if (b, a) in verdicts:
            return verdicts[(b, a)]
        raise AggregationError(f"family pair {a}:{b} has no scored McNemar rows")

    family_out = []
    for idx, pairs in enumerate(families or []):
        members = [lookup(a, b) for a, b in pairs]
        adjusted = holm([m["p_max"] for m in members])
        adjusted_alt = holm([m["sensitivity"]["p_max"] for m in members])
        for member, p_adj, p_adj_alt in zip(members, adjusted, adjusted_alt, strict=True):
            member["family"] = idx
            member["p_holm"] = p_adj
            member["sensitivity"]["p_holm"] = p_adj_alt
        family_out.append(
            {"index": idx, "pairs": [f"{a}:{b}" for a, b in pairs], "size": len(pairs)}
        )
    return {"alpha": ALPHA, "pairs": list(verdicts.values()), "families": family_out}


def load_reference(path: str | Path) -> dict[str, dict[str, float]]:
    """Paper reference numbers from a hand-filled CSV: approach,metric,value."""
    reference: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            reference.setdefault(row["approach"], {})[row["metric"]] = float(row["value"])
    return reference


def _fmt(cell: dict) -> str:
    if cell["std"] is None:
        return f"{cell['mean']:.3f}"
    return f"{cell['mean']:.3f}±{cell['std']:.3f}"


def render_markdown(comparison: dict, reference: dict[str, dict[str, float]] | None = None) -> str:
    """Render the comparison as Markdown tables, optionally with paper reference rows."""
    lines = [
        f"# Comparison — split: {comparison['split']}",
        "",
        f"Parser: {comparison.get('parser_version')}. Primary unparseable policy: "
        f"{comparison['policy']}; the alt row is the other policy's macro. "
        "Values are mean±std over seeds.",
        "",
        "## All (Avg) overview",
        "",
        "| approach | seeds | " + " | ".join(METRIC_NAMES) + " | f1 (alt) | parse-fail |",
        "|---" * (len(METRIC_NAMES) + 4) + "|",
    ]
    for name, agg in comparison["approaches"].items():
        alt = _fmt(agg["alt"]["macro"]["f1"]) if "alt" in agg else "—"
        lines.append(
            f"| {name} | {','.join(str(s) for s in agg['seeds'])} | "
            + " | ".join(_fmt(agg["macro"][m]) for m in METRIC_NAMES)
            + f" | {alt} | {_fmt(agg['parse_failure_rate'])} |"
        )
    if reference:
        for name, metrics in reference.items():
            cells = [f"{metrics[m]:.3f}" if m in metrics else "—" for m in METRIC_NAMES]
            lines.append(f"| {name} (paper) | — | " + " | ".join(cells) + " | — | — |")
    for name, agg in comparison["approaches"].items():
        lines += [
            "",
            f"## {name} — per CWE",
            "",
            "| CWE | n | " + " | ".join(METRIC_NAMES) + " |",
            "|---" * (len(METRIC_NAMES) + 2) + "|",
        ]
        for cwe, row in agg["per_cwe"].items():
            lines.append(
                f"| {cwe} | {row['n']} | " + " | ".join(_fmt(row[m]) for m in METRIC_NAMES) + " |"
            )
        lines.append(
            f"| All (macro) | {agg['n']} | "
            + " | ".join(_fmt(agg["macro"][m]) for m in METRIC_NAMES)
            + " |"
        )
        lines.append(
            f"| All (micro) | {agg['n']} | "
            + " | ".join(_fmt(agg["micro"][m]) for m in METRIC_NAMES)
            + " |"
        )
    lines += [
        "",
        "## Paired t-tests (per-CWE F1 means, paper's test; df = #CWEs - 1)",
        "",
        "| a | b | mean diff (a-b) | t | p | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for t in comparison["ttests"]:
        lines.append(
            f"| {t['a']} | {t['b']} | {t['mean_diff']:+.3f} | {t['t']:.3f} | {t['p']:.4f} "
            f"| [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] |"
        )
    lines += [
        "",
        "## McNemar (instance-level, per seed-matched pair; re-parsed with the parser "
        "above, unparseable pairs excluded under drop)",
        "",
    ]
    lines += [
        "| a | b | seed | n | excluded | b(a✓b✗) | c(a✗b✓) | p | dir | p (sensitivity) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in comparison["mcnemar"]:
        if "skipped" in m:
            lines.append(
                f"| {m['a']} | {m['b']} | {m['seed']} | — | — | — | — | {m['skipped']} | — | — |"
            )
        else:
            sens = m["sensitivity"]
            lines.append(
                f"| {m['a']} | {m['b']} | {m['seed']} | {m['n']} | {m['n_excluded']} "
                f"| {m['n_a_correct_b_wrong']} | {m['n_a_wrong_b_correct']} "
                f"| {m['p']:.4f} | {m['direction']} | {sens['p']:.4f} ({sens['policy']}) |"
            )
    verdicts = comparison.get("verdicts")
    if verdicts:
        lines += [
            "",
            f"## Seed rule (Section 6.6): p_max over seeds, direction in every seed, "
            f"alpha = {verdicts['alpha']}; p_holm within named families",
            "",
            "| a | b | seeds | direction | p_max | p_holm | significant | "
            "p_max (sensitivity) | significant (sensitivity) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for v in verdicts["pairs"]:
            p_holm = f"{v['p_holm']:.4f} (family {v['family']})" if "p_holm" in v else "—"
            sens = v["sensitivity"]
            lines.append(
                f"| {v['a']} | {v['b']} | {','.join(str(s) for s in v['seeds'])} "
                f"| {v['direction']} | {v['p_max']:.4f} | {p_holm} | {v['significant']} "
                f"| {sens['p_max']:.4f} ({sens['policy']}) | {sens['significant']} |"
            )
    return "\n".join(lines) + "\n"


def render_latex(comparison: dict) -> str:
    """Render a booktabs fragment of the All (Avg) overview."""
    lines = [
        "% Auto-generated by `make eval` (vulndetect.eval.aggregate) — do not edit.",
        f"% split: {comparison['split']}, parser: {comparison.get('parser_version')}, "
        f"policy: {comparison['policy']}, values: mean±std over seeds",
        r"\begin{tabular}{l" + "c" * len(METRIC_NAMES) + "}",
        r"\toprule",
        "Approach & " + " & ".join(m.capitalize() for m in METRIC_NAMES) + r" \\",
        r"\midrule",
    ]
    for name, agg in comparison["approaches"].items():
        cells = []
        for m in METRIC_NAMES:
            cell = agg["macro"][m]
            if cell["std"] is None:
                cells.append(f"{cell['mean']:.3f}")
            else:
                cells.append(f"{cell['mean']:.3f}\\,$\\pm$\\,{cell['std']:.3f}")
        lines.append(f"{name.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def aggregate(
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    approaches: tuple[str, ...] = DEFAULT_APPROACHES,
    split: str = "test",
    families: list[list[tuple[str, str]]] | None = None,
) -> dict:
    """Collect, check and aggregate all requested approaches into one dict."""
    runs_by_approach: dict[str, list[dict]] = {}
    provenance: dict[str, dict] = {}
    for approach in approaches:
        runs = discover_runs(runs_root, approach)
        provenance[approach] = check_consistency(approach, runs, split)
        runs_by_approach[approach] = runs

    sizes = {name: prov["n"] for name, prov in provenance.items()}
    if len(set(sizes.values())) > 1:
        raise AggregationError(
            f"approaches ran on different dataset sizes: {sizes} — not comparable"
        )

    aggregates = {name: aggregate_approach(runs) for name, runs in runs_by_approach.items()}
    policies = {agg["unparseable_policy"] for agg in aggregates.values()}
    if len(policies) > 1:
        raise AggregationError(f"mixed primary unparseable policies {policies} — not comparable")
    parsers = {prov["parser_version"] for prov in provenance.values()}
    if len(parsers) > 1:
        raise AggregationError(f"mixed parsers {sorted(map(str, parsers))} — not comparable")
    mcnemar_rows = pairwise_mcnemar(runs_by_approach) if len(runs_by_approach) > 1 else []

    return {
        "split": split,
        "n": next(iter(sizes.values())),
        "policy": next(iter(policies)),
        "parser_version": next(iter(parsers)),
        "provenance": provenance,
        "approaches": aggregates,
        "ttests": pairwise_ttests(aggregates) if len(aggregates) > 1 else [],
        "mcnemar": mcnemar_rows,
        "verdicts": seed_verdicts(mcnemar_rows, families) if mcnemar_rows else None,
    }


def write_outputs(
    comparison: dict,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    reference: dict[str, dict[str, float]] | None = None,
) -> dict[str, Path]:
    """Write the comparison as JSON, Markdown and LaTeX and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"comparison_{comparison['split']}"
    paths = {
        "json": out_dir / f"{stem}.json",
        "md": out_dir / f"{stem}.md",
        "tex": out_dir / f"{stem}_table.tex",
    }
    paths["json"].write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["md"].write_text(render_markdown(comparison, reference), encoding="utf-8")
    paths["tex"].write_text(render_latex(comparison), encoding="utf-8")
    return paths


def main() -> None:
    """Parse CLI arguments, aggregate the runs and write the comparison outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Aggregate official runs into the comparison.")
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument(
        "--approaches",
        default=",".join(DEFAULT_APPROACHES),
        help="comma-separated run directory names under --runs-root; snapshot "
        "approaches as <approach>/<snapshot>, e.g. gan/final",
    )
    ap.add_argument("--split", default="test", help="expected dataset split of all runs")
    ap.add_argument(
        "--reference",
        help="optional CSV (approach,metric,value) with the paper's numbers for the "
        "comparison table — numbers are never hardcoded here",
    )
    ap.add_argument(
        "--family",
        action="append",
        default=None,
        help="approach pairs whose p_max values are Holm-adjusted together, as "
        "'a:b,a:c' (repeat the flag for a second family), e.g. "
        "--family gan/final:baseline,gan/final:dual_agent (Section 6.6)",
    )
    args = ap.parse_args()

    approaches = tuple(a.strip() for a in args.approaches.split(",") if a.strip())
    comparison = aggregate(args.runs_root, approaches, args.split, parse_families(args.family))
    reference = load_reference(args.reference) if args.reference else None
    paths = write_outputs(comparison, args.out, reference)
    print(render_markdown(comparison, reference))
    for kind, path in paths.items():
        logger.info("wrote %s -> %s", kind, path)


if __name__ == "__main__":
    main()

"""Label-definition ablation on val: Fig. 1 prompt plus the authors' label-definition sentences.

Usage (from code/):
    uv run python experiments/label_definition/labeldef.py run \\
        --config experiments/label_definition/config.yaml
    uv run python experiments/label_definition/labeldef.py run \\
        --config experiments/label_definition/config.yaml --limit 4      # smoke
    uv run python experiments/label_definition/labeldef.py analyze
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from vulndetect.common import (
    config_sha256,
    get_parser,
    load_config,
    make_prediction,
    set_seed,
    write_run,
)
from vulndetect.common.model import CHAT_DATE_STRING, TextGenerator
from vulndetect.common.parsing import _VULN_CUE_RE
from vulndetect.common.predictions import read_predictions
from vulndetect.common.prompt import PROMPT_TEMPLATE, PROMPT_VERSION, fill_record_fields
from vulndetect.data import load_split
from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
from vulndetect.eval import evaluate, format_report
from vulndetect.eval.metrics import confusion, scores
from vulndetect.eval.stats import mcnemar

logger = logging.getLogger(__name__)

APPROACH = "baseline_labeldef"

# Verbatim from the authors' strict prompt (authors-strict-v1).
DEFINITION = (
    "Vulnerable if the code change is related to a known vulnerability (i.e., the change "
    "either fixes or introduces a security issue). Not Vulnerable when the code change is not "
    "associated with a vulnerability (i.e., it is a regular code update or enhancement without "
    "security implications)."
)
TEMPLATE = PROMPT_TEMPLATE + "\n" + DEFINITION
LABELDEF_PROMPT_VERSION = f"{PROMPT_VERSION}+strict-labeldef-v1"

DEFAULT_RUN = "results/predictions/baseline_labeldef_val/seed42"
REFERENCE = "baseline_val (fig1-v3)"
REFERENCES = {
    REFERENCE: "results/predictions/baseline_val/seed42",
    "baseline_val_date (noise reference)": "results/predictions/baseline_val_date/seed42",
    "baseline_val_strict (authors-strict-v1)": "results/predictions/baseline_val_strict/seed42",
}
THIS = "labeldef (this ablation)"

# Proxy for "the model says the code did not change".
UNCHANGED_RE = re.compile(
    r"\b(?:identical|unchanged|no (?:actual |significant |functional )?(?:changes|modifications))\b"
    r"|\bthe same as (?:the )?(?:original|before)\b",
    re.IGNORECASE,
)


def render(record: dict) -> str:
    """Fig. 1 (byte-identical, same N/A fallbacks) plus the authors' label definition."""
    return fill_record_fields(TEMPLATE, record)


def _token_stats(model, prompts: list[str]) -> dict | None:
    count = getattr(model, "count_tokens", None)
    if count is None or not prompts:
        return None
    n = [count(p) for p in prompts]
    return {"mean": round(sum(n) / len(n), 1), "max": max(n)}


def run(config_path: str | Path, limit: int | None = None, model: TextGenerator | None = None):
    """One val run of the ablation; returns the metrics report."""
    cfg = load_config(config_path)
    ds = cfg["dataset"]
    if ds.get("split") != "val":
        raise ValueError("the label-definition ablation runs on val only")
    if cfg.get("output", {}).get("log_experiments", True):
        raise ValueError("the label-definition ablation must set output.log_experiments: false")
    seed = int(cfg["seed"])
    set_seed(seed)

    records = load_split("val", ds["dir"])
    if limit is not None:
        records = records[:limit]
    logger.info("loaded %d val records", len(records))

    if model is None:
        from vulndetect.common.model import load_model  # lazy: pulls in torch

        model = load_model(cfg["model"])

    parser_version = cfg["parser_version"]
    parse = get_parser(parser_version)
    policy = cfg["eval"]["unparseable"]

    prompts = [render(r) for r in records]
    outputs = model.generate(prompts)
    preds = [make_prediction(r, o, parse(o)) for r, o in zip(records, outputs, strict=True)]
    report = evaluate(preds, policy=policy)
    report["parser_version"] = parser_version

    suffix = "_smoke" if limit is not None else ""
    out_dir = Path(cfg["output"]["predictions_dir"] + suffix) / f"seed{seed}"
    temperature = cfg["model"].get("temperature", 0.0)
    manifest = {
        "approach": APPROACH,
        "experiment": cfg["experiment"],
        "decision": "D85",
        "ablation_of": "baseline_val (fig1-v3, frozen by D81)",
        "model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "prompt_version": LABELDEF_PROMPT_VERSION,
        "parser_version": parser_version,
        "unparseable_policy": policy,
        "dataset": {**ds, "n": len(records), "limit": limit},
        "decoding": {
            "temperature": temperature,
            "top_p": cfg["model"].get("top_p"),
            "do_sample": temperature > 0,
            "max_new_tokens": cfg["model"].get("max_new_tokens"),
            "batch_size": cfg["model"].get("batch_size"),
            "chat_date_string": cfg["model"].get("chat_date_string", CHAT_DATE_STRING),
        },
        "logged": False,
        "prompt_tokens": {
            "labeldef": _token_stats(model, prompts),
            "fig1": _token_stats(model, [fill_record_fields(PROMPT_TEMPLATE, r) for r in records]),
        },
    }
    write_run(out_dir, preds, manifest)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(format_report(report))
    logger.info("wrote run artifacts to %s", out_dir)
    return report


def _load(run_dir: str | Path) -> dict[str, dict]:
    """Load predictions keyed by id, re-parsed with the d09-v2 parser."""
    parse = get_parser("d09-v2")
    preds = read_predictions(Path(run_dir) / "predictions.jsonl")
    return {p["id"]: dict(p, pred_label=parse(p["raw_output"])) for p in preds}


def diagnostics(preds: list[dict]) -> dict:
    """Headline metrics plus the two measures this ablation is about."""
    r = evaluate(preds, "drop")
    fn = [p for p in preds if p["gold_label"] == LABEL_VULN and p["pred_label"] == LABEL_SAFE]
    fix_named = [p for p in fn if _VULN_CUE_RE.search(p["raw_output"])]
    wrote = [bool(UNCHANGED_RE.search(p["raw_output"])) for p in preds]
    as_cls = [
        dict(p, pred_label=LABEL_SAFE if w else LABEL_VULN)
        for p, w in zip(preds, wrote, strict=True)
    ]
    cls = scores(confusion(as_cls, "drop"))

    def p_vulnerable(flag: bool) -> float | None:
        group = [
            p
            for p, w in zip(preds, wrote, strict=True)
            if w == flag and p["pred_label"] is not None
        ]
        if not group:
            return None
        return round(sum(p["pred_label"] == LABEL_VULN for p in group) / len(group), 4)

    keys = ("accuracy", "precision", "recall", "f1", "mcc", "pred_positive_rate")
    return {
        "n": r["n"],
        "n_unparseable": r["n_unparseable"],
        "macro": {k: round(r["macro"][k], 4) for k in keys},
        "n_false_negatives": len(fn),
        "fn_text_names_fix": len(fix_named),
        "fn_text_names_fix_share": round(len(fix_named) / len(fn), 4) if fn else None,
        "unchanged_statement": {
            "n_written": sum(wrote),
            "as_classifier_mcc": round(cls["mcc"], 4),
            "as_classifier_accuracy": round(cls["accuracy"], 4),
            "p_vulnerable_if_written": p_vulnerable(True),
            "p_vulnerable_otherwise": p_vulnerable(False),
        },
    }


def analyze(
    run_dir: str | Path = DEFAULT_RUN,
    references: dict[str, str] | None = None,
    out: str | Path | None = None,
) -> dict:
    """Compare the ablation with the fig1-v3 baseline and the two reference runs."""
    runs = {THIS: str(run_dir), **(REFERENCES if references is None else references)}
    loaded = {n: _load(d) for n, d in runs.items() if (Path(d) / "predictions.jsonl").exists()}
    if THIS not in loaded:
        raise FileNotFoundError(f"{run_dir}/predictions.jsonl missing -- run the ablation first")
    ids = sorted(set.intersection(*(set(v) for v in loaded.values())))
    rows = {name: [v[i] for i in ids] for name, v in loaded.items()}
    result = {
        "n_shared": len(ids),
        "missing_references": sorted(set(runs) - set(loaded)),
        "runs": {name: diagnostics(p) for name, p in rows.items()},
        "mcnemar_vs_baseline_val": {},
    }
    if REFERENCE in rows:
        for name in rows:
            if name != REFERENCE:
                result["mcnemar_vs_baseline_val"][name] = mcnemar(rows[name], rows[REFERENCE])
    out_path = Path(out) if out else Path(run_dir) / "analysis.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _print(result)
    return result


def _print(result: dict) -> None:
    print(f"shared instances: {result['n_shared']}")
    header = f"{'run':42s} {'F1':>6s} {'MCC':>7s} {'recall':>7s} {'PPR':>5s}"
    print(header + f" {'FN':>4s} {'FN fix':>7s} {'unch. MCC':>9s}")
    for name, d in result["runs"].items():
        m, u = d["macro"], d["unchanged_statement"]
        share = d["fn_text_names_fix_share"]
        print(
            f"{name:42s} {m['f1']:6.3f} {m['mcc']:+7.3f} {m['recall']:7.3f} "
            f"{m['pred_positive_rate']:5.2f} {d['n_false_negatives']:4d} "
            f"{share if share is None else f'{share:.2f}':>7} {u['as_classifier_mcc']:+9.3f}"
        )
    for name, x in result["mcnemar_vs_baseline_val"].items():
        print(f"McNemar {name} vs {REFERENCE}: b={x['b']} c={x['c']} p={x['p']:.3f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Label-definition ablation on val.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate on val (GPU)")
    r.add_argument("--config", default="experiments/label_definition/config.yaml")
    r.add_argument("--limit", type=int, help="first N val records only (smoke, own directory)")
    a = sub.add_parser("analyze", help="compare with baseline_val (CPU)")
    a.add_argument("--run-dir", default=DEFAULT_RUN)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.config, limit=args.limit)
    else:
        analyze(args.run_dir)


if __name__ == "__main__":
    main()

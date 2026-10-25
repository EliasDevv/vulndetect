"""Measure token lengths of the rendered Fig. 1 prompt per split and write them to JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from pathlib import Path

import yaml

from vulndetect.common.model import CHAT_DATE_STRING
from vulndetect.common.prompt import PROMPT_VERSION, render_prompt
from vulndetect.data import VALID_SPLITS, load_split


def _stats(values: list[int]) -> dict:
    ordered = sorted(values)
    p99_idx = min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "p99": ordered[p99_idx],
        "max": ordered[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Token lengths of the rendered Fig. 1 prompt.")
    ap.add_argument("--config", default="configs/baseline.yaml", help="config naming the model")
    ap.add_argument("--dataset-dir", default="data/processed/bigvul")
    ap.add_argument("--out", default="results/metrics/prompt_lengths.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # lazy: keeps the import cheap for --help

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], revision=model_cfg.get("revision"))
    date_string = model_cfg.get("chat_date_string", CHAT_DATE_STRING)

    per_split: dict[str, dict] = {}
    for split in VALID_SPLITS:
        tokens, chars = [], []
        for record in load_split(split, args.dataset_dir):
            prompt = render_prompt(record)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                date_string=date_string,
            )
            tokens.append(len(tokenizer(text, add_special_tokens=False).input_ids))
            chars.append(len(prompt))
        per_split[split] = {"tokens": _stats(tokens), "chars": _stats(chars)}
        print(f"{split}: tokens {per_split[split]['tokens']}")

    out = {
        "measured": dt.date.today().isoformat(),
        "model": model_cfg["name"],
        "model_revision": model_cfg.get("revision"),
        "prompt_version": PROMPT_VERSION,
        "chat_date_string": date_string,
        "chat_template": True,
        "dataset_dir": args.dataset_dir,
        "per_split": per_split,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

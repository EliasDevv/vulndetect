"""CLI for the Triple-Agent extension: ``train`` runs the loop, ``eval`` evaluates snapshots."""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    """Parse the CLI arguments and run the Triple-Agent loop or evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Triple-Agent extension (RQ3): GAN loop with the Dual-Agent as detecting party."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run the adversarial loop (Phase A + Phase B)")
    train.add_argument(
        "--config", default="configs/triple_agent.yaml", help="path to the triple-agent YAML"
    )

    ev = sub.add_parser("eval", help="evaluate a snapshot's S_D with the two-stage party")
    ev.add_argument(
        "--config", default="configs/triple_agent.yaml", help="path to the triple-agent YAML"
    )
    ev.add_argument(
        "--snapshot",
        default="final",
        help="'final', 'round0050', a round number, or a path to a snapshot file",
    )
    ev.add_argument(
        "--all-snapshots",
        action="store_true",
        help="evaluate every snapshot of the run (the round-count ablation)",
    )
    ev.add_argument("--limit", type=int, help="only the first N records (smoke run, not logged)")

    args = ap.parse_args()
    if args.command == "train":
        from vulndetect.triple_agent.loop import run_loop

        run_loop(args.config)
        return

    from vulndetect.triple_agent.run import run, run_all_snapshots

    if args.all_snapshots:
        run_all_snapshots(args.config, limit=args.limit)
    else:
        run(args.config, args.snapshot, limit=args.limit)


if __name__ == "__main__":
    main()

"""Render the Chapter 7 figures (overview, round curve, cost) from the metrics files as PDFs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

METRICS = Path("results/metrics")
FIGURES = Path("results/figures")

ACCENT = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#8a8985"
GRID = "#d9d8d3"
ALWAYS_VULNERABLE_F1 = 2 / 3  # balanced split

OVERVIEW = (  # the paper's order, then the extension
    ("rag", "RAG"),
    ("sft", "SFT"),
    ("dual_agent", "Dual-Agent"),
    ("baseline", "Baseline"),
    ("gan/final", "GAN\nfinal"),
    ("triple_agent/final", "Triple-Agent\nfinal"),
)
SNAPSHOTS = (
    ("gan/round0000", "round 0"),
    ("gan/round0010", "round 10"),
    ("gan/round0050", "round 50"),
    ("gan/round0200", "round 200"),
    ("gan/final", "final"),
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def per_seed_f1(c: dict, name: str) -> dict[int, float]:
    """Return the macro-F1 of an approach per seed."""
    a = c["approaches"][name]
    return dict(zip(a["seeds"], a["macro"]["f1"]["values"], strict=True))


def fig_overview(c: dict, reference: dict[str, float]) -> plt.Figure:
    """Plot per-seed and mean macro-F1 of the six approaches with the paper's values."""
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    xs = range(len(OVERVIEW))
    ax.axhline(ALWAYS_VULNERABLE_F1, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axvline(3.5, color=GRID, lw=0.6, zorder=1)
    for x, (name, _) in zip(xs, OVERVIEW, strict=True):
        seeds = per_seed_f1(c, name)
        ax.scatter(
            [x] * len(seeds), list(seeds.values()), s=22, color=MUTED, zorder=3, linewidths=0
        )
        mean = c["approaches"][name]["macro"]["f1"]["mean"]
        ax.hlines(mean, x - 0.22, x + 0.22, color=ACCENT, lw=2.2, zorder=4)
        ax.annotate(
            f"{mean:.3f}",
            (x + 0.26, mean),
            ha="left",
            va="center",
            fontsize=7.5,
            color=INK,
            zorder=5,
        )
        if name in reference:
            ax.scatter(
                [x],
                [reference[name]],
                s=46,
                marker="D",
                facecolors="white",
                edgecolors=INK,
                linewidths=0.9,
                zorder=4,
            )
    ax.set_xticks(list(xs), [label for _, label in OVERVIEW])
    ax.set_xlim(-0.6, len(OVERVIEW) - 0.4)
    ax.set_ylim(0.40, 0.90)
    ax.set_ylabel("Macro-F1, test split")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.text(3.47, 0.885, "Replication", ha="right", va="top", fontsize=8, color=MUTED)
    ax.text(3.53, 0.885, "Extension", ha="left", va="top", fontsize=8, color=MUTED)
    handles = [
        Line2D([], [], marker="o", color=MUTED, ls="", ms=5, label="one seed (42, 43, 44)"),
        Line2D([], [], color=ACCENT, lw=2.2, label="mean over the three seeds"),
        Line2D(
            [],
            [],
            marker="D",
            mfc="white",
            mec=INK,
            ls="",
            ms=6,
            label="reported by the base study",
        ),
        Line2D([], [], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="always Vulnerable (F1 = 0.667)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.8,
    )
    fig.tight_layout(pad=0.4)
    return fig


def fig_round_curve(c: dict) -> plt.Figure:
    """Plot macro-F1 of the GAN snapshots over the loop rounds."""
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    xs = list(range(len(SNAPSHOTS)))
    ax.axvspan(-0.5, 3.0, color="#f1f0ec", zorder=0, lw=0)
    ax.text(1.5, 0.484, "adversarial stage", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(3.75, 0.484, "self-reflection", ha="center", va="bottom", fontsize=8, color=MUTED)
    # seeds 43 and 44 were evaluated at round 0 and final only: markers, no line
    for seed, marker in ((43, "s"), (44, "^")):
        ys = [per_seed_f1(c, name)[seed] for name in ("gan/round0000", "gan/final")]
        ax.scatter([0, 4], ys, s=26, marker=marker, color=MUTED, zorder=3, linewidths=0)
    ys42 = [per_seed_f1(c, name)[42] for name, _ in SNAPSHOTS]
    ax.plot(xs, ys42, color=ACCENT, lw=1.8, zorder=4)
    ax.scatter(xs, ys42, s=30, color=ACCENT, zorder=5, linewidths=0)
    for x, y in zip(xs, ys42, strict=True):
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK,
            zorder=6,
        )
    ax.set_xticks(xs, [label for _, label in SNAPSHOTS])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0.48, 0.60)
    ax.set_ylabel("Macro-F1, test split")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    handles = [
        Line2D([], [], marker="o", color=ACCENT, lw=1.8, ms=5, label="seed 42, every snapshot"),
        Line2D([], [], marker="s", color=MUTED, ls="", ms=4.5, label="seed 43, round 0 and final"),
        Line2D([], [], marker="^", color=MUTED, ls="", ms=5, label="seed 44, round 0 and final"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=2.2)
    fig.tight_layout(pad=0.4)
    return fig


# Cost figure: (display name, token key, comparison key, model calls, label side).
COST = (
    ("SFT", "sft", "sft", 1, "right"),
    ("GAN final", "gan/final", "gan/final", 1, "left"),
    ("RAG", "rag", "rag", 1, "left"),
    ("Baseline", "baseline", "baseline", 1, "left"),
    ("Triple-Agent final", "triple_agent/final", "triple_agent/final", 2, "right"),
    ("Dual-Agent", "dual_agent", "dual_agent", 2, "right"),
)


def fig_cost(c: dict, d: dict) -> plt.Figure:
    """Plot macro-F1 against output tokens per instance for each approach."""
    from matplotlib.ticker import FixedLocator, NullLocator

    tokens = d["cost"]["output_tokens_per_instance"]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    for label, tok_key, key, calls, side in COST:
        x = tokens[tok_key]["mean"]
        y = c["approaches"][key]["macro"]["f1"]["mean"]
        ax.scatter(
            [x],
            [y],
            s=34,
            zorder=3,
            facecolors=ACCENT if calls == 1 else "white",
            edgecolors=ACCENT,
            linewidths=1.3,
        )
        dx = 7 if side == "right" else -7
        ax.annotate(
            f"{label} ({y:.3f})",
            (x, y),
            xytext=(dx, 0),
            textcoords="offset points",
            ha="left" if side == "right" else "right",
            va="center",
            fontsize=7.5,
            color=INK,
            bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"},
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_xlim(3, 3000)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 50, 100, 500, 1000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "50", "100", "500", "1000"])
    ax.set_ylim(0.44, 0.72)
    ax.set_xlabel("Generated tokens per test instance (log scale)")
    ax.set_ylabel("Macro-F1, test split")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", ls="", ms=5.5, mfc=ACCENT, mec=ACCENT, label="one model call"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=5.5,
            mfc="white",
            mec=ACCENT,
            mew=1.3,
            label="two model calls",
        ),
    ]
    legend = ax.legend(
        handles=handles, loc="upper right", frameon=True, handletextpad=0.3, framealpha=1
    )
    legend.get_frame().set_edgecolor("none")
    fig.tight_layout(pad=0.4)
    return fig


def main() -> None:
    """Load the metrics files and write the Chapter 7 figures as PDFs."""
    c = json.loads((METRICS / "comparison_test.json").read_text(encoding="utf-8"))
    reference: dict[str, float] = {}
    for line in (METRICS / "paper_reference.csv").read_text(encoding="utf-8").splitlines()[1:]:
        approach, metric, value = line.split(",")
        if metric == "f1":
            reference[approach.removesuffix("_val")] = float(value)
    missing = [n for n, _ in (*OVERVIEW, *SNAPSHOTS) if n not in c["approaches"]]
    if missing:
        raise SystemExit(f"comparison_test.json lacks {missing} -- rerun the aggregator")
    d = json.loads((METRICS / "ch7_diagnostics.json").read_text(encoding="utf-8"))
    FIGURES.mkdir(exist_ok=True)
    for name, fig in (
        ("ch7_fig_overview.pdf", fig_overview(c, reference)),
        ("ch7_fig_round_curve.pdf", fig_round_curve(c)),
        ("ch7_fig_cost.pdf", fig_cost(c, d)),
    ):
        fig.savefig(FIGURES / name)
        plt.close(fig)
        print(f"wrote {FIGURES / name}")


if __name__ == "__main__":
    main()

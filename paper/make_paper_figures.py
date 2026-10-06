"""Generate paper-specific summary figures from archived aggregate results."""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np



BLUE = "#176B8A"
RED = "#B84A3A"
GRAY = "#5C6670"


def add_labels(ax, bars, digits=3):
    for bar in bars:
        value = bar.get_height()
        inside = value > 0.9
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.055 if inside else value + 0.025,
            f"{value:.{digits}f}",
            ha="center",
            va="top" if inside else "bottom",
            fontsize=7.2,
            color="white" if inside else "black",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    output = args.results / "paper/figures"
    output.mkdir(parents=True, exist_ok=True)
    sweep = json.loads((args.results / "experiment1/sweep.json").read_text())
    selected = [r for r in sweep if r["config"]["occurrences"] == 16 and r["config"]["attr_repeats"] == 128]
    lookup = json.loads((args.results / "experiment2/analysis.json").read_text())
    def scores(relation):
        return np.mean([[r["relations"][relation]["trained_check"]["recall"],
                         r["relations"][relation]["heldout_check"]["recall"],
                         r["relations"][relation]["probe_accuracy"]] for r in selected], axis=0)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), constrained_layout=True)

    ax = axes[0]
    metrics = ["trained\nrecall", "held-out\nrecall", "held-out\nOLS accuracy"]
    structured = scores("semantic")
    random = scores("random")
    x = np.arange(len(metrics))
    width = 0.34
    b1 = ax.bar(
        x - width / 2,
        structured,
        width,
        color=BLUE,
        edgecolor=BLUE,
        label="attribute-defined",
    )
    b2 = ax.bar(
        x + width / 2,
        random,
        width,
        color=RED,
        edgecolor="#6F2C24",
        hatch="//",
        label="matched random",
    )
    add_labels(ax, b1)
    add_labels(ax, b2)
    ax.axhline(1 / 32, color=GRAY, linestyle=":", linewidth=1)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("score")
    ax.set_title("A  Structured relation, $F=16$", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    metrics = ["complete-table\naccuracy", "full-data\nOLS $R^2$"]
    structured = np.array([lookup["summary"][k]["mean"] for k in ("min_structured_accuracy", "min_structured_full_r2")])
    random = np.array([lookup["summary"][k]["mean"] for k in ("arbitrary_accuracy", "arbitrary_full_r2")])
    x = np.arange(len(metrics))
    b1 = ax.bar(
        x - width / 2,
        structured,
        width,
        color=BLUE,
        edgecolor=BLUE,
        label="structured minimum",
    )
    b2 = ax.bar(
        x + width / 2,
        random,
        width,
        color=RED,
        edgecolor="#6F2C24",
        hatch="//",
        label="random relation",
    )
    add_labels(ax, b1)
    add_labels(ax, b2)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.13)
    ax.set_title("B  Random relation, 10$\\times$/control (joint)", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(output / "frequency_not_enough.pdf", bbox_inches="tight")
    fig.savefig(output / "frequency_not_enough.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    for triples, style, label in ((128, "-", "one occurrence"), (0, "--", "no relation triples")):
        points = []
        for repeats in (2, 4, 8, 16, 24):
            rows = [r for r in sweep if r["config"]["n_triples"] == triples
                    and r["config"]["attr_repeats"] == repeats]
            values = [r["relations"]["semantic"]["probe_accuracy"] for r in rows]
            points.append((repeats, np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0))
        x, y, sd = np.array(points).T
        ax.plot(x, y, style, color=BLUE, marker="o", label=label)
        ax.fill_between(x, y - sd, y + sd, color=BLUE, alpha=.12)
    ax.axhline(1 / 32, color=GRAY, linestyle=":")
    ax.set(xlabel="Attribute sentence repetitions", ylabel="Held-out OLS accuracy", ylim=(0, 1.05))
    ax.legend(frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"linear_shift_transition.{suffix}", dpi=240)
    plt.close(fig)
    from factual_recall.symmetric_lookup_transformer_raw_analysis import plot_results, plot_mechanism
    plot_results(lookup["seeds"], output / "goal2_raw_transformer_results",
                 {int(seed): history for seed, history in lookup["histories"].items()})
    plot_mechanism(lookup["seeds"], output / "goal2_raw_transformer_mechanism")
    from factual_recall.paper_audit import write_tables
    write_tables(args.results)



if __name__ == "__main__":
    main()

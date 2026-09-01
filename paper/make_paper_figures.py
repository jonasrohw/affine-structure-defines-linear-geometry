"""Generate paper-specific summary figures from archived aggregate results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

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
    structured = np.array([0.672, 0.635, 0.990])
    random = np.array([0.167, 0.031, 0.036])
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
    structured = np.array([0.99951, 0.99568])
    random = np.array([0.99845, 0.38291])
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

    fig.savefig(OUT / "frequency_not_enough.pdf", bbox_inches="tight")
    fig.savefig(OUT / "frequency_not_enough.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Figure: affine decoding tracks answer sharing, not subject-object co-occurrence.

Panel A contrasts the two marginals on one axis: decoder faithfulness is flat across
co-occurrence quartiles and rises steeply with the number of entities sharing an answer.
Panel B shows the joint, so the contrast is not an artefact of marginalising. Panel C
compares the decoder with a leave-one-out nearest-centroid probe on the same subject
states, per relation.

Inputs are the archived per-fact probe results and the frequency snapshot; no model
inference is performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from factual_recall.figure_style import COLORS, TEXT_WIDTH, apply_style, save_figure  # noqa: E402

SHARING_BINS = [-1, 0, 9, 29, 10 ** 9]
SHARING_LABELS = ["0", "1--9", "10--29", "30+"]
COOC_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def load(results):
    facts = pd.read_csv(ROOT / "paper/data/lment_frequency/facts.csv")
    per_fact = pd.read_csv(results / "probe_perfact_all.csv")
    merged = per_fact.merge(
        facts[["relation", "subject", "object", "lm_prediction_token_id"]],
        on=["relation", "subject", "object"])
    merged["support"] = merged.groupby(
        ["relation", "lm_prediction_token_id"])["subject"].transform("size") - 1
    merged["cooc_bin"] = pd.qcut(merged.cooc, 4, labels=COOC_LABELS, duplicates="drop")
    merged["share_bin"] = pd.cut(merged.support, SHARING_BINS, labels=SHARING_LABELS)
    relations = pd.read_csv(results / "probe_vs_lre.csv")
    cohort = set(pd.read_csv(results / "cohort.csv").relation.unique())
    return merged, relations[relations.relation.isin(cohort)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="External directory with probe_perfact_all.csv, probe_vs_lre.csv and cohort.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/lment-sharing/figures")
    args = parser.parse_args()
    required = ("probe_perfact_all.csv", "probe_vs_lre.csv", "cohort.csv")
    missing = [name for name in required if not (args.inputs / name).is_file()]
    if missing:
        parser.error("External analysis inputs are absent: " + ", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    facts, relations = load(args.inputs)
    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5),
                                gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]},
                                constrained_layout=True)

    # A: the two marginals, same axis.
    by_cooc = facts.groupby("cooc_bin", observed=True).lre_correct.mean()
    by_share = facts.groupby("share_bin", observed=True).lre_correct.mean()
    position = np.arange(4)
    axes[0].plot(position, by_cooc.values, marker="o", color=COLORS["random"],
                 label="subject--object co-occurrence")
    axes[0].plot(position, by_share.values, marker="s", color=COLORS["structured"],
                 label="entities sharing the answer")
    axes[0].set_xticks(position)
    axes[0].set_xticklabels(["1", "2", "3", "4"])
    axes[0].set_xlabel("bin, increasing")
    axes[0].set_ylabel("decoder faithfulness")
    axes[0].set_ylim(0.3, 0.85)
    axes[0].grid(axis="y")
    axes[0].legend(loc="lower right", fontsize=6.5)
    axes[0].set_title("A", loc="left")

    # B: the joint. Rows vary, columns do not.
    table = facts.pivot_table(index="share_bin", columns="cooc_bin",
                              values="lre_correct", aggfunc="mean", observed=True)
    counts = facts.pivot_table(index="share_bin", columns="cooc_bin",
                               values="lre_correct", aggfunc="size", observed=True)
    cmap = LinearSegmentedColormap.from_list(
        "sharing", [COLORS["pale"], COLORS["blue"], COLORS["structured"]])
    image = axes[1].imshow(table.values, cmap=cmap, vmin=0.3, vmax=0.85, aspect="auto")
    axes[1].set_xticks(range(4), COOC_LABELS)
    axes[1].set_yticks(range(4), SHARING_LABELS)
    axes[1].set_xlabel("co-occurrence quartile")
    axes[1].set_ylabel("entities sharing answer")
    axes[1].invert_yaxis()
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            value = table.values[row, column]
            axes[1].text(column, row, f"{value:.2f}", ha="center", va="center",
                         fontsize=6.5,
                         color=COLORS["ink"] if value < 0.68 else "white")
    axes[1].set_title("B", loc="left")

    # C: the decoder against a plain probe on the same states.
    axes[2].plot([0, 1], [0, 1], color=COLORS["muted"], linewidth=0.6, linestyle="--")
    axes[2].scatter(relations.probe, relations.lre, s=18,
                    color=COLORS["lavender"], edgecolor=COLORS["ink"], linewidth=0.4,
                    zorder=3)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("nearest-centroid probe")
    axes[2].set_ylabel("decoder faithfulness")
    axes[2].grid(True)
    axes[2].set_title("C", loc="left")
    axes[2].annotate(f"mean {relations.lre.mean():.3f} vs {relations.probe.mean():.3f}",
                     xy=(0.05, 0.92), xycoords="axes fraction", fontsize=6.5,
                     color=COLORS["muted"])

    print("A marginals -- co-occurrence:", by_cooc.round(3).to_dict())
    print("A marginals -- sharing:      ", by_share.round(3).to_dict())
    print("B cell counts:\n", counts.to_string())
    save_figure(figure, args.output_dir / "lment_sharing")
    print(f"wrote {args.output_dir}/lment_sharing.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()

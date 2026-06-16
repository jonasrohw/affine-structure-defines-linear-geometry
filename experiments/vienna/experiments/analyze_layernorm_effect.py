"""
Analyze the effect of LayerNorm on residual stream vectors.

For each f2 and subject representation at each layer, compute:
  - Cosine similarity between pre-LN and post-LN vectors
  - Mean/norm ratio (fraction of the vector in the mean direction)

Usage (from vienna/):
    python experiments/analyze_layernorm_effect.py [--checkpoint PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config import cfg as default_cfg
from src.model import build_model
from src.representations import (
    get_all_subject_representations,
    get_filler_representations,
    stack_all_filler_representations,
    stack_representations,
)
from src.training import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def cosine_pre_post_ln(vectors, ln_f):
    """Compute cosine similarity between vectors and their LN'd versions.

    Args:
        vectors: (n, d) tensor
        ln_f: LayerNorm module

    Returns:
        cosines: (n,) array
        mean_norm_ratios: (n,) array — |mean| * sqrt(d) / ||v||
    """
    with torch.no_grad():
        normed = ln_f(vectors)
    cos = F.cosine_similarity(vectors, normed, dim=1).numpy()

    d = vectors.shape[1]
    means = vectors.mean(dim=1)
    norms = vectors.norm(dim=1)
    mean_norm_ratios = (means.abs() * np.sqrt(d) / norms).numpy()

    return cos, mean_norm_ratios


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = default_cfg
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.train.checkpoint_dir / "best.pt"

    model = build_model(cfg.model, cfg.vocab)
    load_checkpoint(ckpt_path, model, device)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    all_indices = list(range(cfg.vocab.n_subjects))

    print("Extracting representations...")
    subj_reprs = get_all_subject_representations(model, cfg, all_indices, device)
    filler_reprs = get_filler_representations(model, cfg, all_indices, device)
    n_snapshots = len(subj_reprs[0])
    layer_labels = ["emb"] + [f"L{i+1}" for i in range(n_snapshots - 1)]

    ln_f = model.ln_f.cpu()
    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)
    results_lines = []

    def log(line=""):
        print(line)
        results_lines.append(line)

    log("=" * 60)
    log("Cosine(pre-LN, post-LN) statistics")
    log("=" * 60)
    header = f"{'Layer':>6} | {'Type':>8} | {'mean cos':>10} | {'min cos':>10} | {'max cos':>10} | {'std cos':>10}"
    log(header)
    log("-" * len(header))

    all_data = {}

    for layer in range(n_snapshots):
        S, _ = stack_representations(subj_reprs, all_indices, layer)
        F2, _ = stack_all_filler_representations(filler_reprs, all_indices, cfg.vocab.num_relations, layer)

        s_cos, s_ratio = cosine_pre_post_ln(S, ln_f)
        f_cos, f_ratio = cosine_pre_post_ln(F2, ln_f)

        all_data[layer] = {
            "subj_cos": s_cos, "subj_ratio": s_ratio,
            "f2_cos": f_cos, "f2_ratio": f_ratio,
        }

        log(f"{layer_labels[layer]:>6} | {'subj':>8} | {s_cos.mean():>+10.6f} | {s_cos.min():>+10.6f} | {s_cos.max():>+10.6f} | {s_cos.std():>10.6f}")
        log(f"{'':>6} | {'f2':>8} | {f_cos.mean():>+10.6f} | {f_cos.min():>+10.6f} | {f_cos.max():>+10.6f} | {f_cos.std():>10.6f}")

    log()

    # ── Histograms: one row per layer, two columns (subj, f2) ────────
    fig, axes = plt.subplots(n_snapshots, 2, figsize=(10, 4 * n_snapshots))
    if n_snapshots == 1:
        axes = axes.reshape(1, -1)

    for layer in range(n_snapshots):
        d = all_data[layer]

        ax = axes[layer, 0]
        ax.hist(d["subj_cos"], bins=30, color="steelblue", alpha=0.8, edgecolor="white")
        ax.set_title(f"Subject representations — {layer_labels[layer]}", fontsize=10)
        ax.set_xlabel("cos(pre-LN, post-LN)")
        ax.set_ylabel("count")
        ax.axvline(d["subj_cos"].mean(), color="crimson", ls="--", lw=1.5,
                   label=f"mean={d['subj_cos'].mean():.4f}")
        ax.legend(fontsize=8)

        ax = axes[layer, 1]
        ax.hist(d["f2_cos"], bins=30, color="darkorange", alpha=0.8, edgecolor="white")
        ax.set_title(f"f2 representations — {layer_labels[layer]}", fontsize=10)
        ax.set_xlabel("cos(pre-LN, post-LN)")
        ax.set_ylabel("count")
        ax.axvline(d["f2_cos"].mean(), color="crimson", ls="--", lw=1.5,
                   label=f"mean={d['f2_cos'].mean():.4f}")
        ax.legend(fontsize=8)

    fig.suptitle("Cosine similarity between pre-LN and post-LN vectors", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = cfg.train.results_dir / "layernorm_cosine_histograms.pdf"
    fig.savefig(out)
    log(f"Saved: {out}")

    # ── Scatter plots: cosine vs mean/norm ratio ─────────────────────
    fig, axes = plt.subplots(n_snapshots, 2, figsize=(10, 4 * n_snapshots))
    if n_snapshots == 1:
        axes = axes.reshape(1, -1)

    for layer in range(n_snapshots):
        d = all_data[layer]

        ax = axes[layer, 0]
        ax.scatter(d["subj_ratio"], d["subj_cos"], s=12, alpha=0.6, color="steelblue")
        ax.set_title(f"Subject representations — {layer_labels[layer]}", fontsize=10)
        ax.set_xlabel("|mean| · √d / ||v||")
        ax.set_ylabel("cos(pre-LN, post-LN)")
        ax.set_xlim(left=0)

        ax = axes[layer, 1]
        ax.scatter(d["f2_ratio"], d["f2_cos"], s=12, alpha=0.6, color="darkorange")
        ax.set_title(f"f2 representations — {layer_labels[layer]}", fontsize=10)
        ax.set_xlabel("|mean| · √d / ||v||")
        ax.set_ylabel("cos(pre-LN, post-LN)")
        ax.set_xlim(left=0)

    fig.suptitle("Cosine vs mean/norm ratio: what drives LN's directional change", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = cfg.train.results_dir / "layernorm_cosine_vs_ratio.pdf"
    fig.savefig(out)
    log(f"Saved: {out}")

    plt.close("all")

    results_path = cfg.train.results_dir / "layernorm_effect_results.md"
    with open(results_path, "w") as f:
        f.write("# LayerNorm Effect on Residual Stream Vectors\n\n")
        f.write("Cosine similarity between pre-LN and post-LN vectors for subject and f2 representations.\n\n")
        f.write("```\n")
        f.write("\n".join(results_lines))
        f.write("\n```\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()

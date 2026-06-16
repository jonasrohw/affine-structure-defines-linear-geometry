"""
Linearity analysis for the v2 model (three relations + filler tokens).

Usage (from vienna/):
    python experiments/analyze_linearity_v2.py [--checkpoint PATH] [--device auto]

For each layer (0=embedding through n_layers):
  1. Extract subject representations from [x, a_i] prefix (no relation token).
  2. Extract object representations from [x, b_j] prefix (symmetric).
  3. For each relation R_k: fit affine T_k with 5-fold CV; report Δcos.
  4. Second-order tests: rank(W_1-W_0), rank(W_2-W_0); pure-translation check.
  5. Cross-relation generalization: apply T_0 to predict objects for R1, R2.
  6. Save plots to results/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.affine import cross_relation_eval, fit_affine, kfold_affine, second_order_test
from src.config import Config, cfg as default_cfg
from src.model import build_model
from src.representations import (
    get_all_object_representations,
    get_all_subject_representations,
    stack_representations,
)
from src.training import load_checkpoint

RELATION_LABELS = ["R0 (identity)", "R1 (+50 cyclic)", "R2 (random perm)"]
RELATION_KEYS = ["R0", "R1", "R2"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def build_object_matrix_for_relation(
    obj_reprs: dict,
    pair_indices: list,
    relation_idx: int,
    vocab,
    layer: int,
) -> torch.Tensor:
    """Build the (n, d) object matrix for a given relation at a given layer.

    For each subject index i in pair_indices, the corresponding object is
    b_{R_k(i)}, so we look up obj_reprs[R_k(i)].
    """
    vecs = []
    for i in pair_indices:
        j = vocab.apply_relation(relation_idx, i)
        vecs.append(obj_reprs[j][layer])
    return torch.stack(vecs, dim=0)


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = default_cfg
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.train.checkpoint_dir / "best.pt"

    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}. Run train_v2.py first.")
        sys.exit(1)

    model = build_model(cfg.model, cfg.vocab)
    ckpt = load_checkpoint(ckpt_path, model, device)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")
    if "val_acc_obj" in ckpt:
        print(f"  Best val_acc_obj during training: {ckpt['val_acc_obj']:.4f}")
    print()

    all_indices = list(range(cfg.vocab.n_subjects))  # 0..99

    print("Extracting representations...")
    subj_reprs = get_all_subject_representations(model, cfg, all_indices, device)
    obj_reprs = get_all_object_representations(model, cfg, all_indices, device)
    n_snapshots = len(subj_reprs[0])  # n_layers + 1
    layer_labels = ["emb"] + [f"L{i+1}" for i in range(n_snapshots - 1)]
    print(f"  Snapshots per token: {n_snapshots} (embedding + {n_snapshots-1} layer(s))")
    print()

    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-layer, per-relation Δcos table ─────────────────────────────────
    header = f"{'Layer':>6} | " + " | ".join(f"{k+' CV Δcos':>14}" for k in RELATION_KEYS)
    print(header)
    print("-" * len(header))

    # Store for plots and second-order tests
    layer_cv_means = {k: [] for k in RELATION_KEYS}  # key -> list over layers
    layer_cv_stds  = {k: [] for k in RELATION_KEYS}
    layer_W_full   = {k: [] for k in RELATION_KEYS}  # best-layer W matrices
    layer_b_full   = {k: [] for k in RELATION_KEYS}

    for layer in range(n_snapshots):
        S_all, _ = stack_representations(subj_reprs, all_indices, layer)
        row = f"{layer_labels[layer]:>6}"
        for ri, rkey in enumerate(RELATION_KEYS):
            O_rel = build_object_matrix_for_relation(obj_reprs, all_indices, ri, cfg.vocab, layer)
            mean_dc, std_dc, W_f, b_f = kfold_affine(S_all, O_rel, k=args.kfolds, bias=True)
            layer_cv_means[rkey].append(mean_dc)
            layer_cv_stds[rkey].append(std_dc)
            layer_W_full[rkey].append(W_f)
            layer_b_full[rkey].append(b_f)
            row += f" | {mean_dc:>+8.4f} ± {std_dc:.4f}"
        print(row)

    print()

    # ── Second-order tests (at best layer per relation, and at last layer) ──
    best_layers = {rkey: int(np.argmax(layer_cv_means[rkey])) for rkey in RELATION_KEYS}
    last_layer = n_snapshots - 1

    for analysis_layer, layer_name in [
        (last_layer, f"last layer ({layer_labels[last_layer]})"),
        (best_layers["R0"], f"best layer for R0 ({layer_labels[best_layers['R0']]})"),
    ]:
        print(f"Second-order test at {layer_name}:")
        W_list = [layer_W_full[rkey][analysis_layer] for rkey in RELATION_KEYS]
        so = second_order_test(W_list, RELATION_KEYS)

        for pair, frob in so["frobenius"].items():
            rank = so["diff_rank"][pair]
            print(f"  ||{pair[0]} - {pair[1]}||_F = {frob:.4f}   rank(diff) = {rank}")
        print(f"  Pure-translation hypothesis: {so['pure_translation']}")
        print()

    # ── Cross-relation generalization ───────────────────────────────────────
    print("Cross-relation generalization (T_0 trained on R0, applied to all relations):")
    analysis_layer = best_layers["R0"]
    S_all, _ = stack_representations(subj_reprs, all_indices, analysis_layer)
    O_by_relation = [
        build_object_matrix_for_relation(obj_reprs, all_indices, ri, cfg.vocab, analysis_layer)
        for ri in range(cfg.vocab.num_relations)
    ]
    _, _, W0, b0 = kfold_affine(S_all, O_by_relation[0], k=args.kfolds, bias=True)
    cross = cross_relation_eval(S_all, O_by_relation, W0, b0, "R0", RELATION_KEYS)
    for label, dc in cross.items():
        print(f"  T_0 → {label}: Δcos = {dc:+.4f}")
    print()

    if args.no_plots:
        return

    # ── Per-relation Δcos vs layer plot ────────────────────────────────────
    colors = {"R0": "steelblue", "R1": "darkorange", "R2": "seagreen"}
    x = np.arange(n_snapshots)

    fig, ax = plt.subplots(figsize=(7, 4))
    for rkey in RELATION_KEYS:
        means = np.array(layer_cv_means[rkey])
        stds  = np.array(layer_cv_stds[rkey])
        ri = RELATION_KEYS.index(rkey)
        ax.plot(x, means, "o-", label=RELATION_LABELS[ri], color=colors[rkey])
        ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=colors[rkey])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV Δcos (5-fold)")
    ax.set_title("Per-relation Delta Cosine Similarity per Layer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.train.results_dir / "delta_cos_per_layer_v2.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")

    # ── Frobenius distance heatmap (last layer) ─────────────────────────────
    W_list = [layer_W_full[rkey][last_layer] for rkey in RELATION_KEYS]
    n_rel = len(RELATION_KEYS)
    frob_matrix = np.zeros((n_rel, n_rel))
    for i in range(n_rel):
        for j in range(n_rel):
            frob_matrix[i, j] = (W_list[i] - W_list[j]).norm(p="fro").item()

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(frob_matrix, cmap="viridis")
    ax.set_xticks(range(n_rel))
    ax.set_yticks(range(n_rel))
    ax.set_xticklabels(RELATION_KEYS)
    ax.set_yticklabels(RELATION_KEYS)
    for i in range(n_rel):
        for j in range(n_rel):
            ax.text(j, i, f"{frob_matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if frob_matrix[i, j] > frob_matrix.max() / 2 else "black",
                    fontsize=9)
    ax.set_title(f"||W_i − W_j||_F  ({layer_labels[last_layer]})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = cfg.train.results_dir / "weight_frobenius_v2.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")

    plt.close("all")


if __name__ == "__main__":
    main()

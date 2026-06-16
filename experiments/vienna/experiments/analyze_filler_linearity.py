"""
Linearity probes at filler positions (Experiment 3 from post_v2_analysis_plan.md).

Tests whether linear structure emerges at the f2 position (pos 5), where the
model has seen both a_i and r_k and is under next-token pressure to predict b_j.

Sub-analyses:
  (a) Single affine map from f2 representations to object representations (all relations pooled)
  (b) Per-relation affine maps at f2
  (c) PCA dimensionality of f2 representations

Usage (from vienna/):
    python experiments/analyze_filler_linearity.py [--checkpoint PATH] [--device auto]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.affine import kfold_affine
from src.config import cfg as default_cfg
from src.model import build_model
from src.representations import (
    get_all_object_representations,
    get_all_subject_representations,
    get_filler_representations,
    get_unembedding_object_representations,
    stack_all_filler_representations,
    stack_filler_representations,
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
    p.add_argument("--use-unembedding", action="store_true",
                   help="Use lm_head (unembedding) rows as object representations instead of residual stream from [x, b_j]")
    p.add_argument("--apply-ln-f", action="store_true",
                   help="Apply model's final LayerNorm to source representations before affine probe (use with --use-unembedding)")
    p.add_argument("--normalize", action="store_true",
                   help="L2-normalize all representations to unit vectors before affine fitting")
    return p.parse_args()


def build_pooled_object_matrix(obj_reprs, subject_indices, num_relations, vocab, layer):
    """Build (n_subjects * num_relations, d) object matrix matching the pooled f2 order."""
    vecs = []
    for i in subject_indices:
        for k in range(num_relations):
            j = vocab.apply_relation(k, i)
            vecs.append(obj_reprs[j][layer])
    return torch.stack(vecs, dim=0)


def build_relation_object_matrix(obj_reprs, subject_indices, relation_idx, vocab, layer):
    """Build (n_subjects, d) object matrix for a single relation."""
    vecs = []
    for i in subject_indices:
        j = vocab.apply_relation(relation_idx, i)
        vecs.append(obj_reprs[j][layer])
    return torch.stack(vecs, dim=0)


def pca_effective_dim(X, thresholds=(0.90, 0.95, 0.99)):
    """Return number of PCA components needed for each variance threshold."""
    X_np = X.numpy().astype(np.float64)
    X_centered = X_np - X_np.mean(axis=0)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    var_explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    results = {}
    for t in thresholds:
        results[t] = int(np.searchsorted(var_explained, t) + 1)
    return results


def subject_fold_kfold(S, O, k=5, bias=True):
    """K-fold CV splitting by subject index (every k-th subject in same fold).

    For pooled data where rows are ordered as (subj0,rel0), (subj0,rel1), ...,
    this ensures all relations for a subject stay together.
    """
    from src.evaluation import delta_cos

    n_total = S.shape[0]
    n_relations = 3
    n_subjects = n_total // n_relations
    fold_size = n_subjects // k

    delta_cos_scores = []
    indices = np.arange(n_subjects)

    for fold in range(k):
        val_subj = indices[fold * fold_size: (fold + 1) * fold_size]
        train_subj = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

        val_rows = []
        train_rows = []
        for s in val_subj:
            for r in range(n_relations):
                val_rows.append(s * n_relations + r)
        for s in train_subj:
            for r in range(n_relations):
                train_rows.append(s * n_relations + r)

        val_rows = np.array(val_rows)
        train_rows = np.array(train_rows)

        from src.affine import fit_affine
        W, b = fit_affine(S[train_rows], O[train_rows], bias=bias)
        dc = delta_cos(S[val_rows], O[val_rows], W, b)
        delta_cos_scores.append(dc)

    from src.affine import fit_affine
    W_full, b_full = fit_affine(S, O, bias=bias)
    return float(np.mean(delta_cos_scores)), float(np.std(delta_cos_scores)), W_full, b_full


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

    all_indices = list(range(cfg.vocab.n_subjects))

    obj_mode = "unembedding (lm_head)" if args.use_unembedding else "residual stream ([x, b_j])"
    if args.apply_ln_f:
        obj_mode += " + ln_f applied to source"
    if args.normalize:
        obj_mode += " + L2-normalized"
    print(f"Object representations: {obj_mode}")
    print("Extracting representations...")
    subj_reprs = get_all_subject_representations(model, cfg, all_indices, device)
    if args.use_unembedding:
        obj_reprs = get_unembedding_object_representations(model, cfg, all_indices, cfg.model.n_layers)
    else:
        obj_reprs = get_all_object_representations(model, cfg, all_indices, device)
    filler_reprs = get_filler_representations(model, cfg, all_indices, device)

    if args.apply_ln_f:
        ln_f = model.ln_f.cpu()
        with torch.no_grad():
            for i in subj_reprs:
                subj_reprs[i] = [ln_f(r.unsqueeze(0)).squeeze(0) for r in subj_reprs[i]]
            for key in filler_reprs:
                filler_reprs[key] = [ln_f(r.unsqueeze(0)).squeeze(0) for r in filler_reprs[key]]

    if args.normalize:
        def _norm(v):
            return torch.nn.functional.normalize(v, dim=0)
        for i in subj_reprs:
            subj_reprs[i] = [_norm(r) for r in subj_reprs[i]]
        for key in filler_reprs:
            filler_reprs[key] = [_norm(r) for r in filler_reprs[key]]
        for i in obj_reprs:
            if isinstance(obj_reprs[i], list):
                obj_reprs[i] = [_norm(r) for r in obj_reprs[i]]
            else:
                obj_reprs[i] = _norm(obj_reprs[i])

    n_snapshots = len(subj_reprs[0])
    layer_labels = ["emb"] + [f"L{i+1}" for i in range(n_snapshots - 1)]
    print(f"  Snapshots per token: {n_snapshots}")
    print(f"  Filler representations: {len(filler_reprs)} (subject, relation) pairs")
    if args.normalize:
        print("  All representations L2-normalized to unit vectors")
    print()

    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_unemb" if args.use_unembedding else ""
    if args.apply_ln_f:
        suffix += "_lnf"
    if args.normalize:
        suffix += "_norm"
    results_lines = []

    def log(line=""):
        print(line)
        results_lines.append(line)

    # ══════════════════════════════════════════════════════════════════════
    # 2a. Single affine map from f2 → object (all relations pooled)
    # ══════════════════════════════════════════════════════════════════════
    log("=" * 70)
    log("2a. Single affine map: f2 → object (all relations pooled)")
    log("=" * 70)

    pooled_means = []
    pooled_stds = []
    subj_means = []  # for comparison

    header = f"{'Layer':>6} | {'f2 pooled CV Δcos':>20} | {'subj CV Δcos (avg)':>20}"
    log(header)
    log("-" * len(header))

    for layer in range(n_snapshots):
        F_all = build_pooled_object_matrix(obj_reprs, all_indices, cfg.vocab.num_relations, cfg.vocab, layer)
        # Wait — F_all is the object matrix. Let me name things clearly.
        F2_all, _ = stack_all_filler_representations(filler_reprs, all_indices, cfg.vocab.num_relations, layer)
        O_pooled = build_pooled_object_matrix(obj_reprs, all_indices, cfg.vocab.num_relations, cfg.vocab, layer)

        mean_dc, std_dc, _, _ = subject_fold_kfold(F2_all, O_pooled, k=args.kfolds)
        pooled_means.append(mean_dc)
        pooled_stds.append(std_dc)

        # Subject-position baseline (average across relations, same as v2 analysis)
        S_all, _ = stack_representations(subj_reprs, all_indices, layer)
        rel_dcs = []
        for ri in range(cfg.vocab.num_relations):
            O_rel = build_relation_object_matrix(obj_reprs, all_indices, ri, cfg.vocab, layer)
            m, _, _, _ = kfold_affine(S_all, O_rel, k=args.kfolds, bias=True)
            rel_dcs.append(m)
        subj_avg = np.mean(rel_dcs)
        subj_means.append(subj_avg)

        log(f"{layer_labels[layer]:>6} | {mean_dc:>+12.4f} ± {std_dc:.4f} | {subj_avg:>+12.4f}")

    log()

    # ══════════════════════════════════════════════════════════════════════
    # 2b. Per-relation affine maps at f2
    # ══════════════════════════════════════════════════════════════════════
    log("=" * 70)
    log("2b. Per-relation affine maps at f2")
    log("=" * 70)

    per_rel_means = {k: [] for k in RELATION_KEYS}
    per_rel_stds = {k: [] for k in RELATION_KEYS}

    header = f"{'Layer':>6} | " + " | ".join(f"{k+' f2 Δcos':>14}" for k in RELATION_KEYS)
    log(header)
    log("-" * len(header))

    for layer in range(n_snapshots):
        row = f"{layer_labels[layer]:>6}"
        for ri, rkey in enumerate(RELATION_KEYS):
            F2_rel = stack_filler_representations(filler_reprs, all_indices, ri, layer)
            O_rel = build_relation_object_matrix(obj_reprs, all_indices, ri, cfg.vocab, layer)
            mean_dc, std_dc, _, _ = kfold_affine(F2_rel, O_rel, k=args.kfolds, bias=True)
            per_rel_means[rkey].append(mean_dc)
            per_rel_stds[rkey].append(std_dc)
            row += f" | {mean_dc:>+8.4f} ± {std_dc:.4f}"
        log(row)

    log()

    # ══════════════════════════════════════════════════════════════════════
    # 2c. PCA dimensionality of f2 representations
    # ══════════════════════════════════════════════════════════════════════
    log("=" * 70)
    log("2c. PCA effective dimensionality of f2 representations")
    log("=" * 70)

    pca_results = []
    header = f"{'Layer':>6} | {'90%':>6} | {'95%':>6} | {'99%':>6} | {'n_points':>10}"
    log(header)
    log("-" * len(header))

    for layer in range(n_snapshots):
        F2_all, _ = stack_all_filler_representations(filler_reprs, all_indices, cfg.vocab.num_relations, layer)
        dims = pca_effective_dim(F2_all)
        pca_results.append(dims)
        log(f"{layer_labels[layer]:>6} | {dims[0.90]:>6} | {dims[0.95]:>6} | {dims[0.99]:>6} | {F2_all.shape[0]:>10}")

    log()

    # ── Interpretation ─────────────────────────────────────────────────
    best_pooled_layer = int(np.argmax(pooled_means))
    best_pooled_dc = pooled_means[best_pooled_layer]
    best_subj_dc = max(subj_means)

    log("=" * 70)
    log("Summary")
    log("=" * 70)
    log(f"Best f2 pooled Δcos: {best_pooled_dc:+.4f} at {layer_labels[best_pooled_layer]}")
    log(f"Best subject-pos Δcos (avg over relations): {best_subj_dc:+.4f}")
    log(f"Δcos improvement (f2 vs subject): {best_pooled_dc - best_subj_dc:+.4f}")
    log()

    best_per_rel = {k: max(per_rel_means[k]) for k in RELATION_KEYS}
    for rkey in RELATION_KEYS:
        bl = int(np.argmax(per_rel_means[rkey]))
        log(f"Best per-relation f2 Δcos for {rkey}: {best_per_rel[rkey]:+.4f} at {layer_labels[bl]}")
    log()

    pca_last = pca_results[-1]
    log(f"PCA dims at last layer ({layer_labels[-1]}): 90%={pca_last[0.90]}, 95%={pca_last[0.95]}, 99%={pca_last[0.99]} (of 300 points in {cfg.model.hidden_dim}-D)")
    log()

    if best_pooled_dc > 0 and best_subj_dc < 0:
        log("Interpretation: Linear structure emerges at f2, not at subject position.")
        log("The non-linearity at the subject position is real, not an artifact.")
    elif best_pooled_dc > 0 and best_subj_dc > 0:
        log("Interpretation: Linear structure at both positions (f2 stronger).")
    else:
        log("Interpretation: No linear structure at f2 either — mapping is non-linear throughout.")

    # Save results
    results_path = cfg.train.results_dir / f"filler_linearity_results{suffix}.md"
    with open(results_path, "w") as f:
        f.write("# Filler Linearity Analysis Results\n\n")
        f.write(f"*Experiment 3 from post_v2_analysis_plan.md — object repr: {obj_mode}*\n\n")
        f.write("```\n")
        f.write("\n".join(results_lines))
        f.write("\n```\n")
    print(f"Saved results to {results_path}")

    if args.no_plots:
        return

    # ── Plot 1: f2 pooled vs subject-position Δcos ────────────────────
    x = np.arange(n_snapshots)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, pooled_means, "o-", label="f2 position (pooled)", color="crimson")
    ax.fill_between(x,
                    np.array(pooled_means) - np.array(pooled_stds),
                    np.array(pooled_means) + np.array(pooled_stds),
                    alpha=0.15, color="crimson")
    ax.plot(x, subj_means, "s--", label="subject position (avg)", color="steelblue")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV Δcos (5-fold)")
    ax.set_title("f2-position vs Subject-position Linearity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.train.results_dir / f"filler_vs_subject_dcos{suffix}.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")

    # ── Plot 2: Per-relation f2 Δcos ──────────────────────────────────
    colors = {"R0": "steelblue", "R1": "darkorange", "R2": "seagreen"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for rkey in RELATION_KEYS:
        means = np.array(per_rel_means[rkey])
        stds = np.array(per_rel_stds[rkey])
        ri = RELATION_KEYS.index(rkey)
        ax.plot(x, means, "o-", label=RELATION_LABELS[ri], color=colors[rkey])
        ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=colors[rkey])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV Δcos (5-fold)")
    ax.set_title("Per-relation Δcos at f2 Position")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.train.results_dir / f"filler_per_relation_dcos{suffix}.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")

    # ── Plot 3: PCA effective dimensionality ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, style in [(0.90, "o-"), (0.95, "s--"), (0.99, "^:")]:
        dims = [pca_results[layer][t] for layer in range(n_snapshots)]
        ax.plot(x, dims, style, label=f"{int(t*100)}% variance")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of PCA components")
    ax.set_title("PCA Effective Dimensionality of f2 Representations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.train.results_dir / f"filler_pca_dimensionality{suffix}.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")

    plt.close("all")


if __name__ == "__main__":
    main()

"""
Linearity analysis on a trained model.

Usage (from vienna/):
    python experiments/analyze_linearity.py [--checkpoint PATH] [--device auto]

For each layer (0=embedding through n_layers):
  1. Extracts subject and object representations for all 100 pairs.
  2. Fits affine map with 5-fold cross-validation on training pairs (a0-a79).
  3. Evaluates delta_cos, reconstruction accuracy, MSE on:
       - held-out training pairs (CV eval set inside kfold)
       - heldout pairs (a80-a99) — zero-shot generalization
  4. Prints results table.
  5. Saves PCA plots and delta_cos-vs-layer figure to results/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.affine import fit_affine, kfold_affine, apply_affine
from src.config import Config, cfg as default_cfg
from src.evaluation import (
    delta_cos,
    full_report,
    unembedding_rank_accuracy,
    subject_direct_unembedding_rank,
)
from src.model import build_model
from src.representations import (
    get_all_object_representations,
    get_all_subject_representations,
    get_correct_object_ids,
    stack_representations,
)
from src.training import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = default_cfg
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.train.checkpoint_dir / "best.pt"

    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}. Run train_baseline.py first.")
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
    print(f"  Snapshots per pair: {n_snapshots} (embedding + {n_snapshots-1} layer(s))")
    print(f"  K-fold CV (k={args.kfolds}) splits pairs for affine fit/eval")
    print()

    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-layer analysis ──────────────────────────────────────────────────
    print(f"{'Layer':>6} | {'CV Δcos':>9} ± {'std':>6} | {'CV recon_acc':>12} | {'MSE (full)':>10}")
    print("-" * 60)

    layer_labels = ["emb"] + [f"L{i+1}" for i in range(n_snapshots - 1)]
    cv_means, cv_stds = [], []

    for layer in range(n_snapshots):
        S_all, _ = stack_representations(subj_reprs, all_indices, layer)
        O_all, _ = stack_representations(obj_reprs, all_indices, layer)

        mean_dc, std_dc, W_full, b_full = kfold_affine(
            S_all, O_all, k=args.kfolds, bias=True
        )
        report = full_report(S_all, O_all, W_full, b_full, O_all)

        # CV recon_acc: approximate via per-fold NN accuracy (reuse kfold logic)
        import numpy as np
        n = len(all_indices)
        fold_size = n // args.kfolds
        recon_accs = []
        for fold in range(args.kfolds):
            val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
            train_idx_fold = np.concatenate([np.arange(fold * fold_size), np.arange((fold + 1) * fold_size, n)])
            W_f, b_f = fit_affine(S_all[train_idx_fold], O_all[train_idx_fold], bias=True)
            from src.evaluation import reconstruction_accuracy
            acc = reconstruction_accuracy(S_all[val_idx], O_all[val_idx], W_f, b_f, O_all)
            recon_accs.append(acc)
        cv_recon_acc = float(np.mean(recon_accs))

        cv_means.append(mean_dc)
        cv_stds.append(std_dc)

        print(
            f"{layer_labels[layer]:>6} | {mean_dc:>9.4f} ± {std_dc:>6.4f} | "
            f"{cv_recon_acc:>12.4f} | {report['mse']:>10.5f}"
        )

    print()

    # ── Delta cos vs layer plot ─────────────────────────────────────────────
    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(n_snapshots)
        ax.plot(x, cv_means, "o-", label="CV Δcos (5-fold across pairs)", color="steelblue")
        ax.fill_between(
            x,
            np.array(cv_means) - np.array(cv_stds),
            np.array(cv_means) + np.array(cv_stds),
            alpha=0.2,
            color="steelblue",
        )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δcos")
        ax.set_title("Delta Cosine Similarity per Layer")
        ax.legend()
        fig.tight_layout()
        out = cfg.train.results_dir / "delta_cos_per_layer.pdf"
        fig.savefig(out)
        print(f"Saved: {out}")

        # ── PCA plots for best layer ────────────────────────────────────────
        best_layer = int(np.argmax(cv_means))
        print(f"Best layer by CV Δcos: {layer_labels[best_layer]}")

        S_all, _ = stack_representations(subj_reprs, all_indices, best_layer)
        O_all, _ = stack_representations(obj_reprs, all_indices, best_layer)

        pca = PCA(n_components=2)
        SO = torch.cat([S_all, O_all], dim=0).numpy()
        SO_2d = pca.fit_transform(SO)
        S_2d, O_2d = SO_2d[:100], SO_2d[100:]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(S_2d[:, 0], S_2d[:, 1], c="steelblue", s=30, label="subjects", alpha=0.8)
        ax.scatter(O_2d[:, 0], O_2d[:, 1], c="tomato", s=30, label="objects", alpha=0.8)
        for i in range(0, 100, 10):
            ax.annotate(f"a{i}", S_2d[i], fontsize=7, alpha=0.7)
            ax.annotate(f"b{i}", O_2d[i], fontsize=7, alpha=0.7)
        ax.set_title(f"PCA of subject & object representations (layer {layer_labels[best_layer]})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = cfg.train.results_dir / f"pca_layer{best_layer}.pdf"
        fig.savefig(out)
        print(f"Saved: {out}")

        plt.close("all")

    # ── Per-layer unembedding sweep ─────────────────────────────────────────
    correct_ids = get_correct_object_ids(cfg, all_indices)

    hdr = (
        f"{'Layer':>6} | {'ô top1':>7} | {'ô top5':>7} | {'ô top10':>8} | "
        f"{'ô rank':>8} | {'s top1':>7} | {'s rank':>8} | {'CV top1':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    unemb_top1_ohat = []
    unemb_top1_s = []

    for layer in range(n_snapshots):
        S_all, _ = stack_representations(subj_reprs, all_indices, layer)
        O_all_l, _ = stack_representations(obj_reprs, all_indices, layer)
        W_full, b_full = fit_affine(S_all, O_all_l, bias=True)

        m_ohat = unembedding_rank_accuracy(S_all, correct_ids, W_full, b_full, model)
        m_s = subject_direct_unembedding_rank(S_all, correct_ids, model)

        # 5-fold CV unembedding: fit affine on 4/5, evaluate on 1/5
        n = len(all_indices)
        fold_size = n // args.kfolds
        cv_top1_scores = []
        for fold in range(args.kfolds):
            val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
            train_idx = np.concatenate(
                [np.arange(fold * fold_size), np.arange((fold + 1) * fold_size, n)]
            )
            W_f, b_f = fit_affine(S_all[train_idx], O_all_l[train_idx], bias=True)
            correct_val = [correct_ids[i] for i in val_idx]
            m_cv = unembedding_rank_accuracy(S_all[val_idx], correct_val, W_f, b_f, model)
            cv_top1_scores.append(m_cv["topk_acc/1"])
        cv_top1_mean = float(np.mean(cv_top1_scores))

        unemb_top1_ohat.append(m_ohat["topk_acc/1"])
        unemb_top1_s.append(m_s["topk_acc/1"])

        print(
            f"{layer_labels[layer]:>6} | {m_ohat['topk_acc/1']:>7.4f} | "
            f"{m_ohat['topk_acc/5']:>7.4f} | {m_ohat['topk_acc/10']:>8.4f} | "
            f"{m_ohat['mean_rank']:>8.1f} | {m_s['topk_acc/1']:>7.4f} | "
            f"{m_s['mean_rank']:>8.1f} | {cv_top1_mean:>8.4f}"
        )

    print()

    # ── Unembedding top-1 accuracy vs layer plot ────────────────────────────
    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(n_snapshots)
        ax.plot(x, unemb_top1_ohat, "o-", label="ô top-1 (affine)", color="steelblue")
        ax.plot(x, unemb_top1_s, "s--", label="s top-1 (no affine, baseline)", color="tomato")
        ax.axhline(1.0 / cfg.vocab.vocab_size, color="gray", linewidth=0.8, linestyle=":", label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Top-1 accuracy (unembedding)")
        ax.set_title("Unembedding Top-1 Accuracy per Layer")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = cfg.train.results_dir / "unembedding_rank_per_layer.pdf"
        fig.savefig(out)
        print(f"Saved: {out}")
        plt.close("all")


if __name__ == "__main__":
    main()

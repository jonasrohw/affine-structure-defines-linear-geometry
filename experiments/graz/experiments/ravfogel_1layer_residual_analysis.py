"""
Follow-up diagnostics: relationship between MLP computation and affine structure.

1. Affine fit from h_pre_mlp → h_final (R² of the full residual block)
2. Norm comparison: ||h_pre_mlp|| vs ||mlp_out||
3. Decomposition of affine map: W·h_pre_mlp vs W·mlp_out contributions

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_1layer_residual_analysis.py --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import build_model
from src.affine import fit_affine, apply_affine, reconstruction_accuracy
from experiments.ravfogel_1layer import RavfogelVocab, generate_permutations

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def load_checkpoint(seed, device):
    ckpt_dir = ROOT / f"checkpoints_ravfogel_1l_seed{seed}"
    ckpt = torch.load(ckpt_dir / "final.pt", map_location=device, weights_only=False)
    vocab = RavfogelVocab(n_entities=2025, num_relations=2)
    for k, p in enumerate(ckpt["permutations"]):
        vocab.set_permutation(k, p)
    cfg = ckpt["model_cfg"]
    model_cfg = ModelConfig(n_layers=cfg["n_layers"], hidden_dim=cfg["hidden_dim"],
                            n_heads=cfg["n_heads"])
    model = build_model(model_cfg, vocab)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, vocab, model_cfg


@torch.no_grad()
def extract_states(model, vocab, device, batch_size=512):
    """Extract h_pre_mlp, mlp_out, h_final at subject position for all (i, k)."""
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    block = model.blocks[0]

    h_pre_mlp = {}
    mlp_out = {}
    h_final = {}

    for start in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[start:start + batch_size]
        contexts = torch.tensor(
            [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)]
             for i, k in batch_pairs],
            dtype=torch.long, device=device)

        B, T = contexts.shape
        positions = torch.arange(T, device=device).unsqueeze(0)
        h0 = model.token_emb(contexts) + model.pos_emb(positions)

        attn_out = block.attn(block.ln1(h0))
        h1 = h0 + attn_out  # h_pre_mlp

        mlp_input = block.ln2(h1)
        m_out = block.mlp(mlp_input)
        h2 = h1 + m_out  # h_final

        for idx, (i, k) in enumerate(batch_pairs):
            h_pre_mlp[(i, k)] = h1[idx, 2, :].cpu()
            mlp_out[(i, k)] = m_out[idx, 2, :].cpu()
            h_final[(i, k)] = h2[idx, 2, :].cpu()

    return h_pre_mlp, mlp_out, h_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, vocab, model_cfg = load_checkpoint(args.seed, device)
    lm_head = model.lm_head.weight.detach().cpu()
    O_all = lm_head[vocab.object_offset:vocab.object_offset + vocab.n_entities]

    print("Extracting hidden states...")
    h_pre_mlp, mlp_out, h_final = extract_states(model, vocab, device)

    train_subjs = vocab.train_subject_indices
    test_subjs = vocab.test_subject_indices

    results = {}

    # =========================================================================
    # DIAGNOSTIC 1: Affine fit h_pre_mlp → h_final
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: Is the full residual block affine?")
    print("  Fitting affine map: h_pre_mlp → h_final")
    print("=" * 70)

    for k in range(vocab.num_relations):
        X = torch.stack([h_pre_mlp[(i, k)] for i in range(vocab.n_entities)])
        Y = torch.stack([h_final[(i, k)] for i in range(vocab.n_entities)])

        n = X.shape[0]
        X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)
        sol, _, _, _ = torch.linalg.lstsq(X_aug, Y)
        Y_pred = X_aug @ sol

        ss_res = ((Y - Y_pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(0)) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot

        cos = F.cosine_similarity(Y, Y_pred, dim=1)
        rel_err = (Y - Y_pred).norm(dim=1) / (Y.norm(dim=1) + 1e-8)

        print(f"\n  R{k}: h_pre_mlp → h_final")
        print(f"    R² = {r2:.6f}")
        print(f"    Per-entity cosine: mean={cos.mean():.6f}, min={cos.min():.4f}")
        print(f"    Per-entity rel error: mean={rel_err.mean():.4f}")

        # Check if the affine map is close to identity + small correction
        W_fit = sol[:-1, :]  # (64, 64)
        b_fit = sol[-1, :]   # (64,)
        identity = torch.eye(64)
        deviation = W_fit - identity
        print(f"    ||W_fit - I||_F = {deviation.norm():.4f}")
        print(f"    ||W_fit||_F = {W_fit.norm():.4f}")
        print(f"    ||b_fit|| = {b_fit.norm():.4f}")

        results[f"r{k}_block_affine"] = {
            "r2": round(r2, 6),
            "cos_mean": round(cos.mean().item(), 6),
            "cos_min": round(cos.min().item(), 4),
            "rel_error": round(rel_err.mean().item(), 4),
            "W_minus_I_norm": round(deviation.norm().item(), 4),
            "b_norm": round(b_fit.norm().item(), 4),
        }

    # =========================================================================
    # DIAGNOSTIC 2: Norm comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: Relative magnitudes")
    print("=" * 70)

    for k in range(vocab.num_relations):
        pre_norms = torch.stack([h_pre_mlp[(i, k)].norm() for i in range(vocab.n_entities)])
        mlp_norms = torch.stack([mlp_out[(i, k)].norm() for i in range(vocab.n_entities)])
        final_norms = torch.stack([h_final[(i, k)].norm() for i in range(vocab.n_entities)])
        ratio = mlp_norms / (pre_norms + 1e-8)

        # Cosine between h_pre_mlp and mlp_out
        pre_vecs = torch.stack([h_pre_mlp[(i, k)] for i in range(vocab.n_entities)])
        mlp_vecs = torch.stack([mlp_out[(i, k)] for i in range(vocab.n_entities)])
        cos_pre_mlp = F.cosine_similarity(pre_vecs, mlp_vecs, dim=1)

        print(f"\n  R{k}:")
        print(f"    ||h_pre_mlp||: mean={pre_norms.mean():.3f}, std={pre_norms.std():.3f}")
        print(f"    ||mlp_out||:   mean={mlp_norms.mean():.3f}, std={mlp_norms.std():.3f}")
        print(f"    ||h_final||:   mean={final_norms.mean():.3f}, std={final_norms.std():.3f}")
        print(f"    ||mlp_out|| / ||h_pre_mlp||: mean={ratio.mean():.3f}, std={ratio.std():.3f}")
        print(f"    cos(h_pre_mlp, mlp_out): mean={cos_pre_mlp.mean():.3f}, std={cos_pre_mlp.std():.3f}")

        results[f"r{k}_norms"] = {
            "h_pre_mlp_mean": round(pre_norms.mean().item(), 3),
            "mlp_out_mean": round(mlp_norms.mean().item(), 3),
            "h_final_mean": round(final_norms.mean().item(), 3),
            "ratio_mean": round(ratio.mean().item(), 3),
            "ratio_std": round(ratio.std().item(), 3),
            "cos_pre_mlp_mean": round(cos_pre_mlp.mean().item(), 3),
        }

    # =========================================================================
    # DIAGNOSTIC 3: Decompose the affine map
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: Affine map decomposition")
    print("  W · h_final + b = W · h_pre_mlp + W · mlp_out + b")
    print("=" * 70)

    for k in range(vocab.num_relations):
        # Fit the affine map from h_final to target object embedding (same as the original affine analysis)
        S_train = torch.stack([h_final[(i, k)] for i in train_subjs])
        S_test = torch.stack([h_final[(i, k)] for i in test_subjs])

        target_train = torch.stack([lm_head[vocab.object_id(vocab.relation_object(i, k))]
                                    for i in train_subjs])
        target_test = torch.stack([lm_head[vocab.object_id(vocab.relation_object(i, k))]
                                   for i in test_subjs])

        W, b = fit_affine(S_train, target_train, bias=True)

        train_acc = reconstruction_accuracy(S_train, target_train, W, b, O_all)
        test_acc = reconstruction_accuracy(S_test, target_test, W, b, O_all)
        print(f"\n  R{k} affine map (h_final → target):")
        print(f"    Train Voronoi: {train_acc:.4f}")
        print(f"    Test Voronoi:  {test_acc:.4f}")

        # Now decompose: for test subjects, compute W·h_pre_mlp + b and W·mlp_out separately
        pre_test = torch.stack([h_pre_mlp[(i, k)] for i in test_subjs])
        mlp_test = torch.stack([mlp_out[(i, k)] for i in test_subjs])

        contribution_pre = pre_test @ W.T + b  # W · h_pre_mlp + b
        contribution_mlp = mlp_test @ W.T       # W · mlp_out (no bias — attributed to pre)
        full_pred = contribution_pre + contribution_mlp  # = W · h_final + b

        # Verify decomposition
        direct_pred = apply_affine(S_test, W, b)
        decomp_err = (full_pred - direct_pred).abs().max().item()
        print(f"    Decomposition error: {decomp_err:.2e}")

        # How close is each contribution to the target?
        cos_pre_target = F.cosine_similarity(contribution_pre, target_test, dim=1)
        cos_mlp_target = F.cosine_similarity(contribution_mlp, target_test, dim=1)
        cos_full_target = F.cosine_similarity(full_pred, target_test, dim=1)

        print(f"\n    Cosine with target:")
        print(f"      W·h_pre_mlp + b:  mean={cos_pre_target.mean():.4f}")
        print(f"      W·mlp_out:        mean={cos_mlp_target.mean():.4f}")
        print(f"      Full (W·h_final+b): mean={cos_full_target.mean():.4f}")

        # Voronoi accuracy of each contribution alone
        acc_pre_only = reconstruction_accuracy(pre_test, target_test, W, b, O_all)
        # For mlp-only, we need a separate affine fit or use W with zero input offset
        # Actually, let's just check nearest neighbor of each contribution
        def voronoi_acc(predicted, targets, all_objects):
            correct = 0
            for idx in range(predicted.shape[0]):
                dists = (all_objects - predicted[idx]).norm(dim=1)
                nn_idx = dists.argmin().item()
                target_idx = (all_objects - targets[idx]).norm(dim=1).argmin().item()
                correct += int(nn_idx == target_idx)
            return correct / predicted.shape[0]

        acc_pre = voronoi_acc(contribution_pre, target_test, O_all)
        acc_full = voronoi_acc(full_pred, target_test, O_all)

        print(f"\n    Voronoi accuracy:")
        print(f"      W·h_pre_mlp + b alone: {acc_pre:.4f}")
        print(f"      Full W·h_final + b:     {acc_full:.4f}")

        # Norm decomposition: how much of ||full_pred - bias_direction|| comes from each?
        norm_pre = contribution_pre.norm(dim=1)
        norm_mlp = contribution_mlp.norm(dim=1)
        norm_full = full_pred.norm(dim=1)

        print(f"\n    Norms:")
        print(f"      ||W·h_pre_mlp + b||: mean={norm_pre.mean():.3f}")
        print(f"      ||W·mlp_out||:       mean={norm_mlp.mean():.3f}")
        print(f"      ||W·h_final + b||:   mean={norm_full.mean():.3f}")

        # Variance decomposition: what fraction of the variance in the full
        # prediction is due to h_pre_mlp vs mlp_out?
        full_centered = full_pred - full_pred.mean(0)
        pre_centered = contribution_pre - contribution_pre.mean(0)
        mlp_centered = contribution_mlp - contribution_mlp.mean(0)

        var_full = (full_centered ** 2).sum().item()
        var_pre = (pre_centered ** 2).sum().item()
        var_mlp = (mlp_centered ** 2).sum().item()
        covar = 2 * (pre_centered * mlp_centered).sum().item()

        print(f"\n    Variance decomposition of full prediction:")
        print(f"      Var(W·h_pre_mlp): {var_pre / var_full:.3f} ({var_pre / var_full * 100:.1f}%)")
        print(f"      Var(W·mlp_out):   {var_mlp / var_full:.3f} ({var_mlp / var_full * 100:.1f}%)")
        print(f"      2·Cov:            {covar / var_full:.3f} ({covar / var_full * 100:.1f}%)")

        results[f"r{k}_decomposition"] = {
            "train_voronoi": round(train_acc, 4),
            "test_voronoi": round(test_acc, 4),
            "cos_pre_target": round(cos_pre_target.mean().item(), 4),
            "cos_mlp_target": round(cos_mlp_target.mean().item(), 4),
            "cos_full_target": round(cos_full_target.mean().item(), 4),
            "voronoi_pre_only": round(acc_pre, 4),
            "voronoi_full": round(acc_full, 4),
            "var_frac_pre": round(var_pre / var_full, 3),
            "var_frac_mlp": round(var_mlp / var_full, 3),
            "var_frac_covar": round(covar / var_full, 3),
        }

    # =========================================================================
    # BONUS: Fit affine from h_pre_mlp directly to target (without MLP)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BONUS: Affine fit from h_pre_mlp directly to target")
    print("  (same as layer 0 affine analysis but with explicit R²)")
    print("=" * 70)

    for k in range(vocab.num_relations):
        S_train = torch.stack([h_pre_mlp[(i, k)] for i in train_subjs])
        S_test = torch.stack([h_pre_mlp[(i, k)] for i in test_subjs])
        target_train = torch.stack([lm_head[vocab.object_id(vocab.relation_object(i, k))]
                                    for i in train_subjs])
        target_test = torch.stack([lm_head[vocab.object_id(vocab.relation_object(i, k))]
                                   for i in test_subjs])

        W0, b0 = fit_affine(S_train, target_train, bias=True)
        train_acc = reconstruction_accuracy(S_train, target_train, W0, b0, O_all)
        test_acc = reconstruction_accuracy(S_test, target_test, W0, b0, O_all)

        pred_test = apply_affine(S_test, W0, b0)
        ss_res = ((target_test - pred_test) ** 2).sum().item()
        ss_tot = ((target_test - target_test.mean(0)) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot

        cos = F.cosine_similarity(pred_test, target_test, dim=1)

        print(f"\n  R{k}: h_pre_mlp → target object embedding")
        print(f"    Train Voronoi: {train_acc:.4f}")
        print(f"    Test Voronoi:  {test_acc:.4f}")
        print(f"    R²: {r2:.4f}")
        print(f"    Per-entity cosine: mean={cos.mean():.4f}, min={cos.min():.4f}")

        results[f"r{k}_pre_mlp_to_target"] = {
            "train_voronoi": round(train_acc, 4),
            "test_voronoi": round(test_acc, 4),
            "r2": round(r2, 4),
            "cos_mean": round(cos.mean().item(), 4),
        }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_1layer_residual_analysis_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

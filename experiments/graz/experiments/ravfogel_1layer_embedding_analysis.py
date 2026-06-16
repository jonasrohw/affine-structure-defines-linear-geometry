"""
Check the relationship between learned subject embeddings and target object embeddings.

Key question: is emb(subj_i) ≈ α·o_{π_0(i)} + β·o_{π_1(i)}?
If so, the MLP's job is to "select" the right component based on the relation token.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_1layer_embedding_analysis.py --seed 42
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
from experiments.ravfogel_1layer import RavfogelVocab

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, model_cfg = load_checkpoint(args.seed, device)

    emb = model.token_emb.weight.detach().cpu()  # (V, 64)
    lm_head = model.lm_head.weight.detach().cpu()  # (V, 64)

    n = vocab.n_entities
    results = {}

    # Extract subject embeddings and target object lm_head rows
    S = emb[vocab.subject_offset:vocab.subject_offset + n]  # (2025, 64)
    O_emb = emb[vocab.object_offset:vocab.object_offset + n]  # (2025, 64) token_emb rows
    O_head = lm_head[vocab.object_offset:vocab.object_offset + n]  # (2025, 64) lm_head rows

    # For each subject i, get the two target object indices
    t0 = [vocab.relation_object(i, 0) for i in range(n)]  # π_0(i)
    t1 = [vocab.relation_object(i, 1) for i in range(n)]  # π_1(i)

    O0 = O_head[t0]  # (2025, 64) — lm_head row of R0 target for each subject
    O1 = O_head[t1]  # (2025, 64) — lm_head row of R1 target for each subject

    # =========================================================================
    # Part 1: Basic cosine similarities
    # =========================================================================
    print("=" * 70)
    print("PART 1: Cosine similarity between subject emb and target object emb")
    print("=" * 70)

    cos_s_o0 = F.cosine_similarity(S, O0, dim=1)
    cos_s_o1 = F.cosine_similarity(S, O1, dim=1)
    cos_o0_o1 = F.cosine_similarity(O0, O1, dim=1)

    print(f"\n  cos(emb(subj_i), lm_head[obj_π0(i)]): mean={cos_s_o0.mean():.4f}, std={cos_s_o0.std():.4f}")
    print(f"  cos(emb(subj_i), lm_head[obj_π1(i)]): mean={cos_s_o1.mean():.4f}, std={cos_s_o1.std():.4f}")
    print(f"  cos(lm_head[obj_π0(i)], lm_head[obj_π1(i)]): mean={cos_o0_o1.mean():.4f}, std={cos_o0_o1.std():.4f}")

    # Compare with random baselines
    rng = np.random.RandomState(42)
    rand_idx = rng.choice(n, n, replace=True)
    cos_s_rand = F.cosine_similarity(S, O_head[rand_idx], dim=1)
    print(f"\n  cos(emb(subj_i), lm_head[random_obj]): mean={cos_s_rand.mean():.4f}, std={cos_s_rand.std():.4f}")

    results["cosine"] = {
        "subj_to_r0_target": {"mean": round(cos_s_o0.mean().item(), 4), "std": round(cos_s_o0.std().item(), 4)},
        "subj_to_r1_target": {"mean": round(cos_s_o1.mean().item(), 4), "std": round(cos_s_o1.std().item(), 4)},
        "r0_target_to_r1_target": {"mean": round(cos_o0_o1.mean().item(), 4), "std": round(cos_o0_o1.std().item(), 4)},
        "subj_to_random_obj": {"mean": round(cos_s_rand.mean().item(), 4), "std": round(cos_s_rand.std().item(), 4)},
    }

    # =========================================================================
    # Part 2: Superposition test — is S_i ≈ α·O0_i + β·O1_i ?
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Superposition test")
    print("  Is emb(subj_i) ≈ α·lm_head[obj_π0(i)] + β·lm_head[obj_π1(i)] ?")
    print("=" * 70)

    # Per-subject least-squares fit: S_i = α_i·O0_i + β_i·O1_i
    # For each subject, solve [O0_i, O1_i] @ [α, β]^T = S_i
    alphas = []
    betas = []
    r2_per_subject = []
    for i in range(n):
        A = torch.stack([O0[i], O1[i]], dim=1)  # (64, 2)
        coeffs, _, _, _ = torch.linalg.lstsq(A, S[i].unsqueeze(1))  # (2, 1)
        alpha, beta = coeffs[0, 0].item(), coeffs[1, 0].item()
        alphas.append(alpha)
        betas.append(beta)
        pred = alpha * O0[i] + beta * O1[i]
        ss_res = ((S[i] - pred) ** 2).sum().item()
        ss_tot = ((S[i] - S[i].mean()) ** 2).sum().item()
        r2_per_subject.append(1 - ss_res / max(ss_tot, 1e-10))

    alphas = np.array(alphas)
    betas = np.array(betas)
    r2_per_subject = np.array(r2_per_subject)

    print(f"\n  Per-subject fit: S_i = α_i·O0_i + β_i·O1_i")
    print(f"    α: mean={alphas.mean():.4f}, std={alphas.std():.4f}")
    print(f"    β: mean={betas.mean():.4f}, std={betas.std():.4f}")
    print(f"    R² (per-subject): mean={r2_per_subject.mean():.4f}, median={np.median(r2_per_subject):.4f}, "
          f"min={r2_per_subject.min():.4f}")

    results["superposition_per_subject"] = {
        "alpha_mean": round(float(alphas.mean()), 4),
        "alpha_std": round(float(alphas.std()), 4),
        "beta_mean": round(float(betas.mean()), 4),
        "beta_std": round(float(betas.std()), 4),
        "r2_mean": round(float(r2_per_subject.mean()), 4),
        "r2_median": round(float(np.median(r2_per_subject)), 4),
        "r2_min": round(float(r2_per_subject.min()), 4),
    }

    # Global fit: find single α, β that works for all subjects
    # S = α·O0 + β·O1  (same α, β for all subjects)
    # Flatten: S_flat = α·O0_flat + β·O1_flat
    S_flat = S.reshape(-1)
    O0_flat = O0.reshape(-1)
    O1_flat = O1.reshape(-1)
    A_flat = torch.stack([O0_flat, O1_flat], dim=1)
    coeffs_global, _, _, _ = torch.linalg.lstsq(A_flat, S_flat.unsqueeze(1))
    alpha_g, beta_g = coeffs_global[0, 0].item(), coeffs_global[1, 0].item()

    pred_global = alpha_g * O0 + beta_g * O1
    ss_res_g = ((S - pred_global) ** 2).sum().item()
    ss_tot_g = ((S - S.mean(0)) ** 2).sum().item()
    r2_global = 1 - ss_res_g / ss_tot_g

    cos_global = F.cosine_similarity(S, pred_global, dim=1)

    print(f"\n  Global fit: S = α·O0 + β·O1 (single α, β for all subjects)")
    print(f"    α = {alpha_g:.4f}, β = {beta_g:.4f}")
    print(f"    R² = {r2_global:.4f}")
    print(f"    Per-entity cosine: mean={cos_global.mean():.4f}, min={cos_global.min():.4f}")

    results["superposition_global"] = {
        "alpha": round(alpha_g, 4),
        "beta": round(beta_g, 4),
        "r2": round(r2_global, 4),
        "cos_mean": round(cos_global.mean().item(), 4),
    }

    # =========================================================================
    # Part 3: Richer basis — S_i ≈ α·O0_i + β·O1_i + γ·(mean_S) + ...
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: How much of S is explained by O0 and O1?")
    print("  Regression: S = O0·w0 + O1·w1 + bias  (matrix regression)")
    print("=" * 70)

    # Full matrix regression: for all subjects jointly, S_i = W0·O0_i + W1·O1_i + b
    # Stack: X = [O0, O1, 1] → (2025, 129), Y = S → (2025, 64)
    X = torch.cat([O0, O1, torch.ones(n, 1)], dim=1)  # (2025, 129)
    sol, _, _, _ = torch.linalg.lstsq(X, S)
    S_pred = X @ sol

    ss_res = ((S - S_pred) ** 2).sum().item()
    ss_tot = ((S - S.mean(0)) ** 2).sum().item()
    r2_matrix = 1 - ss_res / ss_tot

    cos_matrix = F.cosine_similarity(S, S_pred, dim=1)

    print(f"\n  Matrix regression: S_i = W0·O0_i + W1·O1_i + b")
    print(f"    R² = {r2_matrix:.4f}")
    print(f"    Per-entity cosine: mean={cos_matrix.mean():.4f}, min={cos_matrix.min():.4f}")

    results["matrix_regression"] = {
        "r2": round(r2_matrix, 4),
        "cos_mean": round(cos_matrix.mean().item(), 4),
        "cos_min": round(cos_matrix.min().item(), 4),
    }

    # =========================================================================
    # Part 4: How much of subject embedding space is spanned by object embeddings?
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: Subspace analysis")
    print("=" * 70)

    # SVD of object embeddings
    O_centered = O_head - O_head.mean(0)
    U_o, S_o, Vh_o = torch.linalg.svd(O_centered, full_matrices=False)
    obj_var = S_o ** 2
    obj_cumvar = obj_var.cumsum(0) / obj_var.sum()

    S_centered = S - S.mean(0)
    U_s, S_s, Vh_s = torch.linalg.svd(S_centered, full_matrices=False)
    subj_var = S_s ** 2
    subj_cumvar = subj_var.cumsum(0) / subj_var.sum()

    print(f"\n  Object embedding (lm_head) dimensionality:")
    print(f"    90% variance in {(obj_cumvar < 0.9).sum().item() + 1} components")
    print(f"    95% variance in {(obj_cumvar < 0.95).sum().item() + 1} components")
    print(f"    99% variance in {(obj_cumvar < 0.99).sum().item() + 1} components")

    print(f"\n  Subject embedding dimensionality:")
    print(f"    90% variance in {(subj_cumvar < 0.9).sum().item() + 1} components")
    print(f"    95% variance in {(subj_cumvar < 0.95).sum().item() + 1} components")
    print(f"    99% variance in {(subj_cumvar < 0.99).sum().item() + 1} components")

    # Projection of subject embeddings onto object subspace
    top_k = 50
    obj_basis = Vh_o[:top_k]  # (k, 64)
    S_proj = S_centered @ obj_basis.T @ obj_basis  # project onto object subspace
    var_explained = (S_proj ** 2).sum().item() / (S_centered ** 2).sum().item()
    print(f"\n  Subject variance in top-{top_k} object PCs: {var_explained:.4f}")

    top_k = 30
    obj_basis = Vh_o[:top_k]
    S_proj = S_centered @ obj_basis.T @ obj_basis
    var_explained_30 = (S_proj ** 2).sum().item() / (S_centered ** 2).sum().item()
    print(f"  Subject variance in top-{top_k} object PCs: {var_explained_30:.4f}")

    # Principal angle between subject and object subspaces
    top_s = Vh_s[:30]
    top_o = Vh_o[:30]
    cos_angles = torch.linalg.svdvals(top_s @ top_o.T)
    print(f"\n  Principal angles between top-30 subj and obj PCs:")
    print(f"    cos(angles): {cos_angles[:5].tolist()}")
    print(f"    Mean cos: {cos_angles.mean():.4f}")
    print(f"    Min cos: {cos_angles.min():.4f}")

    results["subspace"] = {
        "obj_dim_90": int((obj_cumvar < 0.9).sum().item() + 1),
        "subj_dim_90": int((subj_cumvar < 0.9).sum().item() + 1),
        "subj_var_in_obj_top50": round(var_explained, 4),
        "subj_var_in_obj_top30": round(var_explained_30, 4),
        "principal_cos_mean": round(cos_angles.mean().item(), 4),
    }

    # =========================================================================
    # Part 5: Nearest-neighbor analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: Nearest-neighbor — is emb(subj_i) nearest to its target objects?")
    print("=" * 70)

    # For each subject, find nearest object in lm_head space
    # Check if it's π_0(i) or π_1(i)
    correct_r0 = 0
    correct_r1 = 0
    correct_either = 0
    for i in range(n):
        dists = (O_head - S[i]).norm(dim=1)
        nn = dists.argmin().item()
        if nn == t0[i]:
            correct_r0 += 1
            correct_either += 1
        elif nn == t1[i]:
            correct_r1 += 1
            correct_either += 1

    print(f"\n  Nearest object to emb(subj_i) in lm_head space:")
    print(f"    Is π_0(i): {correct_r0}/{n} ({correct_r0/n:.3f})")
    print(f"    Is π_1(i): {correct_r1}/{n} ({correct_r1/n:.3f})")
    print(f"    Is either:  {correct_either}/{n} ({correct_either/n:.3f})")

    results["nearest_neighbor"] = {
        "is_r0_target": round(correct_r0 / n, 4),
        "is_r1_target": round(correct_r1 / n, 4),
        "is_either": round(correct_either / n, 4),
    }

    # =========================================================================
    # Part 6: token_emb vs lm_head — are they the same?
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 6: token_emb vs lm_head (embeddings are NOT tied)")
    print("=" * 70)

    # For subjects
    S_head = lm_head[vocab.subject_offset:vocab.subject_offset + n]
    cos_emb_head_subj = F.cosine_similarity(S, S_head, dim=1)
    print(f"\n  cos(token_emb[subj_i], lm_head[subj_i]): mean={cos_emb_head_subj.mean():.4f}")

    # For objects
    cos_emb_head_obj = F.cosine_similarity(O_emb, O_head, dim=1)
    print(f"  cos(token_emb[obj_j], lm_head[obj_j]):  mean={cos_emb_head_obj.mean():.4f}")

    # Subject token_emb vs target object lm_head
    cos_subj_emb_obj_head_r0 = F.cosine_similarity(S, O_head[t0], dim=1)
    cos_subj_emb_obj_head_r1 = F.cosine_similarity(S, O_head[t1], dim=1)
    print(f"\n  cos(token_emb[subj_i], lm_head[obj_π0(i)]): mean={cos_subj_emb_obj_head_r0.mean():.4f}")
    print(f"  cos(token_emb[subj_i], lm_head[obj_π1(i)]): mean={cos_subj_emb_obj_head_r1.mean():.4f}")

    # Subject lm_head vs target object lm_head
    cos_subj_head_obj_head_r0 = F.cosine_similarity(S_head, O_head[t0], dim=1)
    cos_subj_head_obj_head_r1 = F.cosine_similarity(S_head, O_head[t1], dim=1)
    print(f"  cos(lm_head[subj_i], lm_head[obj_π0(i)]): mean={cos_subj_head_obj_head_r0.mean():.4f}")
    print(f"  cos(lm_head[subj_i], lm_head[obj_π1(i)]): mean={cos_subj_head_obj_head_r1.mean():.4f}")

    results["emb_vs_head"] = {
        "subj_emb_head_cos": round(cos_emb_head_subj.mean().item(), 4),
        "obj_emb_head_cos": round(cos_emb_head_obj.mean().item(), 4),
        "subj_emb_to_r0_obj_head": round(cos_subj_emb_obj_head_r0.mean().item(), 4),
        "subj_emb_to_r1_obj_head": round(cos_subj_emb_obj_head_r1.mean().item(), 4),
    }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_1layer_embedding_analysis_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

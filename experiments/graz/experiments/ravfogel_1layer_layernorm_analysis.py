"""
Test whether the affine structure is explained by LayerNorm being approximately affine.

If ln_f has approximately constant μ and σ across inputs, then the readout
lm_head ∘ ln_f is approximately affine, and 100% model accuracy automatically
implies 100% affine Voronoi — regardless of how complex the internal computation is.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_1layer_layernorm_analysis.py --seed 42
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


@torch.no_grad()
def extract_states(model, vocab, device, batch_size=512):
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    block = model.blocks[0]

    h_final = {}
    h_after_ln = {}

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
        h1 = h0 + attn_out
        mlp_out = block.mlp(block.ln2(h1))
        h2 = h1 + mlp_out
        z = model.ln_f(h2)

        for idx, (i, k) in enumerate(batch_pairs):
            h_final[(i, k)] = h2[idx, 2, :].cpu()
            h_after_ln[(i, k)] = z[idx, 2, :].cpu()

    return h_final, h_after_ln


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, model_cfg = load_checkpoint(args.seed, device)
    lm_head = model.lm_head.weight.detach().cpu()
    ln_f_gamma = model.ln_f.weight.detach().cpu()
    ln_f_beta = model.ln_f.bias.detach().cpu()
    O_all = lm_head[vocab.object_offset:vocab.object_offset + vocab.n_entities]

    train_subjs = vocab.train_subject_indices
    test_subjs = vocab.test_subject_indices

    print("Extracting hidden states...")
    h_final, h_after_ln = extract_states(model, vocab, device)

    results = {}

    # =========================================================================
    # Part 1: How constant are μ and σ across h_final vectors?
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: LayerNorm statistics across inputs")
    print("=" * 70)

    for k in range(vocab.num_relations):
        H = torch.stack([h_final[(i, k)] for i in range(vocab.n_entities)])  # (2025, 64)
        # Per-example mean and std (what LayerNorm computes)
        mu = H.mean(dim=1)    # (2025,)
        sigma = H.std(dim=1)  # (2025,)

        print(f"\n  R{k}:")
        print(f"    μ across entities: mean={mu.mean():.6f}, std={mu.std():.6f}, "
              f"range=[{mu.min():.6f}, {mu.max():.6f}]")
        print(f"    σ across entities: mean={sigma.mean():.6f}, std={sigma.std():.6f}, "
              f"range=[{sigma.min():.6f}, {sigma.max():.6f}]")
        print(f"    Coefficient of variation of μ: {mu.std() / (mu.mean().abs() + 1e-10):.3f}")
        print(f"    Coefficient of variation of σ: {sigma.std() / sigma.mean():.4f}")

        results[f"r{k}_ln_stats"] = {
            "mu_mean": round(mu.mean().item(), 6),
            "mu_std": round(mu.std().item(), 6),
            "mu_range": [round(mu.min().item(), 6), round(mu.max().item(), 6)],
            "sigma_mean": round(sigma.mean().item(), 6),
            "sigma_std": round(sigma.std().item(), 6),
            "sigma_range": [round(sigma.min().item(), 6), round(sigma.max().item(), 6)],
            "sigma_cv": round((sigma.std() / sigma.mean()).item(), 4),
        }

    # =========================================================================
    # Part 2: How affine is ln_f?  Fit h_final → ln_f(h_final)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Is ln_f approximately affine?")
    print("  Fitting affine map: h_final → ln_f(h_final)")
    print("=" * 70)

    for k in range(vocab.num_relations):
        H = torch.stack([h_final[(i, k)] for i in range(vocab.n_entities)])
        Z = torch.stack([h_after_ln[(i, k)] for i in range(vocab.n_entities)])

        n = H.shape[0]
        H_aug = torch.cat([H, torch.ones(n, 1)], dim=1)
        sol, _, _, _ = torch.linalg.lstsq(H_aug, Z)
        Z_pred = H_aug @ sol

        ss_res = ((Z - Z_pred) ** 2).sum().item()
        ss_tot = ((Z - Z.mean(0)) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot

        cos = F.cosine_similarity(Z, Z_pred, dim=1)

        print(f"\n  R{k}: h_final → ln_f(h_final)")
        print(f"    R² = {r2:.6f}")
        print(f"    Per-entity cosine: mean={cos.mean():.6f}, min={cos.min():.4f}")

        results[f"r{k}_ln_affine"] = {
            "r2": round(r2, 6),
            "cos_mean": round(cos.mean().item(), 6),
            "cos_min": round(cos.min().item(), 4),
        }

    # =========================================================================
    # Part 3: Affine fit from ln_f(h_final) to target — should be ~identity
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: Affine fit from ln_f(h_final) to target")
    print("  If readout = lm_head ∘ ln_f is ~affine and model is 100% accurate,")
    print("  then ln_f(h_final) should already be in correct Voronoi cells.")
    print("=" * 70)

    for k in range(vocab.num_relations):
        Z_train = torch.stack([h_after_ln[(i, k)] for i in train_subjs])
        Z_test = torch.stack([h_after_ln[(i, k)] for i in test_subjs])
        target_train = torch.stack([lm_head[vocab.object_id(vocab.relation_object(i, k))]
                                    for i in train_subjs])
        target_test = torch.stack([lm_head[vocab.object_id(vocab.relation_object(i, k))]
                                   for i in test_subjs])

        W, b = fit_affine(Z_train, target_train, bias=True)
        train_acc = reconstruction_accuracy(Z_train, target_train, W, b, O_all)
        test_acc = reconstruction_accuracy(Z_test, target_test, W, b, O_all)

        # Also check: is ln_f(h_final) already in the correct Voronoi cell WITHOUT any affine map?
        # i.e., is the nearest object embedding to z_i the correct one?
        correct_no_map = 0
        for i_idx, i in enumerate(test_subjs):
            z = Z_test[i_idx]
            dists = (O_all - z).norm(dim=1)
            nn_idx = dists.argmin().item()
            obj_idx = vocab.relation_object(i, k)
            correct_no_map += int(nn_idx == obj_idx)
        raw_voronoi = correct_no_map / len(test_subjs)

        print(f"\n  R{k}:")
        print(f"    Raw Voronoi (no affine map): {raw_voronoi:.4f}")
        print(f"    Affine Voronoi (train): {train_acc:.4f}")
        print(f"    Affine Voronoi (test):  {test_acc:.4f}")

        results[f"r{k}_post_ln"] = {
            "raw_voronoi": round(raw_voronoi, 4),
            "affine_train_voronoi": round(train_acc, 4),
            "affine_test_voronoi": round(test_acc, 4),
        }

    # =========================================================================
    # Part 4: Chain test — compose ln_f affine fit with lm_head
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: Full chain test")
    print("  If ln_f is ~affine (h → L·h + l), then lm_head ∘ ln_f ≈ lm_head·L·h + lm_head·l")
    print("  Checking if the composed affine map explains the Voronoi structure.")
    print("=" * 70)

    for k in range(vocab.num_relations):
        H = torch.stack([h_final[(i, k)] for i in range(vocab.n_entities)])
        Z = torch.stack([h_after_ln[(i, k)] for i in range(vocab.n_entities)])

        # Fit ln_f as affine: Z ≈ L·H + l
        n = H.shape[0]
        H_aug = torch.cat([H, torch.ones(n, 1)], dim=1)
        sol, _, _, _ = torch.linalg.lstsq(H_aug, Z)
        L = sol[:-1, :]  # (64, 64)
        l = sol[-1, :]    # (64,)

        # Composed map: lm_head ∘ (L·h + l) → for object j, logit = o_j · (L·h + l)
        # This gives an affine map from h to "approximate z"
        # Then check: for each test subject, is the approximate z nearest to correct object?

        Z_approx_test = torch.stack([h_final[(i, k)] for i in test_subjs]) @ L.T + l
        correct_approx = 0
        for i_idx, i in enumerate(test_subjs):
            z_approx = Z_approx_test[i_idx]
            dists = (O_all - z_approx).norm(dim=1)
            nn_idx = dists.argmin().item()
            obj_idx = vocab.relation_object(i, k)
            correct_approx += int(nn_idx == obj_idx)
        approx_voronoi = correct_approx / len(test_subjs)

        # Compare: using actual ln_f output
        Z_actual_test = torch.stack([h_after_ln[(i, k)] for i in test_subjs])
        correct_actual = 0
        for i_idx, i in enumerate(test_subjs):
            z = Z_actual_test[i_idx]
            dists = (O_all - z).norm(dim=1)
            nn_idx = dists.argmin().item()
            obj_idx = vocab.relation_object(i, k)
            correct_actual += int(nn_idx == obj_idx)
        actual_voronoi = correct_actual / len(test_subjs)

        print(f"\n  R{k}:")
        print(f"    Voronoi using actual ln_f(h):     {actual_voronoi:.4f}")
        print(f"    Voronoi using affine approx L·h+l: {approx_voronoi:.4f}")

        results[f"r{k}_chain"] = {
            "actual_voronoi": round(actual_voronoi, 4),
            "approx_voronoi": round(approx_voronoi, 4),
        }

    # =========================================================================
    # Part 5: Control — does a RANDOM permutation also get high affine Voronoi?
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: Control — random permutation baseline")
    print("  If affine structure is trivial (any arrangement works), a random")
    print("  permutation of target assignments should also get high Voronoi.")
    print("=" * 70)

    import random
    rng = random.Random(42)
    for k in range(vocab.num_relations):
        H_train = torch.stack([h_final[(i, k)] for i in train_subjs])
        H_test = torch.stack([h_final[(i, k)] for i in test_subjs])

        # True targets
        target_train = torch.stack([lm_head[vocab.object_id(vocab.relation_object(i, k))]
                                    for i in train_subjs])
        target_test = torch.stack([lm_head[vocab.object_id(vocab.relation_object(i, k))]
                                   for i in test_subjs])

        W_true, b_true = fit_affine(H_train, target_train, bias=True)
        true_acc = reconstruction_accuracy(H_test, target_test, W_true, b_true, O_all)

        # Random permutation of object assignments
        random_accs = []
        for trial in range(5):
            perm = list(range(vocab.n_entities))
            rng.shuffle(perm)
            rand_target_train = torch.stack([lm_head[vocab.object_id(perm[i])]
                                             for i in train_subjs])
            rand_target_test = torch.stack([lm_head[vocab.object_id(perm[i])]
                                            for i in test_subjs])
            W_rand, b_rand = fit_affine(H_train, rand_target_train, bias=True)
            rand_acc = reconstruction_accuracy(H_test, rand_target_test, W_rand, b_rand, O_all)
            random_accs.append(rand_acc)

        mean_rand = np.mean(random_accs)
        print(f"\n  R{k}:")
        print(f"    True permutation Voronoi:   {true_acc:.4f}")
        print(f"    Random permutation Voronoi: {mean_rand:.4f} "
              f"(range {min(random_accs):.4f}–{max(random_accs):.4f})")

        results[f"r{k}_control"] = {
            "true_voronoi": round(true_acc, 4),
            "random_voronoi_mean": round(float(mean_rand), 4),
            "random_voronoi_range": [round(min(random_accs), 4), round(max(random_accs), 4)],
        }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_1layer_layernorm_analysis_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

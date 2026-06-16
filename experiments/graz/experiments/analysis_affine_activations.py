"""
Affine approximation of the mapping from MLP input to post-GELU activations.

For each relation context separately, fits a_t ≈ A @ x_t + b
where x_t is the MLP input (64-dim) and a_t is the post-GELU activation (256-dim).

Evaluates fit quality via R², cosine similarity, and residual analysis.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/analysis_affine_activations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config import ModelConfig, VocabConfig
from src.model import build_model

ROOT = Path(__file__).parent.parent
CHECKPOINT = ROOT / "checkpoints_v7_seed42" / "final.pt"
RESULTS_DIR = ROOT / "results"


@torch.no_grad()
def extract_mlp_io(model, vocab, device):
    """Extract MLP input and post-GELU activations for all (entity, relation) pairs."""
    model.eval()
    n = vocab.n_entities
    contexts = []
    keys = []
    for i in range(n):
        for k in range(vocab.num_relations):
            keys.append((i, k))
            contexts.append([vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)])

    batch = torch.tensor(contexts, dtype=torch.long, device=device)

    with model.capture_residuals() as residuals:
        with model.capture_sublayers() as sublayers:
            model(batch)

    h0 = residuals[0][:, 2, :]
    attn_out = sublayers["attn"][0][:, 2, :]
    h0_plus_attn = h0 + attn_out

    ln2 = model.blocks[0].ln2
    fc1 = model.blocks[0].mlp.fc1

    mlp_input = ln2(h0_plus_attn)
    pre_gelu = fc1(mlp_input)
    acts = F.gelu(pre_gelu)

    return keys, mlp_input.cpu().numpy(), acts.cpu().numpy(), pre_gelu.cpu().numpy()


def fit_affine(X, Y):
    """Fit Y ≈ X @ A + b via least squares. X: (n, d_in), Y: (n, d_out)."""
    n = X.shape[0]
    X_aug = np.concatenate([X, np.ones((n, 1))], axis=1)  # (n, d_in+1)
    # Solve X_aug @ W = Y, where W = [A; b]
    W, residuals, rank, sv = np.linalg.lstsq(X_aug, Y, rcond=None)
    A = W[:-1, :]  # (d_in, d_out)
    b = W[-1, :]    # (d_out,)
    Y_pred = X_aug @ W
    return A, b, Y_pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = VocabConfig(
        n_f_attrs=10, n_g_attrs=10, k_per_group=2,
        num_relations=2, n_entities=2025,
        n_train_subjects=2025, r0_shift=7, r1_shift=13,
    )
    model_cfg = ModelConfig(n_layers=1, hidden_dim=64, n_heads=4)
    model = build_model(model_cfg, vocab)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    print("Loaded checkpoint")

    print("Extracting MLP inputs and activations...")
    keys, mlp_inputs, acts, pre_gelu = extract_mlp_io(model, vocab, device)
    key_to_idx = {k: idx for idx, k in enumerate(keys)}
    n = vocab.n_entities

    # Separate by relation
    r0_idx = [key_to_idx[(i, 0)] for i in range(n)]
    r1_idx = [key_to_idx[(i, 1)] for i in range(n)]

    X_r0 = mlp_inputs[r0_idx]  # (2025, 64)
    Y_r0 = acts[r0_idx]        # (2025, 256)
    X_r1 = mlp_inputs[r1_idx]
    Y_r1 = acts[r1_idx]

    # Also get pre-GELU for comparison
    Z_r0 = pre_gelu[r0_idx]    # (2025, 256)
    Z_r1 = pre_gelu[r1_idx]

    print(f"\nMLP input shape: {X_r0.shape}")
    print(f"Activation shape: {Y_r0.shape}")

    # --- Fit affine maps ---
    print("\n=== Affine fit: mlp_input -> post-GELU activations ===")
    for name, X, Y in [("R0", X_r0, Y_r0), ("R1", X_r1, Y_r1)]:
        A, b, Y_pred = fit_affine(X, Y)

        # Per-neuron R²
        ss_res = ((Y - Y_pred) ** 2).sum(axis=0)
        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
        r2_per_neuron = 1 - ss_res / (ss_tot + 1e-12)

        # Per-entity cosine similarity
        cos_per_entity = np.array([
            np.dot(Y[i], Y_pred[i]) / (np.linalg.norm(Y[i]) * np.linalg.norm(Y_pred[i]) + 1e-12)
            for i in range(len(Y))
        ])

        # Per-entity relative error
        rel_err = np.linalg.norm(Y - Y_pred, axis=1) / (np.linalg.norm(Y, axis=1) + 1e-12)

        # Overall R²
        ss_res_total = ((Y - Y_pred) ** 2).sum()
        ss_tot_total = ((Y - Y.mean(axis=0)) ** 2).sum()
        r2_overall = 1 - ss_res_total / ss_tot_total

        print(f"\n{name}:")
        print(f"  Overall R²:            {r2_overall:.6f}")
        print(f"  Per-neuron R²:         mean={r2_per_neuron.mean():.4f}, "
              f"min={r2_per_neuron.min():.4f}, median={np.median(r2_per_neuron):.4f}")
        print(f"  Per-entity cosine:     mean={cos_per_entity.mean():.6f}, "
              f"min={cos_per_entity.min():.4f}, std={cos_per_entity.std():.6f}")
        print(f"  Per-entity rel error:  mean={rel_err.mean():.4f}, "
              f"max={rel_err.max():.4f}")
        print(f"  Residual norm:         mean={np.linalg.norm(Y - Y_pred, axis=1).mean():.4f}")
        print(f"  Actual norm:           mean={np.linalg.norm(Y, axis=1).mean():.4f}")

    # --- Compare: affine fit to pre-GELU (should be perfect since fc1 is linear) ---
    print("\n=== Affine fit: mlp_input -> pre-GELU (sanity check, should be perfect) ===")
    for name, X, Z in [("R0", X_r0, Z_r0), ("R1", X_r1, Z_r1)]:
        _, _, Z_pred = fit_affine(X, Z)
        ss_res = ((Z - Z_pred) ** 2).sum()
        ss_tot = ((Z - Z.mean(axis=0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        print(f"  {name}: R² = {r2:.10f}")

    # --- Fit affine to the END-TO-END map: mlp_input -> MLP output ---
    print("\n=== Affine fit: mlp_input -> MLP output (acts @ fc2.T + fc2_bias) ===")
    fc2_weight = model.blocks[0].mlp.fc2.weight.detach().cpu().numpy()  # (64, 256)
    fc2_bias = model.blocks[0].mlp.fc2.bias.detach().cpu().numpy()      # (64,)

    for name, X, Y_act in [("R0", X_r0, Y_r0), ("R1", X_r1, Y_r1)]:
        # MLP output = acts @ fc2.T + fc2_bias
        mlp_out = Y_act @ fc2_weight.T + fc2_bias  # (2025, 64)
        A, b_vec, out_pred = fit_affine(X, mlp_out)

        ss_res = ((mlp_out - out_pred) ** 2).sum()
        ss_tot = ((mlp_out - mlp_out.mean(axis=0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        cos_per_entity = np.array([
            np.dot(mlp_out[i], out_pred[i]) / (np.linalg.norm(mlp_out[i]) * np.linalg.norm(out_pred[i]) + 1e-12)
            for i in range(len(mlp_out))
        ])
        rel_err = np.linalg.norm(mlp_out - out_pred, axis=1) / (np.linalg.norm(mlp_out, axis=1) + 1e-12)

        print(f"\n{name}:")
        print(f"  Overall R²:            {r2:.6f}")
        print(f"  Per-entity cosine:     mean={cos_per_entity.mean():.6f}, "
              f"min={cos_per_entity.min():.4f}, std={cos_per_entity.std():.6f}")
        print(f"  Per-entity rel error:  mean={rel_err.mean():.4f}, max={rel_err.max():.4f}")

    # --- Theoretical affine map vs fitted ---
    # If GELU were identity, the map would be: acts = fc1(mlp_input) = mlp_input @ W1.T + b1
    # so the affine map mlp_input -> acts would have A = W1.T, b = b1
    # The fitted A should be close to a scaled version of W1.T
    print("\n=== Comparison with fc1 weights (GELU=identity baseline) ===")
    fc1_weight = model.blocks[0].mlp.fc1.weight.detach().cpu().numpy()  # (256, 64)
    fc1_bias = model.blocks[0].mlp.fc1.bias.detach().cpu().numpy()      # (256,)

    for name, X, Y in [("R0", X_r0, Y_r0), ("R1", X_r1, Y_r1)]:
        A_fit, b_fit, _ = fit_affine(X, Y)
        # A_fit: (64, 256), fc1_weight.T: (64, 256)
        # Per-neuron cosine between fitted column and fc1 column
        cos_cols = np.array([
            np.dot(A_fit[:, j], fc1_weight[j, :]) /
            (np.linalg.norm(A_fit[:, j]) * np.linalg.norm(fc1_weight[j, :]) + 1e-12)
            for j in range(256)
        ])
        scale_ratio = np.array([
            np.linalg.norm(A_fit[:, j]) / (np.linalg.norm(fc1_weight[j, :]) + 1e-12)
            for j in range(256)
        ])
        print(f"\n{name}:")
        print(f"  Cosine(fitted col, fc1 col): mean={cos_cols.mean():.4f}, "
              f"std={cos_cols.std():.4f}, min={cos_cols.min():.4f}")
        print(f"  Scale ratio ||A_fit||/||W1||: mean={scale_ratio.mean():.4f}, "
              f"std={scale_ratio.std():.4f}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top row: scatter of actual vs predicted activations (sample of neurons)
    for col, (name, X, Y) in enumerate([("R0", X_r0, Y_r0), ("R1", X_r1, Y_r1)]):
        _, _, Y_pred = fit_affine(X, Y)
        ax = axes[0, col]
        # Flatten all (entity, neuron) pairs, subsample for plotting
        rng = np.random.RandomState(42)
        flat_actual = Y.flatten()
        flat_pred = Y_pred.flatten()
        idx = rng.choice(len(flat_actual), size=min(50000, len(flat_actual)), replace=False)
        ax.scatter(flat_actual[idx], flat_pred[idx], s=0.5, alpha=0.3, color="steelblue")
        lims = [min(flat_actual.min(), flat_pred.min()), max(flat_actual.max(), flat_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Actual activation")
        ax.set_ylabel("Predicted (affine)")
        ax.set_title(f"{name}: Actual vs Affine-predicted activations")

    # Bottom row: per-neuron R² distribution
    for col, (name, X, Y) in enumerate([("R0", X_r0, Y_r0), ("R1", X_r1, Y_r1)]):
        _, _, Y_pred = fit_affine(X, Y)
        ss_res = ((Y - Y_pred) ** 2).sum(axis=0)
        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
        r2_per_neuron = 1 - ss_res / (ss_tot + 1e-12)

        ax = axes[1, col]
        ax.hist(r2_per_neuron, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("R² per neuron")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}: Per-neuron R² (affine fit)")
        ax.axvline(x=r2_per_neuron.mean(), color="red", linestyle="--",
                   label=f"mean={r2_per_neuron.mean():.3f}")
        ax.legend()

    plt.suptitle("Affine approximation: MLP input → post-GELU activations", fontsize=13)
    plt.tight_layout()
    out_path = RESULTS_DIR / "affine_activation_fit.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()

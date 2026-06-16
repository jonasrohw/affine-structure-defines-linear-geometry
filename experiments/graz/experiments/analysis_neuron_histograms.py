"""
Histogram of MLP activations at relation-specific neurons for each type.

Identifies R0-only and R1-only neurons (mean activation > 0.05 threshold),
then plots the distribution of type-mean activations at those neurons.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/analysis_neuron_histograms.py
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


def precompute_relation_objects(vocab):
    f_type_to_objs = {}
    g_type_to_objs = {}
    for j in range(vocab.n_entities):
        ft = vocab.object_f_type(j)
        gt = vocab.object_g_type(j)
        f_type_to_objs.setdefault(ft, []).append(j)
        g_type_to_objs.setdefault(gt, []).append(j)
    table = {}
    for i in range(vocab.n_entities):
        table[(i, 0)] = f_type_to_objs[vocab.subject_f_type(i)]
        table[(i, 1)] = g_type_to_objs[vocab.subject_g_type(i)]
    return table


@torch.no_grad()
def extract_mlp_activations(model, vocab, device):
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

    return keys, acts.cpu(), pre_gelu.cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = VocabConfig(
        n_f_attrs=10, n_g_attrs=10, k_per_group=2,
        num_relations=2, n_entities=2025,
        n_train_subjects=2025, r0_shift=7, r1_shift=13,
    )
    model_cfg = ModelConfig(n_layers=1, hidden_dim=64, n_heads=4)
    rel_objects_table = precompute_relation_objects(vocab)
    vocab.relation_objects = lambda subj_idx, rel_k: rel_objects_table[(subj_idx, rel_k)]

    model = build_model(model_cfg, vocab)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    print("Loaded checkpoint")

    print("Extracting activations...")
    keys, acts_all, pre_gelu_all = extract_mlp_activations(model, vocab, device)
    key_to_idx = {k: idx for idx, k in enumerate(keys)}
    n = vocab.n_entities

    # Group subjects by type for each relation
    type_to_subjects = {0: {}, 1: {}}
    for i in range(n):
        t0 = vocab.subject_f_type(i)
        t1 = vocab.subject_g_type(i)
        type_to_subjects[0].setdefault(t0, []).append(i)
        type_to_subjects[1].setdefault(t1, []).append(i)

    # Compute type-mean activations for each relation
    type_mean_acts = {}
    for k in range(2):
        type_list = sorted(type_to_subjects[k].keys())
        means = []
        for t in type_list:
            subjs = type_to_subjects[k][t]
            type_acts = torch.stack([acts_all[key_to_idx[(i, k)]] for i in subjs])
            means.append(type_acts.mean(dim=0))
        type_mean_acts[k] = torch.stack(means)  # (45, 256)

    # Identify R0-only and R1-only neurons (same threshold as original analysis)
    r0_usage = type_mean_acts[0].mean(dim=0)  # (256,)
    r1_usage = type_mean_acts[1].mean(dim=0)  # (256,)
    threshold = 0.05

    r0_active = r0_usage > threshold
    r1_active = r1_usage > threshold
    r0_only = r0_active & ~r1_active
    r1_only = r1_active & ~r0_active

    r0_neuron_ids = r0_only.nonzero(as_tuple=True)[0].numpy()
    r1_neuron_ids = r1_only.nonzero(as_tuple=True)[0].numpy()
    print(f"R0-only neurons: {len(r0_neuron_ids)}")
    print(f"R1-only neurons: {len(r1_neuron_ids)}")

    # Per-entity activations (not type-means)
    # R0 contexts: acts_all for all (i, 0) pairs — shape (2025, 256)
    r0_all = torch.stack([acts_all[key_to_idx[(i, 0)]] for i in range(n)])
    r1_all = torch.stack([acts_all[key_to_idx[(i, 1)]] for i in range(n)])

    r0_at_r0_neurons = r0_all[:, r0_only].numpy()  # (2025, 85)
    r1_at_r0_neurons = r1_all[:, r0_only].numpy()  # (2025, 85)
    r0_at_r1_neurons = r0_all[:, r1_only].numpy()  # (2025, 82)
    r1_at_r1_neurons = r1_all[:, r1_only].numpy()  # (2025, 82)

    print(f"\nR0 contexts at R0-only neurons: mean={r0_at_r0_neurons.mean():.4f}, "
          f"std={r0_at_r0_neurons.std():.4f}")
    print(f"R0 contexts at R1-only neurons: mean={r0_at_r1_neurons.mean():.4f}, "
          f"std={r0_at_r1_neurons.std():.4f}")
    print(f"R1 contexts at R1-only neurons: mean={r1_at_r1_neurons.mean():.4f}, "
          f"std={r1_at_r1_neurons.std():.4f}")
    print(f"R1 contexts at R0-only neurons: mean={r1_at_r0_neurons.mean():.4f}, "
          f"std={r1_at_r0_neurons.std():.4f}")

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R0 histogram
    ax = axes[0]
    ax.hist(r0_at_r0_neurons.flatten(), bins=80, alpha=0.7,
            label=f"R0 contexts at R0 neurons (n={len(r0_neuron_ids)})",
            color="steelblue", edgecolor="white", linewidth=0.5)
    ax.hist(r1_at_r0_neurons.flatten(), bins=80, alpha=0.7,
            label=f"R1 contexts at R0 neurons",
            color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Activation value (post-GELU)")
    ax.set_ylabel("Count (2025 entities × neurons)")
    ax.set_title(f"R0-only neurons ({len(r0_neuron_ids)} neurons)")
    ax.legend(fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # R1 histogram
    ax = axes[1]
    ax.hist(r1_at_r1_neurons.flatten(), bins=80, alpha=0.7,
            label=f"R1 contexts at R1 neurons (n={len(r1_neuron_ids)})",
            color="steelblue", edgecolor="white", linewidth=0.5)
    ax.hist(r0_at_r1_neurons.flatten(), bins=80, alpha=0.7,
            label=f"R0 contexts at R1 neurons",
            color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Activation value (post-GELU)")
    ax.set_ylabel("Count (2025 entities × neurons)")
    ax.set_title(f"R1-only neurons ({len(r1_neuron_ids)} neurons)")
    ax.legend(fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("MLP activations at relation-specific neurons (per entity, 2025 entities)",
                 fontsize=12)
    plt.tight_layout()
    out_path = RESULTS_DIR / "neuron_activation_histograms.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")

    # --- Derivative histograms ---
    # GELU'(x) = Phi(x) + x * phi(x)
    # where Phi = standard normal CDF, phi = standard normal PDF
    def gelu_derivative(x):
        std_normal = torch.distributions.Normal(0, 1)
        phi = torch.exp(std_normal.log_prob(x))
        Phi = std_normal.cdf(x)
        return Phi + x * phi

    # Pre-GELU values per entity
    r0_pre = torch.stack([pre_gelu_all[key_to_idx[(i, 0)]] for i in range(n)])
    r1_pre = torch.stack([pre_gelu_all[key_to_idx[(i, 1)]] for i in range(n)])

    # Derivatives at pre-GELU values for relation-specific neurons
    r0_deriv_at_r0 = gelu_derivative(r0_pre[:, r0_only]).numpy()
    r1_deriv_at_r0 = gelu_derivative(r1_pre[:, r0_only]).numpy()
    r0_deriv_at_r1 = gelu_derivative(r0_pre[:, r1_only]).numpy()
    r1_deriv_at_r1 = gelu_derivative(r1_pre[:, r1_only]).numpy()

    print(f"\nGELU derivatives:")
    print(f"R0 contexts at R0-only neurons: mean={r0_deriv_at_r0.mean():.4f}, "
          f"std={r0_deriv_at_r0.std():.4f}")
    print(f"R1 contexts at R0-only neurons: mean={r1_deriv_at_r0.mean():.4f}, "
          f"std={r1_deriv_at_r0.std():.4f}")
    print(f"R1 contexts at R1-only neurons: mean={r1_deriv_at_r1.mean():.4f}, "
          f"std={r1_deriv_at_r1.std():.4f}")
    print(f"R0 contexts at R1-only neurons: mean={r0_deriv_at_r1.mean():.4f}, "
          f"std={r0_deriv_at_r1.std():.4f}")

    # --- Pre-activation histograms ---
    r0_pre_at_r0 = r0_pre[:, r0_only].numpy()
    r1_pre_at_r0 = r1_pre[:, r0_only].numpy()
    r0_pre_at_r1 = r0_pre[:, r1_only].numpy()
    r1_pre_at_r1 = r1_pre[:, r1_only].numpy()

    print(f"\nPre-activations:")
    print(f"R0 contexts at R0-only neurons: mean={r0_pre_at_r0.mean():.4f}, "
          f"std={r0_pre_at_r0.std():.4f}")
    print(f"R1 contexts at R0-only neurons: mean={r1_pre_at_r0.mean():.4f}, "
          f"std={r1_pre_at_r0.std():.4f}")
    print(f"R1 contexts at R1-only neurons: mean={r1_pre_at_r1.mean():.4f}, "
          f"std={r1_pre_at_r1.std():.4f}")
    print(f"R0 contexts at R1-only neurons: mean={r0_pre_at_r1.mean():.4f}, "
          f"std={r0_pre_at_r1.std():.4f}")

    fig_pre, axes_pre = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes_pre[0]
    ax.hist(r0_pre_at_r0.flatten(), bins=80, alpha=0.7,
            label=f"R0 contexts at R0 neurons",
            color="steelblue", edgecolor="white", linewidth=0.5)
    ax.hist(r1_pre_at_r0.flatten(), bins=80, alpha=0.7,
            label=f"R1 contexts at R0 neurons",
            color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Pre-activation value (before GELU)")
    ax.set_ylabel("Count (2025 entities × neurons)")
    ax.set_title(f"R0-only neurons ({len(r0_neuron_ids)} neurons)")
    ax.legend(fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    ax = axes_pre[1]
    ax.hist(r1_pre_at_r1.flatten(), bins=80, alpha=0.7,
            label=f"R1 contexts at R1 neurons",
            color="steelblue", edgecolor="white", linewidth=0.5)
    ax.hist(r0_pre_at_r1.flatten(), bins=80, alpha=0.7,
            label=f"R0 contexts at R1 neurons",
            color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Pre-activation value (before GELU)")
    ax.set_ylabel("Count (2025 entities × neurons)")
    ax.set_title(f"R1-only neurons ({len(r1_neuron_ids)} neurons)")
    ax.legend(fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Pre-GELU activations at relation-specific neurons (per entity, 2025 entities)",
                 fontsize=12)
    plt.tight_layout()
    out_pre = RESULTS_DIR / "neuron_preactivation_histograms.png"
    plt.savefig(out_pre, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_pre}")

    # --- Second derivative histograms ---
    # GELU''(x) = 2*phi(x) - x^2*phi(x) = phi(x)*(2 - x^2)
    # More precisely: GELU'(x) = Phi(x) + x*phi(x)
    # GELU''(x) = phi(x) + phi(x) + x*(-x*phi(x)) = 2*phi(x) - x^2*phi(x)
    #           = phi(x)*(2 - x^2)
    def gelu_second_derivative(x):
        std_normal = torch.distributions.Normal(0, 1)
        phi = torch.exp(std_normal.log_prob(x))
        return phi * (2 - x ** 2)

    r0_d2_at_r0 = gelu_second_derivative(r0_pre[:, r0_only]).numpy()
    r1_d2_at_r0 = gelu_second_derivative(r1_pre[:, r0_only]).numpy()
    r0_d2_at_r1 = gelu_second_derivative(r0_pre[:, r1_only]).numpy()
    r1_d2_at_r1 = gelu_second_derivative(r1_pre[:, r1_only]).numpy()

    print(f"\nGELU second derivatives:")
    print(f"R0 contexts at R0-only neurons: mean={r0_d2_at_r0.mean():.4f}, "
          f"std={r0_d2_at_r0.std():.4f}")
    print(f"R1 contexts at R0-only neurons: mean={r1_d2_at_r0.mean():.4f}, "
          f"std={r1_d2_at_r0.std():.4f}")
    print(f"R1 contexts at R1-only neurons: mean={r1_d2_at_r1.mean():.4f}, "
          f"std={r1_d2_at_r1.std():.4f}")
    print(f"R0 contexts at R1-only neurons: mean={r0_d2_at_r1.mean():.4f}, "
          f"std={r0_d2_at_r1.std():.4f}")

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes3[0]
    ax.hist(r0_d2_at_r0.flatten(), bins=80, alpha=0.7,
            label=f"R0 contexts at R0 neurons",
            color="steelblue", edgecolor="white", linewidth=0.5)
    ax.hist(r1_d2_at_r0.flatten(), bins=80, alpha=0.7,
            label=f"R1 contexts at R0 neurons",
            color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("GELU''(x) at pre-activation x")
    ax.set_ylabel("Count (2025 entities × neurons)")
    ax.set_title(f"R0-only neurons ({len(r0_neuron_ids)} neurons)")
    ax.legend(fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    ax = axes3[1]
    ax.hist(r1_d2_at_r1.flatten(), bins=80, alpha=0.7,
            label=f"R1 contexts at R1 neurons",
            color="steelblue", edgecolor="white", linewidth=0.5)
    ax.hist(r0_d2_at_r1.flatten(), bins=80, alpha=0.7,
            label=f"R0 contexts at R1 neurons",
            color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("GELU''(x) at pre-activation x")
    ax.set_ylabel("Count (2025 entities × neurons)")
    ax.set_title(f"R1-only neurons ({len(r1_neuron_ids)} neurons)")
    ax.legend(fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("GELU second derivative at relation-specific neurons (per entity, 2025 entities)",
                 fontsize=12)
    plt.tight_layout()
    out_d2 = RESULTS_DIR / "neuron_second_derivative_histograms.png"
    plt.savefig(out_d2, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_d2}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes2[0]
    ax.hist(r0_deriv_at_r0.flatten(), bins=80, alpha=0.7,
            label=f"R0 contexts at R0 neurons",
            color="steelblue", edgecolor="white", linewidth=0.5)
    ax.hist(r1_deriv_at_r0.flatten(), bins=80, alpha=0.7,
            label=f"R1 contexts at R0 neurons",
            color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("GELU'(x) at pre-activation x")
    ax.set_ylabel("Count (2025 entities × neurons)")
    ax.set_title(f"R0-only neurons ({len(r0_neuron_ids)} neurons)")
    ax.legend(fontsize=9)

    ax = axes2[1]
    ax.hist(r1_deriv_at_r1.flatten(), bins=80, alpha=0.7,
            label=f"R1 contexts at R1 neurons",
            color="steelblue", edgecolor="white", linewidth=0.5)
    ax.hist(r0_deriv_at_r1.flatten(), bins=80, alpha=0.7,
            label=f"R0 contexts at R1 neurons",
            color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("GELU'(x) at pre-activation x")
    ax.set_ylabel("Count (2025 entities × neurons)")
    ax.set_title(f"R1-only neurons ({len(r1_neuron_ids)} neurons)")
    ax.legend(fontsize=9)

    plt.suptitle("GELU derivative at relation-specific neurons (per entity, 2025 entities)",
                 fontsize=12)
    plt.tight_layout()
    out_path2 = RESULTS_DIR / "neuron_derivative_histograms.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path2}")


if __name__ == "__main__":
    main()

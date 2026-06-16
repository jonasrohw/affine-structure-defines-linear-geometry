"""
Per-neuron input decomposition: relation component vs subject component.

For each MLP neuron j, the pre-activation decomposes as:
    pre_j = fc1_j · LN(h0 + attn_out) + bias_j
          ≈ (fc1_j · attn_out) + (fc1_j · h0) + ...

We measure:
1. Per-neuron gating SNR: how large is the relation shift vs subject variation?
2. Geometric decomposition: projection of fc1 rows onto relation direction vs subject subspace
3. Interference rate: how often does subject variation flip a neuron's gate?

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/analysis_neuron_input_decomposition_v7.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ModelConfig, VocabConfig
from src.model import build_model

ROOT = Path(__file__).parent.parent
CHECKPOINT = ROOT / "checkpoints_v7_seed42" / "final.pt"


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
def extract_components(model, vocab, device):
    """Extract h0, attn_out, LN(h0+attn_out), and pre-GELU values for all contexts."""
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

    h0 = residuals[0][:, 2, :].cpu()
    attn_out = sublayers["attn"][0][:, 2, :].cpu()

    block = model.blocks[0]
    ln2 = block.ln2
    fc1 = block.mlp.fc1

    h0_plus_attn = h0 + attn_out
    # Apply LN on device then bring back
    h0_plus_attn_dev = (residuals[0][:, 2, :] + sublayers["attn"][0][:, 2, :])
    mlp_input = ln2(h0_plus_attn_dev).cpu()
    pre_gelu = fc1(mlp_input.to(device)).cpu()

    fc1_weight = fc1.weight.detach().cpu()  # (256, 64)
    fc1_bias = fc1.bias.detach().cpu()      # (256,)

    return keys, h0, attn_out, mlp_input, pre_gelu, fc1_weight, fc1_bias


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    print(f"Loaded checkpoint from {CHECKPOINT}")

    print("Extracting components...")
    keys, h0, attn_out, mlp_input, pre_gelu, fc1_weight, fc1_bias = extract_components(
        model, vocab, device)

    n = vocab.n_entities
    key_to_idx = {k: i for i, k in enumerate(keys)}
    n_neurons = fc1_weight.shape[0]  # 256

    r0_idx = [key_to_idx[(i, 0)] for i in range(n)]
    r1_idx = [key_to_idx[(i, 1)] for i in range(n)]

    h0_r0 = h0[r0_idx]        # (2025, 64)
    h0_r1 = h0[r1_idx]        # (2025, 64)
    attn_r0 = attn_out[r0_idx]  # (2025, 64)
    attn_r1 = attn_out[r1_idx]  # (2025, 64)
    mlp_in_r0 = mlp_input[r0_idx]  # (2025, 64)
    mlp_in_r1 = mlp_input[r1_idx]  # (2025, 64)
    pre_gelu_r0 = pre_gelu[r0_idx]  # (2025, 256)
    pre_gelu_r1 = pre_gelu[r1_idx]  # (2025, 256)

    # Mean attention output per relation
    mean_attn_r0 = attn_r0.mean(dim=0)  # (64,)
    mean_attn_r1 = attn_r1.mean(dim=0)  # (64,)

    # Relation direction in residual stream space
    rel_dir = mean_attn_r0 - mean_attn_r1  # (64,)
    rel_dir_norm = rel_dir.norm().item()
    rel_dir_unit = rel_dir / rel_dir.norm()

    print(f"\nRelation direction norm: {rel_dir_norm:.4f}")

    # =====================================================================
    # ANALYSIS 1: Per-neuron decomposition on the ACTUAL MLP input (post-LN)
    # =====================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Per-neuron pre-activation decomposition (post-LayerNorm)")
    print("=" * 70)

    # For each neuron j, pre_j = fc1_j · mlp_input + bias_j
    # We decompose mlp_input into:
    #   - relation mean: mean(mlp_input | relation k)
    #   - subject residual: mlp_input - mean(mlp_input | relation k)

    mean_mlp_in_r0 = mlp_in_r0.mean(dim=0)  # (64,)
    mean_mlp_in_r1 = mlp_in_r1.mean(dim=0)  # (64,)

    # Per-neuron relation component (the constant part that differs by relation)
    # = fc1_j · mean_mlp_in_Rk + bias_j
    rel_comp_r0 = fc1_weight @ mean_mlp_in_r0 + fc1_bias  # (256,)
    rel_comp_r1 = fc1_weight @ mean_mlp_in_r1 + fc1_bias  # (256,)

    # Per-neuron subject component (the varying part)
    # = fc1_j · (mlp_input - mean_mlp_input_Rk)
    subj_residual_r0 = mlp_in_r0 - mean_mlp_in_r0  # (2025, 64)
    subj_residual_r1 = mlp_in_r1 - mean_mlp_in_r1  # (2025, 64)
    subj_comp_r0 = subj_residual_r0 @ fc1_weight.T  # (2025, 256)
    subj_comp_r1 = subj_residual_r1 @ fc1_weight.T  # (2025, 256)

    # Verify decomposition
    reconstructed_r0 = rel_comp_r0.unsqueeze(0) + subj_comp_r0
    max_err = (pre_gelu_r0 - reconstructed_r0).abs().max().item()
    print(f"Decomposition verification (max error): {max_err:.2e}")

    # Gating shift per neuron: difference in relation component between R0 and R1
    gating_shift = rel_comp_r0 - rel_comp_r1  # (256,) — positive means R0 > R1

    # Subject variation per neuron (pooled across both relations)
    subj_std = torch.cat([subj_comp_r0, subj_comp_r1], dim=0).std(dim=0)  # (256,)

    # Gating SNR
    gating_snr = gating_shift.abs() / (subj_std + 1e-10)

    print(f"\nGating shift (|rel_comp_R0 - rel_comp_R1|):")
    print(f"  Mean: {gating_shift.abs().mean():.4f}")
    print(f"  Median: {gating_shift.abs().median():.4f}")
    print(f"  Std: {gating_shift.std():.4f}")

    print(f"\nSubject variation (std of subject component, pooled):")
    print(f"  Mean: {subj_std.mean():.4f}")
    print(f"  Median: {subj_std.median():.4f}")

    print(f"\nGating SNR (|gating_shift| / subject_std):")
    print(f"  Mean: {gating_snr.mean():.4f}")
    print(f"  Median: {gating_snr.median():.4f}")
    print(f"  Min: {gating_snr.min():.4f}")
    print(f"  Max: {gating_snr.max():.4f}")
    print(f"  Neurons with SNR > 2: {(gating_snr > 2).sum().item()}/256")
    print(f"  Neurons with SNR > 3: {(gating_snr > 3).sum().item()}/256")
    print(f"  Neurons with SNR > 5: {(gating_snr > 5).sum().item()}/256")
    print(f"  Neurons with SNR < 1: {(gating_snr < 1).sum().item()}/256")

    # =====================================================================
    # ANALYSIS 2: Interference — how often does subject variation flip the gate?
    # =====================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Gate interference (subject variation flipping the gate)")
    print("=" * 70)

    # Identify R0-only and R1-only neurons using the activation threshold
    acts_r0 = F.gelu(pre_gelu_r0)  # (2025, 256)
    acts_r1 = F.gelu(pre_gelu_r1)  # (2025, 256)

    type_mean_r0 = acts_r0.mean(dim=0)
    type_mean_r1 = acts_r1.mean(dim=0)
    threshold = 0.05

    r0_active = type_mean_r0 > threshold
    r1_active = type_mean_r1 > threshold
    r0_only = r0_active & ~r1_active
    r1_only = r1_active & ~r0_active

    print(f"R0-only neurons: {r0_only.sum().item()}")
    print(f"R1-only neurons: {r1_only.sum().item()}")

    # For R0-only neurons: they should be ON for R0, OFF for R1
    # Interference = R1 context pushing pre-activation above 0 (turning it on)
    # or R0 context pushing pre-activation below ~-0.5 (GELU saturation)

    # Count how often R1 contexts activate R0-only neurons (pre_gelu > 0)
    r0_neuron_ids = r0_only.nonzero(as_tuple=True)[0]
    r1_neuron_ids = r1_only.nonzero(as_tuple=True)[0]

    # R1 contexts at R0 neurons: should be off (pre < 0)
    r1_at_r0_pre = pre_gelu_r1[:, r0_only]  # (2025, 85)
    wrongly_on = (r1_at_r0_pre > 0).float().mean().item()
    print(f"\nR0-only neurons activated by R1 context (pre > 0): {100*wrongly_on:.2f}%")

    # R0 contexts at R1 neurons: should be off
    r0_at_r1_pre = pre_gelu_r0[:, r1_only]
    wrongly_on_r1 = (r0_at_r1_pre > 0).float().mean().item()
    print(f"R1-only neurons activated by R0 context (pre > 0): {100*wrongly_on_r1:.2f}%")

    # R0 contexts at R0 neurons: should be on (pre > 0 ideally)
    r0_at_r0_pre = pre_gelu_r0[:, r0_only]
    wrongly_off = (r0_at_r0_pre < 0).float().mean().item()
    print(f"R0-only neurons suppressed in R0 context (pre < 0): {100*wrongly_off:.2f}%")

    r1_at_r1_pre = pre_gelu_r1[:, r1_only]
    wrongly_off_r1 = (r1_at_r1_pre < 0).float().mean().item()
    print(f"R1-only neurons suppressed in R1 context (pre < 0): {100*wrongly_off_r1:.2f}%")

    # More nuanced: even if pre > 0, the activation may be negligible
    # Use GELU output threshold instead
    r1_at_r0_act = acts_r1[:, r0_only]
    wrongly_active = (r1_at_r0_act > 0.05).float().mean().item()
    print(f"\nR0-only neurons with act > 0.05 in R1 context: {100*wrongly_active:.2f}%")

    r0_at_r1_act = acts_r0[:, r1_only]
    wrongly_active_r1 = (r0_at_r1_act > 0.05).float().mean().item()
    print(f"R1-only neurons with act > 0.05 in R0 context: {100*wrongly_active_r1:.2f}%")

    # =====================================================================
    # ANALYSIS 3: fc1 row geometry — relation vs subject alignment
    # =====================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: fc1 row geometry (relation direction vs subject subspace)")
    print("=" * 70)

    # Relation direction in MLP input space (post-LN)
    mlp_rel_dir = mean_mlp_in_r0 - mean_mlp_in_r1
    mlp_rel_dir_unit = mlp_rel_dir / mlp_rel_dir.norm()

    # Project each fc1 row onto relation direction
    fc1_rel_proj = fc1_weight @ mlp_rel_dir_unit  # (256,) — signed projection

    # Subject subspace: PCA of mlp_input residuals (after removing relation means)
    all_residuals = torch.cat([subj_residual_r0, subj_residual_r1], dim=0)  # (4050, 64)
    U, S, Vt = torch.linalg.svd(all_residuals, full_matrices=False)
    subj_basis = Vt[:10]  # top-10 subject PCs, shape (10, 64)

    # Project fc1 rows onto subject subspace
    fc1_subj_proj = fc1_weight @ subj_basis.T  # (256, 10)
    fc1_subj_norm = fc1_subj_proj.norm(dim=1)  # (256,)

    # fc1 row norms
    fc1_row_norms = fc1_weight.norm(dim=1)  # (256,)

    # Fraction of fc1 row in relation direction vs subject subspace
    rel_frac = fc1_rel_proj.abs() / fc1_row_norms
    subj_frac = fc1_subj_norm / fc1_row_norms

    print(f"\nRelation direction norm in MLP input space: {mlp_rel_dir.norm():.4f}")
    print(f"Subject subspace variance explained (top 10 PCs): "
          f"{S[:10].pow(2).sum() / S.pow(2).sum() * 100:.1f}%")
    print(f"Top 5 singular values: {S[:5].tolist()}")

    print(f"\nfc1 row projections:")
    print(f"  |fc1_j · rel_dir|:  mean={fc1_rel_proj.abs().mean():.4f}, "
          f"std={fc1_rel_proj.abs().std():.4f}")
    print(f"  ||fc1_j in subj_subspace||: mean={fc1_subj_norm.mean():.4f}, "
          f"std={fc1_subj_norm.std():.4f}")
    print(f"  fc1 row norm: mean={fc1_row_norms.mean():.4f}, "
          f"std={fc1_row_norms.std():.4f}")

    print(f"\n  Fraction of fc1 row in relation dir: mean={rel_frac.mean():.4f}, "
          f"median={rel_frac.median():.4f}")
    print(f"  Fraction of fc1 row in subject subspace: mean={subj_frac.mean():.4f}, "
          f"median={subj_frac.median():.4f}")

    # Correlation between relation projection and gating shift
    corr = torch.corrcoef(torch.stack([fc1_rel_proj, gating_shift]))[0, 1].item()
    print(f"\n  Correlation(fc1_rel_proj, gating_shift): {corr:.4f}")

    # Per-neuron category breakdown
    print(f"\nBy neuron category:")
    for label, mask in [("R0-only", r0_only), ("R1-only", r1_only),
                        ("Dead", ~r0_active & ~r1_active)]:
        if mask.sum() == 0:
            continue
        m_rel = fc1_rel_proj[mask]
        m_subj = fc1_subj_norm[mask]
        m_snr = gating_snr[mask]
        m_shift = gating_shift[mask]
        print(f"  {label} ({mask.sum().item()} neurons):")
        print(f"    fc1·rel_dir: mean={m_rel.mean():.4f} ± {m_rel.std():.4f} "
              f"(sign: {(m_rel > 0).sum().item()}+ / {(m_rel < 0).sum().item()}-)")
        print(f"    fc1 in subj_space: mean={m_subj.mean():.4f} ± {m_subj.std():.4f}")
        print(f"    gating shift: mean={m_shift.mean():.4f} ± {m_shift.std():.4f}")
        print(f"    gating SNR: mean={m_snr.mean():.4f}, median={m_snr.median():.4f}")

    # =====================================================================
    # ANALYSIS 4: Orthogonality of relation and subject signals
    # =====================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Relation direction vs subject subspace orthogonality")
    print("=" * 70)

    # Project relation direction onto subject PCs
    rel_in_subj = subj_basis @ mlp_rel_dir_unit  # (10,) — projections
    rel_in_subj_norm = rel_in_subj.norm().item()
    print(f"\nRelation direction projected onto subject subspace:")
    print(f"  ||projection|| = {rel_in_subj_norm:.4f} (of unit vector)")
    print(f"  This means {100*rel_in_subj_norm**2:.1f}% of relation direction variance "
          f"is in the subject subspace")
    print(f"  Per-PC projections: {rel_in_subj.tolist()[:5]}")

    # Also: cosine between relation direction and each subject PC
    cos_with_pcs = (subj_basis @ mlp_rel_dir_unit)
    print(f"  Cosine with top 5 PCs: {cos_with_pcs[:5].tolist()}")

    # Effective orthogonality: the relation direction is in a subspace
    # approximately orthogonal to the subject variation
    print(f"\n  Fraction of relation dir ORTHOGONAL to subject subspace: "
          f"{1 - rel_in_subj_norm**2:.4f}")

    # =====================================================================
    # ANALYSIS 5: Per-neuron detailed example
    # =====================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Example neurons — the full story")
    print("=" * 70)

    # Pick top R0-specific, top R1-specific, and a shared/low-SNR neuron
    # Sort by gating shift (R0-preferring = large positive shift)
    sorted_by_shift = gating_shift.argsort(descending=True)

    examples = [
        ("Top R0 neuron", sorted_by_shift[0].item()),
        ("2nd R0 neuron", sorted_by_shift[1].item()),
        ("Top R1 neuron", sorted_by_shift[-1].item()),
        ("2nd R1 neuron", sorted_by_shift[-2].item()),
        ("Lowest SNR", gating_snr.argsort()[0].item()),
    ]

    for label, j in examples:
        print(f"\n  {label} (neuron {j}):")
        print(f"    fc1 row norm: {fc1_row_norms[j]:.4f}")
        print(f"    fc1 · rel_dir: {fc1_rel_proj[j]:.4f} "
              f"({100*rel_frac[j]:.1f}% of row norm)")
        print(f"    fc1 in subj_space: {fc1_subj_norm[j]:.4f} "
              f"({100*subj_frac[j]:.1f}% of row norm)")
        print(f"    Gating shift: {gating_shift[j]:.4f}")
        print(f"    Subject std: {subj_std[j]:.4f}")
        print(f"    Gating SNR: {gating_snr[j]:.4f}")
        print(f"    Relation component: R0={rel_comp_r0[j]:.4f}, R1={rel_comp_r1[j]:.4f}")
        print(f"    Pre-GELU stats: R0 mean={pre_gelu_r0[:, j].mean():.4f} ± {pre_gelu_r0[:, j].std():.4f}, "
              f"R1 mean={pre_gelu_r1[:, j].mean():.4f} ± {pre_gelu_r1[:, j].std():.4f}")

    # =====================================================================
    # SUMMARY STATISTICS
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
The pre-activation of each neuron decomposes into:
  pre_j = (relation component) + (subject component)

Relation component: constant per relation, determines gating (on/off)
  Mean |shift|: {gating_shift.abs().mean():.3f}
  Produced by fc1_j projecting onto the relation direction in MLP input space

Subject component: varies across entities, carries type information
  Mean std: {subj_std.mean():.3f}

Gating SNR (|relation shift| / subject std):
  Mean: {gating_snr.mean():.2f}
  Median: {gating_snr.median():.2f}
  >2: {(gating_snr > 2).sum().item()}/256 neurons
  >3: {(gating_snr > 3).sum().item()}/256 neurons
  <1: {(gating_snr < 1).sum().item()}/256 neurons

Interference rate:
  Wrong-relation contexts activating (pre > 0): {100*wrongly_on:.1f}% / {100*wrongly_on_r1:.1f}%
  Matching-relation contexts suppressed (pre < 0): {100*wrongly_off:.1f}% / {100*wrongly_off_r1:.1f}%

Geometry: relation direction is {'mostly' if 1 - rel_in_subj_norm**2 > 0.7 else 'partially'} orthogonal to subject subspace
  {100*(1-rel_in_subj_norm**2):.1f}% of relation direction is orthogonal to top-10 subject PCs
  This near-orthogonality explains why the two signals don't interfere:
  fc1 rows can have independent projections onto the relation direction (for gating)
  and onto the subject subspace (for type-varying activation).
""")


if __name__ == "__main__":
    main()

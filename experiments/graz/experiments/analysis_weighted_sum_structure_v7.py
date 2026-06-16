"""
Structure of the MLP weighted sum: how does the network choose activations
to produce type centroids from the fixed fc2 weight dictionary?

For each type t (within a relation), the MLP computation reduces to:
    a_t @ W ≈ target_direction_t
where:
    a_t ∈ R^256: mean post-GELU activation for type t
    W = fc2.weight.T ��� R^(256×64): fixed output dictionary
    target_direction_t ∈ R^64: what the MLP needs to add

Three analyses:
1. Comparison to least-squares optimal activations
2. Effective dimensionality of type-activation matrix
3. Dictionary geometry (fc2 weight columns)

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/analysis_weighted_sum_structure_v7.py
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


def build_type_centroids(model, vocab, rel_objects_table):
    weight = model.lm_head.weight.detach().cpu()
    cache = {}
    targets = {}
    for i in range(vocab.n_entities):
        for k in range(vocab.num_relations):
            t = vocab.subject_f_type(i) if k == 0 else vocab.subject_g_type(i)
            key = (k, t)
            if key not in cache:
                obj_indices = rel_objects_table[(i, k)]
                obj_tokens = [vocab.object_id(j) for j in obj_indices]
                cache[key] = weight[obj_tokens].mean(dim=0)
            targets[(i, k)] = cache[key]
    return targets, cache


@torch.no_grad()
def extract_all(model, vocab, device):
    """Extract activations and residual stream states for all (subject, relation) pairs."""
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
    fc2 = model.blocks[0].mlp.fc2

    mlp_input = ln2(h0_plus_attn)
    pre_gelu = fc1(mlp_input)
    acts = F.gelu(pre_gelu)  # (B, 256)

    return keys, h0_plus_attn.cpu(), acts.cpu()


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
    model.eval()
    print(f"Loaded checkpoint")

    centroid_targets, centroid_cache = build_type_centroids(model, vocab, rel_objects_table)

    print("Extracting activations...")
    keys, h0_plus_attn_all, acts_all = extract_all(model, vocab, device)
    key_to_idx = {k: idx for idx, k in enumerate(keys)}

    # fc2 weight matrix: W = fc2.weight.T, shape (256, 64)
    fc2 = model.blocks[0].mlp.fc2
    W = fc2.weight.detach().cpu().T  # (256, 64)
    fc2_bias = fc2.bias.detach().cpu() if fc2.bias is not None else torch.zeros(64)

    n = vocab.n_entities

    for k in range(vocab.num_relations):
        print(f"\n{'='*80}")
        print(f"RELATION R{k}")
        print(f"{'='*80}")

        # Group subjects by type
        type_to_subjects = {}
        for i in range(n):
            t = vocab.subject_f_type(i) if k == 0 else vocab.subject_g_type(i)
            type_to_subjects.setdefault(t, []).append(i)

        type_list = sorted(type_to_subjects.keys())
        n_types = len(type_list)

        # Compute type-mean activations and target directions
        type_mean_acts = []  # (n_types, 256)
        target_directions = []  # (n_types, 64) — what MLP should add
        actual_outputs = []  # (n_types, 64) — what MLP actually produces

        for t in type_list:
            subjs = type_to_subjects[t]
            # Mean activation for this type
            type_acts = torch.stack([acts_all[key_to_idx[(i, k)]] for i in subjs])
            mean_act = type_acts.mean(dim=0)
            type_mean_acts.append(mean_act)

            # Target direction: centroid - mean(h0+attn) for this type
            mean_h = torch.stack([h0_plus_attn_all[key_to_idx[(i, k)]] for i in subjs]).mean(dim=0)
            centroid = centroid_targets[(subjs[0], k)]
            target_directions.append(centroid - mean_h)

            # Actual MLP output for this type
            actual_out = mean_act @ W + fc2_bias
            actual_outputs.append(actual_out)

        type_mean_acts = torch.stack(type_mean_acts)  # (45, 256)
        target_directions = torch.stack(target_directions)  # (45, 64)
        actual_outputs = torch.stack(actual_outputs)  # (45, 64)

        # ================================================================
        # ANALYSIS 1: Comparison to least-squares optimal
        # ================================================================
        print(f"\n{'─'*60}")
        print(f"ANALYSIS 1: LEARNED vs OPTIMAL ACTIVATIONS")
        print(f"{'─'*60}")
        print(f"For each type, compute optimal a* = target @ pinv(W)")
        print(f"Compare to actual learned activations.")

        # Optimal activations: a* = target_directions @ pinv(W)
        # pinv(W): W is (256, 64), pinv is (64, 256)
        W_np = W.numpy().astype(np.float64)
        W_pinv = torch.tensor(np.linalg.pinv(W_np), dtype=torch.float32)  # (64, 256)

        # Target for optimal: what we want a @ W to produce
        # This is target_direction - fc2_bias (since actual output = a @ W + bias)
        targets_minus_bias = target_directions - fc2_bias.unsqueeze(0)
        optimal_acts = targets_minus_bias @ W_pinv  # (45, 256)

        # Compare learned vs optimal
        # Cosine similarity of activation vectors
        cos_act = (F.normalize(type_mean_acts, dim=1) *
                   F.normalize(optimal_acts, dim=1)).sum(dim=1)
        print(f"\nCosine similarity (learned activations vs optimal):")
        print(f"  Mean: {cos_act.mean():.4f}")
        print(f"  Std:  {cos_act.std():.4f}")
        print(f"  Min:  {cos_act.min():.4f}")
        print(f"  Max:  {cos_act.max():.4f}")

        # How well does each produce the target?
        learned_output = type_mean_acts @ W + fc2_bias  # (45, 64)
        optimal_output = optimal_acts @ W + fc2_bias  # (45, 64)

        cos_learned_target = (F.normalize(learned_output, dim=1) *
                              F.normalize(target_directions, dim=1)).sum(dim=1)
        cos_optimal_target = (F.normalize(optimal_output, dim=1) *
                              F.normalize(target_directions, dim=1)).sum(dim=1)

        print(f"\nReconstruction quality (cosine with target direction):")
        print(f"  Learned:  mean={cos_learned_target.mean():.4f}, "
              f"std={cos_learned_target.std():.4f}")
        print(f"  Optimal:  mean={cos_optimal_target.mean():.4f}, "
              f"std={cos_optimal_target.std():.4f}")

        # MSE comparison
        mse_learned = ((learned_output - targets_minus_bias) ** 2).sum(dim=1)
        mse_optimal = ((optimal_output - targets_minus_bias) ** 2).sum(dim=1)
        print(f"\nMSE to target (lower=better):")
        print(f"  Learned:  mean={mse_learned.mean():.4f}")
        print(f"  Optimal:  mean={mse_optimal.mean():.4f}")
        print(f"  Ratio (learned/optimal): {mse_learned.mean()/mse_optimal.mean():.2f}")

        # Sparsity comparison
        learned_sparsity = (type_mean_acts < 0.01).float().mean(dim=1)
        optimal_sparsity = (optimal_acts.abs() < 0.01).float().mean(dim=1)
        print(f"\nSparsity (fraction of near-zero activations):")
        print(f"  Learned:  {learned_sparsity.mean():.3f}")
        print(f"  Optimal:  {optimal_sparsity.mean():.3f}")

        # Norm comparison
        print(f"\nActivation norms:")
        print(f"  Learned L1: {type_mean_acts.abs().sum(dim=1).mean():.3f}")
        print(f"  Optimal L1: {optimal_acts.abs().sum(dim=1).mean():.3f}")
        print(f"  Learned L2: {type_mean_acts.norm(dim=1).mean():.3f}")
        print(f"  Optimal L2: {optimal_acts.norm(dim=1).mean():.3f}")

        # Key difference: optimal can have negative values, learned cannot (GELU ≥ 0 mostly)
        print(f"\n  Optimal has negative values: "
              f"{(optimal_acts < 0).float().mean()*100:.1f}% of entries")
        print(f"  Learned has negative values: "
              f"{(type_mean_acts < 0).float().mean()*100:.1f}% of entries")

        # ================================================================
        # ANALYSIS 2: Effective dimensionality
        # ================================================================
        print(f"\n{'─'*60}")
        print(f"ANALYSIS 2: EFFECTIVE DIMENSIONALITY OF TYPE ACTIVATIONS")
        print(f"{'─'*60}")

        # SVD of type-mean activation matrix (45 × 256)
        # Center first
        centered = type_mean_acts - type_mean_acts.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        # S has min(45, 256) = 45 values
        total_var = (S ** 2).sum()
        cumvar = (S ** 2).cumsum(0) / total_var

        print(f"\nSingular values of (45×256) type-mean activation matrix:")
        print(f"  Top 10 singular values: {S[:10].tolist()}")
        print(f"  Variance explained by top-k components:")
        for k_dim in [1, 2, 3, 5, 10, 15, 20, 30, 44]:
            if k_dim <= len(S):
                print(f"    k={k_dim:>2}: {cumvar[k_dim-1]*100:.1f}%")

        # Effective rank (number of components for 95% variance)
        for threshold in [0.90, 0.95, 0.99]:
            n_needed = (cumvar < threshold).sum().item() + 1
            print(f"  Components for {threshold*100:.0f}% variance: {n_needed}/45")

        # Same analysis on the output side: type-mean outputs (45 × 64)
        output_centered = actual_outputs - actual_outputs.mean(dim=0, keepdim=True)
        _, S_out, _ = torch.linalg.svd(output_centered, full_matrices=False)
        total_var_out = (S_out ** 2).sum()
        cumvar_out = (S_out ** 2).cumsum(0) / total_var_out

        print(f"\n  Output space (45×64) singular values:")
        print(f"  Top 10: {S_out[:10].tolist()}")
        for k_dim in [1, 2, 3, 5, 10, 15, 20]:
            if k_dim <= len(S_out):
                print(f"    k={k_dim:>2}: {cumvar_out[k_dim-1]*100:.1f}%")

        # Also: what's the effective rank of the TARGET directions?
        target_centered = target_directions - target_directions.mean(dim=0, keepdim=True)
        _, S_tgt, _ = torch.linalg.svd(target_centered, full_matrices=False)
        total_var_tgt = (S_tgt ** 2).sum()
        cumvar_tgt = (S_tgt ** 2).cumsum(0) / total_var_tgt

        print(f"\n  Target direction space (45×64) singular values:")
        print(f"  Top 10: {S_tgt[:10].tolist()}")
        for k_dim in [1, 2, 3, 5, 10, 15, 20]:
            if k_dim <= len(S_tgt):
                print(f"    k={k_dim:>2}: {cumvar_tgt[k_dim-1]*100:.1f}%")

        # ================================================================
        # ANALYSIS 3: Dictionary geometry
        # ================================================================
        print(f"\n{'─'*60}")
        print(f"ANALYSIS 3: DICTIONARY GEOMETRY (fc2 columns)")
        print(f"{'─'*60}")
        print(f"W = fc2.weight.T has shape (256, 64): 256 output direction vectors")

        # Basic stats on W columns
        W_norms = W.norm(dim=1)  # (256,)
        print(f"\nColumn norms:")
        print(f"  Mean: {W_norms.mean():.4f}")
        print(f"  Std:  {W_norms.std():.4f}")
        print(f"  Min:  {W_norms.min():.4f}")
        print(f"  Max:  {W_norms.max():.4f}")

        # Pairwise cosine similarity between columns
        W_norm_vecs = F.normalize(W, dim=1)  # (256, 64)
        cos_matrix = W_norm_vecs @ W_norm_vecs.T  # (256, 256)
        # Off-diagonal
        mask = ~torch.eye(256).bool()
        off_diag = cos_matrix[mask]
        print(f"\nPairwise cosine similarity between columns:")
        print(f"  Mean: {off_diag.mean():.4f}")
        print(f"  Std:  {off_diag.std():.4f}")
        print(f"  Max (most similar pair): {off_diag.max():.4f}")
        print(f"  Min (most opposing pair): {off_diag.min():.4f}")

        # SVD of W itself — effective rank of the dictionary
        _, S_W, _ = torch.linalg.svd(W, full_matrices=False)
        total_var_W = (S_W ** 2).sum()
        cumvar_W = (S_W ** 2).cumsum(0) / total_var_W
        print(f"\nSingular values of W (256×64):")
        print(f"  All 64: condition number = {S_W[0]/S_W[-1]:.2f}")
        print(f"  Top 10: {S_W[:10].tolist()}")
        print(f"  Variance by top-k:")
        for k_dim in [1, 2, 5, 10, 20, 32, 64]:
            if k_dim <= len(S_W):
                print(f"    k={k_dim:>2}: {cumvar_W[k_dim-1]*100:.1f}%")

        # Which columns are most used by each relation?
        # "Usage" = mean activation × column norm
        usage = type_mean_acts.mean(dim=0) * W_norms  # (256,)
        sorted_usage = usage.argsort(descending=True)
        print(f"\nTop 20 most-used dictionary columns (mean_act × col_norm):")
        print(f"{'Neuron':>8} {'usage':>8} {'mean_act':>9} {'col_norm':>9}")
        print("-" * 38)
        for rank in range(20):
            j = sorted_usage[rank].item()
            print(f"{j:>8d} {usage[j]:>8.4f} "
                  f"{type_mean_acts.mean(dim=0)[j]:>9.4f} {W_norms[j]:>9.4f}")

        # Columns that point toward vs away from centroid cluster
        # Mean centroid direction
        mean_centroid_dir = F.normalize(target_directions.mean(dim=0, keepdim=True), dim=1)
        col_centroid_align = (W_norm_vecs @ mean_centroid_dir.T).squeeze()  # (256,)
        print(f"\nColumn alignment with mean centroid direction:")
        print(f"  Mean alignment: {col_centroid_align.mean():.4f}")
        print(f"  Std: {col_centroid_align.std():.4f}")
        print(f"  Columns aligned (>0.3): {(col_centroid_align > 0.3).sum().item()}/256")
        print(f"  Columns opposing (<-0.3): {(col_centroid_align < -0.3).sum().item()}/256")

        # Do active neurons preferentially align with centroid direction?
        active_mask = type_mean_acts.mean(dim=0) > 0.1
        active_align = col_centroid_align[active_mask].mean()
        inactive_align = col_centroid_align[~active_mask].mean()
        print(f"\n  Active neurons ({active_mask.sum()} total) mean alignment: "
              f"{active_align:.4f}")
        print(f"  Inactive neurons ({(~active_mask).sum()} total) mean alignment: "
              f"{inactive_align:.4f}")

        # Clustering: do columns cluster by type-preference?
        # For each column, find which type uses it most
        # type_mean_acts is (45, 256): for each column, which type has highest activation?
        max_type_per_col = type_mean_acts.argmax(dim=0)  # (256,) — index of type
        # How many columns does each type "own" (have highest activation)?
        type_ownership = torch.bincount(max_type_per_col, minlength=n_types)
        print(f"\n  Column ownership by types (which type activates each column most):")
        print(f"    Mean columns per type: {type_ownership.float().mean():.1f}")
        print(f"    Std: {type_ownership.float().std():.1f}")
        print(f"    Max: {type_ownership.max().item()}")
        print(f"    Min: {type_ownership.min().item()}")
        print(f"    Types with 0 columns: {(type_ownership == 0).sum().item()}")

    # ================================================================
    # CROSS-RELATION ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print(f"CROSS-RELATION DICTIONARY ANALYSIS")
    print(f"{'='*80}")

    # The dictionary W is shared between relations.
    # How do R0 vs R1 use it differently?
    type_to_subjects_r0 = {}
    type_to_subjects_r1 = {}
    for i in range(n):
        t0 = vocab.subject_f_type(i)
        t1 = vocab.subject_g_type(i)
        type_to_subjects_r0.setdefault(t0, []).append(i)
        type_to_subjects_r1.setdefault(t1, []).append(i)

    # Mean activation per type for each relation
    r0_type_acts = []
    for t in sorted(type_to_subjects_r0.keys()):
        subjs = type_to_subjects_r0[t]
        acts = torch.stack([acts_all[key_to_idx[(i, 0)]] for i in subjs])
        r0_type_acts.append(acts.mean(dim=0))
    r0_type_acts = torch.stack(r0_type_acts)  # (45, 256)

    r1_type_acts = []
    for t in sorted(type_to_subjects_r1.keys()):
        subjs = type_to_subjects_r1[t]
        acts = torch.stack([acts_all[key_to_idx[(i, 1)]] for i in subjs])
        r1_type_acts.append(acts.mean(dim=0))
    r1_type_acts = torch.stack(r1_type_acts)  # (45, 256)

    # Per-column: is it used by R0, R1, or both?
    r0_usage = r0_type_acts.mean(dim=0)  # (256,) mean across R0 types
    r1_usage = r1_type_acts.mean(dim=0)  # (256,) mean across R1 types

    r0_active = r0_usage > 0.05
    r1_active = r1_usage > 0.05
    both_active = r0_active & r1_active
    r0_only = r0_active & ~r1_active
    r1_only = r1_active & ~r0_active
    neither = ~r0_active & ~r1_active

    print(f"\nDictionary column usage by relation (threshold: mean act > 0.05):")
    print(f"  R0-only columns: {r0_only.sum().item()}/256")
    print(f"  R1-only columns: {r1_only.sum().item()}/256")
    print(f"  Shared columns:  {both_active.sum().item()}/256")
    print(f"  Neither:         {neither.sum().item()}/256")

    # Correlation of column usage across relations
    usage_corr = np.corrcoef(r0_usage.numpy(), r1_usage.numpy())[0, 1]
    print(f"  Correlation of column usage (R0 vs R1): {usage_corr:.4f}")

    # Do shared columns point in a "general" direction vs relation-specific ones?
    if both_active.sum() > 0 and r0_only.sum() > 0 and r1_only.sum() > 0:
        shared_norms = W_norms[both_active]
        r0_norms = W_norms[r0_only]
        r1_norms = W_norms[r1_only]
        print(f"\n  Column norms by category:")
        print(f"    Shared: {shared_norms.mean():.4f}")
        print(f"    R0-only: {r0_norms.mean():.4f}")
        print(f"    R1-only: {r1_norms.mean():.4f}")


if __name__ == "__main__":
    main()

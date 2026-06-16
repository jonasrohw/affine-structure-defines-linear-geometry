"""
Neuron-level decomposition of MLP contribution to type-centroid mapping.

Two analyses:
  1. Per-neuron contribution decomposition: for each of 256 MLP neurons,
     measure how much its output aligns with the direction needed to reach
     the type centroid.
  2. Neuron type-selectivity: for each neuron, compute F-ratio
     (between-type variance / within-type variance) of activations.

Architecture reminder (1-layer, dim=64):
  MLP: fc1 (64→256), GELU, fc2 (256→64)
  Input to MLP: ln2(h0 + attn_out)
  mlp_out = fc2(gelu(fc1(ln2(h0 + attn_out))))

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/analysis_neuron_decomposition_v7.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config, ModelConfig, TrainConfig, VocabConfig
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
def extract_mlp_internals(model, vocab, device):
    """Extract MLP inputs, neuron activations, and per-neuron contributions.

    Returns:
        mlp_input: dict (i, k) -> tensor(64,)  [input to MLP = ln2(h0+attn)]
        neuron_acts: dict (i, k) -> tensor(256,)  [post-GELU activations]
        neuron_contribs: dict (i, k) -> tensor(256, 64)  [per-neuron output vectors]
        h0_plus_attn: dict (i, k) -> tensor(64,)  [pre-MLP residual]
    """
    model.eval()
    contexts = []
    keys = []
    for i in range(vocab.n_entities):
        for k in range(vocab.num_relations):
            keys.append((i, k))
            contexts.append([vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)])

    batch = torch.tensor(contexts, dtype=torch.long, device=device)

    with model.capture_residuals() as residuals:
        with model.capture_sublayers() as sublayers:
            model(batch)

    h0 = residuals[0][:, 2, :]  # (B, 64) on device
    attn_out = sublayers["attn"][0][:, 2, :]  # (B, 64) on device

    h0_plus_attn_tensor = h0 + attn_out  # (B, 64)

    # Compute MLP input: ln2(h0 + attn_out)
    ln2 = model.blocks[0].ln2
    mlp_input_tensor = ln2(h0_plus_attn_tensor)  # (B, 64)

    # Manually compute neuron activations
    fc1 = model.blocks[0].mlp.fc1
    fc2 = model.blocks[0].mlp.fc2

    pre_gelu = fc1(mlp_input_tensor)  # (B, 256)
    acts = F.gelu(pre_gelu)  # (B, 256)

    # Per-neuron contributions: acts[:, j] * fc2.weight[:, j]
    # fc2.weight shape: (64, 256), so fc2.weight[:, j] is (64,)
    # contribution_j = acts[:, j:j+1] * fc2.weight[:, j]  -> (B, 64)
    # Full: acts.unsqueeze(2) * fc2.weight.T.unsqueeze(0) -> (B, 256, 64)
    fc2_w = fc2.weight.detach()  # (64, 256)
    contribs = acts.unsqueeze(2) * fc2_w.T.unsqueeze(0)  # (B, 256, 64)

    # Move to CPU and build dicts
    h0_plus_attn_cpu = h0_plus_attn_tensor.cpu()
    mlp_input_cpu = mlp_input_tensor.cpu()
    acts_cpu = acts.cpu()
    contribs_cpu = contribs.cpu()

    h0_plus_attn_dict = {}
    mlp_input_dict = {}
    neuron_acts_dict = {}
    neuron_contribs_dict = {}

    for idx, key in enumerate(keys):
        h0_plus_attn_dict[key] = h0_plus_attn_cpu[idx]
        mlp_input_dict[key] = mlp_input_cpu[idx]
        neuron_acts_dict[key] = acts_cpu[idx]
        neuron_contribs_dict[key] = contribs_cpu[idx]

    return mlp_input_dict, neuron_acts_dict, neuron_contribs_dict, h0_plus_attn_dict


def analysis_decomposition(neuron_contribs, h0_plus_attn, centroid_targets, vocab):
    """For each neuron, measure alignment of its contribution with the needed direction.

    needed_direction = centroid - (h0 + attn_out)
    alignment_j = neuron_contrib_j · normalized(needed_direction)

    Returns per-neuron mean alignment (256,) for each relation.
    """
    n_neurons = 256
    results = {}

    for k in range(vocab.num_relations):
        alignments = torch.zeros(n_neurons)
        magnitudes = torch.zeros(n_neurons)
        count = 0

        for i in range(vocab.n_entities):
            target = centroid_targets[(i, k)]
            current = h0_plus_attn[(i, k)]
            needed = target - current  # (64,)
            needed_norm = needed / (needed.norm() + 1e-10)

            contrib = neuron_contribs[(i, k)]  # (256, 64)
            # Projection of each neuron's contribution onto needed direction
            proj = contrib @ needed_norm  # (256,)
            alignments += proj
            magnitudes += contrib.norm(dim=1)
            count += 1

        alignments /= count
        magnitudes /= count
        results[k] = {
            "mean_alignment": alignments,
            "mean_magnitude": magnitudes,
            "fractional_alignment": alignments / (magnitudes + 1e-10),
        }

    return results


def analysis_selectivity(neuron_acts, vocab):
    """Compute F-ratio (between-type / within-type variance) per neuron.

    For R0: group subjects by f-type.
    For R1: group subjects by g-type.
    """
    n_neurons = 256
    results = {}

    for k in range(vocab.num_relations):
        # Collect activations grouped by type
        type_to_acts = {}
        for i in range(vocab.n_entities):
            t = vocab.subject_f_type(i) if k == 0 else vocab.subject_g_type(i)
            if t not in type_to_acts:
                type_to_acts[t] = []
            type_to_acts[t].append(neuron_acts[(i, k)])

        # Stack within each group
        type_groups = {}
        for t, act_list in type_to_acts.items():
            type_groups[t] = torch.stack(act_list)  # (n_per_type, 256)

        # Compute F-ratio per neuron
        all_acts = torch.stack([neuron_acts[(i, k)] for i in range(vocab.n_entities)])
        global_mean = all_acts.mean(dim=0)  # (256,)

        n_types = len(type_groups)
        between_var = torch.zeros(n_neurons)
        within_var = torch.zeros(n_neurons)

        for t, group in type_groups.items():
            n_g = group.shape[0]
            group_mean = group.mean(dim=0)  # (256,)
            between_var += n_g * (group_mean - global_mean) ** 2
            within_var += ((group - group_mean) ** 2).sum(dim=0)

        between_var /= (n_types - 1)
        within_var /= (vocab.n_entities - n_types)

        f_ratio = between_var / (within_var + 1e-10)

        results[k] = {
            "f_ratio": f_ratio,
            "between_var": between_var,
            "within_var": within_var,
        }

    return results


def analysis_relation_specificity(neuron_acts, vocab):
    """Determine whether each neuron is relation-specific or shared.

    For each neuron j, compare its activation vector across all subjects
    under R0 vs R1. Metrics:
      - Mean activation under R0 vs R1
      - Cohen's d: (mean_R0 - mean_R1) / pooled_std
      - Pearson correlation of activation profiles across subjects
      - Relation F-ratio: between-relation var / within-relation var
    """
    n_neurons = 256
    n = vocab.n_entities

    # Stack activations: (n_entities, 256) for each relation
    acts_r0 = torch.stack([neuron_acts[(i, 0)] for i in range(n)])
    acts_r1 = torch.stack([neuron_acts[(i, 1)] for i in range(n)])

    mean_r0 = acts_r0.mean(dim=0)  # (256,)
    mean_r1 = acts_r1.mean(dim=0)  # (256,)
    std_r0 = acts_r0.std(dim=0)
    std_r1 = acts_r1.std(dim=0)

    # Cohen's d
    pooled_std = torch.sqrt((std_r0**2 + std_r1**2) / 2)
    cohens_d = (mean_r0 - mean_r1) / (pooled_std + 1e-10)

    # Pearson correlation across subjects for each neuron
    correlations = torch.zeros(n_neurons)
    for j in range(n_neurons):
        x = acts_r0[:, j]
        y = acts_r1[:, j]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        corr = (x_centered * y_centered).sum() / (
            x_centered.norm() * y_centered.norm() + 1e-10)
        correlations[j] = corr

    # Relation F-ratio (1-way ANOVA: R0 vs R1 as groups)
    global_mean = (acts_r0.mean(dim=0) + acts_r1.mean(dim=0)) / 2
    between_var = n * ((mean_r0 - global_mean)**2 + (mean_r1 - global_mean)**2)
    within_var = ((acts_r0 - mean_r0)**2).sum(dim=0) + ((acts_r1 - mean_r1)**2).sum(dim=0)
    within_var /= (2 * n - 2)
    between_var /= 1  # df_between = 2 - 1 = 1
    rel_f_ratio = between_var / (within_var + 1e-10)

    # Activation ratio (which relation dominates)
    act_ratio = mean_r0 / (mean_r1 + 1e-10)

    return {
        "mean_r0": mean_r0,
        "mean_r1": mean_r1,
        "cohens_d": cohens_d,
        "correlations": correlations,
        "rel_f_ratio": rel_f_ratio,
        "act_ratio": act_ratio,
    }


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

    print("Precomputing relation objects...")
    rel_objects_table = precompute_relation_objects(vocab)
    vocab.relation_objects = lambda subj_idx, rel_k: rel_objects_table[(subj_idx, rel_k)]

    model = build_model(model_cfg, vocab)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {CHECKPOINT}")

    # Build type centroids
    centroid_targets, _ = build_type_centroids(model, vocab, rel_objects_table)

    # Extract MLP internals
    print("Extracting MLP internals (neuron activations + contributions)...")
    mlp_input, neuron_acts, neuron_contribs, h0_plus_attn = \
        extract_mlp_internals(model, vocab, device)

    # === Analysis 1: Per-neuron contribution decomposition ===
    print("\n" + "=" * 80)
    print("ANALYSIS 1: PER-NEURON CONTRIBUTION DECOMPOSITION")
    print("=" * 80)
    print("For each neuron j, measure alignment of its output with the")
    print("direction needed to reach the type centroid from (h0 + attn_out).")
    print("=" * 80)

    decomp = analysis_decomposition(neuron_contribs, h0_plus_attn, centroid_targets, vocab)

    for k in range(vocab.num_relations):
        align = decomp[k]["mean_alignment"]
        mag = decomp[k]["mean_magnitude"]

        print(f"\n--- Relation R{k} ---")
        print(f"Total MLP alignment with centroid direction: {align.sum():.4f}")
        print(f"Total MLP magnitude: {mag.sum():.4f}")

        # Top contributing neurons (by alignment)
        sorted_idx = align.argsort(descending=True)
        print(f"\nTop 20 neurons by alignment with centroid direction:")
        print(f"{'Neuron':>8} {'Alignment':>10} {'Magnitude':>10} {'Frac':>8}")
        print("-" * 40)
        for rank in range(20):
            j = sorted_idx[rank].item()
            print(f"{j:>8d} {align[j]:>10.4f} {mag[j]:>10.4f} "
                  f"{align[j]/mag[j] if mag[j] > 1e-6 else 0:>8.3f}")

        # Bottom neurons (opposing centroid direction)
        print(f"\nBottom 10 neurons (opposing centroid direction):")
        print(f"{'Neuron':>8} {'Alignment':>10} {'Magnitude':>10} {'Frac':>8}")
        print("-" * 40)
        for rank in range(10):
            j = sorted_idx[-(rank+1)].item()
            print(f"{j:>8d} {align[j]:>10.4f} {mag[j]:>10.4f} "
                  f"{align[j]/mag[j] if mag[j] > 1e-6 else 0:>8.3f}")

        # Cumulative alignment (how many neurons needed for 90% of total)
        cumulative = align[sorted_idx].cumsum(0)
        total = align.sum()
        for threshold in [0.5, 0.75, 0.9, 0.95]:
            n_needed = (cumulative < threshold * total).sum().item() + 1
            print(f"  Neurons for {threshold*100:.0f}% of total alignment: {n_needed}/256")

    # === Analysis 2: Neuron type-selectivity ===
    print("\n" + "=" * 80)
    print("ANALYSIS 2: NEURON TYPE-SELECTIVITY (F-ratio)")
    print("=" * 80)
    print("F-ratio = between-type variance / within-type variance")
    print("High F-ratio = neuron fires selectively for certain types.")
    print("=" * 80)

    selectivity = analysis_selectivity(neuron_acts, vocab)

    for k in range(vocab.num_relations):
        f_ratio = selectivity[k]["f_ratio"]
        type_name = "f-type" if k == 0 else "g-type"

        print(f"\n--- Relation R{k} (grouped by {type_name}) ---")
        print(f"Mean F-ratio: {f_ratio.mean():.2f}")
        print(f"Max F-ratio: {f_ratio.max():.2f}")
        print(f"Neurons with F > 10: {(f_ratio > 10).sum().item()}/256")
        print(f"Neurons with F > 50: {(f_ratio > 50).sum().item()}/256")
        print(f"Neurons with F > 100: {(f_ratio > 100).sum().item()}/256")

        sorted_idx = f_ratio.argsort(descending=True)
        print(f"\nTop 20 most type-selective neurons:")
        print(f"{'Neuron':>8} {'F-ratio':>10} {'Between-var':>12} {'Within-var':>12}")
        print("-" * 46)
        for rank in range(20):
            j = sorted_idx[rank].item()
            print(f"{j:>8d} {f_ratio[j]:>10.2f} "
                  f"{selectivity[k]['between_var'][j]:>12.4f} "
                  f"{selectivity[k]['within_var'][j]:>12.6f}")

    # === Cross-analysis: overlap between high-alignment and high-selectivity ===
    print("\n" + "=" * 80)
    print("CROSS-ANALYSIS: ALIGNMENT vs SELECTIVITY")
    print("=" * 80)

    for k in range(vocab.num_relations):
        align = decomp[k]["mean_alignment"]
        f_ratio = selectivity[k]["f_ratio"]

        top_align = set(align.argsort(descending=True)[:30].tolist())
        top_select = set(f_ratio.argsort(descending=True)[:30].tolist())
        overlap = top_align & top_select

        print(f"\nR{k}: Top-30 alignment ∩ Top-30 selectivity = {len(overlap)} neurons")
        if overlap:
            print(f"  Neurons: {sorted(overlap)}")

        # Correlation
        corr = np.corrcoef(align.numpy(), f_ratio.numpy())[0, 1]
        print(f"  Pearson correlation (alignment, F-ratio): {corr:.4f}")

    # === Analysis 3: Relation specificity ===
    print("\n" + "=" * 80)
    print("ANALYSIS 3: RELATION SPECIFICITY")
    print("=" * 80)
    print("For each neuron, compare activations under R0 vs R1 contexts.")
    print("Cohen's d: effect size of relation on activation level.")
    print("Correlation: do activations track subject identity across relations?")
    print("=" * 80)

    rel_spec = analysis_relation_specificity(neuron_acts, vocab)

    cohens_d = rel_spec["cohens_d"]
    correlations = rel_spec["correlations"]
    rel_f = rel_spec["rel_f_ratio"]
    mean_r0 = rel_spec["mean_r0"]
    mean_r1 = rel_spec["mean_r1"]

    abs_d = cohens_d.abs()
    print(f"\nCohen's d (|R0 - R1| / pooled_std):")
    print(f"  Mean |d|: {abs_d.mean():.3f}")
    print(f"  Max |d|: {abs_d.max():.3f}")
    print(f"  Neurons with |d| > 0.8 (large effect): {(abs_d > 0.8).sum().item()}/256")
    print(f"  Neurons with |d| > 0.5 (medium effect): {(abs_d > 0.5).sum().item()}/256")
    print(f"  Neurons with |d| > 0.2 (small effect): {(abs_d > 0.2).sum().item()}/256")
    print(f"  Neurons with |d| < 0.2 (negligible): {(abs_d < 0.2).sum().item()}/256")

    print(f"\nCross-relation correlation (per neuron, across subjects):")
    print(f"  Mean correlation: {correlations.mean():.4f}")
    print(f"  Neurons with corr > 0.8 (shared): {(correlations > 0.8).sum().item()}/256")
    print(f"  Neurons with corr > 0.5: {(correlations > 0.5).sum().item()}/256")
    print(f"  Neurons with corr < 0.2 (relation-specific): {(correlations < 0.2).sum().item()}/256")
    print(f"  Neurons with corr < 0 (anti-correlated): {(correlations < 0).sum().item()}/256")

    # Classify neurons
    r0_specific = (abs_d > 0.8) & (cohens_d > 0)  # much more active for R0
    r1_specific = (abs_d > 0.8) & (cohens_d < 0)  # much more active for R1
    shared_high = (abs_d < 0.2) & (correlations > 0.5)
    shared_low = (abs_d < 0.2) & (correlations < 0.2)

    print(f"\nClassification (|d|>0.8 = relation-specific, |d|<0.2 + corr>0.5 = shared):")
    print(f"  R0-specific (much more active for R0): {r0_specific.sum().item()}/256")
    print(f"  R1-specific (much more active for R1): {r1_specific.sum().item()}/256")
    print(f"  Shared (similar activation, correlated profiles): {shared_high.sum().item()}/256")
    print(f"  Dormant/uncorrelated (low d, low corr): {shared_low.sum().item()}/256")

    # Top R0-specific neurons
    r0_sorted = cohens_d.argsort(descending=True)
    print(f"\nTop 15 R0-preferring neurons (highest Cohen's d):")
    print(f"{'Neuron':>8} {'d':>8} {'mean_R0':>8} {'mean_R1':>8} {'corr':>8} {'F':>8}")
    print("-" * 52)
    for rank in range(15):
        j = r0_sorted[rank].item()
        print(f"{j:>8d} {cohens_d[j]:>8.3f} {mean_r0[j]:>8.4f} "
              f"{mean_r1[j]:>8.4f} {correlations[j]:>8.3f} {rel_f[j]:>8.1f}")

    # Top R1-specific neurons
    r1_sorted = cohens_d.argsort()
    print(f"\nTop 15 R1-preferring neurons (most negative Cohen's d):")
    print(f"{'Neuron':>8} {'d':>8} {'mean_R0':>8} {'mean_R1':>8} {'corr':>8} {'F':>8}")
    print("-" * 52)
    for rank in range(15):
        j = r1_sorted[rank].item()
        print(f"{j:>8d} {cohens_d[j]:>8.3f} {mean_r0[j]:>8.4f} "
              f"{mean_r1[j]:>8.4f} {correlations[j]:>8.3f} {rel_f[j]:>8.1f}")

    # Most shared neurons (high correlation, low d)
    # Sort by correlation descending among neurons with |d| < 0.5
    shared_mask = abs_d < 0.5
    shared_corr = correlations.clone()
    shared_corr[~shared_mask] = -999
    shared_sorted = shared_corr.argsort(descending=True)
    print(f"\nTop 15 shared neurons (high cross-relation correlation, |d|<0.5):")
    print(f"{'Neuron':>8} {'corr':>8} {'d':>8} {'mean_R0':>8} {'mean_R1':>8}")
    print("-" * 44)
    for rank in range(15):
        j = shared_sorted[rank].item()
        if shared_corr[j] < -900:
            break
        print(f"{j:>8d} {correlations[j]:>8.3f} {cohens_d[j]:>8.3f} "
              f"{mean_r0[j]:>8.4f} {mean_r1[j]:>8.4f}")

    # Cross-reference with alignment analysis
    print(f"\nRelation specificity of top alignment neurons:")
    print(f"{'Neuron':>8} {'R0-align':>9} {'R1-align':>9} {'d':>8} {'corr':>8} {'Role':>12}")
    print("-" * 60)
    # Union of top-10 from each relation
    top_r0_align = decomp[0]["mean_alignment"].argsort(descending=True)[:10].tolist()
    top_r1_align = decomp[1]["mean_alignment"].argsort(descending=True)[:10].tolist()
    shown = set()
    for j in top_r0_align + top_r1_align:
        if j in shown:
            continue
        shown.add(j)
        a0 = decomp[0]["mean_alignment"][j].item()
        a1 = decomp[1]["mean_alignment"][j].item()
        d = cohens_d[j].item()
        c = correlations[j].item()
        if abs(d) > 0.8:
            role = "R0-spec" if d > 0 else "R1-spec"
        elif c > 0.5:
            role = "shared"
        else:
            role = "mixed"
        print(f"{j:>8d} {a0:>9.4f} {a1:>9.4f} {d:>8.3f} {c:>8.3f} {role:>12}")

    # === Summary statistics ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for k in range(vocab.num_relations):
        align = decomp[k]["mean_alignment"]
        mag = decomp[k]["mean_magnitude"]
        f_ratio = selectivity[k]["f_ratio"]

        total_align = align.sum().item()
        total_mag = mag.sum().item()

        # How concentrated is the alignment?
        pos_align = align[align > 0]
        neg_align = align[align < 0]

        print(f"\nR{k}:")
        print(f"  Total alignment: {total_align:.4f}")
        print(f"  Positive contributors: {len(pos_align)}/256 "
              f"(sum={pos_align.sum():.4f})")
        print(f"  Negative contributors: {len(neg_align)}/256 "
              f"(sum={neg_align.sum():.4f})")
        print(f"  Top-10 neurons account for: "
              f"{align[align.argsort(descending=True)[:10]].sum()/total_align*100:.1f}% "
              f"of total alignment")
        print(f"  Top-30 neurons account for: "
              f"{align[align.argsort(descending=True)[:30]].sum()/total_align*100:.1f}% "
              f"of total alignment")
        print(f"  Highly selective neurons (F>50): "
              f"{(f_ratio > 50).sum().item()}/256")


if __name__ == "__main__":
    main()

"""
Per-subject neuron contribution analysis.

Question: does the MLP use the same computation for all subjects of a given
type, or are there subject-specific differences in how neurons contribute
to the centroid direction?

If the MLP implements a uniform affine-like map, then within a type group
(same target centroid), the per-neuron contribution profile should be similar
across subjects (just scaled by input magnitude). If the computation is
subject-specific, we'd see high within-type variance of neuron contributions.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/analysis_neuron_per_subject_v7.py
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
def compute_per_subject_contributions(model, vocab, device):
    """Compute per-neuron alignment contribution for each (subject, relation).

    Returns:
        contributions: tensor (n_entities, num_relations, 256) — per-neuron
            projection onto centroid direction for each subject
        activations: tensor (n_entities, num_relations, 256) — raw post-GELU
        magnitudes: tensor (n_entities, num_relations, 256) — ||neuron_j output||
    """
    model.eval()
    n = vocab.n_entities
    n_neurons = 4 * 64  # 256

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

    fc2_w = fc2.weight.detach()  # (64, 256)
    # Per-neuron output vectors: (B, 256, 64)
    contribs_vec = acts.unsqueeze(2) * fc2_w.T.unsqueeze(0)

    h0_plus_attn_cpu = h0_plus_attn.cpu()
    acts_cpu = acts.cpu()
    contribs_vec_cpu = contribs_vec.cpu()

    return keys, h0_plus_attn_cpu, acts_cpu, contribs_vec_cpu


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

    print("Computing per-subject neuron contributions...")
    keys, h0_plus_attn_all, acts_all, contribs_vec_all = \
        compute_per_subject_contributions(model, vocab, device)

    # Build index: keys[idx] = (i, k)
    key_to_idx = {k: idx for idx, k in enumerate(keys)}

    n = vocab.n_entities
    n_neurons = 256

    for k in range(vocab.num_relations):
        print(f"\n{'='*80}")
        print(f"RELATION R{k}")
        print(f"{'='*80}")

        # Group subjects by type
        type_to_subjects = {}
        for i in range(n):
            t = vocab.subject_f_type(i) if k == 0 else vocab.subject_g_type(i)
            type_to_subjects.setdefault(t, []).append(i)

        n_types = len(type_to_subjects)
        type_list = sorted(type_to_subjects.keys())

        # For each subject, compute per-neuron alignment with centroid direction
        # alignment_j(i) = contrib_vec[i,j,:] · normalized(centroid - h0_plus_attn)
        alignments = torch.zeros(n, n_neurons)  # (n_entities, 256)
        raw_acts = torch.zeros(n, n_neurons)

        for i in range(n):
            idx = key_to_idx[(i, k)]
            target = centroid_targets[(i, k)]
            current = h0_plus_attn_all[idx]
            needed = target - current
            needed_norm = needed / (needed.norm() + 1e-10)

            contrib = contribs_vec_all[idx]  # (256, 64)
            alignments[i] = contrib @ needed_norm
            raw_acts[i] = acts_all[idx]

        # === Within-type vs between-type variance of neuron contributions ===
        global_mean = alignments.mean(dim=0)  # (256,)

        between_var = torch.zeros(n_neurons)
        within_var = torch.zeros(n_neurons)

        for t in type_list:
            subjs = type_to_subjects[t]
            group = alignments[subjs]  # (n_per_type, 256)
            group_mean = group.mean(dim=0)
            n_g = len(subjs)
            between_var += n_g * (group_mean - global_mean) ** 2
            within_var += ((group - group_mean) ** 2).sum(dim=0)

        between_var /= (n_types - 1)
        within_var /= (n - n_types)
        contribution_f_ratio = between_var / (within_var + 1e-10)

        # Overall stats
        total_within = within_var.sum().item()
        total_between = between_var.sum().item()
        print(f"\nVariance decomposition of per-neuron centroid alignment:")
        print(f"  Total between-type variance: {total_between:.4f}")
        print(f"  Total within-type variance:  {total_within:.4f}")
        print(f"  Ratio (between/within):      {total_between/total_within:.2f}")
        print(f"  Mean F-ratio per neuron:     {contribution_f_ratio.mean():.2f}")

        # What fraction of total variance is within-type?
        total_var = alignments.var(dim=0)  # (256,)
        # within_type_fraction per neuron
        within_frac = within_var / (total_var * (n-1)/(n-n_types) + 1e-10)

        print(f"\n  Mean within-type fraction of total variance: "
              f"{within_frac.mean():.4f}")
        print(f"  Neurons where within-type > 50% of variance: "
              f"{(within_frac > 0.5).sum().item()}/256")
        print(f"  Neurons where within-type > 80% of variance: "
              f"{(within_frac > 0.8).sum().item()}/256")

        # === Per-neuron detail: which neurons vary most within type? ===
        within_std = within_var.sqrt()
        neuron_mean_align = alignments.mean(dim=0)
        cv = within_std / (neuron_mean_align.abs() + 1e-10)  # coefficient of variation

        print(f"\nPer-neuron within-type variability (top 20 by within-type std):")
        print(f"{'Neuron':>8} {'mean_align':>11} {'within_std':>11} "
              f"{'between_std':>12} {'F-ratio':>8} {'CV':>8}")
        print("-" * 64)
        sorted_within = within_std.argsort(descending=True)
        for rank in range(20):
            j = sorted_within[rank].item()
            print(f"{j:>8d} {neuron_mean_align[j]:>11.4f} "
                  f"{within_std[j]:>11.4f} {between_var[j].sqrt():>12.4f} "
                  f"{contribution_f_ratio[j]:>8.2f} {cv[j]:>8.2f}")

        # === Contribution profile similarity within type ===
        # Pick a few example types and show how similar neuron profiles are
        # within vs across types
        print(f"\nContribution profile similarity (cosine) within vs across types:")

        within_cosines = []
        across_cosines = []

        # Sample 10 types
        sample_types = type_list[:10]
        for t_idx, t in enumerate(sample_types):
            subjs = type_to_subjects[t]
            if len(subjs) < 2:
                continue
            # Within-type: cosine between contribution profiles of subjects in same type
            profiles = alignments[subjs]  # (n_per_type, 256)
            profiles_norm = F.normalize(profiles, dim=1)
            sim_matrix = profiles_norm @ profiles_norm.T
            # Upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones(len(subjs), len(subjs)), diagonal=1).bool()
            within_cosines.extend(sim_matrix[mask].tolist())

            # Across-type: compare with next type
            if t_idx + 1 < len(sample_types):
                t2 = sample_types[t_idx + 1]
                subjs2 = type_to_subjects[t2]
                profiles2 = alignments[subjs2]
                profiles2_norm = F.normalize(profiles2, dim=1)
                cross_sim = profiles_norm @ profiles2_norm.T
                across_cosines.extend(cross_sim.flatten().tolist())

        if within_cosines and across_cosines:
            print(f"  Within-type mean cosine: {np.mean(within_cosines):.4f} "
                  f"(std={np.std(within_cosines):.4f})")
            print(f"  Across-type mean cosine: {np.mean(across_cosines):.4f} "
                  f"(std={np.std(across_cosines):.4f})")

        # Do this for ALL types (not just sample)
        all_within_cos = []
        all_across_cos = []
        # Compute mean profile per type
        type_mean_profiles = []
        for t in type_list:
            subjs = type_to_subjects[t]
            profiles = alignments[subjs]
            type_mean_profiles.append(profiles.mean(dim=0))

            # Within-type cosines (sample pairs to avoid O(n^2))
            if len(subjs) >= 2:
                profiles_norm = F.normalize(profiles, dim=1)
                sim = profiles_norm @ profiles_norm.T
                mask = torch.triu(torch.ones(len(subjs), len(subjs)), diagonal=1).bool()
                all_within_cos.extend(sim[mask].tolist())

        type_mean_profiles = torch.stack(type_mean_profiles)
        type_profiles_norm = F.normalize(type_mean_profiles, dim=1)
        across_sim = type_profiles_norm @ type_profiles_norm.T
        mask = torch.triu(torch.ones(n_types, n_types), diagonal=1).bool()
        all_across_cos = across_sim[mask].tolist()

        print(f"\n  Full dataset:")
        print(f"  Within-type mean cosine (all pairs): {np.mean(all_within_cos):.4f} "
              f"(std={np.std(all_within_cos):.4f}, n={len(all_within_cos)})")
        print(f"  Across-type mean cosine (type means): {np.mean(all_across_cos):.4f} "
              f"(std={np.std(all_across_cos):.4f}, n={len(all_across_cos)})")

        # === Activation pattern analysis ===
        # Are neurons operating in linear regime of GELU?
        # In linear regime: gelu(x) ≈ x for x >> 0, gelu(x) ≈ 0 for x << 0
        # Check what fraction of activations are in the "transition zone"
        print(f"\n  GELU regime analysis (raw pre-GELU values):")
        # Need pre-gelu values
        # We have acts (post-GELU). Let's check raw activations for regime.
        # Actually we stored post-GELU acts. Let's analyze their distribution.
        rel_acts = torch.stack([acts_all[key_to_idx[(i, k)]] for i in range(n)])

        # For GELU: linear when input > ~2, dead when input < ~-2
        # Post-GELU: if act > 0.5 likely in linear regime
        # if act ≈ 0 could be dead or transition
        mean_act_per_neuron = rel_acts.mean(dim=0)
        std_act_per_neuron = rel_acts.std(dim=0)

        # Fraction of inputs where neuron is essentially dead (act < 0.01)
        dead_frac = (rel_acts < 0.01).float().mean(dim=0)
        # Fraction where neuron is in confident linear regime (act > 0.3)
        linear_frac = (rel_acts > 0.3).float().mean(dim=0)
        # Transition zone
        transition_frac = 1 - dead_frac - linear_frac

        print(f"  Mean fraction dead (act<0.01):       {dead_frac.mean():.3f}")
        print(f"  Mean fraction linear (act>0.3):      {linear_frac.mean():.3f}")
        print(f"  Mean fraction transition:            {transition_frac.mean():.3f}")

        # Neurons that are always active vs sometimes dead
        always_active = (dead_frac < 0.01).sum().item()
        sometimes_dead = ((dead_frac > 0.1) & (dead_frac < 0.9)).sum().item()
        mostly_dead = (dead_frac > 0.9).sum().item()
        print(f"  Neurons always active (dead<1%):     {always_active}/256")
        print(f"  Neurons sometimes dead (10-90%):     {sometimes_dead}/256")
        print(f"  Neurons mostly dead (dead>90%):      {mostly_dead}/256")

        # === Key question: within a type, is the variation due to GELU nonlinearity? ===
        # If neurons switch between active/dead for different subjects of same type,
        # that's subject-specific gating
        print(f"\n  Within-type activation consistency:")
        type_consistency = []
        for t in type_list[:20]:
            subjs = type_to_subjects[t]
            group_acts = rel_acts[subjs]  # (n_per_type, 256)
            # For each neuron: is it consistently active or dead within the type?
            # Measure: std of binary(act > 0.01) within the type
            active_binary = (group_acts > 0.01).float()
            # If std=0, neuron is always same state. If std>0, it switches.
            consistency = 1 - active_binary.std(dim=0)  # 1 = perfectly consistent
            type_consistency.append(consistency)

        type_consistency = torch.stack(type_consistency)
        mean_consistency = type_consistency.mean(dim=0)
        print(f"  Mean neuron on/off consistency within type: "
              f"{mean_consistency.mean():.4f}")
        print(f"  Neurons with perfect consistency (>0.99):  "
              f"{(mean_consistency > 0.99).sum().item()}/256")
        print(f"  Neurons with low consistency (<0.9):       "
              f"{(mean_consistency < 0.9).sum().item()}/256")


if __name__ == "__main__":
    main()

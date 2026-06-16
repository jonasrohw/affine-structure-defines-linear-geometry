"""
Detailed implementation analysis for the Ravfogel bijective 1-layer model.

Replicates the full v7 analysis suite:
  1. Attention decomposition (per-head outputs, relation direction)
  2. Neuron-level decomposition (alignment with target, relation specificity)
  3. Neuron input decomposition (gating SNR, gate interference, fc1 geometry)
  4. Activation histograms (distributions by relation, GELU derivatives)
  5. Weighted sum structure (effective dimensionality, dictionary partition)
  6. Affine activation fit (R² of MLP input → activations → output)

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_1layer_analysis.py --seed 42
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import build_model

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
def extract_all_states(model, vocab, device, batch_size=512):
    """Extract embeddings, attention outputs, MLP inputs/outputs at subject position."""
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]

    embeddings = {}
    attn_outputs = {}
    mlp_inputs = {}
    mlp_outputs = {}
    pre_gelu = {}
    post_gelu = {}

    block = model.blocks[0]
    fc1 = block.mlp.fc1
    fc2 = block.mlp.fc2
    ln1 = block.ln1
    ln2 = block.ln2

    for start in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[start:start + batch_size]
        contexts = torch.tensor(
            [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)]
             for i, k in batch_pairs],
            dtype=torch.long, device=device)

        B, T = contexts.shape
        positions = torch.arange(T, device=device).unsqueeze(0)
        h0 = model.token_emb(contexts) + model.pos_emb(positions)

        attn_out = block.attn(ln1(h0))
        h1 = h0 + attn_out

        mlp_in = ln2(h1)
        pre_g = fc1(mlp_in)
        post_g = F.gelu(pre_g)
        mlp_out = fc2(post_g)

        for idx, (i, k) in enumerate(batch_pairs):
            embeddings[(i, k)] = h0[idx, 2, :].cpu()
            attn_outputs[(i, k)] = attn_out[idx, 2, :].cpu()
            mlp_inputs[(i, k)] = mlp_in[idx, 2, :].cpu()
            mlp_outputs[(i, k)] = mlp_out[idx, 2, :].cpu()
            pre_gelu[(i, k)] = pre_g[idx, 2, :].cpu()
            post_gelu[(i, k)] = post_g[idx, 2, :].cpu()

    return embeddings, attn_outputs, mlp_inputs, mlp_outputs, pre_gelu, post_gelu


@torch.no_grad()
def extract_per_head_attn(model, vocab, device, batch_size=512):
    """Extract per-head attention outputs and attention weights at subject position."""
    import math
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]

    block = model.blocks[0]
    ln1 = block.ln1
    attn = block.attn
    n_heads = attn.n_heads
    head_dim = attn.head_dim

    head_outputs = {h: {} for h in range(n_heads)}
    attn_weights_all = {h: {} for h in range(n_heads)}

    for start in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[start:start + batch_size]
        contexts = torch.tensor(
            [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)]
             for i, k in batch_pairs],
            dtype=torch.long, device=device)

        B, T = contexts.shape
        positions = torch.arange(T, device=device).unsqueeze(0)
        h0 = model.token_emb(contexts) + model.pos_emb(positions)

        x = ln1(h0)
        C = x.shape[-1]
        qkv = attn.qkv(x)
        q, k_proj, v = qkv.split(C, dim=2)

        def reshape(t):
            return t.view(B, T, n_heads, head_dim).transpose(1, 2)

        q, k_proj, v = reshape(q), reshape(k_proj), reshape(v)
        att = (q @ k_proj.transpose(-2, -1)) / math.sqrt(head_dim)
        causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, n_heads, T, head_dim)

        # Per-head: project through out_proj slice
        W_o = attn.out_proj.weight  # (hidden_dim, hidden_dim)
        for h in range(n_heads):
            W_o_h = W_o[:, h * head_dim:(h + 1) * head_dim]  # (hidden_dim, head_dim)
            head_out = y[:, h, 2, :] @ W_o_h.T  # (B, hidden_dim)
            for idx, (i, kk) in enumerate(batch_pairs):
                head_outputs[h][(i, kk)] = head_out[idx].cpu()
                attn_weights_all[h][(i, kk)] = att[idx, h, 2, :].cpu()

    return head_outputs, attn_weights_all


def analysis_1_attention_decomposition(model, vocab, device, results):
    """Per-head attention output analysis, relation direction, subject-specific info."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: ATTENTION DECOMPOSITION")
    print("=" * 70)

    head_outputs, attn_weights = extract_per_head_attn(model, vocab, device)
    n_heads = len(head_outputs)

    # Full attention output (sum of heads)
    full_attn = {}
    for i in range(vocab.n_entities):
        for k in range(vocab.num_relations):
            full_attn[(i, k)] = sum(head_outputs[h][(i, k)] for h in range(n_heads))

    # Relation means
    r0_vecs = torch.stack([full_attn[(i, 0)] for i in range(vocab.n_entities)])
    r1_vecs = torch.stack([full_attn[(i, 1)] for i in range(vocab.n_entities)])
    mean_r0 = r0_vecs.mean(0)
    mean_r1 = r1_vecs.mean(0)

    cos_means = F.cosine_similarity(mean_r0.unsqueeze(0), mean_r1.unsqueeze(0)).item()

    # Within-relation pairwise cosine (sample)
    rng = np.random.RandomState(42)
    n_sample = min(500, vocab.n_entities)
    idxs = rng.choice(vocab.n_entities, n_sample, replace=False)
    r0_sample = r0_vecs[idxs]
    r1_sample = r1_vecs[idxs]
    within_r0 = F.cosine_similarity(r0_sample.unsqueeze(0), r0_sample.unsqueeze(1), dim=2)
    within_r1 = F.cosine_similarity(r1_sample.unsqueeze(0), r1_sample.unsqueeze(1), dim=2)
    mask = torch.triu(torch.ones(n_sample, n_sample), diagonal=1).bool()
    within_r0_cos = within_r0[mask].mean().item()
    within_r1_cos = within_r1[mask].mean().item()

    # Cross-relation cosine (same subject)
    cross_cos = F.cosine_similarity(r0_sample, r1_sample, dim=1).mean().item()

    # Between vs within relation variance
    all_vecs = torch.cat([r0_vecs, r1_vecs], dim=0)
    grand_mean = all_vecs.mean(0)
    total_var = ((all_vecs - grand_mean) ** 2).sum().item()
    between_var = (vocab.n_entities * ((mean_r0 - grand_mean) ** 2).sum().item() +
                   vocab.n_entities * ((mean_r1 - grand_mean) ** 2).sum().item())
    between_pct = between_var / total_var * 100

    print(f"\nFull attention output at subject position:")
    print(f"  Mean norm: R0={r0_vecs.norm(dim=1).mean():.3f}, R1={r1_vecs.norm(dim=1).mean():.3f}")
    print(f"  cos(mean_R0, mean_R1) = {cos_means:.3f}")
    print(f"  Within-R0 pairwise cos = {within_r0_cos:.3f}")
    print(f"  Within-R1 pairwise cos = {within_r1_cos:.3f}")
    print(f"  Cross-relation cos (same subj) = {cross_cos:.3f}")
    print(f"  Between-relation variance: {between_pct:.1f}%")
    print(f"  Within-relation variance: {100 - between_pct:.1f}%")

    attn_result = {
        "full_attn": {
            "r0_norm": round(r0_vecs.norm(dim=1).mean().item(), 4),
            "r1_norm": round(r1_vecs.norm(dim=1).mean().item(), 4),
            "cos_means": round(cos_means, 4),
            "within_r0_cos": round(within_r0_cos, 4),
            "within_r1_cos": round(within_r1_cos, 4),
            "cross_rel_cos": round(cross_cos, 4),
            "between_rel_var_pct": round(between_pct, 1),
        }
    }

    # Per-head analysis
    print(f"\nPer-head analysis:")
    rel_dir = mean_r0 - mean_r1
    rel_dir_norm = rel_dir.norm().item()
    print(f"  Relation direction norm: {rel_dir_norm:.3f}")

    for h in range(n_heads):
        h_r0 = torch.stack([head_outputs[h][(i, 0)] for i in range(vocab.n_entities)])
        h_r1 = torch.stack([head_outputs[h][(i, 1)] for i in range(vocab.n_entities)])
        h_mean_r0 = h_r0.mean(0)
        h_mean_r1 = h_r1.mean(0)
        h_cos = F.cosine_similarity(h_mean_r0.unsqueeze(0), h_mean_r1.unsqueeze(0)).item()

        h_all = torch.cat([h_r0, h_r1], dim=0)
        h_grand = h_all.mean(0)
        h_total = ((h_all - h_grand) ** 2).sum().item()
        h_between = (vocab.n_entities * ((h_mean_r0 - h_grand) ** 2).sum().item() +
                     vocab.n_entities * ((h_mean_r1 - h_grand) ** 2).sum().item())
        h_between_pct = h_between / max(h_total, 1e-10) * 100

        h_within_r0 = F.cosine_similarity(h_r0[idxs].unsqueeze(0), h_r0[idxs].unsqueeze(1), dim=2)
        h_within_r1 = F.cosine_similarity(h_r1[idxs].unsqueeze(0), h_r1[idxs].unsqueeze(1), dim=2)
        h_wr0 = h_within_r0[mask].mean().item()
        h_wr1 = h_within_r1[mask].mean().item()

        # Contribution to relation direction
        h_diff = h_mean_r0 - h_mean_r1
        proj = (h_diff @ rel_dir).item() / (rel_dir_norm ** 2)
        proj_pct = proj * rel_dir_norm / rel_dir_norm * 100

        print(f"\n  Head {h}:")
        print(f"    Norm: R0={h_r0.norm(dim=1).mean():.3f}, R1={h_r1.norm(dim=1).mean():.3f}")
        print(f"    cos(mean_R0, mean_R1) = {h_cos:.3f}")
        print(f"    Between-rel variance: {h_between_pct:.1f}%")
        print(f"    Within-rel cos: R0={h_wr0:.3f}, R1={h_wr1:.3f}")
        print(f"    Rel direction contribution: {proj:.3f} ({proj_pct:.1f}%)")

        attn_result[f"head{h}"] = {
            "r0_norm": round(h_r0.norm(dim=1).mean().item(), 3),
            "r1_norm": round(h_r1.norm(dim=1).mean().item(), 3),
            "cos_means": round(h_cos, 3),
            "between_rel_var_pct": round(h_between_pct, 1),
            "within_r0_cos": round(h_wr0, 3),
            "within_r1_cos": round(h_wr1, 3),
            "rel_dir_proj": round(proj, 4),
            "rel_dir_pct": round(proj_pct, 1),
        }

    # Attention weight patterns
    print(f"\n  Attention weights at pos 2 (subject → [sep, rel, self]):")
    for h in range(n_heads):
        r0_weights = torch.stack([attn_weights[h][(i, 0)] for i in range(vocab.n_entities)])
        r1_weights = torch.stack([attn_weights[h][(i, 1)] for i in range(vocab.n_entities)])
        r0_mean = r0_weights.mean(0)
        r1_mean = r1_weights.mean(0)
        print(f"    Head {h}: R0=[{r0_mean[0]:.3f}, {r0_mean[1]:.3f}, {r0_mean[2]:.3f}] "
              f"R1=[{r1_mean[0]:.3f}, {r1_mean[1]:.3f}, {r1_mean[2]:.3f}]")
        attn_result[f"head{h}"]["attn_weights_r0"] = [round(x, 3) for x in r0_mean.tolist()]
        attn_result[f"head{h}"]["attn_weights_r1"] = [round(x, 3) for x in r1_mean.tolist()]

    results["attention_decomposition"] = attn_result
    return full_attn


def analysis_2_neuron_decomposition(model, vocab, device, embeddings, attn_outputs,
                                    mlp_inputs, mlp_outputs, post_gelu, results):
    """Per-neuron contribution alignment with target direction, relation specificity."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: NEURON-LEVEL DECOMPOSITION")
    print("=" * 70)

    block = model.blocks[0]
    fc2_weight = block.mlp.fc2.weight.detach().cpu()  # (64, 256)
    lm_head = model.lm_head.weight.detach().cpu()  # (V, 64)
    ln_f_weight = model.ln_f.weight.detach().cpu()
    ln_f_bias = model.ln_f.bias.detach().cpu()
    n_neurons = fc2_weight.shape[1]

    # For each (i, k), compute needed direction = target_obj_emb - pre_mlp_residual_projected
    # Actually: the model predicts by: logits = lm_head @ ln_f(h0 + attn + mlp)
    # Target direction in residual stream: the direction the MLP must push toward
    # We use: target = lm_head[obj_id] (the object's embedding row)
    # and: needed = target - (contribution from h0 + attn projected through ln_f + lm_head)
    # Simpler approach: measure alignment of each neuron's contribution with the
    # direction from pre-MLP residual to the target object embedding

    neuron_results = {"per_relation": {}}

    for k in range(vocab.num_relations):
        # Per-neuron activations and contributions
        activations = []  # (n_entities, n_neurons)
        for i in range(vocab.n_entities):
            activations.append(post_gelu[(i, k)])
        activations = torch.stack(activations)  # (2025, 256)

        # Each neuron j contributes: activation[j] * fc2[:, j]
        # Alignment with "needed direction" toward target object embedding
        target_obj_embs = []
        for i in range(vocab.n_entities):
            obj_idx = vocab.relation_object(i, k)
            target_obj_embs.append(lm_head[vocab.object_id(obj_idx)])
        target_obj_embs = torch.stack(target_obj_embs)  # (2025, 64)

        # Pre-MLP residual
        pre_mlp = torch.stack([embeddings[(i, k)] + attn_outputs[(i, k)]
                               for i in range(vocab.n_entities)])  # (2025, 64)

        # Needed direction (unnormalized): what MLP should add
        needed = target_obj_embs - pre_mlp  # approximate
        needed_norm = needed / (needed.norm(dim=1, keepdim=True) + 1e-8)

        # Per-neuron alignment
        alignments = []
        for j in range(n_neurons):
            fc2_col = fc2_weight[:, j]  # (64,)
            contribution = activations[:, j:j+1] * fc2_col.unsqueeze(0)  # (2025, 64)
            align = (contribution * needed_norm).sum(dim=1).mean().item()
            alignments.append(align)
        alignments = np.array(alignments)

        total_align = sum(alignments)
        positive = (alignments > 0).sum()
        negative = (alignments < 0).sum()

        sorted_idx = np.argsort(-alignments)
        cumsum = np.cumsum(alignments[sorted_idx])
        top10_pct = cumsum[9] / total_align * 100 if total_align > 0 else 0
        top30_pct = cumsum[29] / total_align * 100 if total_align > 0 else 0
        for_90 = np.searchsorted(cumsum, 0.9 * total_align) + 1 if total_align > 0 else 256

        print(f"\n  R{k} neuron alignment:")
        print(f"    Total alignment: {total_align:.3f}")
        print(f"    Positive contributors: {positive}/{n_neurons}")
        print(f"    Negative contributors: {negative}/{n_neurons}")
        print(f"    Top-10 neurons: {top10_pct:.1f}% of alignment")
        print(f"    Top-30 neurons: {top30_pct:.1f}% of alignment")
        print(f"    Neurons for 90%: {for_90}/{n_neurons}")

        neuron_results["per_relation"][f"r{k}"] = {
            "total_alignment": round(total_align, 4),
            "positive_contributors": int(positive),
            "negative_contributors": int(negative),
            "top10_pct": round(top10_pct, 1),
            "top30_pct": round(top30_pct, 1),
            "neurons_for_90pct": int(for_90),
        }

    # Relation specificity (Cohen's d)
    print(f"\n  Relation specificity (Cohen's d):")
    r0_acts = torch.stack([post_gelu[(i, 0)] for i in range(vocab.n_entities)])  # (2025, 256)
    r1_acts = torch.stack([post_gelu[(i, 1)] for i in range(vocab.n_entities)])  # (2025, 256)

    r0_mean = r0_acts.mean(0)
    r1_mean = r1_acts.mean(0)
    pooled_std = ((r0_acts.var(0) + r1_acts.var(0)) / 2).sqrt()
    cohens_d = ((r0_mean - r1_mean) / (pooled_std + 1e-8)).numpy()

    abs_d = np.abs(cohens_d)
    r0_specific = ((abs_d > 0.8) & (cohens_d > 0)).sum()
    r1_specific = ((abs_d > 0.8) & (cohens_d < 0)).sum()
    large_d = (abs_d > 0.8).sum()
    negligible_d = (abs_d < 0.2).sum()

    # Cross-relation correlation
    cross_corrs = []
    for j in range(n_neurons):
        c = np.corrcoef(r0_acts[:, j].numpy(), r1_acts[:, j].numpy())[0, 1]
        cross_corrs.append(c if not np.isnan(c) else 0)
    cross_corrs = np.array(cross_corrs)

    shared = ((abs_d < 0.2) & (cross_corrs > 0.5)).sum()

    print(f"    Mean |Cohen's d|: {abs_d.mean():.2f}")
    print(f"    Neurons |d| > 0.8 (relation-specific): {large_d}/{n_neurons}")
    print(f"    Neurons |d| < 0.2 (negligible): {negligible_d}/{n_neurons}")
    print(f"    R0-specific (d > 0.8): {r0_specific}")
    print(f"    R1-specific (d < -0.8): {r1_specific}")
    print(f"    Shared (|d|<0.2, corr>0.5): {shared}")
    print(f"    Mean cross-relation corr: {cross_corrs.mean():.3f}")
    print(f"    Neurons with corr > 0.8: {(cross_corrs > 0.8).sum()}")

    neuron_results["relation_specificity"] = {
        "mean_abs_d": round(float(abs_d.mean()), 3),
        "max_abs_d": round(float(abs_d.max()), 3),
        "large_d": int(large_d),
        "negligible_d": int(negligible_d),
        "r0_specific": int(r0_specific),
        "r1_specific": int(r1_specific),
        "shared": int(shared),
        "mean_cross_corr": round(float(cross_corrs.mean()), 3),
        "high_corr": int((cross_corrs > 0.8).sum()),
    }

    results["neuron_decomposition"] = neuron_results
    return cohens_d, cross_corrs


def analysis_3_input_decomposition(model, vocab, device, mlp_inputs, pre_gelu, post_gelu, results):
    """Pre-activation decomposition into relation + subject, gating SNR, interference."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: NEURON INPUT DECOMPOSITION")
    print("=" * 70)

    block = model.blocks[0]
    fc1_weight = block.mlp.fc1.weight.detach().cpu()  # (256, 64)
    fc1_bias = block.mlp.fc1.bias.detach().cpu()  # (256,)
    n_neurons = fc1_weight.shape[0]

    # MLP input decomposition: relation mean + subject residual
    r0_inputs = torch.stack([mlp_inputs[(i, 0)] for i in range(vocab.n_entities)])
    r1_inputs = torch.stack([mlp_inputs[(i, 1)] for i in range(vocab.n_entities)])
    mean_r0 = r0_inputs.mean(0)
    mean_r1 = r1_inputs.mean(0)

    # Relation direction in MLP input space
    rel_dir = mean_r0 - mean_r1
    rel_dir_unit = rel_dir / rel_dir.norm()

    # Pre-activations by relation
    r0_pre = torch.stack([pre_gelu[(i, 0)] for i in range(vocab.n_entities)])  # (2025, 256)
    r1_pre = torch.stack([pre_gelu[(i, 1)] for i in range(vocab.n_entities)])

    # Relation component = fc1 @ mean_k + bias
    rel_comp_r0 = (fc1_weight @ mean_r0 + fc1_bias)  # (256,)
    rel_comp_r1 = (fc1_weight @ mean_r1 + fc1_bias)

    # Subject component = fc1 @ (x - mean_k)
    subj_comp_r0 = r0_pre - rel_comp_r0.unsqueeze(0)  # (2025, 256)
    subj_comp_r1 = r1_pre - rel_comp_r1.unsqueeze(0)

    # Verify decomposition
    recon_r0 = rel_comp_r0.unsqueeze(0) + subj_comp_r0
    max_err = (recon_r0 - r0_pre).abs().max().item()
    print(f"\n  Decomposition verification: max error = {max_err:.2e}")

    # Gating shift and SNR
    gating_shift = (rel_comp_r0 - rel_comp_r1).numpy()  # (256,)
    subj_std_r0 = subj_comp_r0.std(0).numpy()
    subj_std_r1 = subj_comp_r1.std(0).numpy()
    subj_std = (subj_std_r0 + subj_std_r1) / 2
    gating_snr = np.abs(gating_shift) / (subj_std + 1e-8)

    print(f"\n  Gating SNR:")
    print(f"    Mean gating shift: {np.abs(gating_shift).mean():.3f}")
    print(f"    Mean subject std: {subj_std.mean():.3f}")
    print(f"    Mean SNR: {gating_snr.mean():.2f}")
    print(f"    Median SNR: {np.median(gating_snr):.2f}")
    print(f"    SNR > 2: {(gating_snr > 2).sum()}/{n_neurons}")
    print(f"    SNR > 3: {(gating_snr > 3).sum()}/{n_neurons}")
    print(f"    SNR > 5: {(gating_snr > 5).sum()}/{n_neurons}")
    print(f"    SNR < 1: {(gating_snr < 1).sum()}/{n_neurons}")

    input_results = {
        "gating_snr": {
            "mean_shift": round(float(np.abs(gating_shift).mean()), 3),
            "mean_subj_std": round(float(subj_std.mean()), 3),
            "mean_snr": round(float(gating_snr.mean()), 2),
            "median_snr": round(float(np.median(gating_snr)), 2),
            "snr_gt2": int((gating_snr > 2).sum()),
            "snr_gt3": int((gating_snr > 3).sum()),
            "snr_gt5": int((gating_snr > 5).sum()),
            "snr_lt1": int((gating_snr < 1).sum()),
        }
    }

    # Gate interference
    r0_post = torch.stack([post_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_post = torch.stack([post_gelu[(i, 1)] for i in range(vocab.n_entities)])

    # Classify neurons by relation
    r0_mean_act = r0_post.mean(0).numpy()
    r1_mean_act = r1_post.mean(0).numpy()
    r0_neurons = np.where((r0_mean_act > 0.05) & (r1_mean_act <= 0.05))[0]
    r1_neurons = np.where((r1_mean_act > 0.05) & (r0_mean_act <= 0.05))[0]
    dead_neurons = np.where((r0_mean_act <= 0.05) & (r1_mean_act <= 0.05))[0]

    print(f"\n  Neuron classification:")
    print(f"    R0-only: {len(r0_neurons)}")
    print(f"    R1-only: {len(r1_neurons)}")
    print(f"    Dead: {len(dead_neurons)}")

    # Interference: wrong-relation activation
    if len(r0_neurons) > 0:
        r1_at_r0 = r1_post[:, r0_neurons]  # R1 context at R0 neurons
        r0_at_r0 = r0_post[:, r0_neurons]
        wrong_active = (r1_at_r0 > 0.05).float().mean().item()
        suppressed = (r0_at_r0 < 0).float().mean().item()
        print(f"\n  Gate interference:")
        print(f"    R0 neurons active in R1 context (>0.05): {wrong_active:.3f}")
        print(f"    R0 neurons suppressed in R0 context (<0): {suppressed:.3f}")
    else:
        wrong_active = 0
        suppressed = 0

    if len(r1_neurons) > 0:
        r0_at_r1 = r0_post[:, r1_neurons]
        r1_at_r1 = r1_post[:, r1_neurons]
        wrong_active_r1 = (r0_at_r1 > 0.05).float().mean().item()
        suppressed_r1 = (r1_at_r1 < 0).float().mean().item()
        print(f"    R1 neurons active in R0 context (>0.05): {wrong_active_r1:.3f}")
        print(f"    R1 neurons suppressed in R1 context (<0): {suppressed_r1:.3f}")
    else:
        wrong_active_r1 = 0
        suppressed_r1 = 0

    input_results["neuron_classification"] = {
        "r0_only": int(len(r0_neurons)),
        "r1_only": int(len(r1_neurons)),
        "dead": int(len(dead_neurons)),
    }
    input_results["gate_interference"] = {
        "r0_neurons_active_in_r1": round(wrong_active, 4),
        "r0_neurons_suppressed_in_r0": round(suppressed, 4),
        "r1_neurons_active_in_r0": round(wrong_active_r1, 4),
        "r1_neurons_suppressed_in_r1": round(suppressed_r1, 4),
    }

    # fc1 row geometry
    print(f"\n  fc1 row geometry:")
    fc1_norms = fc1_weight.norm(dim=1).numpy()
    fc1_rel_proj = (fc1_weight @ rel_dir_unit).numpy()

    # Subject subspace: top PCs of subject variation in MLP input
    all_inputs = torch.cat([r0_inputs, r1_inputs], dim=0)
    grand_mean = all_inputs.mean(0)
    centered = all_inputs - grand_mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    subj_pcs = Vh[:10]  # top 10 PCs
    var_explained = (S[:10] ** 2).sum() / (S ** 2).sum()

    fc1_in_subj = (fc1_weight @ subj_pcs.T).norm(dim=1).numpy()  # (256,)
    fc1_rel_frac = np.abs(fc1_rel_proj) / (fc1_norms + 1e-8)
    fc1_subj_frac = fc1_in_subj / (fc1_norms + 1e-8)

    print(f"    Mean fc1 row norm: {fc1_norms.mean():.3f}")
    print(f"    |fc1 · rel_dir|: {np.abs(fc1_rel_proj).mean():.3f}")
    print(f"    ||fc1 in subj_space||: {fc1_in_subj.mean():.3f}")
    print(f"    Fraction in relation dir: {fc1_rel_frac.mean():.0%}")
    print(f"    Fraction in subject space: {fc1_subj_frac.mean():.0%}")
    print(f"    Subject PCs variance explained: {var_explained:.3f}")

    # Orthogonality of relation and subject signals
    rel_in_subj = (rel_dir_unit @ subj_pcs.T).norm().item() ** 2
    print(f"\n  Orthogonality:")
    print(f"    Relation direction in subject subspace: {rel_in_subj:.1%}")
    print(f"    Relation direction orthogonal: {1 - rel_in_subj:.1%}")

    input_results["fc1_geometry"] = {
        "mean_norm": round(float(fc1_norms.mean()), 3),
        "mean_rel_proj": round(float(np.abs(fc1_rel_proj).mean()), 3),
        "mean_subj_proj": round(float(fc1_in_subj.mean()), 3),
        "rel_fraction": round(float(fc1_rel_frac.mean()), 3),
        "subj_fraction": round(float(fc1_subj_frac.mean()), 3),
        "subj_pc_variance": round(float(var_explained), 3),
    }
    input_results["orthogonality"] = {
        "rel_in_subj_pct": round(rel_in_subj * 100, 1),
        "rel_orthogonal_pct": round((1 - rel_in_subj) * 100, 1),
    }

    results["input_decomposition"] = input_results
    return r0_neurons, r1_neurons, dead_neurons


def analysis_4_activation_histograms(model, vocab, device, post_gelu, pre_gelu,
                                     r0_neurons, r1_neurons, results):
    """Activation distribution statistics by relation context."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: ACTIVATION DISTRIBUTIONS")
    print("=" * 70)

    r0_post = torch.stack([post_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_post = torch.stack([post_gelu[(i, 1)] for i in range(vocab.n_entities)])
    r0_pre = torch.stack([pre_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_pre = torch.stack([pre_gelu[(i, 1)] for i in range(vocab.n_entities)])

    hist_results = {}

    # Post-GELU activation statistics
    print(f"\n  Post-GELU activation statistics:")
    if len(r0_neurons) > 0:
        r0_at_r0 = r0_post[:, r0_neurons]
        r1_at_r0 = r1_post[:, r0_neurons]
        print(f"    R0 contexts at R0 neurons: mean={r0_at_r0.mean():.3f}, std={r0_at_r0.std():.3f}")
        print(f"    R1 contexts at R0 neurons: mean={r1_at_r0.mean():.3f}, std={r1_at_r0.std():.3f}")
        hist_results["r0_at_r0"] = {"mean": round(r0_at_r0.mean().item(), 3),
                                     "std": round(r0_at_r0.std().item(), 3)}
        hist_results["r1_at_r0"] = {"mean": round(r1_at_r0.mean().item(), 3),
                                     "std": round(r1_at_r0.std().item(), 3)}

    if len(r1_neurons) > 0:
        r1_at_r1 = r1_post[:, r1_neurons]
        r0_at_r1 = r0_post[:, r1_neurons]
        print(f"    R1 contexts at R1 neurons: mean={r1_at_r1.mean():.3f}, std={r1_at_r1.std():.3f}")
        print(f"    R0 contexts at R1 neurons: mean={r0_at_r1.mean():.3f}, std={r0_at_r1.std():.3f}")
        hist_results["r1_at_r1"] = {"mean": round(r1_at_r1.mean().item(), 3),
                                     "std": round(r1_at_r1.std().item(), 3)}
        hist_results["r0_at_r1"] = {"mean": round(r0_at_r1.mean().item(), 3),
                                     "std": round(r0_at_r1.std().item(), 3)}

    # Pre-GELU statistics
    print(f"\n  Pre-GELU statistics:")
    if len(r0_neurons) > 0:
        r0_pre_r0 = r0_pre[:, r0_neurons]
        r1_pre_r0 = r1_pre[:, r0_neurons]
        print(f"    R0 contexts at R0 neurons: mean={r0_pre_r0.mean():.3f}, std={r0_pre_r0.std():.3f}")
        print(f"    R1 contexts at R0 neurons: mean={r1_pre_r0.mean():.3f}, std={r1_pre_r0.std():.3f}")
        gating_shift_r0 = r0_pre_r0.mean().item() - r1_pre_r0.mean().item()
        print(f"    Gating shift (R0 neurons): {gating_shift_r0:.3f}")
        hist_results["pre_gelu_r0_at_r0"] = {"mean": round(r0_pre_r0.mean().item(), 3),
                                               "std": round(r0_pre_r0.std().item(), 3)}
        hist_results["pre_gelu_r1_at_r0"] = {"mean": round(r1_pre_r0.mean().item(), 3),
                                               "std": round(r1_pre_r0.std().item(), 3)}

    if len(r1_neurons) > 0:
        r1_pre_r1 = r1_pre[:, r1_neurons]
        r0_pre_r1 = r0_pre[:, r1_neurons]
        print(f"    R1 contexts at R1 neurons: mean={r1_pre_r1.mean():.3f}, std={r1_pre_r1.std():.3f}")
        print(f"    R0 contexts at R1 neurons: mean={r0_pre_r1.mean():.3f}, std={r0_pre_r1.std():.3f}")
        gating_shift_r1 = r1_pre_r1.mean().item() - r0_pre_r1.mean().item()
        print(f"    Gating shift (R1 neurons): {gating_shift_r1:.3f}")
        hist_results["pre_gelu_r1_at_r1"] = {"mean": round(r1_pre_r1.mean().item(), 3),
                                               "std": round(r1_pre_r1.std().item(), 3)}
        hist_results["pre_gelu_r0_at_r1"] = {"mean": round(r0_pre_r1.mean().item(), 3),
                                               "std": round(r0_pre_r1.std().item(), 3)}

    # GELU derivative analysis
    print(f"\n  GELU derivative analysis:")

    def gelu_derivative(x):
        phi = torch.distributions.Normal(0, 1).log_prob(x).exp()
        Phi = 0.5 * (1 + torch.erf(x / np.sqrt(2)))
        return Phi + x * phi

    if len(r0_neurons) > 0:
        d_r0_r0 = gelu_derivative(r0_pre[:, r0_neurons])
        d_r1_r0 = gelu_derivative(r1_pre[:, r0_neurons])
        print(f"    GELU'(x) R0 at R0 neurons: mean={d_r0_r0.mean():.3f}")
        print(f"    GELU'(x) R1 at R0 neurons: mean={d_r1_r0.mean():.3f}")
        hist_results["gelu_deriv_r0_at_r0"] = round(d_r0_r0.mean().item(), 3)
        hist_results["gelu_deriv_r1_at_r0"] = round(d_r1_r0.mean().item(), 3)

    if len(r1_neurons) > 0:
        d_r1_r1 = gelu_derivative(r1_pre[:, r1_neurons])
        d_r0_r1 = gelu_derivative(r0_pre[:, r1_neurons])
        print(f"    GELU'(x) R1 at R1 neurons: mean={d_r1_r1.mean():.3f}")
        print(f"    GELU'(x) R0 at R1 neurons: mean={d_r0_r1.mean():.3f}")
        hist_results["gelu_deriv_r1_at_r1"] = round(d_r1_r1.mean().item(), 3)
        hist_results["gelu_deriv_r0_at_r1"] = round(d_r0_r1.mean().item(), 3)

    results["activation_histograms"] = hist_results


def analysis_5_weighted_sum(model, vocab, device, post_gelu, mlp_outputs, results):
    """Effective dimensionality, dictionary geometry, relation partition."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: WEIGHTED SUM STRUCTURE")
    print("=" * 70)

    block = model.blocks[0]
    fc2_weight = block.mlp.fc2.weight.detach().cpu()  # (64, 256)
    fc2_bias = block.mlp.fc2.bias.detach().cpu()  # (64,)
    lm_head = model.lm_head.weight.detach().cpu()
    n_neurons = fc2_weight.shape[1]

    ws_results = {}

    # Effective dimensionality of activation patterns
    print(f"\n  Effective dimensionality of activations:")
    for k in range(vocab.num_relations):
        acts = torch.stack([post_gelu[(i, k)] for i in range(vocab.n_entities)])  # (2025, 256)
        acts_centered = acts - acts.mean(0)
        U, S, Vh = torch.linalg.svd(acts_centered, full_matrices=False)
        var = S ** 2
        cumvar = var.cumsum(0) / var.sum()

        for_90 = (cumvar < 0.90).sum().item() + 1
        for_95 = (cumvar < 0.95).sum().item() + 1
        for_99 = (cumvar < 0.99).sum().item() + 1

        print(f"\n  R{k} activation SVD:")
        print(f"    Components for 90% variance: {for_90}")
        print(f"    Components for 95% variance: {for_95}")
        print(f"    Components for 99% variance: {for_99}")
        print(f"    Top-5 variance: {cumvar[4]:.3f}")
        print(f"    Top-10 variance: {cumvar[9]:.3f}")
        print(f"    Top-20 variance: {cumvar[19]:.3f}")

        ws_results[f"r{k}_dimensionality"] = {
            "for_90pct": int(for_90),
            "for_95pct": int(for_95),
            "for_99pct": int(for_99),
            "top5_var": round(cumvar[4].item(), 3),
            "top10_var": round(cumvar[9].item(), 3),
            "top20_var": round(cumvar[19].item(), 3),
        }

    # MLP output dimensionality
    print(f"\n  MLP output dimensionality:")
    for k in range(vocab.num_relations):
        outs = torch.stack([mlp_outputs[(i, k)] for i in range(vocab.n_entities)])
        outs_centered = outs - outs.mean(0)
        U, S, Vh = torch.linalg.svd(outs_centered, full_matrices=False)
        var = S ** 2
        cumvar = var.cumsum(0) / var.sum()
        for_90 = (cumvar < 0.90).sum().item() + 1
        for_95 = (cumvar < 0.95).sum().item() + 1
        print(f"    R{k}: 90% in {for_90} components, 95% in {for_95}")
        ws_results[f"r{k}_output_dim"] = {"for_90pct": int(for_90), "for_95pct": int(for_95)}

    # Dictionary geometry (fc2 columns)
    print(f"\n  Dictionary geometry (fc2 columns):")
    W = fc2_weight.T  # (256, 64) — each row is a dictionary element
    col_norms = W.norm(dim=1)
    print(f"    Mean column norm: {col_norms.mean():.3f}")
    print(f"    Std column norm: {col_norms.std():.3f}")

    # Pairwise cosines (sample)
    W_normed = W / (col_norms.unsqueeze(1) + 1e-8)
    cos_mat = W_normed @ W_normed.T
    mask_upper = torch.triu(torch.ones(n_neurons, n_neurons), diagonal=1).bool()
    pairwise_cos = cos_mat[mask_upper]
    print(f"    Mean pairwise cosine: {pairwise_cos.mean():.3f}")
    print(f"    Max pairwise cosine: {pairwise_cos.max():.3f}")
    print(f"    Min pairwise cosine: {pairwise_cos.min():.3f}")

    # Condition number
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    cond = (S[0] / S[-1]).item()
    print(f"    Condition number: {cond:.2f}")

    ws_results["dictionary"] = {
        "mean_norm": round(col_norms.mean().item(), 3),
        "std_norm": round(col_norms.std().item(), 3),
        "mean_pairwise_cos": round(pairwise_cos.mean().item(), 3),
        "max_pairwise_cos": round(pairwise_cos.max().item(), 3),
        "min_pairwise_cos": round(pairwise_cos.min().item(), 3),
        "condition_number": round(cond, 2),
    }

    # Column usage by relation
    r0_acts = torch.stack([post_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_acts = torch.stack([post_gelu[(i, 1)] for i in range(vocab.n_entities)])
    r0_usage = (r0_acts.mean(0) > 0.05).numpy()
    r1_usage = (r1_acts.mean(0) > 0.05).numpy()

    r0_only_cols = (r0_usage & ~r1_usage).sum()
    r1_only_cols = (~r0_usage & r1_usage).sum()
    shared_cols = (r0_usage & r1_usage).sum()
    dead_cols = (~r0_usage & ~r1_usage).sum()

    print(f"\n  Column usage by relation:")
    print(f"    R0-only: {r0_only_cols}")
    print(f"    R1-only: {r1_only_cols}")
    print(f"    Shared: {shared_cols}")
    print(f"    Dead: {dead_cols}")

    usage_corr = np.corrcoef(r0_usage.astype(float), r1_usage.astype(float))[0, 1]
    print(f"    Usage correlation: {usage_corr:.3f}")

    ws_results["column_usage"] = {
        "r0_only": int(r0_only_cols),
        "r1_only": int(r1_only_cols),
        "shared": int(shared_cols),
        "dead": int(dead_cols),
        "usage_corr": round(float(usage_corr), 3),
    }

    # Learned vs optimal activations
    print(f"\n  Learned vs optimal activations:")
    for k in range(vocab.num_relations):
        acts = torch.stack([post_gelu[(i, k)] for i in range(vocab.n_entities)])
        outs = torch.stack([mlp_outputs[(i, k)] for i in range(vocab.n_entities)])

        # Target: what MLP output should be (object embedding - pre-mlp contribution)
        # Use actual MLP output as an approximation of the target
        target = outs.mean(0)  # mean output direction for this relation

        # Optimal activation for the mean target
        # a* = (target - bias) @ pinv(fc2_weight.T)  => (target - bias) @ pinv(W)
        W_pinv = torch.linalg.pinv(fc2_weight)  # (256, 64)
        a_opt = (target - fc2_bias) @ W_pinv.T  # (256,)
        a_learned = acts.mean(0)

        cos_a = F.cosine_similarity(a_learned.unsqueeze(0), a_opt.unsqueeze(0)).item()
        out_learned = a_learned @ fc2_weight.T + fc2_bias
        out_opt = a_opt @ fc2_weight.T + fc2_bias
        cos_out_learned = F.cosine_similarity(out_learned.unsqueeze(0), target.unsqueeze(0)).item()
        cos_out_opt = F.cosine_similarity(out_opt.unsqueeze(0), target.unsqueeze(0)).item()

        sparsity_learned = (a_learned.abs() < 0.01).float().mean().item()
        sparsity_opt = (a_opt.abs() < 0.01).float().mean().item()

        print(f"\n  R{k}:")
        print(f"    cos(learned a, optimal a*): {cos_a:.3f}")
        print(f"    cos(learned output, target): {cos_out_learned:.3f}")
        print(f"    cos(optimal output, target): {cos_out_opt:.3f}")
        print(f"    Learned sparsity: {sparsity_learned:.3f}")
        print(f"    Optimal sparsity: {sparsity_opt:.3f}")
        print(f"    Learned L1 norm: {a_learned.abs().sum():.1f}")
        print(f"    Optimal L1 norm: {a_opt.abs().sum():.1f}")

        ws_results[f"r{k}_learned_vs_optimal"] = {
            "cos_activations": round(cos_a, 3),
            "cos_output_learned": round(cos_out_learned, 3),
            "cos_output_optimal": round(cos_out_opt, 3),
            "sparsity_learned": round(sparsity_learned, 3),
            "sparsity_optimal": round(sparsity_opt, 3),
        }

    results["weighted_sum_structure"] = ws_results


def analysis_6_affine_fit(model, vocab, device, mlp_inputs, post_gelu, mlp_outputs, results):
    """R² of affine approximation: MLP input → activations and MLP input → output."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: AFFINE ACTIVATION FIT")
    print("=" * 70)

    affine_results = {}

    for k in range(vocab.num_relations):
        X = torch.stack([mlp_inputs[(i, k)] for i in range(vocab.n_entities)])  # (2025, 64)
        Y_act = torch.stack([post_gelu[(i, k)] for i in range(vocab.n_entities)])  # (2025, 256)
        Y_out = torch.stack([mlp_outputs[(i, k)] for i in range(vocab.n_entities)])  # (2025, 64)

        n = X.shape[0]

        # Fit affine: Y = X @ W + b  =>  [X, 1] @ [W; b] = Y
        X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)  # (2025, 65)

        # MLP input → post-GELU activations
        sol_act, _, _, _ = torch.linalg.lstsq(X_aug, Y_act)
        Y_act_pred = X_aug @ sol_act
        ss_res_act = ((Y_act - Y_act_pred) ** 2).sum().item()
        ss_tot_act = ((Y_act - Y_act.mean(0)) ** 2).sum().item()
        r2_act = 1 - ss_res_act / ss_tot_act

        # Per-entity cosine
        cos_act = F.cosine_similarity(Y_act, Y_act_pred, dim=1)
        rel_err_act = (Y_act - Y_act_pred).norm(dim=1) / (Y_act.norm(dim=1) + 1e-8)

        # Per-neuron R²
        per_neuron_r2 = []
        for j in range(Y_act.shape[1]):
            ss_r = ((Y_act[:, j] - Y_act_pred[:, j]) ** 2).sum().item()
            ss_t = ((Y_act[:, j] - Y_act[:, j].mean()) ** 2).sum().item()
            per_neuron_r2.append(1 - ss_r / max(ss_t, 1e-10))
        per_neuron_r2 = np.array(per_neuron_r2)

        # MLP input → MLP output (end-to-end)
        sol_out, _, _, _ = torch.linalg.lstsq(X_aug, Y_out)
        Y_out_pred = X_aug @ sol_out
        ss_res_out = ((Y_out - Y_out_pred) ** 2).sum().item()
        ss_tot_out = ((Y_out - Y_out.mean(0)) ** 2).sum().item()
        r2_out = 1 - ss_res_out / ss_tot_out

        cos_out = F.cosine_similarity(Y_out, Y_out_pred, dim=1)
        rel_err_out = (Y_out - Y_out_pred).norm(dim=1) / (Y_out.norm(dim=1) + 1e-8)

        print(f"\n  R{k} — MLP input → post-GELU activations:")
        print(f"    Overall R²: {r2_act:.4f}")
        print(f"    Per-neuron R² mean: {per_neuron_r2.mean():.3f}, median: {np.median(per_neuron_r2):.3f}")
        print(f"    Per-neuron R² min: {per_neuron_r2.min():.3f}")
        print(f"    Per-entity cosine: mean={cos_act.mean():.4f}, min={cos_act.min():.3f}")
        print(f"    Per-entity rel error: mean={rel_err_act.mean():.3f}")

        print(f"\n  R{k} — MLP input → MLP output (end-to-end):")
        print(f"    Overall R²: {r2_out:.4f}")
        print(f"    Per-entity cosine: mean={cos_out.mean():.4f}, min={cos_out.min():.3f}")
        print(f"    Per-entity rel error: mean={rel_err_out.mean():.3f}")

        affine_results[f"r{k}"] = {
            "input_to_activations": {
                "r2": round(r2_act, 4),
                "per_neuron_r2_mean": round(float(per_neuron_r2.mean()), 3),
                "per_neuron_r2_median": round(float(np.median(per_neuron_r2)), 3),
                "per_neuron_r2_min": round(float(per_neuron_r2.min()), 3),
                "per_entity_cos_mean": round(cos_act.mean().item(), 4),
                "per_entity_cos_min": round(cos_act.min().item(), 3),
                "per_entity_rel_error": round(rel_err_act.mean().item(), 3),
            },
            "input_to_output": {
                "r2": round(r2_out, 4),
                "per_entity_cos_mean": round(cos_out.mean().item(), 4),
                "per_entity_cos_min": round(cos_out.min().item(), 3),
                "per_entity_rel_error": round(rel_err_out.mean().item(), 3),
            }
        }

    # Sanity check: MLP input → pre-GELU should be perfect
    from experiments.ravfogel_1layer import RavfogelVocab  # already imported
    block = model.blocks[0]
    fc1 = block.mlp.fc1
    print(f"\n  Sanity: MLP input → pre-GELU is exactly affine (by construction)")

    results["affine_fit"] = affine_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading checkpoint...")
    model, vocab, model_cfg = load_checkpoint(args.seed, device)
    print(f"Model: dim={model_cfg.hidden_dim}, layers={model_cfg.n_layers}, "
          f"heads={model_cfg.n_heads}")
    print(f"Entities: {vocab.n_entities}, Relations: {vocab.num_relations}")
    print(f"MLP neurons: {4 * model_cfg.hidden_dim}")

    print("\nExtracting hidden states for all 4050 (subject, relation) pairs...")
    embeddings, attn_outputs, mlp_inputs, mlp_outputs, pre_gelu, post_gelu = \
        extract_all_states(model, vocab, device)

    results = {
        "experiment": "ravfogel_1layer_analysis",
        "seed": args.seed,
        "n_entities": vocab.n_entities,
        "dim": model_cfg.hidden_dim,
        "n_neurons": 4 * model_cfg.hidden_dim,
    }

    # Run all analyses
    full_attn = analysis_1_attention_decomposition(model, vocab, device, results)
    cohens_d, cross_corrs = analysis_2_neuron_decomposition(
        model, vocab, device, embeddings, attn_outputs,
        mlp_inputs, mlp_outputs, post_gelu, results)
    r0_neurons, r1_neurons, dead_neurons = analysis_3_input_decomposition(
        model, vocab, device, mlp_inputs, pre_gelu, post_gelu, results)
    analysis_4_activation_histograms(
        model, vocab, device, post_gelu, pre_gelu, r0_neurons, r1_neurons, results)
    analysis_5_weighted_sum(model, vocab, device, post_gelu, mlp_outputs, results)
    analysis_6_affine_fit(model, vocab, device, mlp_inputs, post_gelu, mlp_outputs, results)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_1layer_analysis_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

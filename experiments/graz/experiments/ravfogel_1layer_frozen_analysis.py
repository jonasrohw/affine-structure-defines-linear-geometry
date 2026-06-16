"""
Detailed implementation analysis for the frozen-affine-embedding 1-layer model.

Same 6-analysis suite as ravfogel_1layer_analysis.py, applied to the checkpoint
where subject/object embeddings are frozen with provably-affine structure.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_1layer_frozen_analysis.py --seed 42
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
from experiments.ravfogel_1layer import RavfogelVocab, generate_permutations

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def load_checkpoint(seed, device):
    ckpt_dir = ROOT / f"checkpoints_ravfogel_1l_frozen_seed{seed}"
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

    base_vectors = ckpt["base_vectors"]
    return model, vocab, model_cfg, base_vectors


@torch.no_grad()
def extract_all_states(model, vocab, device, batch_size=512):
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

        y = att @ v

        W_o = attn.out_proj.weight
        for h in range(n_heads):
            W_o_h = W_o[:, h * head_dim:(h + 1) * head_dim]
            head_out = y[:, h, 2, :] @ W_o_h.T
            for idx, (i, kk) in enumerate(batch_pairs):
                head_outputs[h][(i, kk)] = head_out[idx].cpu()
                attn_weights_all[h][(i, kk)] = att[idx, h, 2, :].cpu()

    return head_outputs, attn_weights_all


def analysis_1_attention(model, vocab, device, results):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: ATTENTION DECOMPOSITION")
    print("=" * 70)

    head_outputs, attn_weights = extract_per_head_attn(model, vocab, device)
    n_heads = len(head_outputs)

    full_attn = {}
    for i in range(vocab.n_entities):
        for k in range(vocab.num_relations):
            full_attn[(i, k)] = sum(head_outputs[h][(i, k)] for h in range(n_heads))

    r0_vecs = torch.stack([full_attn[(i, 0)] for i in range(vocab.n_entities)])
    r1_vecs = torch.stack([full_attn[(i, 1)] for i in range(vocab.n_entities)])
    mean_r0 = r0_vecs.mean(0)
    mean_r1 = r1_vecs.mean(0)

    cos_means = F.cosine_similarity(mean_r0.unsqueeze(0), mean_r1.unsqueeze(0)).item()

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
    cross_cos = F.cosine_similarity(r0_sample, r1_sample, dim=1).mean().item()

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

    rel_dir = mean_r0 - mean_r1
    rel_dir_norm = rel_dir.norm().item()
    print(f"  Relation direction norm: {rel_dir_norm:.3f}")

    attn_result = {
        "between_rel_var_pct": round(between_pct, 1),
        "cos_means": round(cos_means, 4),
        "within_r0_cos": round(within_r0_cos, 4),
        "within_r1_cos": round(within_r1_cos, 4),
        "cross_rel_cos": round(cross_cos, 4),
        "rel_dir_norm": round(rel_dir_norm, 4),
        "r0_norm": round(r0_vecs.norm(dim=1).mean().item(), 4),
        "r1_norm": round(r1_vecs.norm(dim=1).mean().item(), 4),
    }

    # Per-head
    print(f"\nPer-head analysis:")
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

        h_diff = h_mean_r0 - h_mean_r1
        proj = (h_diff @ rel_dir).item() / (rel_dir_norm ** 2) if rel_dir_norm > 1e-8 else 0

        print(f"  Head {h}: norm_R0={h_r0.norm(dim=1).mean():.3f}  "
              f"cos_means={h_cos:.3f}  between_rel={h_between_pct:.1f}%  "
              f"rel_dir_contrib={proj*100:.1f}%")

        attn_result[f"head{h}"] = {
            "r0_norm": round(h_r0.norm(dim=1).mean().item(), 3),
            "r1_norm": round(h_r1.norm(dim=1).mean().item(), 3),
            "cos_means": round(h_cos, 3),
            "between_rel_pct": round(h_between_pct, 1),
            "rel_dir_contrib_pct": round(proj * 100, 1),
        }

    # Attention weights
    print(f"\n  Attention weights at pos 2 (subject → [sep, rel, self]):")
    for h in range(n_heads):
        r0_w = torch.stack([attn_weights[h][(i, 0)] for i in range(vocab.n_entities)]).mean(0)
        r1_w = torch.stack([attn_weights[h][(i, 1)] for i in range(vocab.n_entities)]).mean(0)
        print(f"    Head {h}: R0=[{r0_w[0]:.3f}, {r0_w[1]:.3f}, {r0_w[2]:.3f}] "
              f"R1=[{r1_w[0]:.3f}, {r1_w[1]:.3f}, {r1_w[2]:.3f}]")
        attn_result[f"head{h}"]["attn_weights_r0"] = [round(x, 3) for x in r0_w.tolist()]
        attn_result[f"head{h}"]["attn_weights_r1"] = [round(x, 3) for x in r1_w.tolist()]

    results["attention_decomposition"] = attn_result


def analysis_2_neurons(model, vocab, device, embeddings, attn_outputs,
                       mlp_inputs, mlp_outputs, post_gelu, results):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: NEURON-LEVEL DECOMPOSITION")
    print("=" * 70)

    block = model.blocks[0]
    fc2_weight = block.mlp.fc2.weight.detach().cpu()
    lm_head = model.lm_head.weight.detach().cpu()
    n_neurons = fc2_weight.shape[1]

    neuron_results = {"per_relation": {}}

    for k in range(vocab.num_relations):
        activations = torch.stack([post_gelu[(i, k)] for i in range(vocab.n_entities)])

        target_obj_embs = torch.stack([
            lm_head[vocab.object_id(vocab.relation_object(i, k))]
            for i in range(vocab.n_entities)])

        pre_mlp = torch.stack([embeddings[(i, k)] + attn_outputs[(i, k)]
                               for i in range(vocab.n_entities)])
        needed = target_obj_embs - pre_mlp
        needed_norm = needed / (needed.norm(dim=1, keepdim=True) + 1e-8)

        alignments = []
        for j in range(n_neurons):
            fc2_col = fc2_weight[:, j]
            contribution = activations[:, j:j+1] * fc2_col.unsqueeze(0)
            align = (contribution * needed_norm).sum(dim=1).mean().item()
            alignments.append(align)
        alignments = np.array(alignments)

        total_align = sum(alignments)
        positive = (alignments > 0).sum()
        sorted_idx = np.argsort(-alignments)
        cumsum = np.cumsum(alignments[sorted_idx])
        top10_pct = cumsum[9] / total_align * 100 if total_align > 0 else 0
        top30_pct = cumsum[29] / total_align * 100 if total_align > 0 else 0
        for_90 = np.searchsorted(cumsum, 0.9 * total_align) + 1 if total_align > 0 else 256

        print(f"\n  R{k} alignment: total={total_align:.3f}, positive={positive}/{n_neurons}, "
              f"top-10={top10_pct:.1f}%, for 90%={for_90}")

        neuron_results["per_relation"][f"r{k}"] = {
            "total_alignment": round(total_align, 4),
            "positive_contributors": int(positive),
            "top10_pct": round(top10_pct, 1),
            "top30_pct": round(top30_pct, 1),
            "neurons_for_90pct": int(for_90),
        }

    # Cohen's d
    r0_acts = torch.stack([post_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_acts = torch.stack([post_gelu[(i, 1)] for i in range(vocab.n_entities)])
    r0_mean = r0_acts.mean(0)
    r1_mean = r1_acts.mean(0)
    pooled_std = ((r0_acts.var(0) + r1_acts.var(0)) / 2).sqrt()
    cohens_d = ((r0_mean - r1_mean) / (pooled_std + 1e-8)).numpy()
    abs_d = np.abs(cohens_d)

    r0_specific = ((abs_d > 0.8) & (cohens_d > 0)).sum()
    r1_specific = ((abs_d > 0.8) & (cohens_d < 0)).sum()
    large_d = (abs_d > 0.8).sum()
    negligible_d = (abs_d < 0.2).sum()

    cross_corrs = []
    for j in range(n_neurons):
        c = np.corrcoef(r0_acts[:, j].numpy(), r1_acts[:, j].numpy())[0, 1]
        cross_corrs.append(c if not np.isnan(c) else 0)
    cross_corrs = np.array(cross_corrs)
    shared = ((abs_d < 0.2) & (cross_corrs > 0.5)).sum()

    print(f"\n  Cohen's d: mean|d|={abs_d.mean():.2f}, |d|>0.8={large_d}, "
          f"|d|<0.2={negligible_d}, R0-spec={r0_specific}, R1-spec={r1_specific}")
    print(f"  Shared (|d|<0.2, corr>0.5): {shared}")
    print(f"  Mean cross-relation corr: {cross_corrs.mean():.3f}")

    neuron_results["relation_specificity"] = {
        "mean_abs_d": round(float(abs_d.mean()), 3),
        "max_abs_d": round(float(abs_d.max()), 3),
        "large_d": int(large_d),
        "negligible_d": int(negligible_d),
        "r0_specific": int(r0_specific),
        "r1_specific": int(r1_specific),
        "shared": int(shared),
        "mean_cross_corr": round(float(cross_corrs.mean()), 3),
    }

    results["neuron_decomposition"] = neuron_results
    return cohens_d, cross_corrs


def analysis_3_input(model, vocab, device, mlp_inputs, pre_gelu, post_gelu, results):
    print("\n" + "=" * 70)
    print("ANALYSIS 3: NEURON INPUT DECOMPOSITION")
    print("=" * 70)

    block = model.blocks[0]
    fc1_weight = block.mlp.fc1.weight.detach().cpu()
    fc1_bias = block.mlp.fc1.bias.detach().cpu()
    n_neurons = fc1_weight.shape[0]

    r0_inputs = torch.stack([mlp_inputs[(i, 0)] for i in range(vocab.n_entities)])
    r1_inputs = torch.stack([mlp_inputs[(i, 1)] for i in range(vocab.n_entities)])
    mean_r0 = r0_inputs.mean(0)
    mean_r1 = r1_inputs.mean(0)
    rel_dir = mean_r0 - mean_r1
    rel_dir_unit = rel_dir / rel_dir.norm()

    r0_pre = torch.stack([pre_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_pre = torch.stack([pre_gelu[(i, 1)] for i in range(vocab.n_entities)])

    rel_comp_r0 = fc1_weight @ mean_r0 + fc1_bias
    rel_comp_r1 = fc1_weight @ mean_r1 + fc1_bias
    subj_comp_r0 = r0_pre - rel_comp_r0.unsqueeze(0)
    subj_comp_r1 = r1_pre - rel_comp_r1.unsqueeze(0)

    gating_shift = (rel_comp_r0 - rel_comp_r1).numpy()
    subj_std = ((subj_comp_r0.std(0).numpy() + subj_comp_r1.std(0).numpy()) / 2)
    gating_snr = np.abs(gating_shift) / (subj_std + 1e-8)

    print(f"\n  Gating SNR:")
    print(f"    Mean shift: {np.abs(gating_shift).mean():.3f}")
    print(f"    Mean subject std: {subj_std.mean():.3f}")
    print(f"    Mean SNR: {gating_snr.mean():.2f}, Median: {np.median(gating_snr):.2f}")
    print(f"    SNR > 2: {(gating_snr > 2).sum()}/{n_neurons}")
    print(f"    SNR > 3: {(gating_snr > 3).sum()}/{n_neurons}")
    print(f"    SNR < 1: {(gating_snr < 1).sum()}/{n_neurons}")

    input_results = {
        "gating_snr": {
            "mean_shift": round(float(np.abs(gating_shift).mean()), 3),
            "mean_subj_std": round(float(subj_std.mean()), 3),
            "mean_snr": round(float(gating_snr.mean()), 2),
            "median_snr": round(float(np.median(gating_snr)), 2),
            "snr_gt2": int((gating_snr > 2).sum()),
            "snr_gt3": int((gating_snr > 3).sum()),
            "snr_lt1": int((gating_snr < 1).sum()),
        }
    }

    # Neuron classification
    r0_post = torch.stack([post_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_post = torch.stack([post_gelu[(i, 1)] for i in range(vocab.n_entities)])
    r0_mean_act = r0_post.mean(0).numpy()
    r1_mean_act = r1_post.mean(0).numpy()
    r0_neurons = np.where((r0_mean_act > 0.05) & (r1_mean_act <= 0.05))[0]
    r1_neurons = np.where((r1_mean_act > 0.05) & (r0_mean_act <= 0.05))[0]
    dead_neurons = np.where((r0_mean_act <= 0.05) & (r1_mean_act <= 0.05))[0]
    shared_neurons = np.where((r0_mean_act > 0.05) & (r1_mean_act > 0.05))[0]

    print(f"\n  Neuron classification:")
    print(f"    R0-only: {len(r0_neurons)}")
    print(f"    R1-only: {len(r1_neurons)}")
    print(f"    Shared: {len(shared_neurons)}")
    print(f"    Dead: {len(dead_neurons)}")

    # Gate interference
    if len(r0_neurons) > 0:
        wrong_active = (r1_post[:, r0_neurons] > 0.05).float().mean().item()
        suppressed = (r0_post[:, r0_neurons] < 0).float().mean().item()
        print(f"\n  Gate interference:")
        print(f"    R0 neurons active in R1: {wrong_active:.3f}")
        print(f"    R0 neurons suppressed in R0: {suppressed:.3f}")
    else:
        wrong_active = 0
        suppressed = 0

    if len(r1_neurons) > 0:
        wrong_active_r1 = (r0_post[:, r1_neurons] > 0.05).float().mean().item()
        suppressed_r1 = (r1_post[:, r1_neurons] < 0).float().mean().item()
        print(f"    R1 neurons active in R0: {wrong_active_r1:.3f}")
        print(f"    R1 neurons suppressed in R1: {suppressed_r1:.3f}")
    else:
        wrong_active_r1 = 0
        suppressed_r1 = 0

    input_results["neuron_classification"] = {
        "r0_only": int(len(r0_neurons)),
        "r1_only": int(len(r1_neurons)),
        "shared": int(len(shared_neurons)),
        "dead": int(len(dead_neurons)),
    }
    input_results["gate_interference"] = {
        "r0_active_in_r1": round(wrong_active, 4),
        "r0_suppressed_in_r0": round(suppressed, 4),
        "r1_active_in_r0": round(wrong_active_r1, 4),
        "r1_suppressed_in_r1": round(suppressed_r1, 4),
    }

    # fc1 geometry
    all_inputs = torch.cat([r0_inputs, r1_inputs], dim=0)
    centered = all_inputs - all_inputs.mean(0)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    subj_pcs = Vh[:10]

    fc1_norms = fc1_weight.norm(dim=1).numpy()
    fc1_rel_proj = (fc1_weight @ rel_dir_unit).numpy()
    fc1_in_subj = (fc1_weight @ subj_pcs.T).norm(dim=1).numpy()
    fc1_rel_frac = np.abs(fc1_rel_proj) / (fc1_norms + 1e-8)
    fc1_subj_frac = fc1_in_subj / (fc1_norms + 1e-8)

    rel_in_subj = (rel_dir_unit @ subj_pcs.T).norm().item() ** 2

    print(f"\n  fc1 geometry:")
    print(f"    Mean norm: {fc1_norms.mean():.3f}")
    print(f"    Fraction in relation dir: {fc1_rel_frac.mean():.0%}")
    print(f"    Fraction in subject space: {fc1_subj_frac.mean():.0%}")
    print(f"\n  Orthogonality:")
    print(f"    Relation dir in subject subspace: {rel_in_subj:.1%}")
    print(f"    Relation dir orthogonal: {1 - rel_in_subj:.1%}")

    input_results["fc1_geometry"] = {
        "mean_norm": round(float(fc1_norms.mean()), 3),
        "rel_fraction": round(float(fc1_rel_frac.mean()), 3),
        "subj_fraction": round(float(fc1_subj_frac.mean()), 3),
    }
    input_results["orthogonality"] = {
        "rel_in_subj_pct": round(rel_in_subj * 100, 1),
        "rel_orthogonal_pct": round((1 - rel_in_subj) * 100, 1),
    }

    results["input_decomposition"] = input_results
    return r0_neurons, r1_neurons, dead_neurons


def analysis_4_activations(vocab, post_gelu, pre_gelu, r0_neurons, r1_neurons, results):
    print("\n" + "=" * 70)
    print("ANALYSIS 4: ACTIVATION DISTRIBUTIONS")
    print("=" * 70)

    r0_post = torch.stack([post_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_post = torch.stack([post_gelu[(i, 1)] for i in range(vocab.n_entities)])
    r0_pre = torch.stack([pre_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_pre = torch.stack([pre_gelu[(i, 1)] for i in range(vocab.n_entities)])

    hist_results = {}

    def gelu_derivative(x):
        phi = torch.distributions.Normal(0, 1).log_prob(x).exp()
        Phi = 0.5 * (1 + torch.erf(x / np.sqrt(2)))
        return Phi + x * phi

    print(f"\n  Post-GELU / Pre-GELU / GELU derivative:")
    if len(r0_neurons) > 0:
        r0_at_r0_post = r0_post[:, r0_neurons]
        r1_at_r0_post = r1_post[:, r0_neurons]
        r0_at_r0_pre = r0_pre[:, r0_neurons]
        r1_at_r0_pre = r1_pre[:, r0_neurons]
        d_r0_r0 = gelu_derivative(r0_at_r0_pre)
        d_r1_r0 = gelu_derivative(r1_at_r0_pre)

        print(f"    R0 neurons — post-GELU: R0 mean={r0_at_r0_post.mean():.3f}, "
              f"R1 mean={r1_at_r0_post.mean():.3f}")
        print(f"    R0 neurons — pre-GELU:  R0 mean={r0_at_r0_pre.mean():.3f}, "
              f"R1 mean={r1_at_r0_pre.mean():.3f}")
        print(f"    R0 neurons — GELU':     R0 mean={d_r0_r0.mean():.3f}, "
              f"R1 mean={d_r1_r0.mean():.3f}")

        hist_results["r0_neurons"] = {
            "post_gelu_r0": round(r0_at_r0_post.mean().item(), 3),
            "post_gelu_r1": round(r1_at_r0_post.mean().item(), 3),
            "pre_gelu_r0": round(r0_at_r0_pre.mean().item(), 3),
            "pre_gelu_r1": round(r1_at_r0_pre.mean().item(), 3),
            "gelu_deriv_r0": round(d_r0_r0.mean().item(), 3),
            "gelu_deriv_r1": round(d_r1_r0.mean().item(), 3),
        }

    if len(r1_neurons) > 0:
        r1_at_r1_post = r1_post[:, r1_neurons]
        r0_at_r1_post = r0_post[:, r1_neurons]
        r1_at_r1_pre = r1_pre[:, r1_neurons]
        r0_at_r1_pre = r0_pre[:, r1_neurons]
        d_r1_r1 = gelu_derivative(r1_at_r1_pre)
        d_r0_r1 = gelu_derivative(r0_at_r1_pre)

        print(f"    R1 neurons — post-GELU: R1 mean={r1_at_r1_post.mean():.3f}, "
              f"R0 mean={r0_at_r1_post.mean():.3f}")
        print(f"    R1 neurons �� pre-GELU:  R1 mean={r1_at_r1_pre.mean():.3f}, "
              f"R0 mean={r0_at_r1_pre.mean():.3f}")
        print(f"    R1 neurons — GELU':     R1 mean={d_r1_r1.mean():.3f}, "
              f"R0 mean={d_r0_r1.mean():.3f}")

        hist_results["r1_neurons"] = {
            "post_gelu_r1": round(r1_at_r1_post.mean().item(), 3),
            "post_gelu_r0": round(r0_at_r1_post.mean().item(), 3),
            "pre_gelu_r1": round(r1_at_r1_pre.mean().item(), 3),
            "pre_gelu_r0": round(r0_at_r1_pre.mean().item(), 3),
            "gelu_deriv_r1": round(d_r1_r1.mean().item(), 3),
            "gelu_deriv_r0": round(d_r0_r1.mean().item(), 3),
        }

    results["activation_histograms"] = hist_results


def analysis_5_weighted_sum(model, vocab, post_gelu, mlp_outputs, results):
    print("\n" + "=" * 70)
    print("ANALYSIS 5: WEIGHTED SUM STRUCTURE")
    print("=" * 70)

    block = model.blocks[0]
    fc2_weight = block.mlp.fc2.weight.detach().cpu()
    n_neurons = fc2_weight.shape[1]

    ws_results = {}

    for k in range(vocab.num_relations):
        acts = torch.stack([post_gelu[(i, k)] for i in range(vocab.n_entities)])
        acts_centered = acts - acts.mean(0)
        U, S, Vh = torch.linalg.svd(acts_centered, full_matrices=False)
        var = S ** 2
        cumvar = var.cumsum(0) / var.sum()

        for_90 = (cumvar < 0.90).sum().item() + 1
        for_95 = (cumvar < 0.95).sum().item() + 1
        for_99 = (cumvar < 0.99).sum().item() + 1

        print(f"\n  R{k} activation SVD: 90% in {for_90}, 95% in {for_95}, 99% in {for_99}")
        print(f"    Top-5: {cumvar[4]:.3f}, Top-10: {cumvar[9]:.3f}")

        ws_results[f"r{k}_dimensionality"] = {
            "for_90pct": int(for_90),
            "for_95pct": int(for_95),
            "for_99pct": int(for_99),
            "top5_var": round(cumvar[4].item(), 3),
            "top10_var": round(cumvar[9].item(), 3),
        }

    # Column usage
    r0_acts = torch.stack([post_gelu[(i, 0)] for i in range(vocab.n_entities)])
    r1_acts = torch.stack([post_gelu[(i, 1)] for i in range(vocab.n_entities)])
    r0_usage = (r0_acts.mean(0) > 0.05).numpy()
    r1_usage = (r1_acts.mean(0) > 0.05).numpy()

    r0_only = (r0_usage & ~r1_usage).sum()
    r1_only = (~r0_usage & r1_usage).sum()
    shared = (r0_usage & r1_usage).sum()
    dead = (~r0_usage & ~r1_usage).sum()
    usage_corr = np.corrcoef(r0_usage.astype(float), r1_usage.astype(float))[0, 1]

    print(f"\n  Column usage: R0-only={r0_only}, R1-only={r1_only}, "
          f"shared={shared}, dead={dead}")
    print(f"  Usage correlation: {usage_corr:.3f}")

    # Dictionary geometry
    W = fc2_weight.T
    col_norms = W.norm(dim=1)
    W_normed = W / (col_norms.unsqueeze(1) + 1e-8)
    cos_mat = W_normed @ W_normed.T
    mask_upper = torch.triu(torch.ones(n_neurons, n_neurons), diagonal=1).bool()
    pairwise_cos = cos_mat[mask_upper]
    U_d, S_d, _ = torch.linalg.svd(W, full_matrices=False)
    cond = (S_d[0] / S_d[-1]).item()

    print(f"  Dictionary: mean_norm={col_norms.mean():.3f}, "
          f"mean_cos={pairwise_cos.mean():.3f}, cond={cond:.2f}")

    ws_results["column_usage"] = {
        "r0_only": int(r0_only), "r1_only": int(r1_only),
        "shared": int(shared), "dead": int(dead),
        "usage_corr": round(float(usage_corr), 3),
    }
    ws_results["dictionary"] = {
        "mean_norm": round(col_norms.mean().item(), 3),
        "mean_pairwise_cos": round(pairwise_cos.mean().item(), 3),
        "condition_number": round(cond, 2),
    }

    results["weighted_sum_structure"] = ws_results


def analysis_6_affine_fit(model, vocab, mlp_inputs, post_gelu, mlp_outputs, results):
    print("\n" + "=" * 70)
    print("ANALYSIS 6: AFFINE ACTIVATION FIT")
    print("=" * 70)

    affine_results = {}

    for k in range(vocab.num_relations):
        X = torch.stack([mlp_inputs[(i, k)] for i in range(vocab.n_entities)])
        Y_act = torch.stack([post_gelu[(i, k)] for i in range(vocab.n_entities)])
        Y_out = torch.stack([mlp_outputs[(i, k)] for i in range(vocab.n_entities)])

        n = X.shape[0]
        X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)

        # Input → activations
        sol_act, _, _, _ = torch.linalg.lstsq(X_aug, Y_act)
        Y_act_pred = X_aug @ sol_act
        ss_res_act = ((Y_act - Y_act_pred) ** 2).sum().item()
        ss_tot_act = ((Y_act - Y_act.mean(0)) ** 2).sum().item()
        r2_act = 1 - ss_res_act / ss_tot_act

        cos_act = F.cosine_similarity(Y_act, Y_act_pred, dim=1)

        # Input → output
        sol_out, _, _, _ = torch.linalg.lstsq(X_aug, Y_out)
        Y_out_pred = X_aug @ sol_out
        ss_res_out = ((Y_out - Y_out_pred) ** 2).sum().item()
        ss_tot_out = ((Y_out - Y_out.mean(0)) ** 2).sum().item()
        r2_out = 1 - ss_res_out / ss_tot_out

        cos_out = F.cosine_similarity(Y_out, Y_out_pred, dim=1)

        print(f"\n  R{k} — input → activations: R²={r2_act:.4f}, "
              f"cos mean={cos_act.mean():.4f}, min={cos_act.min():.3f}")
        print(f"  R{k} — input → output:      R²={r2_out:.4f}, "
              f"cos mean={cos_out.mean():.4f}, min={cos_out.min():.3f}")

        affine_results[f"r{k}"] = {
            "input_to_activations": {
                "r2": round(r2_act, 4),
                "cos_mean": round(cos_act.mean().item(), 4),
                "cos_min": round(cos_act.min().item(), 3),
            },
            "input_to_output": {
                "r2": round(r2_out, 4),
                "cos_mean": round(cos_out.mean().item(), 4),
                "cos_min": round(cos_out.min().item(), 3),
            }
        }

    results["affine_fit"] = affine_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading frozen-affine checkpoint...")
    model, vocab, model_cfg, base_vectors = load_checkpoint(args.seed, device)
    print(f"Model: dim={model_cfg.hidden_dim}, layers={model_cfg.n_layers}, heads={model_cfg.n_heads}")
    print(f"Entities: {vocab.n_entities}, Relations: {vocab.num_relations}")
    print(f"MLP neurons: {4 * model_cfg.hidden_dim}")

    print("\nExtracting hidden states...")
    embeddings, attn_outputs, mlp_inputs, mlp_outputs, pre_gelu, post_gelu = \
        extract_all_states(model, vocab, device)

    results = {
        "experiment": "ravfogel_1layer_frozen_analysis",
        "seed": args.seed,
        "n_entities": vocab.n_entities,
        "dim": model_cfg.hidden_dim,
        "n_neurons": 4 * model_cfg.hidden_dim,
    }

    analysis_1_attention(model, vocab, device, results)
    analysis_2_neurons(model, vocab, device, embeddings, attn_outputs,
                       mlp_inputs, mlp_outputs, post_gelu, results)
    r0_neurons, r1_neurons, dead_neurons = analysis_3_input(
        model, vocab, device, mlp_inputs, pre_gelu, post_gelu, results)
    analysis_4_activations(vocab, post_gelu, pre_gelu, r0_neurons, r1_neurons, results)
    analysis_5_weighted_sum(model, vocab, post_gelu, mlp_outputs, results)
    analysis_6_affine_fit(model, vocab, mlp_inputs, post_gelu, mlp_outputs, results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_1layer_frozen_analysis_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

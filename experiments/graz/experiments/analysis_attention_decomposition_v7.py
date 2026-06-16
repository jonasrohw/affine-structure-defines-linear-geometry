"""
Per-head attention output analysis for v7 checkpoint.

Decomposes attn_out at the subject position (position 2) into per-head
contributions, then measures:
1. Within-relation similarity of attn_out (is it ~constant per relation?)
2. Between-relation difference (does it differentiate R0 vs R1?)
3. Per-head analysis: which head(s) carry the relation signal?
4. Per-head analysis: which head(s) carry subject-specific information?

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/analysis_attention_decomposition_v7.py
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
def extract_attention_decomposition(model, vocab, device):
    """Extract per-head attention outputs at position 2 for all (subject, relation) pairs.

    Returns:
        attn_out_full: (n_entities * n_relations, dim) — full attention output
        per_head_out: (n_entities * n_relations, n_heads, dim) — per-head contributions
        attn_weights: (n_entities * n_relations, n_heads, 3) — attention weights at pos 2
        keys: list of (subj_idx, rel_k) tuples
    """
    model.eval()
    n = vocab.n_entities
    contexts = []
    keys = []
    for i in range(n):
        for k in range(vocab.num_relations):
            keys.append((i, k))
            contexts.append([vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)])

    batch = torch.tensor(contexts, dtype=torch.long, device=device)
    B, T = batch.shape

    # Forward through embedding + position
    positions = torch.arange(T, device=device).unsqueeze(0)
    x = model.token_emb(batch) + model.pos_emb(positions)

    # Apply ln1
    block = model.blocks[0]
    ln1_out = block.ln1(x)

    # Manually decompose attention
    attn = block.attn
    n_heads = attn.n_heads
    head_dim = attn.head_dim
    C = n_heads * head_dim

    qkv = attn.qkv(ln1_out)
    q, k_proj, v = qkv.split(C, dim=2)

    def reshape(t):
        return t.view(B, T, n_heads, head_dim).transpose(1, 2)

    q, k_proj, v = reshape(q), reshape(k_proj), reshape(v)

    # Attention weights
    att = (q @ k_proj.transpose(-2, -1)) / (head_dim ** 0.5)
    causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
    att = att.masked_fill(~causal_mask, float("-inf"))
    att = F.softmax(att, dim=-1)  # (B, n_heads, T, T)

    # Per-head weighted values: (B, n_heads, T, head_dim)
    head_values = att @ v

    # Per-head output through out_proj
    # out_proj.weight is (C, C), organized as (out_dim, in_dim)
    # The input is concat of heads: (B, T, C)
    # Per-head contribution: head h's value (B, T, head_dim) projected through
    # out_proj.weight[:, h*head_dim:(h+1)*head_dim]
    out_w = attn.out_proj.weight  # (C, C)

    per_head_outputs = []
    for h in range(n_heads):
        w_h = out_w[:, h * head_dim:(h + 1) * head_dim]  # (C, head_dim)
        hv = head_values[:, h, :, :]  # (B, T, head_dim)
        out_h = hv @ w_h.T  # (B, T, C)
        per_head_outputs.append(out_h[:, 2, :])  # position 2 only

    per_head_out = torch.stack(per_head_outputs, dim=1)  # (B, n_heads, C)

    # Full attention output for verification
    y = head_values.transpose(1, 2).contiguous().view(B, T, C)
    full_attn_out = attn.out_proj(y)[:, 2, :]  # (B, C)

    # Verify decomposition
    reconstructed = per_head_out.sum(dim=1)
    max_err = (full_attn_out - reconstructed).abs().max().item()
    print(f"Per-head decomposition max error: {max_err:.2e}")

    # Attention weights at position 2
    attn_at_pos2 = att[:, :, 2, :T]  # (B, n_heads, T=3)

    return (full_attn_out.cpu(), per_head_out.cpu(),
            attn_at_pos2.cpu(), keys)


def analyze_relation_signal(attn_out, per_head_out, attn_weights, keys, vocab):
    """Analyze how attention encodes relation identity."""
    n = vocab.n_entities
    n_heads = per_head_out.shape[1]
    dim = attn_out.shape[1]

    key_to_idx = {k: i for i, k in enumerate(keys)}

    # Separate by relation
    r0_indices = [key_to_idx[(i, 0)] for i in range(n)]
    r1_indices = [key_to_idx[(i, 1)] for i in range(n)]

    attn_r0 = attn_out[r0_indices]  # (n, dim)
    attn_r1 = attn_out[r1_indices]  # (n, dim)

    print("\n" + "=" * 70)
    print("FULL ATTENTION OUTPUT ANALYSIS (attn_out at position 2)")
    print("=" * 70)

    # Mean attention output per relation
    mean_r0 = attn_r0.mean(dim=0)
    mean_r1 = attn_r1.mean(dim=0)

    print(f"\nMean attn_out norm:  R0={mean_r0.norm():.4f}, R1={mean_r1.norm():.4f}")
    print(f"Cosine(mean_R0, mean_R1): {F.cosine_similarity(mean_r0.unsqueeze(0), mean_r1.unsqueeze(0)).item():.4f}")

    # Within-relation similarity: mean pairwise cosine of attn_out for same relation
    def mean_pairwise_cosine(X, sample_n=500):
        """Mean cosine between random pairs."""
        n = X.shape[0]
        rng = np.random.RandomState(42)
        i1 = rng.randint(0, n, sample_n)
        i2 = rng.randint(0, n, sample_n)
        mask = i1 != i2
        i1, i2 = i1[mask], i2[mask]
        cos = F.cosine_similarity(X[i1], X[i2], dim=1)
        return cos.mean().item(), cos.std().item()

    wr0_mean, wr0_std = mean_pairwise_cosine(attn_r0)
    wr1_mean, wr1_std = mean_pairwise_cosine(attn_r1)
    print(f"\nWithin-relation pairwise cosine:")
    print(f"  R0: {wr0_mean:.4f} ± {wr0_std:.4f}")
    print(f"  R1: {wr1_mean:.4f} ± {wr1_std:.4f}")

    # Between-relation similarity: cosine of attn_out for same subject, different relation
    cross_cos = F.cosine_similarity(attn_r0, attn_r1, dim=1)
    print(f"\nBetween-relation cosine (same subject, different relation):")
    print(f"  Mean: {cross_cos.mean():.4f} ± {cross_cos.std():.4f}")
    print(f"  Min: {cross_cos.min():.4f}, Max: {cross_cos.max():.4f}")

    # Variance decomposition: how much of attn_out variance is relation vs subject?
    # Total variance
    all_attn = torch.cat([attn_r0, attn_r1], dim=0)
    total_var = all_attn.var(dim=0).sum().item()

    # Within-relation variance
    within_var = (attn_r0.var(dim=0).sum().item() + attn_r1.var(dim=0).sum().item()) / 2

    # Between-relation variance
    grand_mean = all_attn.mean(dim=0)
    between_var = ((mean_r0 - grand_mean).pow(2).sum().item() +
                   (mean_r1 - grand_mean).pow(2).sum().item()) / 2

    print(f"\nVariance decomposition of attn_out:")
    print(f"  Total variance: {total_var:.4f}")
    print(f"  Between-relation variance: {between_var:.4f} ({100 * between_var / total_var:.1f}%)")
    print(f"  Within-relation variance: {within_var:.4f} ({100 * within_var / total_var:.1f}%)")

    # h0 analysis for comparison
    print("\n" + "=" * 70)
    print("ATTENTION WEIGHT PATTERNS")
    print("=" * 70)

    aw_r0 = attn_weights[r0_indices]  # (n, n_heads, 3)
    aw_r1 = attn_weights[r1_indices]

    print(f"\nAttention weights at position 2 → positions [sep, rel, self]")
    print(f"{'Head':>4} | {'R0→sep':>8} {'R0→rel':>8} {'R0→self':>8} | "
          f"{'R1→sep':>8} {'R1→rel':>8} {'R1→self':>8}")
    print("-" * 72)
    for h in range(n_heads):
        r0_w = aw_r0[:, h, :].mean(dim=0)
        r1_w = aw_r1[:, h, :].mean(dim=0)
        print(f"  {h:>2} | {r0_w[0]:>8.4f} {r0_w[1]:>8.4f} {r0_w[2]:>8.4f} | "
              f"{r1_w[0]:>8.4f} {r1_w[1]:>8.4f} {r1_w[2]:>8.4f}")

    # Attention weight variance across subjects (should be low if pattern is fixed)
    print(f"\nAttention weight std across subjects:")
    print(f"{'Head':>4} | {'R0→sep':>8} {'R0→rel':>8} {'R0→self':>8} | "
          f"{'R1→sep':>8} {'R1→rel':>8} {'R1→self':>8}")
    print("-" * 72)
    for h in range(n_heads):
        r0_s = aw_r0[:, h, :].std(dim=0)
        r1_s = aw_r1[:, h, :].std(dim=0)
        print(f"  {h:>2} | {r0_s[0]:>8.4f} {r0_s[1]:>8.4f} {r0_s[2]:>8.4f} | "
              f"{r1_s[0]:>8.4f} {r1_s[1]:>8.4f} {r1_s[2]:>8.4f}")

    print("\n" + "=" * 70)
    print("PER-HEAD ATTENTION OUTPUT ANALYSIS")
    print("=" * 70)

    for h in range(n_heads):
        ph_r0 = per_head_out[r0_indices, h, :]  # (n, dim)
        ph_r1 = per_head_out[r1_indices, h, :]

        mean_h_r0 = ph_r0.mean(dim=0)
        mean_h_r1 = ph_r1.mean(dim=0)

        # Norms
        norm_r0 = mean_h_r0.norm().item()
        norm_r1 = mean_h_r1.norm().item()

        # Cosine between per-head means
        cos_means = F.cosine_similarity(mean_h_r0.unsqueeze(0), mean_h_r1.unsqueeze(0)).item()

        # Variance decomposition per head
        all_h = torch.cat([ph_r0, ph_r1], dim=0)
        total_var_h = all_h.var(dim=0).sum().item()
        within_var_h = (ph_r0.var(dim=0).sum().item() + ph_r1.var(dim=0).sum().item()) / 2
        grand_mean_h = all_h.mean(dim=0)
        between_var_h = ((mean_h_r0 - grand_mean_h).pow(2).sum().item() +
                         (mean_h_r1 - grand_mean_h).pow(2).sum().item()) / 2

        # Within-relation pairwise cosine
        wr0_h, _ = mean_pairwise_cosine(ph_r0)
        wr1_h, _ = mean_pairwise_cosine(ph_r1)

        # Cross-relation cosine (same subject)
        cross_h = F.cosine_similarity(ph_r0, ph_r1, dim=1).mean().item()

        pct_between = 100 * between_var_h / total_var_h if total_var_h > 1e-10 else 0

        print(f"\nHead {h}:")
        print(f"  Mean output norm:     R0={norm_r0:.4f}, R1={norm_r1:.4f}")
        print(f"  Cosine(mean_R0, mean_R1): {cos_means:.4f}")
        print(f"  Variance: total={total_var_h:.4f}, between-rel={between_var_h:.4f} ({pct_between:.1f}%), "
              f"within-rel={within_var_h:.4f} ({100 - pct_between:.1f}%)")
        print(f"  Within-relation pairwise cos: R0={wr0_h:.4f}, R1={wr1_h:.4f}")
        print(f"  Cross-relation cos (same subj): {cross_h:.4f}")

    # Relation direction analysis
    print("\n" + "=" * 70)
    print("RELATION DIRECTION ANALYSIS")
    print("=" * 70)

    # The "relation direction" is the difference between R0 and R1 mean attn_out
    rel_direction = mean_r0 - mean_r1
    rel_dir_norm = rel_direction.norm().item()
    rel_direction_unit = rel_direction / rel_direction.norm()

    print(f"\nRelation direction (mean_R0 - mean_R1):")
    print(f"  Norm: {rel_dir_norm:.4f}")

    # Project each attn_out onto the relation direction
    proj_r0 = (attn_r0 @ rel_direction_unit).numpy()
    proj_r1 = (attn_r1 @ rel_direction_unit).numpy()

    print(f"  Projection of R0 attn_outs: mean={proj_r0.mean():.4f} ± {proj_r0.std():.4f}")
    print(f"  Projection of R1 attn_outs: mean={proj_r1.mean():.4f} ± {proj_r1.std():.4f}")
    print(f"  Separation (Cohen's d): {(proj_r0.mean() - proj_r1.mean()) / np.sqrt((proj_r0.std()**2 + proj_r1.std()**2) / 2):.2f}")

    # Per-head contribution to the relation direction
    print(f"\nPer-head contribution to relation direction:")
    for h in range(n_heads):
        ph_r0_h = per_head_out[r0_indices, h, :]
        ph_r1_h = per_head_out[r1_indices, h, :]
        head_rel_diff = ph_r0_h.mean(dim=0) - ph_r1_h.mean(dim=0)
        proj_on_rel = (head_rel_diff @ rel_direction_unit).item()
        frac = proj_on_rel / rel_dir_norm * 100
        print(f"  Head {h}: projection={proj_on_rel:.4f} ({frac:.1f}% of total), "
              f"diff_norm={head_rel_diff.norm():.4f}")

    # How much subject-specific information does each head carry?
    print(f"\nPer-head subject information (after subtracting relation mean):")
    for h in range(n_heads):
        ph_r0_h = per_head_out[r0_indices, h, :]
        ph_r1_h = per_head_out[r1_indices, h, :]
        # Subtract relation mean to isolate subject-specific signal
        residual_r0 = ph_r0_h - ph_r0_h.mean(dim=0, keepdim=True)
        residual_r1 = ph_r1_h - ph_r1_h.mean(dim=0, keepdim=True)
        # Cross-relation correlation of residuals: does the same subject produce
        # similar residuals under R0 and R1?
        # Use per-subject cosine between R0 and R1 residuals
        cos_residuals = F.cosine_similarity(residual_r0, residual_r1, dim=1)
        subj_var_r0 = residual_r0.var(dim=0).sum().item()
        subj_var_r1 = residual_r1.var(dim=0).sum().item()
        print(f"  Head {h}: subject_var R0={subj_var_r0:.4f} R1={subj_var_r1:.4f}, "
              f"cross-rel residual cos={cos_residuals.mean():.4f} ± {cos_residuals.std():.4f}")


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

    print("Extracting attention decomposition...")
    attn_out, per_head_out, attn_weights, keys = extract_attention_decomposition(
        model, vocab, device)

    analyze_relation_signal(attn_out, per_head_out, attn_weights, keys, vocab)


if __name__ == "__main__":
    main()

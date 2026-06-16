"""
Analyze how the 1-layer attention-only (no MLP, no dummy) model
implements relation-dependent affine maps.

Sequence format: [sep, r_k, subj_i] (3-token context at prediction time)
The model's only tools: attention weights + value vectors (no MLP).

Key question: how does the model produce different outputs for R0 vs R1
using only linear attention + LayerNorm?
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.model import build_model

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def load_model_and_vocab(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab_kwargs = ckpt["vocab_kwargs"]
    vocab = VocabConfig(**vocab_kwargs)
    mcfg = ckpt["model_cfg"]
    model_cfg = ModelConfig(
        n_layers=mcfg["n_layers"], hidden_dim=mcfg["hidden_dim"],
        n_heads=mcfg["n_heads"], no_mlp=mcfg["no_mlp"],
    )
    model = build_model(model_cfg, vocab)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    rel_objects_table = {}
    for i in range(vocab.n_entities):
        for k in range(vocab.num_relations):
            objs = vocab.relation_objects(i, k)
            rel_objects_table[(i, k)] = objs

    return model, vocab, model_cfg, rel_objects_table, ckpt


@torch.no_grad()
def extract_attention_patterns(model, vocab, device, n_subjects=None):
    """Extract attention weights at subject position (pos 2) for R0 and R1."""
    if n_subjects is None:
        n_subjects = vocab.n_entities

    block = model.blocks[0]
    attn = block.attn
    ln1 = block.ln1

    results = {0: [], 1: []}

    for rel_k in [0, 1]:
        for i in range(n_subjects):
            tokens = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(i)]],
                dtype=torch.long, device=device,
            )
            x = model.token_emb(tokens) + model.pos_emb(torch.arange(3, device=device).unsqueeze(0))
            h = ln1(x)

            B, T, C = h.shape
            qkv = attn.qkv(h)
            q, k, v = qkv.split(C, dim=2)
            n_heads = attn.n_heads
            head_dim = attn.head_dim

            def reshape(t):
                return t.view(B, T, n_heads, head_dim).transpose(1, 2)

            q, k, v = reshape(q), reshape(k), reshape(v)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
            att = att.masked_fill(~causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)

            # att shape: [1, n_heads, T, T]
            # We want pos 2 (subject) attending to pos 0,1,2
            attn_at_subj = att[0, :, 2, :].cpu().numpy()  # [n_heads, 3]
            results[rel_k].append(attn_at_subj)

    return {k: np.array(v) for k, v in results.items()}


@torch.no_grad()
def value_decomposition(model, vocab, device, n_subjects=None):
    """Decompose the attention output at subject position into contributions
    from each source position, per head."""
    if n_subjects is None:
        n_subjects = vocab.n_entities

    block = model.blocks[0]
    attn = block.attn
    ln1 = block.ln1
    n_heads = attn.n_heads
    head_dim = attn.head_dim
    dim = n_heads * head_dim

    W_qkv = attn.qkv.weight  # [3*dim, dim]
    W_v = W_qkv[2*dim:3*dim]  # [dim, dim]
    W_o = attn.out_proj.weight  # [dim, dim]

    # Per-head OV matrices
    OV_heads = []
    for h in range(n_heads):
        Wv_h = W_v[h*head_dim:(h+1)*head_dim, :]  # [head_dim, dim]
        Wo_h = W_o[:, h*head_dim:(h+1)*head_dim]   # [dim, head_dim]
        OV_h = Wo_h @ Wv_h  # [dim, dim]
        OV_heads.append(OV_h)

    results = {}
    for rel_k in [0, 1]:
        contribs_sep = []
        contribs_rel = []
        contribs_subj = []
        attn_outputs = []

        for i in range(n_subjects):
            tokens = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(i)]],
                dtype=torch.long, device=device,
            )
            x = model.token_emb(tokens) + model.pos_emb(torch.arange(3, device=device).unsqueeze(0))
            h = ln1(x)

            B, T, C = h.shape
            qkv = attn.qkv(h)
            q, k, v = qkv.split(C, dim=2)

            def reshape(t):
                return t.view(B, T, n_heads, head_dim).transpose(1, 2)

            q, k, v = reshape(q), reshape(k), reshape(v)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
            att = att.masked_fill(~causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)

            # Compute per-source, per-head contributions
            # For subject position (pos 2):
            contrib_by_source = torch.zeros(3, dim, device=device)
            for head in range(n_heads):
                for src in range(3):
                    alpha = att[0, head, 2, src]
                    v_src = v[0, head, src]  # [head_dim]
                    Wo_h = W_o[:, head*head_dim:(head+1)*head_dim]
                    contrib_by_source[src] += alpha * (Wo_h @ v_src)

            contribs_sep.append(contrib_by_source[0].cpu())
            contribs_rel.append(contrib_by_source[1].cpu())
            contribs_subj.append(contrib_by_source[2].cpu())

            full_attn_out = contrib_by_source.sum(dim=0)
            attn_outputs.append(full_attn_out.cpu())

        results[rel_k] = {
            "sep": torch.stack(contribs_sep),
            "rel": torch.stack(contribs_rel),
            "subj": torch.stack(contribs_subj),
            "total": torch.stack(attn_outputs),
        }

    return results


@torch.no_grad()
def variance_decomposition(model, vocab, device, n_subjects=None):
    """How much variance in the final output comes from each source?"""
    if n_subjects is None:
        n_subjects = vocab.n_entities

    decomp = value_decomposition(model, vocab, device, n_subjects)

    print("\n=== Variance decomposition of attention output at subject pos ===\n")
    for rel_k in [0, 1]:
        d = decomp[rel_k]
        total_var = d["total"].var(dim=0).sum().item()
        sep_var = d["sep"].var(dim=0).sum().item()
        rel_var = d["rel"].var(dim=0).sum().item()
        subj_var = d["subj"].var(dim=0).sum().item()

        print(f"R{rel_k}:")
        print(f"  Total attention output variance: {total_var:.4f}")
        print(f"  From sep (pos 0):     {sep_var:.4f} ({100*sep_var/total_var:.1f}%)")
        print(f"  From rel (pos 1):     {rel_var:.4f} ({100*rel_var/total_var:.1f}%)")
        print(f"  From subj (pos 2):    {subj_var:.4f} ({100*subj_var/total_var:.1f}%)")

    return decomp


@torch.no_grad()
def head_specialization(model, vocab, device, n_subjects=None):
    """Analyze what each head does: attention pattern + contribution variance."""
    if n_subjects is None:
        n_subjects = vocab.n_entities

    block = model.blocks[0]
    attn = block.attn
    ln1 = block.ln1
    n_heads = attn.n_heads
    head_dim = attn.head_dim
    dim = n_heads * head_dim
    W_qkv = attn.qkv.weight
    W_v = W_qkv[2*dim:3*dim]
    W_o = attn.out_proj.weight

    print("\n=== Per-head analysis ===\n")

    for rel_k in [0, 1]:
        head_attn_patterns = [[] for _ in range(n_heads)]
        head_contributions = [[] for _ in range(n_heads)]

        for i in range(n_subjects):
            tokens = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(i)]],
                dtype=torch.long, device=device,
            )
            x = model.token_emb(tokens) + model.pos_emb(torch.arange(3, device=device).unsqueeze(0))
            h = ln1(x)

            B, T, C = h.shape
            qkv = attn.qkv(h)
            q, k, v = qkv.split(C, dim=2)

            def reshape(t):
                return t.view(B, T, n_heads, head_dim).transpose(1, 2)

            q, k, v = reshape(q), reshape(k), reshape(v)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
            att = att.masked_fill(~causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)

            for head in range(n_heads):
                head_attn_patterns[head].append(att[0, head, 2, :].cpu().numpy())
                contrib = torch.zeros(dim, device=device)
                for src in range(3):
                    alpha = att[0, head, 2, src]
                    v_src = v[0, head, src]
                    Wo_h = W_o[:, head*head_dim:(head+1)*head_dim]
                    contrib += alpha * (Wo_h @ v_src)
                head_contributions[head].append(contrib.cpu())

        print(f"R{rel_k}:")
        for head in range(n_heads):
            patterns = np.array(head_attn_patterns[head])
            mean_attn = patterns.mean(axis=0)
            std_attn = patterns.std(axis=0)

            contribs = torch.stack(head_contributions[head])
            contrib_var = contribs.var(dim=0).sum().item()

            print(f"  Head {head}: attn_mean=[sep:{mean_attn[0]:.3f}±{std_attn[0]:.3f}, "
                  f"rel:{mean_attn[1]:.3f}±{std_attn[1]:.3f}, "
                  f"subj:{mean_attn[2]:.3f}±{std_attn[2]:.3f}] | "
                  f"contrib_var={contrib_var:.4f}")
        print()


@torch.no_grad()
def ov_circuit_analysis(model, vocab, device):
    """Analyze the OV circuit: what does each head's OV matrix do to
    subject embeddings vs relation embeddings?"""
    block = model.blocks[0]
    attn = block.attn
    n_heads = attn.n_heads
    head_dim = attn.head_dim
    dim = n_heads * head_dim
    W_qkv = attn.qkv.weight
    W_v = W_qkv[2*dim:3*dim]
    W_o = attn.out_proj.weight

    print("\n=== OV circuit analysis ===\n")

    # Compute per-head OV matrices
    OV_heads = []
    for h in range(n_heads):
        Wv_h = W_v[h*head_dim:(h+1)*head_dim, :]
        Wo_h = W_o[:, h*head_dim:(h+1)*head_dim]
        OV_h = Wo_h @ Wv_h
        OV_heads.append(OV_h)

    # Get the lm_head (unembedding) matrix
    W_u = model.lm_head.weight  # [vocab_size, dim]

    # Get subject embeddings and object embeddings
    subj_embs = model.token_emb.weight[vocab.subject_offset:vocab.subject_offset + vocab.n_entities]
    obj_embs = model.token_emb.weight[vocab.object_offset:vocab.object_offset + vocab.n_entities]

    # For each head, compute OV(subject_emb) and see if it aligns with correct objects
    # via the unembedding matrix
    for h in range(n_heads):
        OV_h = OV_heads[h]
        # Apply OV to all subject embeddings
        ov_subj = (OV_h @ subj_embs.T).T  # [n_entities, dim]

        # How much does OV(subj_i) point toward the correct objects in unembedding space?
        # Compute W_u[obj_tokens] @ OV(subj_i) for correct vs random objects
        correct_scores_r0 = []
        correct_scores_r1 = []
        random_scores = []
        for i in range(min(200, vocab.n_entities)):
            ov_i = ov_subj[i]
            # Scores against all objects
            all_scores = W_u[vocab.object_offset:vocab.object_offset + vocab.n_entities] @ ov_i

            r0_objs = vocab.relation_objects(i, 0)
            r1_objs = vocab.relation_objects(i, 1)
            correct_scores_r0.append(all_scores[r0_objs].mean().item())
            correct_scores_r1.append(all_scores[r1_objs].mean().item())
            random_scores.append(all_scores.mean().item())

        print(f"  Head {h} OV(subj) scores via unembedding:")
        print(f"    vs R0-correct objects: {np.mean(correct_scores_r0):.4f}")
        print(f"    vs R1-correct objects: {np.mean(correct_scores_r1):.4f}")
        print(f"    vs all objects (mean): {np.mean(random_scores):.4f}")

    # Singular value spectrum of each OV matrix
    print(f"\n  OV singular value spectra (top 5):")
    for h in range(n_heads):
        svs = torch.linalg.svdvals(OV_heads[h]).cpu().numpy()
        top5 = ", ".join(f"{s:.3f}" for s in svs[:5])
        print(f"    Head {h}: [{top5}] (rank-1 ratio: {svs[0]/svs.sum():.3f})")


@torch.no_grad()
def effective_linear_map(model, vocab, device, n_subjects=None):
    """Extract the effective linear map from subject embedding to pre-LN output,
    separately for R0 and R1. Since attention weights depend on the input
    (softmax), this is measured empirically."""
    if n_subjects is None:
        n_subjects = vocab.n_entities

    block = model.blocks[0]

    print("\n=== Effective linear map: subject_emb → output ===\n")

    for rel_k in [0, 1]:
        subj_embs = []
        outputs_pre_ln = []
        outputs_post_ln = []
        logit_vecs = []

        for i in range(n_subjects):
            tokens = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(i)]],
                dtype=torch.long, device=device,
            )
            x = model.token_emb(tokens) + model.pos_emb(torch.arange(3, device=device).unsqueeze(0))

            with model.capture_sublayers() as sublayers:
                logits = model(tokens)

            # Get pre-final-LN representation at subject position
            attn_out = sublayers["attn"][0][0, 2]  # [dim]
            residual = x[0, 2] + attn_out
            post_ln = model.ln_f(residual.unsqueeze(0)).squeeze(0)

            subj_emb = model.token_emb.weight[vocab.subject_id(i)]
            subj_embs.append(subj_emb.cpu())
            outputs_pre_ln.append(residual.cpu())
            outputs_post_ln.append(post_ln.cpu())
            logit_vecs.append(logits[0, 2].cpu())

        subj_embs = torch.stack(subj_embs)
        outputs_pre_ln = torch.stack(outputs_pre_ln)
        outputs_post_ln = torch.stack(outputs_post_ln)

        # Fit affine map: output ≈ W @ subj_emb + b
        # Using least squares
        X = torch.cat([subj_embs, torch.ones(n_subjects, 1)], dim=1)  # [N, dim+1]

        for label, Y in [("pre-LN residual", outputs_pre_ln), ("post-LN output", outputs_post_ln)]:
            W_fit, residuals, rank, sv = torch.linalg.lstsq(X, Y)
            Y_pred = X @ W_fit
            ss_res = ((Y - Y_pred) ** 2).sum().item()
            ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot

            print(f"R{rel_k} {label}:")
            print(f"  Affine fit R²: {r2:.6f}")
            print(f"  Top singular values of W: {sv[:5].tolist()}")

        # How well do correct object embeddings match the output?
        obj_embs = model.lm_head.weight[vocab.object_offset:vocab.object_offset + vocab.n_entities]
        correct_cos = []
        wrong_cos = []
        for i in range(n_subjects):
            out = outputs_post_ln[i].to(device)
            correct_objs = vocab.relation_objects(i, rel_k)
            correct_emb = obj_embs[correct_objs].mean(dim=0)
            cos_correct = F.cosine_similarity(out.unsqueeze(0), correct_emb.unsqueeze(0)).item()
            correct_cos.append(cos_correct)

            wrong_k = 1 - rel_k
            wrong_objs = vocab.relation_objects(i, wrong_k)
            wrong_emb = obj_embs[wrong_objs].mean(dim=0)
            cos_wrong = F.cosine_similarity(out.unsqueeze(0), wrong_emb.unsqueeze(0)).item()
            wrong_cos.append(cos_wrong)

        print(f"  Cosine with correct object centroid: {np.mean(correct_cos):.4f}")
        print(f"  Cosine with wrong-relation centroid: {np.mean(wrong_cos):.4f}")
        print()


@torch.no_grad()
def attention_weight_dependence(model, vocab, device, n_subjects=None):
    """How much do attention weights vary across subjects vs between relations?
    If weights are nearly constant per relation, the map is effectively linear."""
    if n_subjects is None:
        n_subjects = vocab.n_entities

    attn_data = extract_attention_patterns(model, vocab, device, n_subjects)

    print("\n=== Attention weight dependence on subject vs relation ===\n")

    for rel_k in [0, 1]:
        patterns = attn_data[rel_k]  # [n_subjects, n_heads, 3]
        mean = patterns.mean(axis=0)  # [n_heads, 3]
        std = patterns.std(axis=0)
        cv = std / (mean + 1e-8)  # coefficient of variation

        print(f"R{rel_k}:")
        n_heads = mean.shape[0]
        for h in range(n_heads):
            print(f"  Head {h}: sep={mean[h,0]:.4f}±{std[h,0]:.4f} (CV={cv[h,0]:.3f}), "
                  f"rel={mean[h,1]:.4f}±{std[h,1]:.4f} (CV={cv[h,1]:.3f}), "
                  f"subj={mean[h,2]:.4f}±{std[h,2]:.4f} (CV={cv[h,2]:.3f})")

    # Between-relation difference
    r0_mean = attn_data[0].mean(axis=0)
    r1_mean = attn_data[1].mean(axis=0)
    diff = np.abs(r0_mean - r1_mean)
    print(f"\n  Between-relation |diff| in mean attention:")
    n_heads = r0_mean.shape[0]
    for h in range(n_heads):
        print(f"    Head {h}: Δsep={diff[h,0]:.4f}, Δrel={diff[h,1]:.4f}, Δsubj={diff[h,2]:.4f}")

    return attn_data


@torch.no_grad()
def residual_stream_decomposition(model, vocab, device, n_subjects=None):
    """Decompose the residual stream at the subject position into:
    embedding + pos_emb + attention_output, and measure each component's
    contribution to the final prediction."""
    if n_subjects is None:
        n_subjects = vocab.n_entities

    print("\n=== Residual stream decomposition ===\n")

    W_u = model.lm_head.weight  # [vocab_size, dim]
    obj_range = slice(vocab.object_offset, vocab.object_offset + vocab.n_entities)

    for rel_k in [0, 1]:
        emb_correct = 0
        attn_correct = 0
        full_correct = 0
        total = 0

        emb_logit_contrib = []
        attn_logit_contrib = []

        for i in range(n_subjects if n_subjects else vocab.n_entities):
            tokens = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(i)]],
                dtype=torch.long, device=device,
            )
            x = model.token_emb(tokens) + model.pos_emb(torch.arange(3, device=device).unsqueeze(0))
            emb_at_subj = x[0, 2]  # [dim]

            with model.capture_sublayers() as sublayers:
                logits = model(tokens)

            attn_out = sublayers["attn"][0][0, 2]

            # Logit contribution from each component (before final LN)
            # This is approximate since LN is nonlinear, but informative
            emb_logits = (W_u[obj_range] @ emb_at_subj)
            attn_logits = (W_u[obj_range] @ attn_out)

            correct_objs = [j for j in vocab.relation_objects(i, rel_k)]

            emb_pred = emb_logits.argmax().item()
            attn_pred = attn_logits.argmax().item()
            full_pred = logits[0, 2, vocab.object_offset:vocab.object_offset + vocab.n_entities].argmax().item()

            emb_correct += int(emb_pred in correct_objs)
            attn_correct += int(attn_pred in correct_objs)
            full_correct += int(full_pred in correct_objs)
            total += 1

            emb_logit_contrib.append(emb_logits[correct_objs].mean().item())
            attn_logit_contrib.append(attn_logits[correct_objs].mean().item())

        print(f"R{rel_k}:")
        print(f"  Embedding-only accuracy (obj logits from emb):   {emb_correct/total:.4f}")
        print(f"  Attn-output-only accuracy (obj logits from attn): {attn_correct/total:.4f}")
        print(f"  Full model accuracy:                              {full_correct/total:.4f}")
        print(f"  Mean logit for correct objs from emb:   {np.mean(emb_logit_contrib):.4f}")
        print(f"  Mean logit for correct objs from attn:  {np.mean(attn_logit_contrib):.4f}")
        print()


@torch.no_grad()
def relation_dependent_qk_analysis(model, vocab, device):
    """Analyze how Q(subject) · K(relation) differs between R0 and R1,
    and how this modulates the attention pattern."""
    block = model.blocks[0]
    attn = block.attn
    ln1 = block.ln1
    n_heads = attn.n_heads
    head_dim = attn.head_dim
    dim = n_heads * head_dim

    W_qkv = attn.qkv.weight
    W_q = W_qkv[:dim]
    W_k = W_qkv[dim:2*dim]

    print("\n=== QK analysis: how relations modulate attention ===\n")

    # Get LN-normalized relation embeddings
    rel0_emb = model.token_emb.weight[vocab.relation_id(0)] + model.pos_emb.weight[1]
    rel1_emb = model.token_emb.weight[vocab.relation_id(1)] + model.pos_emb.weight[1]

    # We need to pass through LN — but LN depends on the full sequence,
    # so let's use a representative input
    for rel_k, rel_raw in [(0, rel0_emb), (1, rel1_emb)]:
        tokens_sample = torch.tensor(
            [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(0)]],
            dtype=torch.long, device=device,
        )
        x = model.token_emb(tokens_sample) + model.pos_emb(torch.arange(3, device=device).unsqueeze(0))
        h = ln1(x)
        rel_ln = h[0, 1]  # LN-normalized relation at pos 1

        # K(relation) per head
        k_rel = W_k @ rel_ln
        k_per_head = k_rel.view(n_heads, head_dim)
        print(f"R{rel_k} K(relation) norms per head: " +
              ", ".join(f"h{h}={k_per_head[h].norm():.3f}" for h in range(n_heads)))

    # Cosine similarity of K(R0) vs K(R1) per head
    tokens_r0 = torch.tensor([[vocab.sep_id, vocab.relation_id(0), vocab.subject_id(0)]],
                             dtype=torch.long, device=device)
    tokens_r1 = torch.tensor([[vocab.sep_id, vocab.relation_id(1), vocab.subject_id(0)]],
                             dtype=torch.long, device=device)
    x_r0 = model.token_emb(tokens_r0) + model.pos_emb(torch.arange(3, device=device).unsqueeze(0))
    x_r1 = model.token_emb(tokens_r1) + model.pos_emb(torch.arange(3, device=device).unsqueeze(0))
    h_r0 = ln1(x_r0)
    h_r1 = ln1(x_r1)

    k_r0 = (W_k @ h_r0[0, 1]).view(n_heads, head_dim)
    k_r1 = (W_k @ h_r1[0, 1]).view(n_heads, head_dim)

    print(f"\nCosine K(R0) vs K(R1) per head: " +
          ", ".join(f"h{h}={F.cosine_similarity(k_r0[h:h+1], k_r1[h:h+1]).item():.3f}"
                    for h in range(n_heads)))

    # V(relation) per head — this is the "relation signal" added to the output
    W_v = W_qkv[2*dim:3*dim]
    W_o = attn.out_proj.weight

    v_r0 = (W_v @ h_r0[0, 1]).view(n_heads, head_dim)
    v_r1 = (W_v @ h_r1[0, 1]).view(n_heads, head_dim)

    print(f"Cosine V(R0) vs V(R1) per head: " +
          ", ".join(f"h{h}={F.cosine_similarity(v_r0[h:h+1], v_r1[h:h+1]).item():.3f}"
                    for h in range(n_heads)))

    # OV(relation) — the actual vector added to the residual stream from attending to relation
    for h in range(n_heads):
        Wo_h = W_o[:, h*head_dim:(h+1)*head_dim]
        ov_r0 = Wo_h @ v_r0[h]
        ov_r1 = Wo_h @ v_r1[h]
        cos = F.cosine_similarity(ov_r0.unsqueeze(0), ov_r1.unsqueeze(0)).item()
        print(f"  Head {h} OV(R0) vs OV(R1): cos={cos:.3f}, "
              f"norm_R0={ov_r0.norm():.3f}, norm_R1={ov_r1.norm():.3f}, "
              f"diff_norm={( ov_r0 - ov_r1).norm():.3f}")


@torch.no_grad()
def type_cluster_analysis(model, vocab, device, n_subjects=None):
    """Check if outputs cluster by the correct type (f-type for R0, g-type for R1)."""
    if n_subjects is None:
        n_subjects = vocab.n_entities

    print("\n=== Type-cluster analysis of outputs ===\n")

    for rel_k in [0, 1]:
        type_fn = vocab.subject_f_type if rel_k == 0 else vocab.subject_g_type
        type_groups = {}

        outputs = []
        for i in range(n_subjects):
            tokens = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(i)]],
                dtype=torch.long, device=device,
            )
            logits = model(tokens)
            obj_logits = logits[0, 2, vocab.object_offset:vocab.object_offset + vocab.n_entities]
            outputs.append(obj_logits.cpu())

            t = type_fn(i)
            type_groups.setdefault(t, []).append(i)

        outputs = torch.stack(outputs)  # [n_subjects, n_entities]

        # Measure within-type vs between-type cosine similarity of logit vectors
        within_cos = []
        between_cos = []
        types_list = list(type_groups.keys())

        for t_idx, t in enumerate(types_list[:30]):
            members = type_groups[t]
            if len(members) < 2:
                continue
            for a in range(min(5, len(members))):
                for b in range(a+1, min(5, len(members))):
                    cos = F.cosine_similarity(
                        outputs[members[a]].unsqueeze(0),
                        outputs[members[b]].unsqueeze(0)
                    ).item()
                    within_cos.append(cos)

            if t_idx + 1 < len(types_list):
                other_t = types_list[t_idx + 1]
                other_members = type_groups[other_t]
                for a in range(min(3, len(members))):
                    for b in range(min(3, len(other_members))):
                        cos = F.cosine_similarity(
                            outputs[members[a]].unsqueeze(0),
                            outputs[other_members[b]].unsqueeze(0)
                        ).item()
                        between_cos.append(cos)

        print(f"R{rel_k} ({'f-type' if rel_k == 0 else 'g-type'}):")
        print(f"  Within-type logit cosine:  {np.mean(within_cos):.4f} ± {np.std(within_cos):.4f}")
        print(f"  Between-type logit cosine: {np.mean(between_cos):.4f} ± {np.std(between_cos):.4f}")
        print(f"  Separation: {np.mean(within_cos) - np.mean(between_cos):.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "checkpoints_attn1l_seed42" / "final.pt"
    print(f"Loading checkpoint from {ckpt_path}")
    model, vocab, model_cfg, rel_objects_table, ckpt = load_model_and_vocab(ckpt_path, device)
    print(f"Model: dim={model_cfg.hidden_dim}, heads={model_cfg.n_heads}, "
          f"head_dim={model_cfg.hidden_dim // model_cfg.n_heads}")

    # Monkey-patch relation_objects for evaluation
    vocab.relation_objects = lambda subj_idx, rel_k: rel_objects_table[(subj_idx, rel_k)]

    n_subj = vocab.n_entities

    print("\n" + "=" * 70)
    print("ANALYSIS: 1-LAYER ATTENTION-ONLY (NO MLP, NO DUMMY)")
    print("=" * 70)

    attention_weight_dependence(model, vocab, device, n_subj)
    head_specialization(model, vocab, device, n_subj)
    variance_decomposition(model, vocab, device, n_subj)
    residual_stream_decomposition(model, vocab, device, n_subj)
    relation_dependent_qk_analysis(model, vocab, device)
    ov_circuit_analysis(model, vocab, device)
    effective_linear_map(model, vocab, device, n_subj)
    type_cluster_analysis(model, vocab, device, n_subj)


if __name__ == "__main__":
    main()

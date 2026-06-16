"""
Δcos decomposition for the Salzburg experiment.

Key question: is there an affine transformation from the subject embedding
(layer 0) to the subject representation at layer 1 (the prediction vector)?

Sources:
  - Layer 0 (embedding): token_emb(a_i) + pos_emb(2), no transformer computation
  - Layer 1 (post-transformer): after attention + MLP, encodes prediction

Targets (2 contexts × 5 stages):
  Artificial context [x, b_j]:
    1. Residual stream h₁ (pre-ln_f)
    2. Attention output (L1 block)
    3. MLP output (L1 block)
    4. Post-ln_f: ln_f(h₁)
    5. Unembedding: lm_head.weight rows

  Natural context (subject at pos 2 in [x, r_k, a_i]):
    6. Subject repr at L1 (pre-ln_f) — the prediction vector
    7. Attention output (L1 block)
    8. MLP output (L1 block)
    9. Post-ln_f: ln_f(prediction vector)
   10. Unembedding: lm_head.weight rows (same as #5)

Usage (from salzburg/):
    python experiments/dcos_decomposition.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.affine import fit_affine
from src.config import cfg as default_cfg
from src.model import build_model
from src.representations import (
    build_pooled_object_matrix,
    build_relation_object_matrix,
    get_object_representations,
    get_object_sublayer_outputs,
    get_post_lnf_object_representations,
    get_subject_representations,
    get_subject_sublayer_outputs,
    get_unembedding_object_representations,
    stack_all_subject_representations,
    stack_subject_representations,
)
from src.training import load_checkpoint

RELATION_KEYS = ["R0", "R1", "R2"]


def delta_cos_decomposed(S, O, W, b):
    """Return (delta_cos, cos_o_ohat, cos_o_s) all as mean floats."""
    O_hat = S @ W.T
    if b is not None:
        O_hat = O_hat + b
    o_norm = F.normalize(O, dim=1)
    o_hat_norm = F.normalize(O_hat, dim=1)
    s_norm = F.normalize(S, dim=1)
    cos_o_ohat = (o_norm * o_hat_norm).sum(dim=1)
    cos_o_s = (o_norm * s_norm).sum(dim=1)
    dcos = (cos_o_ohat - cos_o_s).mean().item()
    return dcos, cos_o_ohat.mean().item(), cos_o_s.mean().item()


def kfold_decomposed(S, O, k=5):
    """K-fold CV returning mean (dcos, cos_o_ohat, cos_o_s)."""
    n = S.shape[0]
    fold_size = n // k
    indices = np.arange(n)
    all_dcos, all_co, all_cs = [], [], []
    for fold in range(k):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        W, b = fit_affine(S[train_idx], O[train_idx], bias=True)
        dc, co, cs = delta_cos_decomposed(S[val_idx], O[val_idx], W, b)
        all_dcos.append(dc)
        all_co.append(co)
        all_cs.append(cs)
    return np.mean(all_dcos), np.mean(all_co), np.mean(all_cs)


def subject_fold_kfold_decomposed(S, O, k=5):
    """K-fold CV splitting by subject index for pooled (300-row) data."""
    n_relations = 3
    n_subjects = S.shape[0] // n_relations
    fold_size = n_subjects // k
    indices = np.arange(n_subjects)
    all_dcos, all_co, all_cs = [], [], []
    for fold in range(k):
        val_subj = indices[fold * fold_size: (fold + 1) * fold_size]
        train_subj = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        val_rows = np.array([s * n_relations + r for s in val_subj for r in range(n_relations)])
        train_rows = np.array([s * n_relations + r for s in train_subj for r in range(n_relations)])
        W, b = fit_affine(S[train_rows], O[train_rows], bias=True)
        dc, co, cs = delta_cos_decomposed(S[val_rows], O[val_rows], W, b)
        all_dcos.append(dc)
        all_co.append(co)
        all_cs.append(cs)
    return np.mean(all_dcos), np.mean(all_co), np.mean(all_cs)


def apply_lnf_to_reprs(model, reprs, layer, device):
    """Apply model.ln_f to subject representations at a given layer."""
    ln_f = model.ln_f
    result = {}
    with torch.no_grad():
        for key, layers in reprs.items():
            vec = layers[layer]
            lnf_vec = ln_f(vec.unsqueeze(0).to(device)).squeeze(0).cpu()
            result[key] = [lnf_vec] * (layer + 1)
    return result


def fmt3(dcos, co, cs):
    return f"Δcos={dcos:+.3f}  cos(o,ô)={co:+.3f}  cos(o,s)={cs:+.3f}"


def main():
    cfg = default_cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.train.checkpoint_dir / "best.pt"

    model = build_model(cfg.model, cfg.vocab)
    load_checkpoint(ckpt_path, model, device)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}\n")

    all_indices = list(range(cfg.vocab.n_subjects))
    n_rel = cfg.vocab.num_relations

    print("Extracting representations...")
    subj_reprs = get_subject_representations(model, cfg, all_indices, device)
    obj_resid = get_object_representations(model, cfg, all_indices, device)
    obj_post_lnf = get_post_lnf_object_representations(model, cfg, all_indices, device)
    obj_unemb = get_unembedding_object_representations(model, cfg, all_indices, cfg.model.n_layers)

    print("Extracting sublayer outputs...")
    obj_sublayers = get_object_sublayer_outputs(model, cfg, all_indices, device)
    subj_sublayers = get_subject_sublayer_outputs(model, cfg, all_indices, device)

    print("Applying ln_f to subject L1 representations...")
    subj_post_lnf = apply_lnf_to_reprs(model, subj_reprs, layer=1, device=device)
    print()

    results_lines = []

    def log(line=""):
        print(line)
        results_lines.append(line)

    # Pre-compute source matrices at both layers
    # Layer 0 = embedding, Layer 1 = post-transformer
    S0_all, _ = stack_all_subject_representations(subj_reprs, all_indices, n_rel, layer=0)
    S1_all, _ = stack_all_subject_representations(subj_reprs, all_indices, n_rel, layer=1)

    S0_per_rel = [stack_subject_representations(subj_reprs, all_indices, ri, layer=0) for ri in range(n_rel)]
    S1_per_rel = [stack_subject_representations(subj_reprs, all_indices, ri, layer=1) for ri in range(n_rel)]

    # Natural-context targets: subject repr at L1
    P_all = S1_all
    P_per_rel = S1_per_rel
    P_lnf_all, _ = stack_all_subject_representations(subj_post_lnf, all_indices, n_rel, layer=0)
    P_lnf_per_rel = [stack_subject_representations(subj_post_lnf, all_indices, ri, layer=0) for ri in range(n_rel)]

    # Subject sublayer targets (attention and MLP outputs at subject position)
    subj_attn_all, _ = stack_all_subject_representations(subj_sublayers["attn"], all_indices, n_rel, layer=0)
    subj_attn_per_rel = [stack_subject_representations(subj_sublayers["attn"], all_indices, ri, layer=0) for ri in range(n_rel)]
    subj_mlp_all, _ = stack_all_subject_representations(subj_sublayers["mlp"], all_indices, n_rel, layer=0)
    subj_mlp_per_rel = [stack_subject_representations(subj_sublayers["mlp"], all_indices, ri, layer=0) for ri in range(n_rel)]

    # ========================================================================
    # ARTIFICIAL CONTEXT: [x, b_j] at position 1
    # ========================================================================
    log("%" * 78)
    log("% ARTIFICIAL CONTEXT: object repr from [x, b_j] at position 1")
    log("%" * 78)
    log()

    art_targets = [
        ("Residual stream h₁ (pre-ln_f)", obj_resid, 1),
        ("Attention output (L1)", obj_sublayers["attn"], 0),
        ("MLP output (L1)", obj_sublayers["mlp"], 0),
        ("Post-ln_f: ln_f(h₁)", obj_post_lnf, 1),
        ("Unembedding: lm_head.weight rows", obj_unemb, 1),
    ]

    for obj_label, obj_r, obj_layer in art_targets:
        log("=" * 78)
        log(f"Target: {obj_label} | Layer L1")
        log("=" * 78)

        O_pooled = build_pooled_object_matrix(obj_r, all_indices, n_rel, cfg.vocab, obj_layer)
        O_rel_fn = lambda ri, o=obj_r, ol=obj_layer: build_relation_object_matrix(o, all_indices, ri, cfg.vocab, ol)

        for src_label, src_pooled, src_per_rel in [
            ("L1", S1_all, S1_per_rel),
            ("L0", S0_all, S0_per_rel),
        ]:
            # pooled
            dc, co, cs = subject_fold_kfold_decomposed(src_pooled, O_pooled)
            log(f"  {src_label} pooled:     {fmt3(dc, co, cs)}")

            # per-relation
            for ri, rkey in enumerate(RELATION_KEYS):
                dc, co, cs = kfold_decomposed(src_per_rel[ri], O_rel_fn(ri))
                log(f"  {src_label} {rkey}:        {fmt3(dc, co, cs)}")

            # average
            dcs, cos_list, css = [], [], []
            for ri in range(n_rel):
                dc, co, cs = kfold_decomposed(src_per_rel[ri], O_rel_fn(ri))
                dcs.append(dc); cos_list.append(co); css.append(cs)
            log(f"  {src_label} (avg):      {fmt3(np.mean(dcs), np.mean(cos_list), np.mean(css))}")

        log()

    # ========================================================================
    # NATURAL CONTEXT: subject repr at position 2 in [x, r_k, a_i]
    # ========================================================================
    log("%" * 78)
    log("% NATURAL CONTEXT: subject repr at position 2 in [x, r_k, a_i]")
    log("%" * 78)
    log()

    nat_targets = [
        ("Subject repr at L1 (pre-ln_f)", P_all, P_per_rel),
        ("Attention output (L1)", subj_attn_all, subj_attn_per_rel),
        ("MLP output (L1)", subj_mlp_all, subj_mlp_per_rel),
        ("Post-ln_f: ln_f(subject repr at L1)", P_lnf_all, P_lnf_per_rel),
        ("Unembedding: lm_head.weight rows", None, None),
    ]

    for obj_label, O_pool, O_rels in nat_targets:
        log("=" * 78)
        log(f"Target: {obj_label} | Layer L1")
        log("=" * 78)

        if O_pool is None:
            O_pool = build_pooled_object_matrix(obj_unemb, all_indices, n_rel, cfg.vocab, 1)
            O_rel_fn = lambda ri: build_relation_object_matrix(obj_unemb, all_indices, ri, cfg.vocab, 1)
        else:
            O_rel_fn = lambda ri, rels=O_rels: rels[ri]

        # Only L0 source (L1 → L1 pre-ln_f would be identity)
        src_label = "L0"
        dc, co, cs = subject_fold_kfold_decomposed(S0_all, O_pool)
        log(f"  {src_label} pooled:     {fmt3(dc, co, cs)}")

        for ri, rkey in enumerate(RELATION_KEYS):
            dc, co, cs = kfold_decomposed(S0_per_rel[ri], O_rel_fn(ri))
            log(f"  {src_label} {rkey}:        {fmt3(dc, co, cs)}")

        dcs, cos_list, css = [], [], []
        for ri in range(n_rel):
            dc, co, cs = kfold_decomposed(S0_per_rel[ri], O_rel_fn(ri))
            dcs.append(dc); cos_list.append(co); css.append(cs)
        log(f"  {src_label} (avg):      {fmt3(np.mean(dcs), np.mean(cos_list), np.mean(css))}")

        log()

    # Save
    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = cfg.train.results_dir / "dcos_decomposition.md"
    with open(results_path, "w") as f:
        f.write("# Salzburg Δcos Decomposition\n\n")
        f.write("Sequence format: `[x, r_k, a_i, b_j]` — no fillers, relation before subject.\n")
        f.write("Subject at position 2. All results at 5-fold CV.\n\n")
        f.write("## Two contexts × five stages\n\n")
        f.write("**Artificial context**: object b_j represented via [x, b_j] at position 1\n")
        f.write("**Natural context**: subject repr at position 2 in [x, r_k, a_i]\n\n")
        f.write("**Five stages**: (1) residual stream pre-ln_f, (2) attention output, (3) MLP output, (4) post-ln_f, (5) unembedding rows\n\n")
        f.write("**Sources**:\n")
        f.write("- **L0**: layer 0 (embedding) = token_emb(a_i) + pos_emb(2)\n")
        f.write("- **L1**: layer 1 (post-transformer) = prediction vector\n\n")
        f.write("- **ô**: affine reconstruction of target from source, i.e., ô = s @ W.T + b\n")
        f.write("- Δcos = cos(o, ô) - cos(o, s)\n")
        f.write("- cos(o, ô): cosine between true target and affine-predicted target\n")
        f.write("- cos(o, s): cosine between true target and source repr\n\n")
        f.write("```\n")
        f.write("\n".join(results_lines))
        f.write("\n```\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()

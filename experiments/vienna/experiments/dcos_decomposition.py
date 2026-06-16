"""
Decompose Δcos into its two components: cos(o, ô) and cos(o, s).

Six target types organized as 2 contexts × 3 stages:

  Artificial context [x, b_j] (object at position 1):
    1. Residual stream h₁ (pre-ln_f)
    2. Post-ln_f: ln_f(h₁)
    3. Unembedding: lm_head.weight rows

  Natural context [x, a_i, r_k, f0, f1, f2] (prediction at position 5):
    4. Prediction vector p = h₁ at f2 (pre-ln_f)
    5. Post-ln_f prediction: ln_f(p)
    6. Unembedding: lm_head.weight rows (same as #3)

Sources: f2 (pos 5), rk (pos 2), subject (pos 1 in [x, a_i]).
For natural-context targets, f2 source is omitted (trivially identity for #4).

Usage (from vienna/):
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
    get_all_object_representations,
    get_all_subject_representations,
    get_filler_representations,
    get_post_lnf_object_representations,
    get_relation_position_representations,
    get_unembedding_object_representations,
    stack_all_filler_representations,
    stack_filler_representations,
    stack_representations,
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


def build_pooled_object_matrix(obj_reprs, subject_indices, num_relations, vocab, layer):
    vecs = []
    for i in subject_indices:
        for k in range(num_relations):
            j = vocab.apply_relation(k, i)
            vecs.append(obj_reprs[j][layer])
    return torch.stack(vecs, dim=0)


def build_relation_object_matrix(obj_reprs, subject_indices, relation_idx, vocab, layer):
    vecs = []
    for i in subject_indices:
        j = vocab.apply_relation(relation_idx, i)
        vecs.append(obj_reprs[j][layer])
    return torch.stack(vecs, dim=0)


def fmt3(dcos, co, cs):
    return f"Δcos={dcos:+.3f}  cos(o,ô)={co:+.3f}  cos(o,s)={cs:+.3f}"


def apply_lnf_to_filler_reprs(model, filler_reprs, layer, device):
    """Apply model.ln_f to filler representations at a given layer.

    Returns a dict with same structure as filler_reprs but with ln_f applied.
    """
    ln_f = model.ln_f
    result = {}
    with torch.no_grad():
        for key, layers in filler_reprs.items():
            vec = layers[layer]
            lnf_vec = ln_f(vec.unsqueeze(0).to(device)).squeeze(0).cpu()
            result[key] = [lnf_vec] * (layer + 1)
    return result


def run_source_block(log, label, S_pooled, S_per_rel, O_pooled, O_per_rel_fn,
                     all_indices, cfg, is_pooled_subject_fold):
    """Run pooled, per-relation, and avg rows for one source type."""
    if is_pooled_subject_fold:
        dc, co, cs = subject_fold_kfold_decomposed(S_pooled, O_pooled)
    else:
        dc, co, cs = kfold_decomposed(S_pooled, O_pooled)
    log(f"  {label} pooled:{'  ' if len(label) <= 4 else ' '}{fmt3(dc, co, cs)}")

    for ri, rkey in enumerate(RELATION_KEYS):
        dc, co, cs = kfold_decomposed(S_per_rel[ri], O_per_rel_fn(ri))
        log(f"  {label} {rkey}:{' ' * (8 - len(label))}{fmt3(dc, co, cs)}")

    dcs, cos_list, css = [], [], []
    for ri in range(cfg.vocab.num_relations):
        dc, co, cs = kfold_decomposed(S_per_rel[ri], O_per_rel_fn(ri))
        dcs.append(dc)
        cos_list.append(co)
        css.append(cs)
    log(f"  {label} (avg):{' ' * (6 - len(label))}{fmt3(np.mean(dcs), np.mean(cos_list), np.mean(css))}")


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
    layer = 1  # L1
    n_rel = cfg.vocab.num_relations

    print("Extracting representations...")
    subj_reprs = get_all_subject_representations(model, cfg, all_indices, device)
    obj_resid = get_all_object_representations(model, cfg, all_indices, device)
    obj_post_lnf = get_post_lnf_object_representations(model, cfg, all_indices, device)
    obj_unemb = get_unembedding_object_representations(model, cfg, all_indices, cfg.model.n_layers)
    filler_reprs = get_filler_representations(model, cfg, all_indices, device)
    rpos_reprs = get_relation_position_representations(model, cfg, all_indices, device)

    print("Applying ln_f to f2 representations...")
    filler_post_lnf = apply_lnf_to_filler_reprs(model, filler_reprs, layer, device)
    print()

    results_lines = []

    def log(line=""):
        print(line)
        results_lines.append(line)

    # Pre-compute source matrices
    F2_all, _ = stack_all_filler_representations(filler_reprs, all_indices, n_rel, layer)
    RK_all, _ = stack_all_filler_representations(rpos_reprs, all_indices, n_rel, layer)
    S_all, _ = stack_representations(subj_reprs, all_indices, layer)
    S_pooled = torch.stack([S_all[i].expand(n_rel, -1) for i in range(len(all_indices))], dim=0)
    S_pooled = S_pooled.reshape(-1, S_all.shape[1])

    F2_per_rel = [stack_filler_representations(filler_reprs, all_indices, ri, layer) for ri in range(n_rel)]
    RK_per_rel = [stack_filler_representations(rpos_reprs, all_indices, ri, layer) for ri in range(n_rel)]

    # Pre-compute natural-context target matrices (f2 = prediction vector)
    P_all = F2_all  # pre-ln_f prediction vectors (pooled)
    P_per_rel = F2_per_rel
    P_lnf_all, _ = stack_all_filler_representations(filler_post_lnf, all_indices, n_rel, 0)
    P_lnf_per_rel = [stack_filler_representations(filler_post_lnf, all_indices, ri, 0) for ri in range(n_rel)]

    # ========================================================================
    # ARTIFICIAL CONTEXT: [x, b_j] at position 1
    # ========================================================================
    log("%" * 78)
    log("% ARTIFICIAL CONTEXT: object repr from [x, b_j] at position 1")
    log("%" * 78)
    log()

    art_targets = [
        ("Residual stream h₁ (pre-ln_f)", obj_resid),
        ("Post-ln_f: ln_f(h₁)", obj_post_lnf),
        ("Unembedding: lm_head.weight rows", obj_unemb),
    ]

    for obj_label, obj_r in art_targets:
        log("=" * 78)
        log(f"Target: {obj_label} | Layer L1")
        log("=" * 78)

        O_pooled = build_pooled_object_matrix(obj_r, all_indices, n_rel, cfg.vocab, layer)
        O_per_rel_fn = lambda ri: build_relation_object_matrix(obj_r, all_indices, ri, cfg.vocab, layer)

        # f2 source
        dc, co, cs = subject_fold_kfold_decomposed(F2_all, O_pooled)
        log(f"  f2 pooled:      {fmt3(dc, co, cs)}")
        for ri, rkey in enumerate(RELATION_KEYS):
            dc, co, cs = kfold_decomposed(F2_per_rel[ri], O_per_rel_fn(ri))
            log(f"  f2 {rkey}:         {fmt3(dc, co, cs)}")

        # rk source
        dc, co, cs = subject_fold_kfold_decomposed(RK_all, O_pooled)
        log(f"  rk pooled:      {fmt3(dc, co, cs)}")
        for ri, rkey in enumerate(RELATION_KEYS):
            dc, co, cs = kfold_decomposed(RK_per_rel[ri], O_per_rel_fn(ri))
            log(f"  rk {rkey}:         {fmt3(dc, co, cs)}")
        dcs, cos_list, css = [], [], []
        for ri in range(n_rel):
            dc, co, cs = kfold_decomposed(RK_per_rel[ri], O_per_rel_fn(ri))
            dcs.append(dc); cos_list.append(co); css.append(cs)
        log(f"  rk (avg):       {fmt3(np.mean(dcs), np.mean(cos_list), np.mean(css))}")

        # subj source
        dcs, cos_list, css = [], [], []
        for ri in range(n_rel):
            dc, co, cs = kfold_decomposed(S_all, O_per_rel_fn(ri))
            dcs.append(dc); cos_list.append(co); css.append(cs)
        log(f"  subj (avg):     {fmt3(np.mean(dcs), np.mean(cos_list), np.mean(css))}")

        log()

    # ========================================================================
    # NATURAL CONTEXT: prediction at f2 position in [x, a_i, r_k, f0, f1, f2]
    # ========================================================================
    log("%" * 78)
    log("% NATURAL CONTEXT: prediction vector at f2 position (pos 5)")
    log("%" * 78)
    log()

    nat_targets = [
        ("Prediction vector p (pre-ln_f)", P_all, P_per_rel),
        ("Post-ln_f prediction: ln_f(p)", P_lnf_all, P_lnf_per_rel),
        ("Unembedding: lm_head.weight rows", None, None),
    ]

    for obj_label, O_pool, O_rels in nat_targets:
        log("=" * 78)
        log(f"Target: {obj_label} | Layer L1")
        log("=" * 78)

        if O_pool is None:
            # Unembedding — same as artificial context
            O_pool = build_pooled_object_matrix(obj_unemb, all_indices, n_rel, cfg.vocab, layer)
            O_rel_fn = lambda ri: build_relation_object_matrix(obj_unemb, all_indices, ri, cfg.vocab, layer)
        else:
            O_rel_fn = lambda ri, rels=O_rels: rels[ri]

        # rk source
        dc, co, cs = subject_fold_kfold_decomposed(RK_all, O_pool)
        log(f"  rk pooled:      {fmt3(dc, co, cs)}")
        for ri, rkey in enumerate(RELATION_KEYS):
            dc, co, cs = kfold_decomposed(RK_per_rel[ri], O_rel_fn(ri))
            log(f"  rk {rkey}:         {fmt3(dc, co, cs)}")
        dcs, cos_list, css = [], [], []
        for ri in range(n_rel):
            dc, co, cs = kfold_decomposed(RK_per_rel[ri], O_rel_fn(ri))
            dcs.append(dc); cos_list.append(co); css.append(cs)
        log(f"  rk (avg):       {fmt3(np.mean(dcs), np.mean(cos_list), np.mean(css))}")

        # subj source
        dc, co, cs = subject_fold_kfold_decomposed(S_pooled, O_pool)
        log(f"  subj pooled:    {fmt3(dc, co, cs)}")
        for ri, rkey in enumerate(RELATION_KEYS):
            dc, co, cs = kfold_decomposed(S_all, O_rel_fn(ri))
            log(f"  subj {rkey}:       {fmt3(dc, co, cs)}")
        dcs, cos_list, css = [], [], []
        for ri in range(n_rel):
            dc, co, cs = kfold_decomposed(S_all, O_rel_fn(ri))
            dcs.append(dc); cos_list.append(co); css.append(cs)
        log(f"  subj (avg):     {fmt3(np.mean(dcs), np.mean(cos_list), np.mean(css))}")

        log()

    # Save
    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = cfg.train.results_dir / "dcos_decomposition.md"
    with open(results_path, "w") as f:
        f.write("# Δcos Decomposition: cos(o, ô) and cos(o, s)\n\n")
        f.write("All results at Layer L1, 5-fold CV.\n\n")
        f.write("## Two contexts × three stages\n\n")
        f.write("**Artificial context**: object b_j represented via [x, b_j] at position 1\n")
        f.write("**Natural context**: prediction vector at f2 position (pos 5) in [x, a_i, r_k, f0, f1, f2]\n\n")
        f.write("**Three stages**: (1) residual stream pre-ln_f, (2) post-ln_f, (3) unembedding rows\n\n")
        f.write("- **source** (s): f2 (pos 5), rk (pos 2), or subject (pos 1 in [x, a_i]) — all pre-ln_f\n")
        f.write("- **target** (o): see section headers\n")
        f.write("- **ô**: affine reconstruction of o from s, i.e., ô = s @ W.T + b\n")
        f.write("- Δcos = cos(o, ô) - cos(o, s)\n")
        f.write("- cos(o, ô): cosine between true target and affine-predicted target\n")
        f.write("- cos(o, s): cosine between true target and source repr\n\n")
        f.write("```\n")
        f.write("\n".join(results_lines))
        f.write("\n```\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()

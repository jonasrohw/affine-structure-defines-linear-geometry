"""
Evaluation for Phase 1: do embeddings reflect attribute structure?

Tests:
1. Graded cosine similarity by f-attribute overlap (0/3, 1/3, 2/3, 3/3)
2. Graded cosine similarity by g-attribute overlap
3. R0/R1 partner similarity (subject ↔ object)
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import COMBOS, VocabConfig
from .model import MiniTransformer


def f_overlap(i: int, j: int, vocab: VocabConfig = None) -> int:
    if vocab is not None:
        return len(set(vocab.subject_f_type(i)) & set(vocab.subject_f_type(j)))
    return len(set(COMBOS[i]) & set(COMBOS[j]))


def g_overlap_subj(i: int, j: int, vocab: VocabConfig = None) -> int:
    if vocab is not None:
        return len(set(vocab.subject_g_type(i)) & set(vocab.subject_g_type(j)))
    return len(set(COMBOS[i]) & set(COMBOS[j]))


def g_overlap_obj(i: int, j: int, vocab: VocabConfig = None, shift: int = 60) -> int:
    if vocab is not None:
        return len(set(vocab.object_g_type(i)) & set(vocab.object_g_type(j)))
    gi = COMBOS[(i + shift) % len(COMBOS)]
    gj = COMBOS[(j + shift) % len(COMBOS)]
    return len(set(gi) & set(gj))


def extract_embeddings(model: MiniTransformer, vocab: VocabConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        subj_emb = model.token_emb.weight[vocab.subject_offset:vocab.subject_offset + vocab.n_entities].cpu()
        obj_emb = model.token_emb.weight[vocab.object_offset:vocab.object_offset + vocab.n_entities].cpu()
    return subj_emb, obj_emb


def pairwise_similarity_by_overlap(embeddings: torch.Tensor, overlap_fn, n: int) -> Dict[int, List[float]]:
    """Group pairwise cosine similarities by attribute overlap level."""
    emb_norm = F.normalize(embeddings, dim=1)
    sims = emb_norm @ emb_norm.T

    by_overlap: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: []}
    for i in range(n):
        for j in range(i + 1, n):
            ov = overlap_fn(i, j)
            by_overlap[ov].append(sims[i, j].item())
    return by_overlap


def similarity_report(model: MiniTransformer, vocab: VocabConfig) -> Dict:
    subj_emb, obj_emb = extract_embeddings(model, vocab)
    n = vocab.n_entities

    # Subject-subject similarity by f-overlap
    def subj_f_fn(i, j):
        return f_overlap(i, j, vocab)
    subj_f = pairwise_similarity_by_overlap(subj_emb, subj_f_fn, n)

    # Subject-subject similarity by g-overlap
    def subj_g_fn(i, j):
        return g_overlap_subj(i, j, vocab)
    subj_g = pairwise_similarity_by_overlap(subj_emb, subj_g_fn, n)

    # Object-object similarity by f-overlap
    def obj_f_fn(i, j):
        return len(set(vocab.object_f_type(i)) & set(vocab.object_f_type(j)))
    obj_f = pairwise_similarity_by_overlap(obj_emb, obj_f_fn, n)

    # Object-object similarity by g-overlap
    def obj_g_fn(i, j):
        return g_overlap_obj(i, j, vocab)
    obj_g = pairwise_similarity_by_overlap(obj_emb, obj_g_fn, n)

    # R0 partner similarity: subject i ↔ first R0 partner
    subj_norm = F.normalize(subj_emb, dim=1)
    obj_norm = F.normalize(obj_emb, dim=1)
    r0_indices = [vocab.relation_objects(i, 0)[0] for i in range(n)]
    r0_obj_norm = obj_norm[r0_indices]
    r0_sims = (subj_norm * r0_obj_norm).sum(dim=1)
    r0_mean = r0_sims.mean().item()

    # R1 partner similarity: subject i ↔ first R1 partner
    r1_indices = [vocab.relation_objects(i, 1)[0] for i in range(n)]
    r1_obj_norm = obj_norm[r1_indices]
    r1_sims = (subj_norm * r1_obj_norm).sum(dim=1)
    r1_mean = r1_sims.mean().item()

    # Random subject-object similarity (non-partner pairs)
    all_cross_sims = subj_norm @ obj_norm.T
    mask = torch.ones(n, n, dtype=torch.bool)
    for i in range(n):
        for j in vocab.relation_objects(i, 0):
            mask[i, j] = False
        for j in vocab.relation_objects(i, 1):
            mask[i, j] = False
    random_mean = all_cross_sims[mask].mean().item()

    return {
        "subj_by_f_overlap": {k: (np.mean(v) if v else float("nan")) for k, v in subj_f.items()},
        "subj_by_g_overlap": {k: (np.mean(v) if v else float("nan")) for k, v in subj_g.items()},
        "obj_by_f_overlap": {k: (np.mean(v) if v else float("nan")) for k, v in obj_f.items()},
        "obj_by_g_overlap": {k: (np.mean(v) if v else float("nan")) for k, v in obj_g.items()},
        "r0_partner_sim": r0_mean,
        "r1_partner_sim": r1_mean,
        "random_cross_sim": random_mean,
        "subj_f_counts": {k: len(v) for k, v in subj_f.items()},
    }


def print_report(report: Dict):
    print("Subject-subject similarity by f-attribute overlap:")
    for ov in [0, 1, 2, 3]:
        n = report["subj_f_counts"].get(ov, 0)
        sim = report["subj_by_f_overlap"].get(ov, float("nan"))
        print(f"  {ov}/3 shared: cos = {sim:+.4f}  (n={n})")
    print()
    print("Subject-subject similarity by g-attribute overlap:")
    for ov in [0, 1, 2, 3]:
        sim = report["subj_by_g_overlap"].get(ov, float("nan"))
        print(f"  {ov}/3 shared: cos = {sim:+.4f}")
    print()
    print("Object-object similarity by f-attribute overlap:")
    for ov in [0, 1, 2, 3]:
        sim = report["obj_by_f_overlap"].get(ov, float("nan"))
        print(f"  {ov}/3 shared: cos = {sim:+.4f}")
    print()
    print("Object-object similarity by g-attribute overlap:")
    for ov in [0, 1, 2, 3]:
        sim = report["obj_by_g_overlap"].get(ov, float("nan"))
        print(f"  {ov}/3 shared: cos = {sim:+.4f}")
    print()
    print("Subject-object partner similarity:")
    print(f"  R0 partners (shared f): cos = {report['r0_partner_sim']:+.4f}")
    print(f"  R1 partners (shared g): cos = {report['r1_partner_sim']:+.4f}")
    print(f"  Random (non-partner):   cos = {report['random_cross_sim']:+.4f}")

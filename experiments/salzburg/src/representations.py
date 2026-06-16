"""
Representation extraction for the Salzburg experiment (no-filler variant).

Sequence format: [x, r_k, a_i, b_j] — subject at position 2.

Key extraction: layer 0 (embedding) vs layer 1 (post-transformer) at position 2.
"""
from typing import Dict, List, Tuple

import torch

from .config import Config
from .model import MiniTransformer


@torch.no_grad()
def _extract_repr_at_pos(
    model: MiniTransformer,
    context: List[int],
    position: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Run model on context, return residual stream at `position` for every layer."""
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    with model.capture_residuals() as residuals:
        model(input_ids)
    return [r[0, position, :].cpu() for r in residuals]


@torch.no_grad()
def _extract_sublayer_at_pos(
    model: MiniTransformer,
    context: List[int],
    position: int,
    device: torch.device,
) -> Dict[str, List[torch.Tensor]]:
    """Run model on context, return attn and mlp outputs at `position` per block."""
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    with model.capture_sublayers() as sublayers:
        model(input_ids)
    return {
        "attn": [t[0, position, :].cpu() for t in sublayers["attn"]],
        "mlp": [t[0, position, :].cpu() for t in sublayers["mlp"]],
    }


def get_subject_representations(
    model: MiniTransformer,
    cfg: Config,
    subject_indices: List[int],
    device: torch.device,
) -> Dict[Tuple[int, int], List[torch.Tensor]]:
    """Extract subject-position representations for all (subject, relation) pairs.

    Context: [x, r_k, a_i] — a_i is at position 2.
    Returns representations at all layers (0 = embedding, 1 = post-transformer).

    Returns:
        {(subject_idx, relation_idx): [repr_layer0, repr_layer1, ...]}
    """
    model.eval()
    reprs = {}
    for i in subject_indices:
        for k in range(cfg.vocab.num_relations):
            context = [
                cfg.vocab.sep_id,
                cfg.vocab.relation_id(k),
                cfg.vocab.subject_id(i),
            ]
            reprs[(i, k)] = _extract_repr_at_pos(model, context, position=2, device=device)
    return reprs


def get_subject_sublayer_outputs(
    model: MiniTransformer,
    cfg: Config,
    subject_indices: List[int],
    device: torch.device,
) -> Dict[str, Dict[Tuple[int, int], List[torch.Tensor]]]:
    """Extract attention and MLP sublayer outputs at subject position (pos 2).

    Context: [x, r_k, a_i] — same as get_subject_representations.

    Returns:
        {"attn": {(subj, rel): [vec_block0, ...]}, "mlp": {(subj, rel): [vec_block0, ...]}}
    """
    model.eval()
    attn_reprs = {}
    mlp_reprs = {}
    for i in subject_indices:
        for k in range(cfg.vocab.num_relations):
            context = [
                cfg.vocab.sep_id,
                cfg.vocab.relation_id(k),
                cfg.vocab.subject_id(i),
            ]
            sub = _extract_sublayer_at_pos(model, context, position=2, device=device)
            attn_reprs[(i, k)] = sub["attn"]
            mlp_reprs[(i, k)] = sub["mlp"]
    return {"attn": attn_reprs, "mlp": mlp_reprs}


def get_object_sublayer_outputs(
    model: MiniTransformer,
    cfg: Config,
    pair_indices: List[int],
    device: torch.device,
) -> Dict[str, Dict[int, List[torch.Tensor]]]:
    """Extract attention and MLP sublayer outputs from [x, b_j] at position 1.

    Returns:
        {"attn": {pair_idx: [vec_block0, ...]}, "mlp": {pair_idx: [vec_block0, ...]}}
    """
    model.eval()
    attn_reprs = {}
    mlp_reprs = {}
    for i in pair_indices:
        context = [cfg.vocab.sep_id, cfg.vocab.object_id(i)]
        sub = _extract_sublayer_at_pos(model, context, position=1, device=device)
        attn_reprs[i] = sub["attn"]
        mlp_reprs[i] = sub["mlp"]
    return {"attn": attn_reprs, "mlp": mlp_reprs}


def get_object_representations(
    model: MiniTransformer,
    cfg: Config,
    pair_indices: List[int],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    """Extract object representations from artificial context [x, b_j] at position 1."""
    model.eval()
    reprs = {}
    for i in pair_indices:
        context = [cfg.vocab.sep_id, cfg.vocab.object_id(i)]
        reprs[i] = _extract_repr_at_pos(model, context, position=1, device=device)
    return reprs


def get_post_lnf_object_representations(
    model: MiniTransformer,
    cfg: Config,
    pair_indices: List[int],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    """Object representations after applying ln_f, from [x, b_j] at position 1."""
    model.eval()
    ln_f = model.ln_f
    reprs = {}
    for i in pair_indices:
        context = [cfg.vocab.sep_id, cfg.vocab.object_id(i)]
        raw = _extract_repr_at_pos(model, context, position=1, device=device)
        with torch.no_grad():
            reprs[i] = [ln_f(r.unsqueeze(0).to(device)).squeeze(0).cpu() for r in raw]
    return reprs


def get_unembedding_object_representations(
    model: MiniTransformer,
    cfg: Config,
    pair_indices: List[int],
    n_layers: int,
) -> Dict[int, List[torch.Tensor]]:
    """Object representations from lm_head.weight rows."""
    weight = model.lm_head.weight.detach().cpu()
    reprs = {}
    for i in pair_indices:
        vec = weight[cfg.vocab.object_id(i)]
        reprs[i] = [vec] * (n_layers + 1)
    return reprs


def stack_subject_representations(
    reprs: Dict[Tuple[int, int], List[torch.Tensor]],
    subject_indices: List[int],
    relation_idx: int,
    layer: int,
) -> torch.Tensor:
    """Stack subject representations for a single relation at a given layer.

    Returns:
        (n_subjects, hidden_dim) tensor.
    """
    vecs = [reprs[(i, relation_idx)][layer] for i in subject_indices]
    return torch.stack(vecs, dim=0)


def stack_all_subject_representations(
    reprs: Dict[Tuple[int, int], List[torch.Tensor]],
    subject_indices: List[int],
    num_relations: int,
    layer: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Stack subject representations for all relations at a given layer.

    Returns:
        (n_subjects * num_relations, hidden_dim) tensor, ordered list of (subj, rel) keys.
    """
    keys = []
    vecs = []
    for i in subject_indices:
        for k in range(num_relations):
            keys.append((i, k))
            vecs.append(reprs[(i, k)][layer])
    return torch.stack(vecs, dim=0), keys


def build_pooled_object_matrix(obj_reprs, subject_indices, num_relations, vocab, layer):
    """Build object target matrix matching pooled subject ordering."""
    vecs = []
    for i in subject_indices:
        for k in range(num_relations):
            j = vocab.apply_relation(k, i)
            vecs.append(obj_reprs[j][layer])
    return torch.stack(vecs, dim=0)


def build_relation_object_matrix(obj_reprs, subject_indices, relation_idx, vocab, layer):
    """Build object target matrix for a single relation."""
    vecs = []
    for i in subject_indices:
        j = vocab.apply_relation(relation_idx, i)
        vecs.append(obj_reprs[j][layer])
    return torch.stack(vecs, dim=0)

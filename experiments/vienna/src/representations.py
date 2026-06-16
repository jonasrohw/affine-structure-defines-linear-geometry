"""
Representation extraction from a trained MiniTransformer.

For each subject a_i, the subject representation at layer L is the residual stream
at the a_i token position when running the context [x, a_i] through the model.

For each object b_i, the object representation at layer L is the residual stream
at the b_i token position when running the context [x, b_i] through the model.
(This is option B from the project plan — symmetric extraction, same positional context.)

residuals[0] = post-embedding (before any transformer block)
residuals[k] = after transformer block k-1  (k = 1 .. n_layers)
So for an n_layers model there are n_layers+1 residual tensors.
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
    """Run model on context, return residual stream at `position` for every layer.

    Returns:
        List of tensors, one per residual snapshot (length = n_layers + 1).
        Each tensor has shape (hidden_dim,).
    """
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    with model.capture_residuals() as residuals:
        model(input_ids)
    # residuals: list of (1, T, hidden_dim) tensors
    return [r[0, position, :].cpu() for r in residuals]


def get_all_subject_representations(
    model: MiniTransformer,
    cfg: Config,
    pair_indices: List[int],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    """Extract subject representations for the given pair indices.

    Context for a_i: [sep, a_i]  — a_i is at position 1.

    Returns:
        {i: [repr_layer0, repr_layer1, ...]} for each i in pair_indices.
    """
    model.eval()
    reprs = {}
    for i in pair_indices:
        context = [cfg.vocab.sep_id, cfg.vocab.subject_id(i)]
        reprs[i] = _extract_repr_at_pos(model, context, position=1, device=device)
    return reprs


def get_all_object_representations(
    model: MiniTransformer,
    cfg: Config,
    pair_indices: List[int],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    """Extract object representations for the given pair indices (option B).

    Context for b_i: [sep, b_i]  — b_i is at position 1.
    Symmetric with subject extraction.

    Returns:
        {i: [repr_layer0, repr_layer1, ...]} for each i in pair_indices.
    """
    model.eval()
    reprs = {}
    for i in pair_indices:
        context = [cfg.vocab.sep_id, cfg.vocab.object_id(i)]
        reprs[i] = _extract_repr_at_pos(model, context, position=1, device=device)
    return reprs


def get_correct_object_ids(cfg: Config, pair_indices: List[int]) -> List[int]:
    """Return vocab token ids of the correct object b_i for each pair index.

    Args:
        cfg: Config with vocab settings.
        pair_indices: List of pair indices (e.g. [0, 1, ..., 99]).

    Returns:
        List of vocab token ids, one per pair (cfg.vocab.object_id(i) = i + 101).
    """
    return [cfg.vocab.object_id(i) for i in pair_indices]


def get_post_lnf_object_representations(
    model: MiniTransformer,
    cfg: Config,
    pair_indices: List[int],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    """Object representations after applying ln_f to the residual stream.

    This is what actually enters the unembedding step: ln_f(h_L).
    Context for b_i: [sep, b_i] — b_i is at position 1.

    Returns:
        {i: [ln_f(repr_layer0), ln_f(repr_layer1), ...]} for each i in pair_indices.
    """
    model.eval()
    ln_f = model.ln_f
    reprs = {}
    for i in pair_indices:
        context = [cfg.vocab.sep_id, cfg.vocab.object_id(i)]
        raw = _extract_repr_at_pos(model, context, position=1, device=device)
        with torch.no_grad():
            reprs[i] = [ln_f(r.unsqueeze(0).to(device)).squeeze(0).cpu() for r in raw]
    return reprs


def get_static_object_embeddings(
    model: MiniTransformer,
    cfg: Config,
    pair_indices: List[int],
) -> Dict[int, torch.Tensor]:
    """Option A: static embedding vectors for b_i tokens (no context).

    Returns:
        {i: embedding_vector} for each i in pair_indices.
    """
    weight = model.token_emb.weight.detach().cpu()
    return {i: weight[cfg.vocab.object_id(i)] for i in pair_indices}


def get_unembedding_object_representations(
    model: MiniTransformer,
    cfg: Config,
    pair_indices: List[int],
    n_layers: int,
) -> Dict[int, List[torch.Tensor]]:
    """Object representations from the unembedding (lm_head) matrix.

    Returns the same dict format as get_all_object_representations, but each
    layer snapshot is the same static vector (the lm_head row for b_i), so
    that downstream code can use it interchangeably.

    Returns:
        {i: [repr] * (n_layers + 1)} for each i in pair_indices.
    """
    weight = model.lm_head.weight.detach().cpu()
    reprs = {}
    for i in pair_indices:
        vec = weight[cfg.vocab.object_id(i)]
        reprs[i] = [vec] * (n_layers + 1)
    return reprs


def get_relation_position_representations(
    model: MiniTransformer,
    cfg: Config,
    subject_indices: List[int],
    device: torch.device,
) -> Dict[Tuple[int, int], List[torch.Tensor]]:
    """Extract position-2 (r_k) representations for all (subject, relation) pairs.

    Context: [x, a_i, r_k, f0, f1, f2] — 6 tokens, position 2 is r_k.
    Due to causal masking, position 2 has seen {x, a_i, r_k}.

    Returns:
        {(subject_idx, relation_idx): [repr_layer0, ...]} for each pair.
    """
    model.eval()
    reprs = {}
    for i in subject_indices:
        for k in range(cfg.vocab.num_relations):
            context = [
                cfg.vocab.sep_id,
                cfg.vocab.subject_id(i),
                cfg.vocab.relation_id(k),
                *cfg.vocab.filler_ids,
            ]
            reprs[(i, k)] = _extract_repr_at_pos(model, context, position=2, device=device)
    return reprs


def get_filler_representations(
    model: MiniTransformer,
    cfg: Config,
    subject_indices: List[int],
    device: torch.device,
) -> Dict[Tuple[int, int], List[torch.Tensor]]:
    """Extract f2-position representations for all (subject, relation) pairs.

    Context: [x, a_i, r_k, f0, f1, f2] — 6 tokens, position 5 is f2.

    Returns:
        {(subject_idx, relation_idx): [repr_layer0, ...]} for each pair.
    """
    model.eval()
    reprs = {}
    for i in subject_indices:
        for k in range(cfg.vocab.num_relations):
            context = [
                cfg.vocab.sep_id,
                cfg.vocab.subject_id(i),
                cfg.vocab.relation_id(k),
                *cfg.vocab.filler_ids,
            ]
            reprs[(i, k)] = _extract_repr_at_pos(model, context, position=5, device=device)
    return reprs


def stack_filler_representations(
    reprs: Dict[Tuple[int, int], List[torch.Tensor]],
    subject_indices: List[int],
    relation_idx: int,
    layer: int,
) -> torch.Tensor:
    """Stack f2 representations for a single relation at a given layer.

    Returns:
        (n_subjects, hidden_dim) tensor.
    """
    vecs = [reprs[(i, relation_idx)][layer] for i in subject_indices]
    return torch.stack(vecs, dim=0)


def stack_all_filler_representations(
    reprs: Dict[Tuple[int, int], List[torch.Tensor]],
    subject_indices: List[int],
    num_relations: int,
    layer: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Stack f2 representations for all relations at a given layer.

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


def stack_representations(
    reprs: Dict[int, List[torch.Tensor]],
    pair_indices: List[int],
    layer: int,
) -> Tuple[torch.Tensor, List[int]]:
    """Stack per-pair representations at a given layer into a matrix.

    Args:
        reprs: Output of get_all_subject/object_representations.
        pair_indices: Ordered list of pair indices to include.
        layer: Which layer snapshot to use (0 = embedding, k = after block k-1).

    Returns:
        (matrix, ordered_indices)
        matrix shape: (len(pair_indices), hidden_dim)
    """
    vecs = [reprs[i][layer] for i in pair_indices]
    return torch.stack(vecs, dim=0), pair_indices

"""
Evaluation metrics for the Vienna experiment.

Primary metric: Delta cosine similarity (Hernandez et al. 2024).
  Δcos = cos(o, ô) - cos(o, s)
  where ô = s @ W.T + b (affine reconstruction of object from subject).

Secondary metrics:
  - Reconstruction accuracy: fraction of pairs where ô is nearest to o
  - MSE of affine approximation
  - Unembedding rank: rank of correct object token in logit-sorted vocab
"""
from typing import List, Optional

import torch
import torch.nn.functional as F


def delta_cos(
    S: torch.Tensor,
    O: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
) -> float:
    """Compute mean delta cosine similarity (Hernandez et al. 2024).

    Δcos = mean_i [ cos(o_i, ô_i) - cos(o_i, s_i) ]

    Args:
        S: (n, d) subject representations (row vectors).
        O: (n, d) object representations (row vectors).
        W: (d, d) weight matrix — reconstruction: ô = s @ W.T + b.
        b: (d,) bias or None.

    Returns:
        Scalar float Δcos, averaged over n pairs.
    """
    O_hat = S @ W.T
    if b is not None:
        O_hat = O_hat + b

    o_norm = F.normalize(O, dim=1)
    o_hat_norm = F.normalize(O_hat, dim=1)
    s_norm = F.normalize(S, dim=1)

    cos_o_ohat = (o_norm * o_hat_norm).sum(dim=1)   # (n,)
    cos_o_s = (o_norm * s_norm).sum(dim=1)           # (n,)

    return (cos_o_ohat - cos_o_s).mean().item()


def reconstruction_accuracy(
    S: torch.Tensor,
    O: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    O_all: Optional[torch.Tensor] = None,
) -> float:
    """Fraction of pairs where affine reconstruction ô is nearest neighbor to o.

    If O_all is provided, nearest neighbor search is over all objects (not just
    the n in S/O). Otherwise, search is over O itself.

    Args:
        S: (n, d) subject representations.
        O: (n, d) ground-truth object representations for this subset.
        W: (d, d) weight matrix.
        b: (d,) bias or None.
        O_all: (m, d) full set of object representations for NN search.

    Returns:
        Accuracy in [0, 1].
    """
    O_hat = S @ W.T
    if b is not None:
        O_hat = O_hat + b

    gallery = O_all if O_all is not None else O

    # Cosine similarity between each ô and each gallery vector
    O_hat_norm = F.normalize(O_hat, dim=1)      # (n, d)
    gallery_norm = F.normalize(gallery, dim=1)  # (m, d)
    sims = O_hat_norm @ gallery_norm.T          # (n, m)

    nn_idx = sims.argmax(dim=1)  # (n,)

    # Ground-truth indices within gallery
    if O_all is None:
        gt_idx = torch.arange(len(O))
    else:
        # Find nearest row in O_all for each o via L2 distance (robust to float precision)
        dists = torch.cdist(O.float(), O_all.float())
        gt_idx = dists.argmin(dim=1)

    return (nn_idx == gt_idx).float().mean().item()


def mse(
    S: torch.Tensor,
    O: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
) -> float:
    """Mean squared error of affine reconstruction."""
    O_hat = S @ W.T
    if b is not None:
        O_hat = O_hat + b
    return F.mse_loss(O_hat, O).item()


def cosine_similarity_mean(
    S: torch.Tensor,
    O: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
) -> float:
    """Mean cosine similarity between ô and o."""
    O_hat = S @ W.T
    if b is not None:
        O_hat = O_hat + b
    o_norm = F.normalize(O, dim=1)
    o_hat_norm = F.normalize(O_hat, dim=1)
    return (o_norm * o_hat_norm).sum(dim=1).mean().item()


def full_report(
    S: torch.Tensor,
    O: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    O_all: Optional[torch.Tensor] = None,
    label: str = "",
) -> dict:
    """Compute all metrics and return as dict."""
    metrics = {
        "delta_cos": delta_cos(S, O, W, b),
        "cos_mean": cosine_similarity_mean(S, O, W, b),
        "mse": mse(S, O, W, b),
        "recon_acc": reconstruction_accuracy(S, O, W, b, O_all),
    }
    if label:
        metrics = {f"{label}/{k}": v for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def decode_ohat_to_logits(O_hat: torch.Tensor, model) -> torch.Tensor:
    """Apply model's unembedding (ln_f + lm_head) to a batch of ô vectors.

    residuals are captured *before* ln_f, so the correct decoding path is:
        model.ln_f(ô) → model.lm_head(...) → logits over vocab.

    Args:
        O_hat: (n, hidden_dim) pre-ln_f representations.
        model: MiniTransformer with .ln_f and .lm_head attributes.

    Returns:
        logits: (n, vocab_size)
    """
    device = next(model.parameters()).device
    O_hat = O_hat.to(device).float()
    normed = model.ln_f(O_hat)
    return model.lm_head(normed)


@torch.no_grad()
def unembedding_rank_accuracy(
    S: torch.Tensor,
    correct_object_ids: List[int],
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    model,
) -> dict:
    """Decode ô via the model's unembedding and measure rank of the correct token.

    For each pair: computes ô = S @ W.T + b, applies ln_f + lm_head,
    finds the rank of the correct object token in the logit-sorted vocabulary.
    Rank 1 means ô decoded to the correct object token (top-1 accuracy).

    Args:
        S: (n, d) subject representations (pre-ln_f).
        correct_object_ids: Vocab token ids for the correct b_i, length n.
        W: (d, d) affine weight matrix.
        b: (d,) bias or None.
        model: MiniTransformer.

    Returns:
        Dict with keys: mean_rank, median_rank, topk_acc/1, topk_acc/5, topk_acc/10
    """
    O_hat = S @ W.T
    if b is not None:
        O_hat = O_hat + b

    logits = decode_ohat_to_logits(O_hat, model).cpu()  # (n, vocab_size)
    sorted_indices = logits.argsort(dim=1, descending=True)  # (n, vocab_size)

    ranks = []
    for i, cid in enumerate(correct_object_ids):
        rank = (sorted_indices[i] == cid).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
        ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float32)
    return {
        "mean_rank": ranks.mean().item(),
        "median_rank": ranks.median().item(),
        "topk_acc/1": (ranks <= 1).float().mean().item(),
        "topk_acc/5": (ranks <= 5).float().mean().item(),
        "topk_acc/10": (ranks <= 10).float().mean().item(),
    }


@torch.no_grad()
def subject_direct_unembedding_rank(
    S: torch.Tensor,
    correct_object_ids: List[int],
    model,
) -> dict:
    """Decode subject representations directly (no affine) and measure rank.

    Baseline: does s already encode the correct object without the affine map?
    Distinguishes whether affine adds value (outcome 1) vs. subject already
    encodes the object (outcome 2).

    Args:
        S: (n, d) subject representations (pre-ln_f).
        correct_object_ids: Vocab token ids for the correct b_i, length n.
        model: MiniTransformer.

    Returns:
        Dict with keys: mean_rank, median_rank, topk_acc/1, topk_acc/5, topk_acc/10
    """
    logits = decode_ohat_to_logits(S, model).cpu()  # (n, vocab_size)
    sorted_indices = logits.argsort(dim=1, descending=True)

    ranks = []
    for i, cid in enumerate(correct_object_ids):
        rank = (sorted_indices[i] == cid).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float32)
    return {
        "mean_rank": ranks.mean().item(),
        "median_rank": ranks.median().item(),
        "topk_acc/1": (ranks <= 1).float().mean().item(),
        "topk_acc/5": (ranks <= 5).float().mean().item(),
        "topk_acc/10": (ranks <= 10).float().mean().item(),
    }

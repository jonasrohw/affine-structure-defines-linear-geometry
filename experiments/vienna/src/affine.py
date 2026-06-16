"""
Affine transformation fitting between subject and object representations.

Convention: representations are row vectors of shape (hidden_dim,).
The affine map is:  ô = s @ W.T + b
which is equivalent to  ô^T = W @ s^T + b^T  (column-vector notation).

W has shape (hidden_dim, hidden_dim).
b has shape (hidden_dim,).

Fitting uses least squares: given S (n x d) and O (n x d), find W, b minimizing
||S @ W.T + 1 @ b - O||_F^2, i.e. ||S_aug @ [W; b]^T - O||_F^2
where S_aug = [S | 1] has shape (n x (d+1)).
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def fit_affine(
    S: torch.Tensor,
    O: torch.Tensor,
    bias: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fit affine (or linear) map W, b such that S @ W.T + b ≈ O via least squares.

    Args:
        S: Subject representations, shape (n, d).
        O: Object representations, shape (n, d).
        bias: If True, fit W and b (affine). If False, fit W only (linear, b=None).

    Returns:
        (W, b): W shape (d, d), b shape (d,) or None.
    """
    S_np = S.numpy().astype(np.float64)
    O_np = O.numpy().astype(np.float64)

    if bias:
        ones = np.ones((S_np.shape[0], 1), dtype=np.float64)
        S_aug = np.concatenate([S_np, ones], axis=1)  # (n, d+1)
    else:
        S_aug = S_np  # (n, d)

    # Solve: S_aug @ X ≈ O  =>  X = pinv(S_aug) @ O
    # X has shape (d+1, d) if bias else (d, d)
    X, _, _, _ = np.linalg.lstsq(S_aug, O_np, rcond=None)

    if bias:
        W = torch.tensor(X[:-1].T, dtype=torch.float32)  # (d, d)
        b = torch.tensor(X[-1], dtype=torch.float32)      # (d,)
        return W, b
    else:
        W = torch.tensor(X.T, dtype=torch.float32)        # (d, d)
        return W, None


def apply_affine(
    S: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply affine map: ô = s @ W.T + b.

    Args:
        S: (n, d) subject representations.
        W: (d, d) weight matrix.
        b: (d,) bias or None.

    Returns:
        (n, d) reconstructed object representations.
    """
    # W.T because W is stored such that ô = s @ W.T + b
    out = S @ W.T
    if b is not None:
        out = out + b
    return out


def kfold_affine(
    S: torch.Tensor,
    O: torch.Tensor,
    k: int = 5,
    bias: bool = True,
) -> Tuple[float, float, torch.Tensor, Optional[torch.Tensor]]:
    """K-fold cross-validation of affine fit across pairs.

    Fits W, b on k-1 folds, evaluates delta_cos on the held-out fold.
    Also returns W, b fit on the full set (for downstream use).

    Args:
        S: (n, d) subject representations.
        O: (n, d) object representations.
        k: Number of folds.
        bias: Whether to include bias term.

    Returns:
        (mean_delta_cos, std_delta_cos, W_full, b_full)
        W_full and b_full are fit on all n pairs.
    """
    from .evaluation import delta_cos  # avoid circular import

    n = S.shape[0]
    fold_size = n // k
    delta_cos_scores = []

    indices = np.arange(n)

    for fold in range(k):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

        S_train, O_train = S[train_idx], O[train_idx]
        S_val, O_val = S[val_idx], O[val_idx]

        W, b = fit_affine(S_train, O_train, bias=bias)
        dc = delta_cos(S_val, O_val, W, b)
        delta_cos_scores.append(dc)

    W_full, b_full = fit_affine(S, O, bias=bias)
    return (
        float(np.mean(delta_cos_scores)),
        float(np.std(delta_cos_scores)),
        W_full,
        b_full,
    )


def second_order_test(
    W_list: List[torch.Tensor],
    labels: List[str],
    rank_threshold: float = 1e-3,
) -> Dict:
    """Test whether the space of per-relation weight matrices is linearly structured.

    For each pair (W_i, W_j) computes:
      - Frobenius distance ||W_i - W_j||_F
      - Numerical rank of (W_i - W_j) at the given threshold

    Also tests the pure-translation hypothesis: whether a single shared W with
    relation-specific bias b_k suffices (i.e. all W_k are identical).

    Args:
        W_list: List of (d, d) weight matrices, one per relation.
        labels: Human-readable names for each relation (e.g. ["R0","R1","R2"]).
        rank_threshold: Singular values below this fraction of the largest are
                        treated as zero when computing numerical rank.

    Returns:
        Dict with keys:
          "frobenius": {(li, lj): float} pairwise Frobenius distances
          "diff_rank": {(li, lj): int} numerical rank of W_i - W_j
          "pure_translation": bool — True if all W_k are within rank_threshold of W_0
    """
    results: Dict = {"frobenius": {}, "diff_rank": {}, "pure_translation": False}
    n = len(W_list)

    for i in range(n):
        for j in range(i + 1, n):
            diff = W_list[i] - W_list[j]
            frob = diff.norm(p="fro").item()
            key = (labels[i], labels[j])
            results["frobenius"][key] = frob

            diff_np = diff.numpy().astype(np.float64)
            singular_values = np.linalg.svd(diff_np, compute_uv=False)
            threshold = rank_threshold * singular_values[0] if singular_values[0] > 0 else 0.0
            rank = int((singular_values > threshold).sum())
            results["diff_rank"][key] = rank

    # Pure-translation hypothesis: all W_k ≈ W_0
    # Test by checking if max pairwise Frobenius distance is small relative to ||W_0||_F
    w0_norm = W_list[0].norm(p="fro").item()
    max_frob = max(results["frobenius"].values()) if results["frobenius"] else 0.0
    results["pure_translation"] = (max_frob / max(w0_norm, 1e-8)) < 0.05

    return results


def cross_relation_eval(
    S: torch.Tensor,
    O_by_relation: List[torch.Tensor],
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    train_label: str,
    target_labels: List[str],
) -> Dict:
    """Apply a single affine map (trained on one relation) to other relations.

    Args:
        S: (n, d) subject representations (same for all relations).
        O_by_relation: List of (n, d) object matrices, one per relation.
                       Index matches target_labels.
        W: (d, d) weight matrix trained on the source relation.
        b: (d,) bias or None.
        train_label: Name of the relation W was trained on.
        target_labels: Names for each entry in O_by_relation.

    Returns:
        Dict mapping target_label -> delta_cos float.
    """
    from .evaluation import delta_cos  # avoid circular import

    results = {}
    for label, O_target in zip(target_labels, O_by_relation):
        results[label] = delta_cos(S, O_target, W, b)
    return results

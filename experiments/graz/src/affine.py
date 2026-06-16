"""
Affine transformation analysis for Graz Phase 2.

Key question: does an affine map fitted on train subject hidden states
generalize to test subjects?

Convention: representations are row vectors of shape (hidden_dim,).
The affine map is:  ô = s @ W.T + b

For each relation k:
  1. Extract hidden state h at subject position for context [x, r_k, a_i]
  2. Target: lm_head weight row for the correct object b_j
  3. Fit W_k, b_k on train subjects via least squares
  4. Evaluate on train and test subjects separately
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, VocabConfig
from .model import MiniTransformer


def fit_affine(
    S: torch.Tensor,
    O: torch.Tensor,
    bias: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    S_np = S.numpy().astype(np.float64)
    O_np = O.numpy().astype(np.float64)

    if bias:
        ones = np.ones((S_np.shape[0], 1), dtype=np.float64)
        S_aug = np.concatenate([S_np, ones], axis=1)
    else:
        S_aug = S_np

    X, _, _, _ = np.linalg.lstsq(S_aug, O_np, rcond=None)

    if bias:
        W = torch.tensor(X[:-1].T, dtype=torch.float32)
        b = torch.tensor(X[-1], dtype=torch.float32)
        return W, b
    else:
        W = torch.tensor(X.T, dtype=torch.float32)
        return W, None


def apply_affine(
    S: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
) -> torch.Tensor:
    out = S @ W.T
    if b is not None:
        out = out + b
    return out


@torch.no_grad()
def extract_subject_hidden(
    model: MiniTransformer,
    cfg: Config,
    subject_indices: List[int],
    device: torch.device,
    layer: int = -1,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Extract hidden state at subject position for [x, r_k, a_i].

    Batches all contexts into a single forward pass for efficiency.

    Args:
        layer: -1 for final (pre-ln_f) residual, 0 for post-embedding.
    """
    model.eval()
    keys = []
    contexts = []
    for i in subject_indices:
        for k in range(cfg.vocab.num_relations):
            keys.append((i, k))
            contexts.append([cfg.vocab.sep_id, cfg.vocab.relation_id(k), cfg.vocab.subject_id(i)])

    batch = torch.tensor(contexts, dtype=torch.long, device=device)
    with model.capture_residuals() as residuals:
        model(batch)
    # residuals[layer] has shape (B, 3, hidden_dim); we want position 2
    hidden = residuals[layer][:, 2, :].cpu()

    reprs = {}
    for idx, key in enumerate(keys):
        reprs[key] = hidden[idx]
    return reprs


def get_object_targets(
    model: MiniTransformer,
    cfg: Config,
    subject_indices: List[int],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Get lm_head weight rows for the correct object of each (subject, relation) pair."""
    weight = model.lm_head.weight.detach().cpu()
    targets = {}
    for i in subject_indices:
        for k in range(cfg.vocab.num_relations):
            obj_idx = cfg.vocab.relation_object(i, k)
            obj_token = cfg.vocab.object_id(obj_idx)
            targets[(i, k)] = weight[obj_token]
    return targets


def stack_pairs(
    h_dict: Dict[Tuple[int, int], torch.Tensor],
    t_dict: Dict[Tuple[int, int], torch.Tensor],
    subject_indices: List[int],
    relation_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    S = torch.stack([h_dict[(i, relation_k)] for i in subject_indices])
    O = torch.stack([t_dict[(i, relation_k)] for i in subject_indices])
    return S, O


def reconstruction_accuracy(
    S: torch.Tensor,
    O: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    O_all: torch.Tensor,
) -> float:
    """Fraction of pairs where affine reconstruction ô is nearest to correct o in O_all."""
    O_hat = apply_affine(S, W, b)
    O_hat_norm = F.normalize(O_hat, dim=1)
    O_all_norm = F.normalize(O_all, dim=1)
    sims = O_hat_norm @ O_all_norm.T
    nn_idx = sims.argmax(dim=1)

    dists = torch.cdist(O.float(), O_all.float())
    gt_idx = dists.argmin(dim=1)
    return (nn_idx == gt_idx).float().mean().item()


def _analyze_one_layer(
    h_all: Dict[Tuple[int, int], torch.Tensor],
    t_all: Dict[Tuple[int, int], torch.Tensor],
    O_all_objects: torch.Tensor,
    vocab,
) -> Dict:
    train_subjs = vocab.train_subject_indices
    test_subjs = vocab.test_subject_indices

    results = {}
    for k in range(vocab.num_relations):
        S_train, O_train = stack_pairs(h_all, t_all, train_subjs, k)
        S_test, O_test = stack_pairs(h_all, t_all, test_subjs, k)

        W, b = fit_affine(S_train, O_train, bias=True)

        train_acc = reconstruction_accuracy(S_train, O_train, W, b, O_all_objects)
        test_acc = reconstruction_accuracy(S_test, O_test, W, b, O_all_objects)

        O_hat_train = apply_affine(S_train, W, b)
        O_hat_test = apply_affine(S_test, W, b)
        train_mse = F.mse_loss(O_hat_train, O_train).item()
        test_mse = F.mse_loss(O_hat_test, O_test).item()

        train_cos = (F.normalize(O_hat_train, dim=1) * F.normalize(O_train, dim=1)).sum(1).mean().item()
        test_cos = (F.normalize(O_hat_test, dim=1) * F.normalize(O_test, dim=1)).sum(1).mean().item()

        results[f"r{k}"] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_cos": train_cos,
            "test_cos": test_cos,
        }

    all_train_acc = np.mean([results[f"r{k}"]["train_acc"] for k in range(vocab.num_relations)])
    all_test_acc = np.mean([results[f"r{k}"]["test_acc"] for k in range(vocab.num_relations)])
    results["overall"] = {
        "train_acc": float(all_train_acc),
        "test_acc": float(all_test_acc),
    }
    return results


def affine_analysis(
    model: MiniTransformer,
    cfg: Config,
    device: torch.device,
    layer: int = -1,
) -> Dict:
    """Full affine analysis: fit on train, evaluate on train + test."""
    vocab = cfg.vocab
    all_subjects = list(range(vocab.n_entities))
    h_all = extract_subject_hidden(model, cfg, all_subjects, device, layer=layer)
    t_all = get_object_targets(model, cfg, all_subjects)
    weight = model.lm_head.weight.detach().cpu()
    O_all_objects = weight[vocab.object_offset:vocab.object_offset + vocab.n_entities]
    return _analyze_one_layer(h_all, t_all, O_all_objects, vocab)


def get_centroid_targets(
    model: MiniTransformer,
    cfg: Config,
    subject_indices: List[int],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Mean lm_head weight of all valid objects for each (subject, relation) pair."""
    weight = model.lm_head.weight.detach().cpu()
    vocab = cfg.vocab
    targets = {}
    cache = {}
    for i in subject_indices:
        for k in range(vocab.num_relations):
            t = vocab.subject_f_type(i) if k == 0 else vocab.subject_g_type(i)
            key = (k, t)
            if key not in cache:
                obj_indices = vocab.relation_objects(i, k)
                obj_tokens = [vocab.object_id(j) for j in obj_indices]
                cache[key] = weight[obj_tokens].mean(dim=0)
            targets[(i, k)] = cache[key]
    return targets


def build_type_centroids(
    model: MiniTransformer,
    vocab: VocabConfig,
) -> Dict[int, torch.Tensor]:
    """Unique type centroids per relation: {rel_k: tensor(n_types, dim)}."""
    weight = model.lm_head.weight.detach().cpu()
    centroids = {}
    for k in range(vocab.num_relations):
        seen = {}
        for i in range(vocab.n_entities):
            t = vocab.subject_f_type(i) if k == 0 else vocab.subject_g_type(i)
            if t not in seen:
                obj_indices = vocab.relation_objects(i, k)
                obj_tokens = [vocab.object_id(j) for j in obj_indices]
                seen[t] = weight[obj_tokens].mean(dim=0)
        centroids[k] = torch.stack(list(seen.values()))
    return centroids


def affine_analysis_centroid(
    model: MiniTransformer,
    cfg: Config,
    device: torch.device,
    layer: int = -1,
) -> Dict:
    """Affine analysis using type centroids as targets instead of individual objects."""
    vocab = cfg.vocab
    all_subjects = list(range(vocab.n_entities))
    h_all = extract_subject_hidden(model, cfg, all_subjects, device, layer=layer)
    t_all = get_centroid_targets(model, cfg, all_subjects)
    type_centroids = build_type_centroids(model, vocab)

    train_subjs = vocab.train_subject_indices
    test_subjs = vocab.test_subject_indices

    results = {}
    for k in range(vocab.num_relations):
        S_train, O_train = stack_pairs(h_all, t_all, train_subjs, k)
        S_test, O_test = stack_pairs(h_all, t_all, test_subjs, k)

        W, b = fit_affine(S_train, O_train, bias=True)

        O_all = type_centroids[k]
        train_acc = reconstruction_accuracy(S_train, O_train, W, b, O_all)
        test_acc = reconstruction_accuracy(S_test, O_test, W, b, O_all)

        results[f"r{k}"] = {"train_acc": train_acc, "test_acc": test_acc}

    all_train = np.mean([results[f"r{k}"]["train_acc"] for k in range(vocab.num_relations)])
    all_test = np.mean([results[f"r{k}"]["test_acc"] for k in range(vocab.num_relations)])
    results["overall"] = {"train_acc": float(all_train), "test_acc": float(all_test)}
    return results


def affine_analysis_all_layers(
    model: MiniTransformer,
    cfg: Config,
    device: torch.device,
) -> Dict[str, Dict]:
    """Run affine analysis for all layers in a single forward pass.

    Returns {"layer0": {...}, "layer1": {...}, ...}
    """
    vocab = cfg.vocab
    all_subjects = list(range(vocab.n_entities))

    # Build batch
    keys = []
    contexts = []
    for i in all_subjects:
        for k in range(vocab.num_relations):
            keys.append((i, k))
            contexts.append([vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)])

    model.eval()
    batch = torch.tensor(contexts, dtype=torch.long, device=device)
    with torch.no_grad():
        with model.capture_residuals() as residuals:
            model(batch)

    t_all = get_object_targets(model, cfg, all_subjects)
    weight = model.lm_head.weight.detach().cpu()
    O_all_objects = weight[vocab.object_offset:vocab.object_offset + vocab.n_entities]

    results = {}
    for layer_idx, layer_residual in enumerate(residuals):
        hidden = layer_residual[:, 2, :].cpu()
        h_all = {key: hidden[idx] for idx, key in enumerate(keys)}
        label = f"layer{layer_idx}"
        results[label] = _analyze_one_layer(h_all, t_all, O_all_objects, vocab)

    return results

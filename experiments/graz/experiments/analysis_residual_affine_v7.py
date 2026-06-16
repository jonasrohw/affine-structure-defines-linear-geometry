"""
Residual stream affine analysis for v7 checkpoint.

At each point in the residual stream, fit an affine map from subject
representations to mean-object (centroid) representations, separately
for each relation.

Residual stream points (1-layer transformer):
  0. h0: token_emb + pos_emb
  1. h0 + attn_out: post-attention residual
  2. h0 + attn_out + mlp_out: post-MLP residual (pre-ln_f)
  3. ln_f(h0 + attn_out + mlp_out): post-final-LayerNorm

Target: mean of lm_head weight rows for all valid objects (type centroid).

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/analysis_residual_affine_v7.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config, ModelConfig, TrainConfig, VocabConfig
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


def build_type_centroids(model, vocab, rel_objects_table):
    """Compute centroid of lm_head weights for all valid objects per (subject, relation)."""
    weight = model.lm_head.weight.detach().cpu()
    cache = {}
    targets = {}
    for i in range(vocab.n_entities):
        for k in range(vocab.num_relations):
            t = vocab.subject_f_type(i) if k == 0 else vocab.subject_g_type(i)
            key = (k, t)
            if key not in cache:
                obj_indices = rel_objects_table[(i, k)]
                obj_tokens = [vocab.object_id(j) for j in obj_indices]
                cache[key] = weight[obj_tokens].mean(dim=0)
            targets[(i, k)] = cache[key]
    return targets, cache


def get_unique_centroids(cache, vocab):
    """Get unique centroids per relation as tensors for Voronoi evaluation."""
    centroids = {}
    for k in range(vocab.num_relations):
        vecs = []
        for key, vec in cache.items():
            if key[0] == k:
                vecs.append(vec)
        centroids[k] = torch.stack(vecs)
    return centroids


def fit_affine(S, O):
    """Fit affine map: O_hat = S @ W.T + b via least squares."""
    S_np = S.numpy().astype(np.float64)
    O_np = O.numpy().astype(np.float64)
    ones = np.ones((S_np.shape[0], 1), dtype=np.float64)
    S_aug = np.concatenate([S_np, ones], axis=1)
    X, _, _, _ = np.linalg.lstsq(S_aug, O_np, rcond=None)
    W = torch.tensor(X[:-1].T, dtype=torch.float32)
    b = torch.tensor(X[-1], dtype=torch.float32)
    return W, b


def voronoi_accuracy(S, O_true, W, b, all_centroids):
    """Fraction where affine prediction lands in correct Voronoi cell."""
    O_hat = S @ W.T + b
    O_hat_norm = F.normalize(O_hat, dim=1)
    all_norm = F.normalize(all_centroids, dim=1)
    sims_pred = O_hat_norm @ all_norm.T
    pred_idx = sims_pred.argmax(dim=1)

    O_true_norm = F.normalize(O_true, dim=1)
    sims_true = O_true_norm @ all_norm.T
    true_idx = sims_true.argmax(dim=1)

    return (pred_idx == true_idx).float().mean().item()


def cosine_similarity(O_hat, O_true):
    """Mean cosine similarity between predicted and true."""
    return (F.normalize(O_hat, dim=1) * F.normalize(O_true, dim=1)).sum(1).mean().item()


@torch.no_grad()
def extract_all_residual_points(model, vocab, device):
    """Extract hidden states at all residual stream points for relation contexts.

    Returns dict: point_name -> {(subj_idx, rel_k): tensor(dim,)}
    """
    model.eval()
    contexts = []
    keys = []
    for i in range(vocab.n_entities):
        for k in range(vocab.num_relations):
            keys.append((i, k))
            contexts.append([vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)])

    batch = torch.tensor(contexts, dtype=torch.long, device=device)

    # Capture both residuals and sublayers
    with model.capture_residuals() as residuals:
        with model.capture_sublayers() as sublayers:
            model(batch)

    # residuals[0] = h0 (post-embedding), shape (B, 3, dim)
    # residuals[1] = h1 (post-block), shape (B, 3, dim)
    # sublayers["attn"][0] = attn_out, shape (B, 3, dim)
    # sublayers["mlp"][0] = mlp_out, shape (B, 3, dim)

    h0 = residuals[0][:, 2, :].cpu()              # post-embedding
    attn_out = sublayers["attn"][0][:, 2, :].cpu()  # attention contribution
    mlp_out = sublayers["mlp"][0][:, 2, :].cpu()    # MLP contribution

    h0_plus_attn = h0 + attn_out                   # post-attention residual
    h1 = residuals[1][:, 2, :].cpu()               # post-block (= h0 + attn + mlp)

    # Apply final layer norm
    ln_f = model.ln_f
    h1_device = residuals[1][:, 2, :]  # still on device
    h1_ln = ln_f(h1_device).cpu()

    points = {
        "h0 (embedding)": h0,
        "h0+attn (post-attention)": h0_plus_attn,
        "h0+attn+mlp (post-MLP)": h1,
        "ln_f(h1) (final)": h1_ln,
    }

    result = {}
    for name, tensor in points.items():
        d = {}
        for idx, key in enumerate(keys):
            d[key] = tensor[idx]
        result[name] = d

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Rebuild config matching v7
    vocab = VocabConfig(
        n_f_attrs=10, n_g_attrs=10, k_per_group=2,
        num_relations=2, n_entities=2025,
        n_train_subjects=2025, r0_shift=7, r1_shift=13,
    )
    model_cfg = ModelConfig(n_layers=1, hidden_dim=64, n_heads=4)

    print("Precomputing relation objects...")
    rel_objects_table = precompute_relation_objects(vocab)
    vocab.relation_objects = lambda subj_idx, rel_k: rel_objects_table[(subj_idx, rel_k)]

    # Build and load model
    model = build_model(model_cfg, vocab)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {CHECKPOINT}")

    # Compute targets (type centroids from lm_head weights)
    centroid_targets, centroid_cache = build_type_centroids(model, vocab, rel_objects_table)
    unique_centroids = get_unique_centroids(centroid_cache, vocab)

    # Extract hidden states at all residual stream points
    print("Extracting residual stream representations...")
    all_points = extract_all_residual_points(model, vocab, device)

    # Use a train/test split for evaluation (first half train, second half test)
    n = vocab.n_entities
    train_subjs = list(range(n // 2))
    test_subjs = list(range(n // 2, n))

    print(f"\nFitting affine maps: {len(train_subjs)} train subjects, "
          f"{len(test_subjs)} test subjects")
    print(f"Unique centroids per relation: {unique_centroids[0].shape[0]}")

    # Analyze each residual stream point
    print(f"\n{'='*80}")
    print(f"RESIDUAL STREAM AFFINE ANALYSIS (v7 checkpoint, seed=42)")
    print(f"{'='*80}")
    print(f"Target: type centroid (mean lm_head weight of 45 valid objects)")
    print(f"Metric: Voronoi accuracy (nearest centroid match)")
    print(f"{'='*80}\n")

    header = f"{'Point':<30} | {'Rel':>3} | {'Train Acc':>9} | {'Test Acc':>8} | {'Train cos':>9} | {'Test cos':>8}"
    print(header)
    print("-" * len(header))

    summary = []

    for point_name, h_dict in all_points.items():
        for k in range(vocab.num_relations):
            # Stack subject representations and targets
            S_train = torch.stack([h_dict[(i, k)] for i in train_subjs])
            O_train = torch.stack([centroid_targets[(i, k)] for i in train_subjs])
            S_test = torch.stack([h_dict[(i, k)] for i in test_subjs])
            O_test = torch.stack([centroid_targets[(i, k)] for i in test_subjs])

            # Fit affine on train
            W, b = fit_affine(S_train, O_train)

            # Evaluate
            all_c = unique_centroids[k]
            train_acc = voronoi_accuracy(S_train, O_train, W, b, all_c)
            test_acc = voronoi_accuracy(S_test, O_test, W, b, all_c)

            # Cosine similarity
            O_hat_train = S_train @ W.T + b
            O_hat_test = S_test @ W.T + b
            train_cos = cosine_similarity(O_hat_train, O_train)
            test_cos = cosine_similarity(O_hat_test, O_test)

            print(f"{point_name:<30} | R{k}  | {train_acc:>9.3f} | {test_acc:>8.3f} | "
                  f"{train_cos:>9.4f} | {test_cos:>8.4f}")

            summary.append({
                "point": point_name,
                "relation": k,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_cos": train_cos,
                "test_cos": test_cos,
            })

        print("-" * len(header))

    # Print condensed summary
    print(f"\n{'='*60}")
    print("CONDENSED SUMMARY (test accuracy per residual stream point)")
    print(f"{'='*60}")
    print(f"{'Point':<30} | {'R0 test':>7} | {'R1 test':>7} | {'Mean':>7}")
    print("-" * 60)
    for point_name in all_points.keys():
        r0 = [s for s in summary if s["point"] == point_name and s["relation"] == 0][0]
        r1 = [s for s in summary if s["point"] == point_name and s["relation"] == 1][0]
        mean_acc = (r0["test_acc"] + r1["test_acc"]) / 2
        print(f"{point_name:<30} | {r0['test_acc']:>7.3f} | {r1['test_acc']:>7.3f} | {mean_acc:>7.3f}")


if __name__ == "__main__":
    main()

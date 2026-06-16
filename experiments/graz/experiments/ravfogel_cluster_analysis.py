"""
Cluster analysis for the dim=64 nonlinear demixing mechanism.

Question: does the model decompose the 2025-entity mapping into clusters
where each cluster uses a more structured (closer to affine) sub-mapping?

Approach:
  1. Train a dim=64 model to convergence (30 epochs)
  2. Extract MLP inputs and outputs for all entities under each relation
  3. Compute global affine R² (baseline)
  4. Cluster entities by MLP input vectors (k-means, sweep k)
  5. Fit separate affine map per cluster, compute per-cluster R²
  6. Also try clustering by post-GELU activation patterns
  7. Compare: does intra-cluster R² >> global R²?

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_cluster_analysis.py --seed 42
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from src.config import ModelConfig
from src.model import build_model
from src.training import get_lr

from experiments.ravfogel_1layer import (
    RavfogelVocab, generate_permutations, build_sequences,
    eval_relation_accuracy, train_epoch, SEQ_LENGTH, REL_SENTENCE_LEN,
)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

ACT_FNS = {
    "gelu": F.gelu,
    "relu": F.relu,
}


def patch_activation(model, act_name):
    """Replace the MLP activation function in all blocks."""
    act_fn = ACT_FNS[act_name]
    for block in model.blocks:
        if hasattr(block, 'mlp'):
            original_forward = block.mlp.forward.__func__ if hasattr(block.mlp.forward, '__func__') else None
            fc1 = block.mlp.fc1
            fc2 = block.mlp.fc2
            def make_forward(fc1, fc2, fn):
                def forward(self, x):
                    return fc2(fn(fc1(x)))
                return forward
            import types
            block.mlp.forward = types.MethodType(make_forward(fc1, fc2, act_fn), block.mlp)


@torch.no_grad()
def extract_all_states(model, vocab, device, act_fn=F.gelu, batch_size=512):
    """Extract MLP inputs, outputs, pre/post-activation at subject position for all (entity, relation) pairs."""
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]

    block = model.blocks[0]
    fc1 = block.mlp.fc1
    fc2 = block.mlp.fc2
    ln1 = block.ln1
    ln2 = block.ln2

    mlp_inputs = {}
    mlp_outputs = {}
    pre_gelu = {}
    post_gelu = {}

    for start in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[start:start + batch_size]
        contexts = torch.tensor(
            [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)]
             for i, k in batch_pairs],
            dtype=torch.long, device=device)

        B, T = contexts.shape
        positions = torch.arange(T, device=device).unsqueeze(0)
        h0 = model.token_emb(contexts) + model.pos_emb(positions)

        attn_out = block.attn(ln1(h0))
        h1 = h0 + attn_out

        mlp_in = ln2(h1)
        pre_g = fc1(mlp_in)
        post_g = act_fn(pre_g)
        mlp_out = fc2(post_g)

        for idx, (i, k) in enumerate(batch_pairs):
            mlp_inputs[(i, k)] = mlp_in[idx, 2, :].cpu()
            mlp_outputs[(i, k)] = mlp_out[idx, 2, :].cpu()
            pre_gelu[(i, k)] = pre_g[idx, 2, :].cpu()
            post_gelu[(i, k)] = post_g[idx, 2, :].cpu()

    return mlp_inputs, mlp_outputs, pre_gelu, post_gelu


def fit_affine_r2(X, Y):
    """Fit Y = X @ W + b via least squares, return R²."""
    n = X.shape[0]
    if n < X.shape[1] + 2:
        return float('nan'), 0
    X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)
    sol, _, _, _ = torch.linalg.lstsq(X_aug, Y)
    Y_pred = X_aug @ sol
    ss_res = ((Y - Y_pred) ** 2).sum().item()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    return r2, n


def cluster_analysis(X, Y, k_values, clustering_basis, basis_name, rng_seed=42):
    """
    Cluster entities and fit per-cluster affine maps.

    Args:
        X: (n_entities, dim) MLP inputs
        Y: (n_entities, dim) MLP outputs
        k_values: list of cluster counts to try
        clustering_basis: (n_entities, feat_dim) vectors to cluster on
        basis_name: label for this clustering basis
        rng_seed: random seed for k-means

    Returns:
        list of dicts with results per k
    """
    n = X.shape[0]
    global_r2, _ = fit_affine_r2(X, Y)

    results = []
    for k in k_values:
        if k >= n:
            continue

        km = KMeans(n_clusters=k, random_state=rng_seed, n_init=10)
        labels = km.fit_predict(clustering_basis.numpy())

        cluster_r2s = []
        cluster_sizes = []
        cluster_details = []

        for c in range(k):
            mask = labels == c
            nc = int(mask.sum())
            if nc < X.shape[1] + 2:
                r2_c = float('nan')
            else:
                r2_c, _ = fit_affine_r2(X[mask], Y[mask])
            cluster_r2s.append(r2_c)
            cluster_sizes.append(nc)
            cluster_details.append({"cluster": c, "size": nc, "r2": round(r2_c, 4) if not np.isnan(r2_c) else None})

        valid_r2s = [r for r in cluster_r2s if not np.isnan(r)]
        valid_sizes = [s for r, s in zip(cluster_r2s, cluster_sizes) if not np.isnan(r)]
        total_valid = sum(valid_sizes)

        if valid_r2s:
            weighted_r2 = sum(r * s for r, s in zip(valid_r2s, valid_sizes)) / total_valid
            mean_r2 = np.mean(valid_r2s)
            min_r2 = min(valid_r2s)
            max_r2 = max(valid_r2s)
        else:
            weighted_r2 = mean_r2 = min_r2 = max_r2 = float('nan')

        results.append({
            "k": k,
            "basis": basis_name,
            "global_r2": round(global_r2, 4),
            "weighted_r2": round(weighted_r2, 4),
            "mean_r2": round(mean_r2, 4),
            "min_r2": round(min_r2, 4),
            "max_r2": round(max_r2, 4),
            "n_valid_clusters": len(valid_r2s),
            "n_underfitted": k - len(valid_r2s),
            "clusters": cluster_details,
        })

    return results


def train_model(args, device):
    """Train a model from scratch and return (model, vocab)."""
    vocab = RavfogelVocab(n_entities=args.n_entities, num_relations=2)
    perms = generate_permutations(args.n_entities, 2, seed=args.seed)
    for k, p in enumerate(perms):
        vocab.set_permutation(k, p)

    model_cfg = ModelConfig(n_layers=1, hidden_dim=args.dim, n_heads=args.n_heads)
    torch.manual_seed(args.seed)
    model = build_model(model_cfg, vocab)
    if args.activation != "gelu":
        patch_activation(model, args.activation)
    model.to(device)

    warmup_steps = 200
    steps_per_epoch = args.num_sequences // args.batch_size
    total_steps = steps_per_epoch * args.n_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def lr_fn(step):
        return get_lr(step, warmup_steps, total_steps, args.lr)

    rng = random.Random(args.seed + 2000)
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.n_epochs + 1):
        sequences = build_sequences(vocab, args.num_sequences, SEQ_LENGTH, rng)
        avg_loss, global_step = train_epoch(
            model, sequences, optimizer, device, args.batch_size,
            lr_fn, global_step)

        if epoch % 5 == 0 or epoch == args.n_epochs:
            acc = eval_relation_accuracy(model, vocab, device)
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | acc={acc:.3f} | "
                  f"t={time.time()-t0:.0f}s", flush=True)

    acc = eval_relation_accuracy(model, vocab, device)
    print(f"\nFinal accuracy: {acc:.3f}")

    act_suffix = f"_{args.activation}" if args.activation != "gelu" else ""
    ckpt_path = ROOT / f"checkpoints_ravfogel_cluster_dim{args.dim}_seed{args.seed}{act_suffix}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "seed": args.seed,
        "dim": args.dim,
        "n_entities": args.n_entities,
        "n_epochs": args.n_epochs,
        "permutations": perms,
        "activation": args.activation,
        "model_cfg": {"n_layers": 1, "hidden_dim": args.dim, "n_heads": args.n_heads},
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    return model, vocab


def load_model(args, device):
    """Load a previously trained model from checkpoint."""
    act_suffix = f"_{args.activation}" if args.activation != "gelu" else ""
    ckpt_path = ROOT / f"checkpoints_ravfogel_cluster_dim{args.dim}_seed{args.seed}{act_suffix}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    vocab = RavfogelVocab(n_entities=ckpt["n_entities"], num_relations=2)
    for k, p in enumerate(ckpt["permutations"]):
        vocab.set_permutation(k, p)

    cfg = ckpt["model_cfg"]
    model_cfg = ModelConfig(n_layers=cfg["n_layers"], hidden_dim=cfg["hidden_dim"],
                            n_heads=cfg["n_heads"])
    model = build_model(model_cfg, vocab)
    if args.activation != "gelu":
        patch_activation(model, args.activation)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")
    return model, vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-entities", type=int, default=2025)
    parser.add_argument("--num-sequences", type=int, default=62500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--activation", type=str, default="gelu",
                        choices=list(ACT_FNS.keys()),
                        help="Activation function (default: gelu)")
    parser.add_argument("--load-checkpoint", action="store_true",
                        help="Load from saved checkpoint instead of retraining")
    args = parser.parse_args()

    act_fn = ACT_FNS[args.activation]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"\n{'='*70}")
    print(f"CLUSTER ANALYSIS: dim={args.dim}, seed={args.seed}, activation={args.activation}")
    print(f"{'='*70}")

    # --- Step 1: Train or load model ---
    if args.load_checkpoint:
        model, vocab = load_model(args, device)
    else:
        print(f"\nTraining model for {args.n_epochs} epochs...")
        model, vocab = train_model(args, device)

    # --- Step 2: Extract states ---
    print(f"\nExtracting MLP states for all {vocab.n_entities} entities × 2 relations...")
    mlp_inputs, mlp_outputs, pre_gelu, post_gelu = extract_all_states(model, vocab, device, act_fn=act_fn)

    k_values = [2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100]

    all_results = {}

    for rel in range(2):
        print(f"\n{'='*70}")
        print(f"RELATION {rel}")
        print(f"{'='*70}")

        X = torch.stack([mlp_inputs[(i, rel)] for i in range(vocab.n_entities)])
        Y = torch.stack([mlp_outputs[(i, rel)] for i in range(vocab.n_entities)])
        PG = torch.stack([post_gelu[(i, rel)] for i in range(vocab.n_entities)])
        PRE = torch.stack([pre_gelu[(i, rel)] for i in range(vocab.n_entities)])

        global_r2, _ = fit_affine_r2(X, Y)
        print(f"\nGlobal affine R² (MLP input → output): {global_r2:.4f}")

        # --- Basis 1: Cluster by MLP input ---
        print(f"\n--- Clustering by MLP input vectors ---")
        res_mlp_in = cluster_analysis(X, Y, k_values, X, "mlp_input")
        for r in res_mlp_in:
            print(f"  k={r['k']:3d} | weighted R²={r['weighted_r2']:.4f} | "
                  f"mean={r['mean_r2']:.4f} | min={r['min_r2']:.4f} | max={r['max_r2']:.4f} | "
                  f"valid={r['n_valid_clusters']}/{r['k']}")

        # --- Basis 2: Cluster by post-GELU activations ---
        print(f"\n--- Clustering by post-GELU activation patterns ---")
        res_post_gelu = cluster_analysis(X, Y, k_values, PG, "post_gelu")
        for r in res_post_gelu:
            print(f"  k={r['k']:3d} | weighted R²={r['weighted_r2']:.4f} | "
                  f"mean={r['mean_r2']:.4f} | min={r['min_r2']:.4f} | max={r['max_r2']:.4f} | "
                  f"valid={r['n_valid_clusters']}/{r['k']}")

        # --- Basis 3: Cluster by affine residual ---
        print(f"\n--- Clustering by affine residual ---")
        n = X.shape[0]
        X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)
        sol, _, _, _ = torch.linalg.lstsq(X_aug, Y)
        residuals = Y - X_aug @ sol
        res_residual = cluster_analysis(X, Y, k_values, residuals, "affine_residual")
        for r in res_residual:
            print(f"  k={r['k']:3d} | weighted R²={r['weighted_r2']:.4f} | "
                  f"mean={r['mean_r2']:.4f} | min={r['min_r2']:.4f} | max={r['max_r2']:.4f} | "
                  f"valid={r['n_valid_clusters']}/{r['k']}")

        all_results[f"relation_{rel}"] = {
            "global_r2": round(global_r2, 4),
            "by_mlp_input": res_mlp_in,
            "by_post_gelu": res_post_gelu,
            "by_affine_residual": res_residual,
        }

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    act_suffix = f"_{args.activation}" if args.activation != "gelu" else ""
    out_path = RESULTS_DIR / f"ravfogel_cluster_dim{args.dim}_seed{args.seed}{act_suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "ravfogel_cluster_analysis",
            "seed": args.seed,
            "dim": args.dim,
            "activation": args.activation,
            "n_entities": args.n_entities,
            "n_epochs": args.n_epochs,
            "k_values": k_values,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

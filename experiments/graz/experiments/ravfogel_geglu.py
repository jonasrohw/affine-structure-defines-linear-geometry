"""
GeGLU experiment: does gate/value separation enable cleaner relation gating?

GeGLU: output = GELU(W_gate @ x + b_gate) * (W_value @ x + b_value)
Then fc2 projects back: fc2(output)

Parameter-matched comparison:
  Standard GELU dim=64: fc1 (64→256) + fc2 (256→64) = 32,768 + 16,384 = 49,152 params (with biases)
  Actually without biases on fc2... let's just count weight params:
    fc1: 64*256 = 16,384  (+ 256 bias = 16,640)
    fc2: 256*64 = 16,384  (+ 64 bias = 16,448)
    Total: 33,088

  GeGLU with hidden=170:
    W_gate: 64*170 = 10,880 (+ 170 bias = 11,050)
    W_value: 64*170 = 10,880 (+ 170 bias = 11,050)
    fc2: 170*64 = 10,880 (+ 64 bias = 10,944)
    Total: 33,044  (≈ matched)

Usage:
    python experiments/ravfogel_geglu.py --seed 42
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
from src.model import build_model, MiniTransformer
from src.training import get_lr

from experiments.ravfogel_1layer import (
    RavfogelVocab, generate_permutations, build_sequences,
    eval_relation_accuracy, train_epoch, SEQ_LENGTH, REL_SENTENCE_LEN,
)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


class GeGLU_MLP(nn.Module):
    """MLP with GeGLU activation: output = GELU(W_gate·x + b_gate) ⊙ (W_value·x + b_value)"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_gate = nn.Linear(input_dim, hidden_dim)
        self.W_value = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        gate = F.gelu(self.W_gate(x))
        value = self.W_value(x)
        return self.fc2(gate * value)


def patch_mlp_geglu(model, hidden_dim):
    """Replace standard MLP with GeGLU MLP in all blocks."""
    import types
    for block in model.blocks:
        if hasattr(block, 'mlp'):
            input_dim = block.mlp.fc1.in_features
            geglu_mlp = GeGLU_MLP(input_dim, hidden_dim)
            nn.init.normal_(geglu_mlp.W_gate.weight, std=0.02)
            nn.init.zeros_(geglu_mlp.W_gate.bias)
            nn.init.normal_(geglu_mlp.W_value.weight, std=0.02)
            nn.init.zeros_(geglu_mlp.W_value.bias)
            nn.init.normal_(geglu_mlp.fc2.weight, std=0.02)
            nn.init.zeros_(geglu_mlp.fc2.bias)
            block.mlp = geglu_mlp


def fit_affine_r2(X, Y):
    n = X.shape[0]
    if n < X.shape[1] + 2:
        return float('nan'), 0
    X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)
    sol, _, _, _ = torch.linalg.lstsq(X_aug, Y)
    Y_pred = X_aug @ sol
    ss_res = ((Y - Y_pred) ** 2).sum().item()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-10), n


def cluster_analysis(X, Y, k_values, clustering_basis, basis_name, rng_seed=42):
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
        for c in range(k):
            mask = labels == c
            nc = int(mask.sum())
            if nc < X.shape[1] + 2:
                r2_c = float('nan')
            else:
                r2_c, _ = fit_affine_r2(X[mask], Y[mask])
            cluster_r2s.append(r2_c)
            cluster_sizes.append(nc)
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
            "k": k, "basis": basis_name, "global_r2": round(global_r2, 4),
            "weighted_r2": round(weighted_r2, 4), "mean_r2": round(mean_r2, 4),
            "min_r2": round(min_r2, 4), "max_r2": round(max_r2, 4),
            "n_valid_clusters": len(valid_r2s),
        })
    return results


@torch.no_grad()
def extract_geglu_states(model, vocab, device, batch_size=512):
    """Extract MLP inputs, gate activations, value activations, outputs."""
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    block = model.blocks[0]
    mlp = block.mlp  # GeGLU_MLP
    ln1 = block.ln1
    ln2 = block.ln2

    mlp_inputs = {}
    mlp_outputs = {}
    gate_acts = {}
    value_acts = {}
    pre_gate = {}

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

        # GeGLU forward decomposed
        pre_g = mlp.W_gate(mlp_in)
        gate = F.gelu(pre_g)
        value = mlp.W_value(mlp_in)
        mlp_out = mlp.fc2(gate * value)

        for idx, (i, k) in enumerate(batch_pairs):
            mlp_inputs[(i, k)] = mlp_in[idx, 2, :].cpu()
            mlp_outputs[(i, k)] = mlp_out[idx, 2, :].cpu()
            gate_acts[(i, k)] = gate[idx, 2, :].cpu()
            value_acts[(i, k)] = value[idx, 2, :].cpu()
            pre_gate[(i, k)] = pre_g[idx, 2, :].cpu()

    return mlp_inputs, mlp_outputs, gate_acts, value_acts, pre_gate


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
    parser.add_argument("--geglu-hidden", type=int, default=170,
                        help="GeGLU hidden dim (170 ≈ param-matched with standard 256)")
    parser.add_argument("--load-checkpoint", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"GeGLU CLUSTER ANALYSIS: dim={args.dim}, seed={args.seed}, geglu_hidden={args.geglu_hidden}")
    print(f"{'='*70}")

    ckpt_path = ROOT / f"checkpoints_ravfogel_cluster_dim{args.dim}_seed{args.seed}_geglu{args.geglu_hidden}.pt"

    if args.load_checkpoint and ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        vocab = RavfogelVocab(n_entities=ckpt["n_entities"], num_relations=2)
        for k, p in enumerate(ckpt["permutations"]):
            vocab.set_permutation(k, p)
        cfg = ckpt["model_cfg"]
        model_cfg = ModelConfig(n_layers=cfg["n_layers"], hidden_dim=cfg["hidden_dim"],
                                n_heads=cfg["n_heads"])
        model = build_model(model_cfg, vocab)
        patch_mlp_geglu(model, args.geglu_hidden)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
    else:
        # Train from scratch
        vocab = RavfogelVocab(n_entities=args.n_entities, num_relations=2)
        perms = generate_permutations(args.n_entities, 2, seed=args.seed)
        for k, p in enumerate(perms):
            vocab.set_permutation(k, p)

        model_cfg = ModelConfig(n_layers=1, hidden_dim=args.dim, n_heads=args.n_heads)
        torch.manual_seed(args.seed)
        model = build_model(model_cfg, vocab)
        patch_mlp_geglu(model, args.geglu_hidden)
        model.to(device)

        # Count parameters
        mlp_params = sum(p.numel() for p in model.blocks[0].mlp.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  GeGLU MLP params: {mlp_params:,}")
        print(f"  Total model params: {total_params:,}")

        warmup_steps = 200
        steps_per_epoch = args.num_sequences // args.batch_size
        total_steps = steps_per_epoch * args.n_epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

        def lr_fn(step):
            return get_lr(step, warmup_steps, total_steps, args.lr)

        rng = random.Random(args.seed + 2000)
        global_step = 0
        t0 = time.time()

        print(f"\nTraining for {args.n_epochs} epochs...")
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

        torch.save({
            "model_state": model.state_dict(),
            "seed": args.seed,
            "dim": args.dim,
            "n_entities": args.n_entities,
            "n_epochs": args.n_epochs,
            "permutations": perms,
            "activation": "geglu",
            "geglu_hidden": args.geglu_hidden,
            "model_cfg": {"n_layers": 1, "hidden_dim": args.dim, "n_heads": args.n_heads},
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    # --- Extract states ---
    print(f"\nExtracting GeGLU states for all {vocab.n_entities} entities × 2 relations...")
    mlp_inputs, mlp_outputs, gate_acts, value_acts, pre_gate = extract_geglu_states(model, vocab, device)

    k_values = [2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100]
    all_results = {}

    for rel in range(2):
        print(f"\n{'='*70}")
        print(f"RELATION {rel}")
        print(f"{'='*70}")

        X = torch.stack([mlp_inputs[(i, rel)] for i in range(vocab.n_entities)])
        Y = torch.stack([mlp_outputs[(i, rel)] for i in range(vocab.n_entities)])
        G = torch.stack([gate_acts[(i, rel)] for i in range(vocab.n_entities)])
        V = torch.stack([value_acts[(i, rel)] for i in range(vocab.n_entities)])
        PG = torch.stack([pre_gate[(i, rel)] for i in range(vocab.n_entities)])

        global_r2, _ = fit_affine_r2(X, Y)
        print(f"\nGlobal affine R² (MLP input → output): {global_r2:.4f}")

        # Cluster by MLP input
        print(f"\n--- Clustering by MLP input vectors ---")
        res_mlp_in = cluster_analysis(X, Y, k_values, X, "mlp_input")
        for r in res_mlp_in:
            print(f"  k={r['k']:3d} | weighted R²={r['weighted_r2']:.4f} | "
                  f"mean={r['mean_r2']:.4f} | min={r['min_r2']:.4f} | max={r['max_r2']:.4f} | "
                  f"valid={r['n_valid_clusters']}/{r['k']}")

        # Cluster by gate activations
        print(f"\n--- Clustering by gate activations ---")
        res_gate = cluster_analysis(X, Y, k_values, G, "gate_activations")
        for r in res_gate:
            print(f"  k={r['k']:3d} | weighted R²={r['weighted_r2']:.4f} | "
                  f"mean={r['mean_r2']:.4f} | min={r['min_r2']:.4f} | max={r['max_r2']:.4f} | "
                  f"valid={r['n_valid_clusters']}/{r['k']}")

        # Cluster by gate * value (effective activation)
        print(f"\n--- Clustering by gate*value (effective activation) ---")
        GV = G * V
        res_gv = cluster_analysis(X, Y, k_values, GV, "gate_times_value")
        for r in res_gv:
            print(f"  k={r['k']:3d} | weighted R²={r['weighted_r2']:.4f} | "
                  f"mean={r['mean_r2']:.4f} | min={r['min_r2']:.4f} | max={r['max_r2']:.4f} | "
                  f"valid={r['n_valid_clusters']}/{r['k']}")

        all_results[f"relation_{rel}"] = {
            "global_r2": round(global_r2, 4),
            "by_mlp_input": res_mlp_in,
            "by_gate": res_gate,
            "by_gate_times_value": res_gv,
        }

    # --- Gate analysis: does the gate align with relation? ---
    print(f"\n{'='*70}")
    print("GATE ANALYSIS: does the gate separate by relation?")
    print(f"{'='*70}")

    for i_ent in range(min(5, vocab.n_entities)):
        g_r0 = gate_acts[(i_ent, 0)]
        g_r1 = gate_acts[(i_ent, 1)]
        cos = F.cosine_similarity(g_r0.unsqueeze(0), g_r1.unsqueeze(0)).item()
        diff_norm = (g_r0 - g_r1).norm().item()
        print(f"  Entity {i_ent}: gate cos(R0,R1)={cos:.4f}, ||diff||={diff_norm:.4f}")

    # Mean gate per relation
    G_r0 = torch.stack([gate_acts[(i, 0)] for i in range(vocab.n_entities)])
    G_r1 = torch.stack([gate_acts[(i, 1)] for i in range(vocab.n_entities)])
    mean_g0 = G_r0.mean(0)
    mean_g1 = G_r1.mean(0)
    cos_means = F.cosine_similarity(mean_g0.unsqueeze(0), mean_g1.unsqueeze(0)).item()
    print(f"\n  Mean gate cos(R0, R1): {cos_means:.4f}")
    print(f"  Mean gate norm R0: {mean_g0.norm():.4f}, R1: {mean_g1.norm():.4f}")

    # How many neurons have significantly different gate activations?
    gate_diff = (G_r0 - G_r1).mean(0)  # per-neuron mean difference
    gate_std = (G_r0 - G_r1).std(0)
    snr = gate_diff.abs() / (gate_std + 1e-8)
    n_high_snr = (snr > 2).sum().item()
    print(f"  Neurons with |gate_diff|/std > 2 (relation-sensitive): {n_high_snr}/{args.geglu_hidden}")

    # PCA on MLP inputs
    print(f"\n{'='*70}")
    print("PCA on MLP inputs (combined R0+R1)")
    print(f"{'='*70}")
    X_all = torch.cat([
        torch.stack([mlp_inputs[(i, 0)] for i in range(vocab.n_entities)]),
        torch.stack([mlp_inputs[(i, 1)] for i in range(vocab.n_entities)])
    ]).numpy()
    X_centered = X_all - X_all.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_var = S**2 / (S**2).sum()
    print(f"  PC1: {explained_var[0]*100:.2f}%")
    print(f"  PC2: {explained_var[1]*100:.2f}%")
    print(f"  PC3: {explained_var[2]*100:.2f}%")
    print(f"  Top-5: {explained_var[:5].sum()*100:.2f}%")

    # Check bimodality of PC1
    proj = X_centered @ Vt[0]
    from scipy.stats import kurtosis
    kurt = kurtosis(proj)
    n_ent = vocab.n_entities
    proj_r0 = proj[:n_ent]
    proj_r1 = proj[n_ent:]
    flip_frac = ((proj_r0 > np.median(proj)) != (proj_r1 > np.median(proj))).mean()
    print(f"  PC1 kurtosis: {kurt:.4f} (negative = bimodal)")
    print(f"  Fraction flipping R0→R1: {flip_frac:.4f}")

    # Per-relation PCA
    for rel in range(2):
        X_rel = torch.stack([mlp_inputs[(i, rel)] for i in range(vocab.n_entities)]).numpy()
        X_c = X_rel - X_rel.mean(axis=0)
        _, S_rel, _ = np.linalg.svd(X_c, full_matrices=False)
        ev = S_rel**2 / (S_rel**2).sum()
        print(f"  R{rel} per-relation PC1: {ev[0]*100:.2f}%, PC2: {ev[1]*100:.2f}%")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_geglu_dim{args.dim}_seed{args.seed}_h{args.geglu_hidden}.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "ravfogel_geglu",
            "seed": args.seed, "dim": args.dim,
            "geglu_hidden": args.geglu_hidden,
            "n_entities": args.n_entities, "n_epochs": args.n_epochs,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

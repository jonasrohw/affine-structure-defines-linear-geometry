"""
Trace the time course of mechanism development in Ravfogel bijective training.

Re-trains the dim=128 model and computes mechanism metrics AFTER EVERY EPOCH:
  1. Accuracy (relation recall)
  2. Attention: between-relation variance, within-relation cosine
  3. Neuron relation specificity: mean |Cohen's d|, fraction |d|>0.8
  4. Neuron classification: R0-only, R1-only, shared, dead
  5. Gating SNR: mean SNR, fraction SNR>2
  6. MLP linearity: per-relation R² of affine fit (input→output)

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_timecourse.py --seed 42
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
from torch.utils.data import TensorDataset, DataLoader

from src.config import ModelConfig
from src.model import build_model
from src.training import get_lr

from experiments.ravfogel_1layer import (
    RavfogelVocab, generate_permutations, build_sequences,
    eval_relation_accuracy, train_epoch, SEQ_LENGTH, REL_SENTENCE_LEN,
)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


@torch.no_grad()
def extract_mechanism_states(model, vocab, device, batch_size=512):
    """Extract attention outputs, MLP pre/post-GELU at subject position for all pairs."""
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]

    block = model.blocks[0]
    fc1 = block.mlp.fc1
    fc2 = block.mlp.fc2
    ln1 = block.ln1
    ln2 = block.ln2

    attn_outputs = {}
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
        post_g = F.gelu(pre_g)
        mlp_out = fc2(post_g)

        for idx, (i, k) in enumerate(batch_pairs):
            attn_outputs[(i, k)] = attn_out[idx, 2, :].cpu()
            mlp_inputs[(i, k)] = mlp_in[idx, 2, :].cpu()
            mlp_outputs[(i, k)] = mlp_out[idx, 2, :].cpu()
            pre_gelu[(i, k)] = pre_g[idx, 2, :].cpu()
            post_gelu[(i, k)] = post_g[idx, 2, :].cpu()

    return attn_outputs, mlp_inputs, mlp_outputs, pre_gelu, post_gelu


@torch.no_grad()
def compute_epoch_metrics(model, vocab, device):
    """Compute all mechanism metrics for the current model state."""
    model.eval()
    n_entities = vocab.n_entities
    n_neurons = 4 * model.blocks[0].mlp.fc1.weight.shape[1]  # infer from weight shape

    attn_outputs, mlp_inputs, mlp_outputs, pre_gelu, post_gelu = \
        extract_mechanism_states(model, vocab, device)

    metrics = {}

    # --- 1. Attention as relation signal ---
    r0_attn = torch.stack([attn_outputs[(i, 0)] for i in range(n_entities)])
    r1_attn = torch.stack([attn_outputs[(i, 1)] for i in range(n_entities)])
    mean_r0 = r0_attn.mean(0)
    mean_r1 = r1_attn.mean(0)

    all_attn = torch.cat([r0_attn, r1_attn], dim=0)
    grand_mean = all_attn.mean(0)
    total_var = ((all_attn - grand_mean) ** 2).sum().item()
    between_var = (n_entities * ((mean_r0 - grand_mean) ** 2).sum().item() +
                   n_entities * ((mean_r1 - grand_mean) ** 2).sum().item())
    between_pct = between_var / max(total_var, 1e-10) * 100

    cos_means = F.cosine_similarity(mean_r0.unsqueeze(0), mean_r1.unsqueeze(0)).item()

    rng = np.random.RandomState(42)
    n_sample = min(500, n_entities)
    idxs = rng.choice(n_entities, n_sample, replace=False)
    r0_s = r0_attn[idxs]
    r1_s = r1_attn[idxs]
    mask = torch.triu(torch.ones(n_sample, n_sample), diagonal=1).bool()
    within_r0 = F.cosine_similarity(r0_s.unsqueeze(0), r0_s.unsqueeze(1), dim=2)[mask].mean().item()
    within_r1 = F.cosine_similarity(r1_s.unsqueeze(0), r1_s.unsqueeze(1), dim=2)[mask].mean().item()

    metrics["attn_between_rel_var_pct"] = round(between_pct, 2)
    metrics["attn_cos_means"] = round(cos_means, 4)
    metrics["attn_within_r0_cos"] = round(within_r0, 4)
    metrics["attn_within_r1_cos"] = round(within_r1, 4)

    # --- 2. Neuron relation specificity (Cohen's d) ---
    r0_post = torch.stack([post_gelu[(i, 0)] for i in range(n_entities)])
    r1_post = torch.stack([post_gelu[(i, 1)] for i in range(n_entities)])

    r0_mean_act = r0_post.mean(0)
    r1_mean_act = r1_post.mean(0)
    pooled_std = ((r0_post.var(0) + r1_post.var(0)) / 2).sqrt()
    cohens_d = ((r0_mean_act - r1_mean_act) / (pooled_std + 1e-8)).numpy()
    abs_d = np.abs(cohens_d)

    metrics["mean_abs_cohens_d"] = round(float(abs_d.mean()), 3)
    metrics["neurons_d_gt_0.8"] = int((abs_d > 0.8).sum())
    metrics["neurons_d_lt_0.2"] = int((abs_d < 0.2).sum())
    metrics["r0_specific"] = int(((abs_d > 0.8) & (cohens_d > 0)).sum())
    metrics["r1_specific"] = int(((abs_d > 0.8) & (cohens_d < 0)).sum())

    # --- 3. Neuron classification (by mean activation threshold) ---
    r0_mean_np = r0_mean_act.numpy()
    r1_mean_np = r1_mean_act.numpy()
    r0_only = int(((r0_mean_np > 0.05) & (r1_mean_np <= 0.05)).sum())
    r1_only = int(((r1_mean_np > 0.05) & (r0_mean_np <= 0.05)).sum())
    shared = int(((r0_mean_np > 0.05) & (r1_mean_np > 0.05)).sum())
    dead = int(((r0_mean_np <= 0.05) & (r1_mean_np <= 0.05)).sum())

    metrics["r0_only_neurons"] = r0_only
    metrics["r1_only_neurons"] = r1_only
    metrics["shared_neurons"] = shared
    metrics["dead_neurons"] = dead

    # --- 4. Gating SNR ---
    r0_pre = torch.stack([pre_gelu[(i, 0)] for i in range(n_entities)])
    r1_pre = torch.stack([pre_gelu[(i, 1)] for i in range(n_entities)])

    block = model.blocks[0]
    fc1_weight = block.mlp.fc1.weight.detach().cpu()
    fc1_bias = block.mlp.fc1.bias.detach().cpu()

    r0_mlp_in = torch.stack([mlp_inputs[(i, 0)] for i in range(n_entities)])
    r1_mlp_in = torch.stack([mlp_inputs[(i, 1)] for i in range(n_entities)])
    mean_r0_in = r0_mlp_in.mean(0)
    mean_r1_in = r1_mlp_in.mean(0)

    rel_comp_r0 = fc1_weight @ mean_r0_in + fc1_bias
    rel_comp_r1 = fc1_weight @ mean_r1_in + fc1_bias
    gating_shift = (rel_comp_r0 - rel_comp_r1).numpy()

    subj_comp_r0 = r0_pre - rel_comp_r0.unsqueeze(0)
    subj_comp_r1 = r1_pre - rel_comp_r1.unsqueeze(0)
    subj_std = ((subj_comp_r0.std(0).numpy() + subj_comp_r1.std(0).numpy()) / 2)
    gating_snr = np.abs(gating_shift) / (subj_std + 1e-8)

    metrics["mean_gating_shift"] = round(float(np.abs(gating_shift).mean()), 4)
    metrics["mean_subj_std"] = round(float(subj_std.mean()), 4)
    metrics["mean_gating_snr"] = round(float(gating_snr.mean()), 3)
    metrics["median_gating_snr"] = round(float(np.median(gating_snr)), 3)
    metrics["snr_gt_2"] = int((gating_snr > 2).sum())

    # --- 5. MLP linearity (affine fit R²) ---
    for k in range(vocab.num_relations):
        X = torch.stack([mlp_inputs[(i, k)] for i in range(n_entities)])
        Y = torch.stack([mlp_outputs[(i, k)] for i in range(n_entities)])
        n = X.shape[0]
        X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)
        sol, _, _, _ = torch.linalg.lstsq(X_aug, Y)
        Y_pred = X_aug @ sol
        ss_res = ((Y - Y_pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(0)) ** 2).sum().item()
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        metrics[f"mlp_r2_r{k}"] = round(r2, 4)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-entities", type=int, default=2025)
    parser.add_argument("--num-sequences", type=int, default=62500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    vocab = RavfogelVocab(n_entities=args.n_entities, num_relations=2)
    perms = generate_permutations(args.n_entities, 2, seed=args.seed)
    for k, p in enumerate(perms):
        vocab.set_permutation(k, p)

    model_cfg = ModelConfig(n_layers=1, hidden_dim=args.dim, n_heads=args.n_heads)

    print(f"\n{'='*70}")
    print(f"RAVFOGEL MECHANISM TIME COURSE (seed={args.seed})")
    print(f"{'='*70}")
    print(f"Entities: {args.n_entities}, Relations: 2 (random bijections)")
    print(f"Model: dim={args.dim}, layers=1, heads={args.n_heads}, WITH MLP")
    print(f"Epochs: {args.n_epochs}, computing metrics after EVERY epoch")

    torch.manual_seed(args.seed)
    model = build_model(model_cfg, vocab)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    n_neurons = 4 * args.dim
    print(f"MLP neurons: {n_neurons}")

    warmup_steps = 200
    steps_per_epoch = args.num_sequences // args.batch_size
    total_steps = steps_per_epoch * args.n_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def lr_fn(step):
        return get_lr(step, warmup_steps, total_steps, args.lr)

    # Pre-training metrics
    print(f"\nComputing pre-training metrics (epoch 0)...", flush=True)
    acc_0 = eval_relation_accuracy(model, vocab, device)
    metrics_0 = compute_epoch_metrics(model, vocab, device)
    metrics_0["accuracy"] = round(acc_0, 4)
    metrics_0["loss"] = None
    metrics_0["epoch"] = 0

    all_metrics = [metrics_0]
    print(f"  Ep  0 | acc={acc_0:.3f} | attn_rel={metrics_0['attn_between_rel_var_pct']:.1f}% | "
          f"|d|={metrics_0['mean_abs_cohens_d']:.2f} | "
          f"R0={metrics_0['r0_only_neurons']} R1={metrics_0['r1_only_neurons']} "
          f"sh={metrics_0['shared_neurons']} | "
          f"SNR={metrics_0['mean_gating_snr']:.2f} | "
          f"R²={metrics_0['mlp_r2_r0']:.3f}/{metrics_0['mlp_r2_r1']:.3f}", flush=True)

    rng = random.Random(args.seed + 2000)
    global_step = 0
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"{'Ep':>3} | {'loss':>6} | {'acc':>5} | {'attn%':>5} | {'|d|':>5} | "
          f"{'R0':>3} {'R1':>3} {'sh':>3} | {'SNR':>5} | {'R²(0)':>6} {'R²(1)':>6} | {'t':>4}")
    print(f"{'-'*70}")

    for epoch in range(1, args.n_epochs + 1):
        sequences = build_sequences(vocab, args.num_sequences, SEQ_LENGTH, rng)
        avg_loss, global_step = train_epoch(
            model, sequences, optimizer, device, args.batch_size,
            lr_fn, global_step)

        acc = eval_relation_accuracy(model, vocab, device)
        metrics = compute_epoch_metrics(model, vocab, device)
        metrics["accuracy"] = round(acc, 4)
        metrics["loss"] = round(avg_loss, 4)
        metrics["epoch"] = epoch
        all_metrics.append(metrics)

        elapsed = time.time() - t0
        print(f"{epoch:3d} | {avg_loss:.4f} | {acc:.3f} | "
              f"{metrics['attn_between_rel_var_pct']:5.1f} | "
              f"{metrics['mean_abs_cohens_d']:5.2f} | "
              f"{metrics['r0_only_neurons']:3d} {metrics['r1_only_neurons']:3d} "
              f"{metrics['shared_neurons']:3d} | "
              f"{metrics['mean_gating_snr']:5.2f} | "
              f"{metrics['mlp_r2_r0']:.4f} {metrics['mlp_r2_r1']:.4f} | "
              f"{elapsed:4.0f}s", flush=True)

    print(f"\n{'='*70}")
    print(f"DONE — {args.n_epochs} epochs in {time.time() - t0:.0f}s")
    print(f"{'='*70}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_timecourse_dim{args.dim}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "ravfogel_timecourse",
            "seed": args.seed,
            "dim": args.dim,
            "n_entities": args.n_entities,
            "n_epochs": args.n_epochs,
            "n_params": n_params,
            "n_neurons": n_neurons,
            "epochs": all_metrics,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

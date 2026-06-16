"""
Ravfogel-style bijective factual recall with 1-layer standard transformer + dummy token.

Same as ravfogel_1layer.py but with a dummy token between subject and object:
  - Sequence format: [sep, r_k, subj_i, dummy, obj_j] (5 tokens per fact)
  - Prediction at dummy position (position 3)

The dummy token gives attention an extra position to combine relation and subject
signals before the MLP processes the result.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_1layer_dummy.py --seed 42
"""
import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.affine import fit_affine, apply_affine, reconstruction_accuracy
from src.config import ModelConfig
from src.model import build_model
from src.training import get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

SEQ_LENGTH = 50
REL_SENTENCE_LEN = 5


@dataclass
class RavfogelVocabDummy:
    n_entities: int = 2025
    num_relations: int = 2
    sep_id: int = 0

    def __post_init__(self):
        self._perm = {}
        self.n_train_subjects = self.n_entities // 2

    @property
    def subject_offset(self) -> int:
        return 1

    @property
    def object_offset(self) -> int:
        return 1 + self.n_entities

    @property
    def relation_offset(self) -> int:
        return 1 + 2 * self.n_entities

    @property
    def dummy_id(self) -> int:
        return self.relation_offset + self.num_relations

    @property
    def vocab_size(self) -> int:
        return 1 + 2 * self.n_entities + self.num_relations + 1  # +1 for dummy

    @property
    def train_subject_indices(self) -> List[int]:
        return list(range(self.n_train_subjects))

    @property
    def test_subject_indices(self) -> List[int]:
        return list(range(self.n_train_subjects, self.n_entities))

    def subject_id(self, i: int) -> int:
        return self.subject_offset + i

    def object_id(self, j: int) -> int:
        return self.object_offset + j

    def relation_id(self, k: int) -> int:
        return self.relation_offset + k

    def set_permutation(self, k: int, perm: List[int]):
        self._perm[k] = perm

    def relation_object(self, i: int, k: int) -> int:
        return self._perm[k][i]

    def relation_objects(self, i: int, k: int) -> List[int]:
        return [self._perm[k][i]]


def generate_permutations(n: int, num_relations: int, seed: int):
    rng = random.Random(seed)
    perms = []
    for _ in range(num_relations):
        p = list(range(n))
        rng.shuffle(p)
        perms.append(p)
    return perms


def build_sequences(vocab, n_sequences, seq_length, rng):
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    sequences = []
    for _ in range(n_sequences):
        tokens = []
        while len(tokens) + REL_SENTENCE_LEN <= seq_length:
            subj_idx, rel_k = rng.choice(all_pairs)
            obj_idx = vocab.relation_object(subj_idx, rel_k)
            tokens.extend([vocab.sep_id, vocab.relation_id(rel_k),
                           vocab.subject_id(subj_idx), vocab.dummy_id,
                           vocab.object_id(obj_idx)])
        while len(tokens) < seq_length:
            tokens.append(vocab.sep_id)
        sequences.append(tokens[:seq_length])
    return sequences


def eval_relation_accuracy(model, vocab, device, batch_size=512):
    """Evaluate accuracy at dummy position: [sep, r_k, subj_i, dummy] → predict obj."""
    model.eval()
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    correct = 0
    total = len(all_pairs)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_pairs = all_pairs[start:start + batch_size]
            contexts = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i), vocab.dummy_id]
                 for i, k in batch_pairs],
                dtype=torch.long, device=device,
            )
            logits = model(contexts)
            preds = logits[:, -1].argmax(dim=-1).cpu().tolist()
            for (i, k), pred in zip(batch_pairs, preds):
                correct += int(pred == vocab.object_id(vocab.relation_object(i, k)))
    return correct / total


def train_epoch(model, sequences, optimizer, device, batch_size, lr_fn, global_step):
    x_all = torch.stack([torch.tensor(s[:-1], dtype=torch.long) for s in sequences])
    y_all = torch.stack([torch.tensor(s[1:], dtype=torch.long) for s in sequences])
    dataset = TensorDataset(x_all, y_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        current_lr = lr_fn(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        logits = model(x)
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
        global_step += 1
    return epoch_loss / max(1, n_batches), global_step


@torch.no_grad()
def extract_hidden_states(model, vocab, device, subject_indices, batch_size=512):
    """Extract residual stream at dummy position (position 3)."""
    model.eval()
    keys = []
    contexts = []
    for i in subject_indices:
        for k in range(vocab.num_relations):
            keys.append((i, k))
            contexts.append([vocab.sep_id, vocab.relation_id(k),
                             vocab.subject_id(i), vocab.dummy_id])

    all_residuals = {}
    for start in range(0, len(contexts), batch_size):
        batch_ctx = contexts[start:start + batch_size]
        batch_keys = keys[start:start + batch_size]
        batch = torch.tensor(batch_ctx, dtype=torch.long, device=device)
        with model.capture_residuals() as residuals:
            model(batch)
        for layer_idx, layer_res in enumerate(residuals):
            if layer_idx not in all_residuals:
                all_residuals[layer_idx] = {}
            hidden = layer_res[:, 3, :].cpu()  # position 3 = dummy
            for i, key in enumerate(batch_keys):
                all_residuals[layer_idx][key] = hidden[i]

    return all_residuals


def affine_analysis(model, vocab, device):
    train_subjs = vocab.train_subject_indices
    test_subjs = vocab.test_subject_indices
    all_subjs = train_subjs + test_subjs

    layer_residuals = extract_hidden_states(model, vocab, device, all_subjs)

    weight = model.lm_head.weight.detach().cpu()
    O_all = weight[vocab.object_offset:vocab.object_offset + vocab.n_entities]

    target = {}
    for i in all_subjs:
        for k in range(vocab.num_relations):
            obj_idx = vocab.relation_object(i, k)
            target[(i, k)] = weight[vocab.object_id(obj_idx)]

    results = {}
    for layer_idx, h_dict in sorted(layer_residuals.items()):
        layer_results = {}
        for k in range(vocab.num_relations):
            S_train = torch.stack([h_dict[(i, k)] for i in train_subjs])
            O_train = torch.stack([target[(i, k)] for i in train_subjs])
            S_test = torch.stack([h_dict[(i, k)] for i in test_subjs])
            O_test = torch.stack([target[(i, k)] for i in test_subjs])

            W, b = fit_affine(S_train, O_train, bias=True)
            train_acc = reconstruction_accuracy(S_train, O_train, W, b, O_all)
            test_acc = reconstruction_accuracy(S_test, O_test, W, b, O_all)

            layer_results[f"r{k}"] = {
                "train_acc": round(train_acc, 4),
                "test_acc": round(test_acc, 4),
            }

        avg_train = np.mean([layer_results[f"r{k}"]["train_acc"] for k in range(vocab.num_relations)])
        avg_test = np.mean([layer_results[f"r{k}"]["test_acc"] for k in range(vocab.num_relations)])
        layer_results["overall"] = {
            "train_acc": round(float(avg_train), 4),
            "test_acc": round(float(avg_test), 4),
        }
        results[f"layer{layer_idx}"] = layer_results

    return results


@torch.no_grad()
def mlp_affine_analysis(model, vocab, device, batch_size=512):
    """Per-relation affine fit of MLP input → MLP output at dummy position."""
    model.eval()
    block = model.blocks[0]

    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    mlp_inputs = {}
    mlp_outputs = {}
    attn_outputs = {}
    pre_gelu_all = {}
    post_gelu_all = {}

    for start in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[start:start + batch_size]
        contexts = torch.tensor(
            [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i), vocab.dummy_id]
             for i, k in batch_pairs],
            dtype=torch.long, device=device)

        B, T = contexts.shape
        positions = torch.arange(T, device=device).unsqueeze(0)
        h0 = model.token_emb(contexts) + model.pos_emb(positions)
        attn_out = block.attn(block.ln1(h0))
        h1 = h0 + attn_out
        mlp_in = block.ln2(h1)
        pre_g = block.mlp.fc1(mlp_in)
        post_g = F.gelu(pre_g)
        mlp_out = block.mlp.fc2(post_g)

        for idx, (i, k) in enumerate(batch_pairs):
            mlp_inputs[(i, k)] = mlp_in[idx, 3, :].cpu()
            mlp_outputs[(i, k)] = mlp_out[idx, 3, :].cpu()
            attn_outputs[(i, k)] = attn_out[idx, 3, :].cpu()
            pre_gelu_all[(i, k)] = pre_g[idx, 3, :].cpu()
            post_gelu_all[(i, k)] = post_g[idx, 3, :].cpu()

    results = {}
    for k in range(vocab.num_relations):
        X = torch.stack([mlp_inputs[(i, k)] for i in range(vocab.n_entities)])
        Y = torch.stack([mlp_outputs[(i, k)] for i in range(vocab.n_entities)])
        n = X.shape[0]
        X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)
        sol, _, _, _ = torch.linalg.lstsq(X_aug, Y)
        Y_pred = X_aug @ sol
        ss_res = ((Y - Y_pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(0)) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot
        cos = F.cosine_similarity(Y, Y_pred, dim=1)
        results[f"r{k}"] = {"r2": round(r2, 4), "cos_mean": round(cos.mean().item(), 4)}

    # Attention output analysis
    r0_attn = torch.stack([attn_outputs[(i, 0)] for i in range(vocab.n_entities)])
    r1_attn = torch.stack([attn_outputs[(i, 1)] for i in range(vocab.n_entities)])
    mean_r0 = r0_attn.mean(0)
    mean_r1 = r1_attn.mean(0)
    all_attn = torch.cat([r0_attn, r1_attn], dim=0)
    grand_mean = all_attn.mean(0)
    total_var = ((all_attn - grand_mean) ** 2).sum().item()
    between_var = (vocab.n_entities * ((mean_r0 - grand_mean) ** 2).sum().item() +
                   vocab.n_entities * ((mean_r1 - grand_mean) ** 2).sum().item())
    results["attn_between_rel_var_pct"] = round(between_var / total_var * 100, 1)

    # Neuron relation specificity
    r0_post = torch.stack([post_gelu_all[(i, 0)] for i in range(vocab.n_entities)])
    r1_post = torch.stack([post_gelu_all[(i, 1)] for i in range(vocab.n_entities)])
    r0_mean_act = r0_post.mean(0).numpy()
    r1_mean_act = r1_post.mean(0).numpy()
    pooled_std = ((r0_post.var(0) + r1_post.var(0)) / 2).sqrt().numpy()
    cohens_d = (r0_mean_act - r1_mean_act) / (pooled_std + 1e-8)
    abs_d = np.abs(cohens_d)
    results["neurons_d_gt_0.8"] = int((abs_d > 0.8).sum())
    results["mean_abs_d"] = round(float(abs_d.mean()), 3)

    r0_only = ((r0_mean_act > 0.05) & (r1_mean_act <= 0.05)).sum()
    r1_only = ((r1_mean_act > 0.05) & (r0_mean_act <= 0.05)).sum()
    shared = ((r0_mean_act > 0.05) & (r1_mean_act > 0.05)).sum()
    dead = ((r0_mean_act <= 0.05) & (r1_mean_act <= 0.05)).sum()
    results["r0_only_neurons"] = int(r0_only)
    results["r1_only_neurons"] = int(r1_only)
    results["shared_neurons"] = int(shared)
    results["dead_neurons"] = int(dead)

    # Gating SNR
    r0_mlp = torch.stack([mlp_inputs[(i, 0)] for i in range(vocab.n_entities)])
    r1_mlp = torch.stack([mlp_inputs[(i, 1)] for i in range(vocab.n_entities)])
    mean_r0_mlp = r0_mlp.mean(0)
    mean_r1_mlp = r1_mlp.mean(0)
    rel_dir = mean_r0_mlp - mean_r1_mlp

    fc1 = block.mlp.fc1.weight.detach().cpu()
    fc1_bias = block.mlp.fc1.bias.detach().cpu()
    rel_comp_r0 = fc1 @ mean_r0_mlp + fc1_bias
    rel_comp_r1 = fc1 @ mean_r1_mlp + fc1_bias
    gating_shift = (rel_comp_r0 - rel_comp_r1).numpy()

    r0_pre = torch.stack([pre_gelu_all[(i, 0)] for i in range(vocab.n_entities)])
    r1_pre = torch.stack([pre_gelu_all[(i, 1)] for i in range(vocab.n_entities)])
    subj_std = ((r0_pre - rel_comp_r0).std(0).numpy() +
                (r1_pre - rel_comp_r1).std(0).numpy()) / 2
    gating_snr = np.abs(gating_shift) / (subj_std + 1e-8)
    results["mean_gating_snr"] = round(float(gating_snr.mean()), 2)
    results["median_gating_snr"] = round(float(np.median(gating_snr)), 2)
    results["snr_gt_2"] = int((gating_snr > 2).sum())

    # GELU derivative at active neurons
    def gelu_derivative(x):
        phi = torch.distributions.Normal(0, 1).log_prob(x).exp()
        Phi = 0.5 * (1 + torch.erf(x / np.sqrt(2)))
        return Phi + x * phi

    if r0_only > 0:
        r0_neuron_idx = np.where((r0_mean_act > 0.05) & (r1_mean_act <= 0.05))[0]
        d_match = gelu_derivative(r0_pre[:, r0_neuron_idx]).mean().item()
        d_wrong = gelu_derivative(r1_pre[:, r0_neuron_idx]).mean().item()
        results["gelu_deriv_matching"] = round(d_match, 3)
        results["gelu_deriv_wrong"] = round(d_wrong, 3)

    return results, attn_outputs, post_gelu_all


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    vocab = RavfogelVocabDummy(n_entities=args.n_entities, num_relations=2)
    perms = generate_permutations(args.n_entities, 2, seed=args.seed)
    for k, p in enumerate(perms):
        vocab.set_permutation(k, p)

    model_cfg = ModelConfig(n_layers=1, hidden_dim=args.dim, n_heads=args.n_heads)

    print(f"\n{'='*70}")
    print(f"RAVFOGEL BIJECTIVE — 1-LAYER + DUMMY TOKEN (seed={args.seed})")
    print(f"{'='*70}")
    print(f"Entities: {args.n_entities}, Relations: 2 (random bijections)")
    print(f"Total facts: {args.n_entities * 2}")
    print(f"Model: dim={args.dim}, layers=1, heads={args.n_heads}, WITH MLP")
    print(f"Sequence: [sep, r_k, subj_i, dummy, obj_j] — predict at dummy pos")
    print(f"Vocab size: {vocab.vocab_size}, Dummy ID: {vocab.dummy_id}")

    torch.manual_seed(args.seed)
    model = build_model(model_cfg, vocab)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    warmup_steps = 200
    steps_per_epoch = args.num_sequences // args.batch_size
    total_steps = steps_per_epoch * args.n_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def lr_fn(step):
        return get_lr(step, warmup_steps, total_steps, args.lr)

    print(f"\nTraining: {args.n_epochs} epochs, lr={args.lr}")
    print(f"Steps/epoch: {steps_per_epoch:,}, Total steps: {total_steps:,}")
    print(f"Sequences: {args.num_sequences:,} × {SEQ_LENGTH} tokens "
          f"({SEQ_LENGTH // REL_SENTENCE_LEN} facts/seq)")
    print(f"Affine split: train={vocab.n_train_subjects}, "
          f"test={args.n_entities - vocab.n_train_subjects}")
    print(f"{'='*70}\n", flush=True)

    pre_acc = eval_relation_accuracy(model, vocab, device)
    print(f"Pre-training accuracy: {pre_acc:.3f} (chance: {1/args.n_entities:.4f})\n", flush=True)

    results = {
        "experiment": "ravfogel_1layer_dummy",
        "seed": args.seed,
        "n_entities": args.n_entities,
        "dim": args.dim,
        "n_layers": 1,
        "n_heads": args.n_heads,
        "has_mlp": True,
        "has_dummy": True,
        "n_params": n_params,
        "vocab_size": vocab.vocab_size,
        "n_epochs": args.n_epochs,
        "pre_acc": round(pre_acc, 4),
        "epochs": {},
    }

    rng = random.Random(args.seed + 2000)
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.n_epochs + 1):
        sequences = build_sequences(vocab, args.num_sequences, SEQ_LENGTH, rng)
        avg_loss, global_step = train_epoch(
            model, sequences, optimizer, device, args.batch_size,
            lr_fn, global_step)
        acc = eval_relation_accuracy(model, vocab, device)
        elapsed = time.time() - t0
        results["epochs"][epoch] = {"loss": round(avg_loss, 4), "accuracy": round(acc, 4)}
        print(f"Ep {epoch:2d}/{args.n_epochs} | loss={avg_loss:.4f} | "
              f"acc={acc:.3f} | {elapsed:.0f}s", flush=True)

    # Affine analysis at dummy position
    print(f"\n--- Affine analysis at dummy position (nearest-neighbor accuracy) ---", flush=True)
    try:
        affine_results = affine_analysis(model, vocab, device)
        for label, res in sorted(affine_results.items()):
            ov = res["overall"]
            print(f"  {label}: train={ov['train_acc']:.3f}  test={ov['test_acc']:.3f}")
        results["affine"] = affine_results
    except Exception as e:
        print(f"  Affine analysis failed: {e}")
        import traceback; traceback.print_exc()
        results["affine"] = str(e)

    # MLP mechanism analysis
    print(f"\n--- MLP mechanism analysis at dummy position ---", flush=True)
    try:
        mlp_results, _, _ = mlp_affine_analysis(model, vocab, device)
        print(f"  Attention between-rel variance: {mlp_results['attn_between_rel_var_pct']}%")
        for k in range(vocab.num_relations):
            r = mlp_results[f"r{k}"]
            print(f"  MLP affine R² (R{k}): {r['r2']}")
        print(f"  Neuron partition: R0-only={mlp_results['r0_only_neurons']}, "
              f"R1-only={mlp_results['r1_only_neurons']}, "
              f"shared={mlp_results['shared_neurons']}, dead={mlp_results['dead_neurons']}")
        print(f"  Mean |Cohen's d|: {mlp_results['mean_abs_d']}")
        print(f"  Neurons |d|>0.8: {mlp_results['neurons_d_gt_0.8']}/256")
        print(f"  Gating SNR: mean={mlp_results['mean_gating_snr']}, "
              f"median={mlp_results['median_gating_snr']}, "
              f"SNR>2: {mlp_results['snr_gt_2']}/256")
        if "gelu_deriv_matching" in mlp_results:
            print(f"  GELU'(x) matching: {mlp_results['gelu_deriv_matching']}, "
                  f"wrong: {mlp_results['gelu_deriv_wrong']}")
        results["mlp_mechanism"] = mlp_results
    except Exception as e:
        print(f"  MLP analysis failed: {e}")
        import traceback; traceback.print_exc()

    final_acc = eval_relation_accuracy(model, vocab, device)
    results["final_acc"] = round(final_acc, 4)
    print(f"\n{'='*70}")
    print(f"FINAL accuracy: {final_acc:.3f}")
    print(f"{'='*70}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_1layer_dummy_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    ckpt_dir = ROOT / f"checkpoints_ravfogel_1l_dummy_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "seed": args.seed,
        "permutations": perms,
        "model_cfg": {"n_layers": 1, "hidden_dim": args.dim, "n_heads": args.n_heads},
    }, ckpt_dir / "final.pt")
    print(f"Checkpoint saved to {ckpt_dir / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()

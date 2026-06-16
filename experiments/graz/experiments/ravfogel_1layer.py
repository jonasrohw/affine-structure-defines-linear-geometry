"""
Ravfogel-style bijective factual recall with 1-layer standard transformer.

Setup:
  - 2025 subjects, 2025 objects
  - 2 relations, each a random bijection (permutation)
  - All 4050 (subject, relation) pairs in training
  - No attribute sentences — relation triples only
  - Sequence format: [sep, r_k, subj_i, obj_j] (4 tokens per fact)

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_1layer.py --seed 42
"""
import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.affine import fit_affine, apply_affine, reconstruction_accuracy
from src.config import ModelConfig
from src.model import build_model
from src.training import get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

SEQ_LENGTH = 48
REL_SENTENCE_LEN = 4


@dataclass
class RavfogelVocab:
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
    def vocab_size(self) -> int:
        return 1 + 2 * self.n_entities + self.num_relations

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
                           vocab.subject_id(subj_idx), vocab.object_id(obj_idx)])
        while len(tokens) < seq_length:
            tokens.append(vocab.sep_id)
        sequences.append(tokens[:seq_length])
    return sequences


def eval_relation_accuracy(model, vocab, device, batch_size=512):
    model.eval()
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    correct = 0
    total = len(all_pairs)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_pairs = all_pairs[start:start + batch_size]
            contexts = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)]
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
    model.eval()
    keys = []
    contexts = []
    for i in subject_indices:
        for k in range(vocab.num_relations):
            keys.append((i, k))
            contexts.append([vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)])

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
            hidden = layer_res[:, 2, :].cpu()
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

            O_hat_test = apply_affine(S_test, W, b)
            test_cos = (nn.functional.normalize(O_hat_test, dim=1) *
                        nn.functional.normalize(O_test, dim=1)).sum(1).mean().item()

            layer_results[f"r{k}"] = {
                "train_acc": round(train_acc, 4),
                "test_acc": round(test_acc, 4),
                "test_cos": round(test_cos, 4),
            }

        avg_train = np.mean([layer_results[f"r{k}"]["train_acc"] for k in range(vocab.num_relations)])
        avg_test = np.mean([layer_results[f"r{k}"]["test_acc"] for k in range(vocab.num_relations)])
        layer_results["overall"] = {
            "train_acc": round(float(avg_train), 4),
            "test_acc": round(float(avg_test), 4),
        }
        results[f"layer{layer_idx}"] = layer_results

    return results


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

    vocab = RavfogelVocab(n_entities=args.n_entities, num_relations=2)
    perms = generate_permutations(args.n_entities, 2, seed=args.seed)
    for k, p in enumerate(perms):
        vocab.set_permutation(k, p)

    model_cfg = ModelConfig(
        n_layers=1, hidden_dim=args.dim, n_heads=args.n_heads,
    )

    print(f"\n{'='*70}")
    print(f"RAVFOGEL BIJECTIVE RECALL — 1-LAYER STANDARD (seed={args.seed})")
    print(f"{'='*70}")
    print(f"Entities: {args.n_entities}, Relations: 2 (random bijections)")
    print(f"Total facts: {args.n_entities * 2}")
    print(f"Model: dim={args.dim}, layers=1, heads={args.n_heads}, WITH MLP")
    print(f"Vocab size: {vocab.vocab_size}")

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
    print(f"Affine split: train={vocab.n_train_subjects}, test={args.n_entities - vocab.n_train_subjects}")
    print(f"{'='*70}\n", flush=True)

    pre_acc = eval_relation_accuracy(model, vocab, device)
    print(f"Pre-training accuracy: {pre_acc:.3f} (chance: {1/args.n_entities:.4f})\n", flush=True)

    results = {
        "experiment": "ravfogel_1layer",
        "seed": args.seed,
        "n_entities": args.n_entities,
        "dim": args.dim,
        "n_layers": 1,
        "n_heads": args.n_heads,
        "has_mlp": True,
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

        results["epochs"][epoch] = {
            "loss": round(avg_loss, 4),
            "accuracy": round(acc, 4),
        }

        print(f"Ep {epoch:2d}/{args.n_epochs} | loss={avg_loss:.4f} | "
              f"acc={acc:.3f} | {elapsed:.0f}s", flush=True)

    # Affine analysis
    print(f"\n--- Affine analysis (nearest-neighbor accuracy) ---", flush=True)
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

    final_acc = eval_relation_accuracy(model, vocab, device)
    results["final_acc"] = round(final_acc, 4)
    print(f"\n{'='*70}")
    print(f"FINAL accuracy: {final_acc:.3f}")
    print(f"{'='*70}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_1layer_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    ckpt_dir = ROOT / f"checkpoints_ravfogel_1l_seed{args.seed}"
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

"""
Embedding probe: measure how much relational structure is in the learned embeddings.

For each relation, train a minimal model (linear or small MLP) on frozen embeddings
to predict the correct object from the subject embedding alone. No attention, no
context — just a feedforward map from subject_emb to object classification.

Probe hierarchy:
  1. Linear: W·e_subj + b → logits over objects (measures affine structure)
  2. MLP: e_subj → hidden → logits (measures nonlinear structure)
  3. Original model: 100% accuracy (the full computation)

The gap between these levels decomposes the model's computation into:
  - Affine structure in embeddings (linear probe accuracy)
  - Nonlinear structure in embeddings (MLP probe - linear probe)
  - Computation-dependent (100% - MLP probe)

Usage:
    cd graz
    CUDA_VISIBLE_DEVICES=0 python experiments/embedding_probe.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import build_model
from experiments.ravfogel_1layer import RavfogelVocab
from experiments.ravfogel_geglu import GeGLU_MLP, patch_mlp_geglu

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_entities = ckpt.get("n_entities") or len(ckpt["permutations"][0])
    vocab = RavfogelVocab(n_entities=n_entities, num_relations=2)
    for k, p in enumerate(ckpt["permutations"]):
        vocab.set_permutation(k, p)

    cfg = ckpt["model_cfg"]
    model_cfg = ModelConfig(n_layers=cfg["n_layers"], hidden_dim=cfg["hidden_dim"],
                            n_heads=cfg["n_heads"])
    model = build_model(model_cfg, vocab)

    activation = ckpt.get("activation", "gelu")
    geglu_hidden = ckpt.get("geglu_hidden", None)
    if activation == "geglu" and geglu_hidden:
        patch_mlp_geglu(model, geglu_hidden)
    elif activation == "relu":
        for block in model.blocks:
            if hasattr(block, 'mlp'):
                original_forward = block.mlp.forward
                block.mlp.forward = lambda x, fc1=block.mlp.fc1, fc2=block.mlp.fc2: fc2(F.relu(fc1(x)))

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    dim = cfg["hidden_dim"]
    return model, vocab, dim, activation, geglu_hidden


class LinearProbe(nn.Module):
    def __init__(self, input_dim, n_objects, object_embeddings):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.register_buffer("object_embs", object_embeddings)

    def forward(self, x):
        h = self.linear(x)
        return h @ self.object_embs.T


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_objects, object_embeddings):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.register_buffer("object_embs", object_embeddings)

    def forward(self, x):
        h = self.fc2(F.gelu(self.fc1(x)))
        return h @ self.object_embs.T


def train_probe(probe, subject_embs, target_indices, n_epochs=500, lr=1e-3):
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    n = subject_embs.shape[0]

    for epoch in range(1, n_epochs + 1):
        logits = probe(subject_embs)
        loss = F.cross_entropy(logits, target_indices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == n_epochs:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == target_indices).float().mean().item()
            if acc >= 1.0:
                break

    with torch.no_grad():
        logits = probe(subject_embs)
        preds = logits.argmax(dim=1)
        acc = (preds == target_indices).float().mean().item()
        loss = F.cross_entropy(logits, target_indices).item()
        top5 = (target_indices.unsqueeze(1) == logits.topk(5, dim=1).indices).any(dim=1).float().mean().item()

    return acc, top5, loss


def run_probes(model, vocab, dim, device, mlp_hidden_mult=4):
    n_ent = vocab.n_entities

    with torch.no_grad():
        subject_ids = torch.arange(n_ent, device=device) + vocab.subject_offset
        object_ids = torch.arange(n_ent, device=device) + vocab.object_offset
        subject_embs = model.token_emb(subject_ids)
        object_embs = model.token_emb(object_ids)

    results = {}

    for rel in range(vocab.num_relations):
        print(f"\n  Relation {rel}:")
        targets = torch.tensor(
            [vocab.relation_object(i, rel) for i in range(n_ent)],
            dtype=torch.long, device=device)

        # Linear probe
        torch.manual_seed(42)
        linear_probe = LinearProbe(dim, n_ent, object_embs.detach()).to(device)
        lin_acc, lin_top5, lin_loss = train_probe(
            linear_probe, subject_embs.detach(), targets, n_epochs=1000, lr=1e-3)
        print(f"    Linear:  acc={lin_acc:.4f}  top5={lin_top5:.4f}  loss={lin_loss:.4f}")

        # MLP probe (hidden = 4x dim, same as the transformer's MLP)
        mlp_hidden = dim * mlp_hidden_mult
        torch.manual_seed(42)
        mlp_probe = MLPProbe(dim, mlp_hidden, n_ent, object_embs.detach()).to(device)
        mlp_acc, mlp_top5, mlp_loss = train_probe(
            mlp_probe, subject_embs.detach(), targets, n_epochs=1000, lr=1e-3)
        print(f"    MLP:     acc={mlp_acc:.4f}  top5={mlp_top5:.4f}  loss={mlp_loss:.4f}")

        # MLP probe with larger hidden (8x dim)
        torch.manual_seed(42)
        mlp_big = MLPProbe(dim, dim * 8, n_ent, object_embs.detach()).to(device)
        big_acc, big_top5, big_loss = train_probe(
            mlp_big, subject_embs.detach(), targets, n_epochs=2000, lr=1e-3)
        print(f"    MLP-8x:  acc={big_acc:.4f}  top5={big_top5:.4f}  loss={big_loss:.4f}")

        results[f"R{rel}"] = {
            "linear": {"acc": lin_acc, "top5": lin_top5, "loss": lin_loss},
            "mlp_4x": {"acc": mlp_acc, "top5": mlp_top5, "loss": mlp_loss},
            "mlp_8x": {"acc": big_acc, "top5": big_top5, "loss": big_loss},
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    checkpoints = [
        ("GELU dim=64", ROOT / "checkpoints_ravfogel_cluster_dim64_seed42.pt"),
        ("ReLU dim=64", ROOT / "checkpoints_ravfogel_cluster_dim64_seed42_relu.pt"),
        ("GeGLU dim=64 h=170", ROOT / "checkpoints_ravfogel_cluster_dim64_seed42_geglu170.pt"),
        ("GeGLU dim=128 h=340", ROOT / "checkpoints_ravfogel_cluster_dim128_seed42_geglu340.pt"),
    ]

    all_results = {}

    for name, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"\n{'='*70}")
            print(f"SKIPPING {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")

        model, vocab, dim, activation, geglu_hidden = load_model(ckpt_path, device)
        n_ent = vocab.n_entities
        random_acc = 1.0 / n_ent
        print(f"  dim={dim}, n_entities={n_ent}, activation={activation}")
        print(f"  Random baseline: {random_acc:.6f} ({random_acc*100:.4f}%)")

        results = run_probes(model, vocab, dim, device)
        all_results[name] = results

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Rel':>3} {'Linear':>8} {'MLP-4x':>8} {'MLP-8x':>8} {'Original':>8}")
    print("-" * 70)
    for name, results in all_results.items():
        for rel_key, rel_data in results.items():
            print(f"{name:<25} {rel_key:>3} "
                  f"{rel_data['linear']['acc']:>8.4f} "
                  f"{rel_data['mlp_4x']['acc']:>8.4f} "
                  f"{rel_data['mlp_8x']['acc']:>8.4f} "
                  f"{'1.0000':>8}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "embedding_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

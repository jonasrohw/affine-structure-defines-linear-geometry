"""
Ablation studies for the Vienna experiment.

Usage (from vienna/):
    python experiments/ablations.py --study {size,depth,data,relations} [options]

Studies:
  size      — hidden_dim sweep: 32, 64, 128, 256
  depth     — n_layers sweep: 1, 2, 3, 4
  data      — training data size sweep (num_sequences)
  relations — add second relation R2(a_i) = b_{(i+50) mod 100}
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.affine import kfold_affine
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader, SequenceDataset, generate_sequence
from src.evaluation import delta_cos
from src.model import build_model
from src.representations import (
    get_all_object_representations,
    get_all_subject_representations,
    stack_representations,
)
from src.training import evaluate, train


def run_experiment(cfg: Config, device: torch.device, tag: str) -> dict:
    """Train a model and return best-layer CV delta_cos on training and heldout pairs."""
    torch.manual_seed(cfg.seed)

    train_loader = make_dataloader(cfg, split="train")
    model = build_model(cfg.model, cfg.vocab)

    history = train(
        model=model,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        val_loader=train_loader,
        verbose=False,
    )

    model.to(device)
    model.eval()

    train_idx = cfg.vocab.train_indices
    heldout_idx = cfg.vocab.heldout_indices

    subj_reprs = get_all_subject_representations(model, cfg, list(range(100)), device)
    obj_reprs = get_all_object_representations(model, cfg, list(range(100)), device)
    n_snapshots = len(subj_reprs[0])

    best_cv = -999.0
    best_layer = 0
    results_by_layer = []

    for layer in range(n_snapshots):
        S_train, _ = stack_representations(subj_reprs, train_idx, layer)
        O_train, _ = stack_representations(obj_reprs, train_idx, layer)
        S_held, _ = stack_representations(subj_reprs, heldout_idx, layer)
        O_held, _ = stack_representations(obj_reprs, heldout_idx, layer)

        mean_dc, std_dc, W_full, b_full = kfold_affine(S_train, O_train, k=5, bias=True)
        held_dc = delta_cos(S_held, O_held, W_full, b_full)
        results_by_layer.append((mean_dc, std_dc, held_dc))

        if mean_dc > best_cv:
            best_cv = mean_dc
            best_layer = layer

    # Final training accuracy
    train_metrics = evaluate(model, train_loader, device, cfg, max_batches=50)

    return {
        "tag": tag,
        "best_layer": best_layer,
        "best_cv_delta_cos": best_cv,
        "heldout_delta_cos_at_best": results_by_layer[best_layer][2],
        "train_acc_obj": train_metrics["acc_obj"],
        "results_by_layer": results_by_layer,
    }


def study_size(device, cfg_base, args):
    dims = [32, 64, 128, 256]
    results = []
    for d in dims:
        print(f"  dim={d}...")
        cfg = Config(
            vocab=cfg_base.vocab,
            model=ModelConfig(n_layers=1, hidden_dim=d, n_heads=max(1, d // 32)),
            train=cfg_base.train,
            seed=cfg_base.seed,
        )
        r = run_experiment(cfg, device, tag=f"dim={d}")
        results.append((d, r))
        print(f"    train_acc_obj={r['train_acc_obj']:.3f}  best_cv_Δcos={r['best_cv_delta_cos']:.4f}  heldout_Δcos={r['heldout_delta_cos_at_best']:.4f}")

    if not args.no_plots:
        xs = [d for d, _ in results]
        cv = [r["best_cv_delta_cos"] for _, r in results]
        held = [r["heldout_delta_cos_at_best"] for _, r in results]
        _save_ablation_plot(xs, cv, held, "Hidden Dimension", "size_ablation", cfg_base)


def study_depth(device, cfg_base, args):
    depths = [1, 2, 3, 4]
    results = []
    for d in depths:
        print(f"  n_layers={d}...")
        cfg = Config(
            vocab=cfg_base.vocab,
            model=ModelConfig(n_layers=d, hidden_dim=cfg_base.model.hidden_dim, n_heads=cfg_base.model.n_heads),
            train=cfg_base.train,
            seed=cfg_base.seed,
        )
        r = run_experiment(cfg, device, tag=f"layers={d}")
        results.append((d, r))
        print(f"    train_acc_obj={r['train_acc_obj']:.3f}  best_cv_Δcos={r['best_cv_delta_cos']:.4f}  best_layer={r['best_layer']}")

    if not args.no_plots:
        xs = [d for d, _ in results]
        cv = [r["best_cv_delta_cos"] for _, r in results]
        held = [r["heldout_delta_cos_at_best"] for _, r in results]
        _save_ablation_plot(xs, cv, held, "Number of Layers", "depth_ablation", cfg_base)


def study_data(device, cfg_base, args):
    sizes = [500, 1000, 2000, 5000, 10000]
    results = []
    for n in sizes:
        print(f"  num_sequences={n}...")
        train_cfg = TrainConfig(
            num_sequences=n,
            seq_length=cfg_base.train.seq_length,
            batch_size=cfg_base.train.batch_size,
            max_epochs=cfg_base.train.max_epochs,
            learning_rate=cfg_base.train.learning_rate,
        )
        cfg = Config(vocab=cfg_base.vocab, model=cfg_base.model, train=train_cfg, seed=cfg_base.seed)
        r = run_experiment(cfg, device, tag=f"n={n}")
        results.append((n, r))
        print(f"    train_acc_obj={r['train_acc_obj']:.3f}  best_cv_Δcos={r['best_cv_delta_cos']:.4f}")

    if not args.no_plots:
        xs = [n for n, _ in results]
        cv = [r["best_cv_delta_cos"] for _, r in results]
        held = [r["heldout_delta_cos_at_best"] for _, r in results]
        _save_ablation_plot(xs, cv, held, "Number of Training Sequences", "data_ablation", cfg_base)


def study_relations(device, cfg_base, args):
    """Add a second relation R2(a_i) = b_{(i+50) mod 100} and check if linearity still holds."""
    import random
    from torch.utils.data import DataLoader

    print("  Training with two relations (R1: a_i->b_i, R2: a_i->b_{(i+50)%100})...")

    vocab = cfg_base.vocab

    class TwoRelationDataset:
        def __init__(self, num_sequences, seq_length, pair_indices, seed):
            rng = random.Random(seed)
            import torch
            self.sequences = []
            for _ in range(num_sequences):
                tokens = []
                n_triplets = seq_length // 3
                for _ in range(n_triplets):
                    i = rng.choice(pair_indices)
                    relation = rng.choice([0, 1])
                    j = i if relation == 0 else (i + 50) % 100
                    tokens.append(vocab.sep_id)
                    tokens.append(vocab.subject_id(i))
                    tokens.append(vocab.object_id(j))
                self.sequences.append(tokens)

        def __len__(self): return len(self.sequences)
        def __getitem__(self, idx):
            import torch
            t = self.sequences[idx]
            return torch.tensor(t[:-1], dtype=torch.long), torch.tensor(t[1:], dtype=torch.long)

    ds = TwoRelationDataset(
        cfg_base.train.num_sequences,
        cfg_base.train.seq_length,
        cfg_base.vocab.train_indices,
        cfg_base.seed,
    )
    loader = DataLoader(ds, batch_size=cfg_base.train.batch_size, shuffle=True, drop_last=True)

    model = build_model(cfg_base.model, vocab)
    torch.manual_seed(cfg_base.seed)
    history = train(model, loader, cfg_base, device, val_loader=loader, verbose=True)

    model.to(device)
    model.eval()
    subj_reprs = get_all_subject_representations(model, cfg_base, cfg_base.vocab.train_indices, device)
    obj_reprs_r1 = get_all_object_representations(model, cfg_base, cfg_base.vocab.train_indices, device)

    print("\nPer-layer CV Δcos (R1: a_i -> b_i, two-relation model):")
    n_snapshots = len(subj_reprs[0])
    layer_labels = ["emb"] + [f"L{i+1}" for i in range(n_snapshots - 1)]
    for layer in range(n_snapshots):
        S, _ = stack_representations(subj_reprs, cfg_base.vocab.train_indices, layer)
        O, _ = stack_representations(obj_reprs_r1, cfg_base.vocab.train_indices, layer)
        mean_dc, std_dc, _, _ = kfold_affine(S, O, k=5, bias=True)
        print(f"  {layer_labels[layer]}: {mean_dc:.4f} ± {std_dc:.4f}")


def _save_ablation_plot(xs, cv, held, xlabel, name, cfg):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, cv, "o-", label="CV Δcos (train pairs)", color="steelblue")
    ax.plot(xs, held, "s--", label="Δcos (heldout pairs)", color="tomato")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Δcos")
    ax.set_title(f"Ablation: {xlabel}")
    ax.legend()
    fig.tight_layout()
    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.train.results_dir / f"{name}.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=["size", "depth", "data", "relations"], required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg_base = Config(
        train=TrainConfig(max_epochs=args.epochs),
        seed=args.seed,
    )

    print(f"Study: {args.study} | Device: {device}")
    studies = {
        "size": study_size,
        "depth": study_depth,
        "data": study_data,
        "relations": study_relations,
    }
    studies[args.study](device, cfg_base, args)


if __name__ == "__main__":
    main()

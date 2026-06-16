"""
Train the baseline model.

Usage (from vienna/):
    python experiments/train_baseline.py [--layers N] [--dim D] [--epochs E] [--seed S]

Trains on 80 subject-object pairs (a0-a79), evaluates separately on:
  - train pairs (a0-a79): should reach >99% accuracy
  - heldout pairs (a80-a99): verifies the model hasn't "seen" these

Saves best checkpoint to checkpoints/best.pt.
"""
import argparse
import sys
from pathlib import Path

# Allow running from experiments/ or from vienna/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.model import build_model
from src.training import evaluate, load_checkpoint, train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--num_seq", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = Config(
        vocab=VocabConfig(),
        model=ModelConfig(
            n_layers=args.layers,
            hidden_dim=args.dim,
            n_heads=args.heads,
        ),
        train=TrainConfig(
            max_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch,
            seq_length=args.seq_len,
            num_sequences=args.num_seq,
        ),
        seed=args.seed,
    )

    torch.manual_seed(cfg.seed)

    print(f"Device: {device}")
    print(f"Model: {args.layers} layers, dim={args.dim}, heads={args.heads}")
    print(f"Training pairs: {cfg.vocab.n_train_pairs} (a0-a{cfg.vocab.n_train_pairs-1})")
    print(f"Held-out pairs: {cfg.vocab.n_heldout_pairs} (a{cfg.vocab.n_train_pairs}-a99)")
    print()

    train_loader = make_dataloader(cfg, split="train", seed_offset=0)

    model = build_model(cfg.model, cfg.vocab)
    print(f"Parameters: {model.n_parameters():,}")
    print()

    history = train(
        model=model,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        val_loader=train_loader,
        verbose=True,
    )

    # Final evaluation
    print("\n--- Final Evaluation ---")
    best_ckpt = cfg.train.checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, device)
        print(f"Loaded best checkpoint from {best_ckpt}")

    model.to(device)
    train_metrics = evaluate(model, train_loader, device, cfg, max_batches=50)

    print(f"Train — loss={train_metrics['loss']:.4f} | acc_obj={train_metrics['acc_obj']:.4f} | acc_sep={train_metrics['acc_sep']:.4f}")

    if train_metrics["acc_obj"] >= 0.99:
        print("\n✓ Training success: >99% object prediction accuracy on training pairs.")
    else:
        print(f"\n✗ Object prediction accuracy {train_metrics['acc_obj']:.3f} below 99% target.")
        print("  Try more epochs or a larger model.")


if __name__ == "__main__":
    main()

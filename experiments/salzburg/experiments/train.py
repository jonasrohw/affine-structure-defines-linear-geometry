"""
Train the Salzburg model (no fillers, relation before subject).

Sequence format: x r_k a_i b_j  (4 tokens per triple, 10 triples per sequence)

Usage (from salzburg/):
    python experiments/train.py [--layers N] [--dim D] [--epochs E]
"""
import argparse
import sys
from pathlib import Path

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
    p.add_argument("--seq_len", type=int, default=40)
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
            checkpoint_dir=Path(__file__).parent.parent / "checkpoints",
        ),
        seed=args.seed,
    )

    torch.manual_seed(cfg.seed)

    print(f"Device: {device}")
    print(f"Model: {args.layers} layer(s), dim={args.dim}, heads={args.heads}")
    print(f"Vocab size: {cfg.vocab.vocab_size} tokens (no fillers)")
    print(f"Relations: R0=identity, R1=+50 cyclic, R2=hard-coded permutation")
    print(f"Sequence format: x r_k a_i b_j  ({args.seq_len // 4} triples/seq)")
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

    best_ckpt = cfg.train.checkpoint_dir / "best.pt"
    print("\n--- Final Evaluation ---")
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, device)
        print(f"Loaded best checkpoint from {best_ckpt}")

    model.to(device)
    train_metrics = evaluate(model, train_loader, device, cfg, max_batches=50)

    print(
        f"acc_obj={train_metrics['acc_obj']:.4f} | "
        f"acc_rel={train_metrics['acc_rel']:.4f} | "
        f"loss={train_metrics['loss']:.4f}"
    )

    if train_metrics["acc_obj"] >= 0.99:
        print("\nTraining success: >99% object prediction accuracy.")
    else:
        print(f"\nObject accuracy {train_metrics['acc_obj']:.3f} below 99% target.")
        print("  Try more epochs (--epochs 400) or a larger model (--dim 256).")


if __name__ == "__main__":
    main()

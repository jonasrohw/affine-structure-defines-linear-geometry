"""Command-line interface for configurable factual recall experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config_loader import load_experiment_config
from .datasets import make_dataloader
from .model import build_model
from .registry import known_configs
from .training import evaluate, load_checkpoint, train


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cmd_list_configs(_: argparse.Namespace) -> None:
    print(json.dumps(known_configs(), indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_experiment_config(args.config)
    device = resolve_device(args.device)
    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, split="train", seed_offset=0)
    model = build_model(cfg.model, cfg.vocab)

    print(f"Device: {device}")
    print(
        "Model: "
        f"{cfg.model.n_layers} layer(s), dim={cfg.model.hidden_dim}, heads={cfg.model.n_heads}"
    )
    print(f"Dataset: vocab={cfg.vocab.vocab_size}, relations={cfg.vocab.relation_names()}")
    print(f"Parameters: {model.n_parameters():,}")

    if args.dry_run:
        return

    history = train(
        model=model,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        val_loader=train_loader,
        verbose=not args.quiet,
    )
    best_ckpt = cfg.train.checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, device)
    metrics = evaluate(model, train_loader, device, cfg, max_batches=args.eval_batches)
    print(json.dumps({"history": history, "final_metrics": metrics}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="factual-recall")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--eval-batches", type=int, default=20)
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument("--quiet", action="store_true")
    train_parser.set_defaults(func=cmd_train)

    list_parser = sub.add_parser("list-configs")
    list_parser.set_defaults(func=cmd_list_configs)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

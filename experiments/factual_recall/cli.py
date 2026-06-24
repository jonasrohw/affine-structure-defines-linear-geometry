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
from .synthetic_curriculum import (
    load_synthetic_config,
    run_experiment1,
    run_experiment2,
    run_frequency_contrast,
    run_mixed_relations,
)
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


def cmd_synthetic_exp1(args: argparse.Namespace) -> None:
    cfg = load_synthetic_config(args.config)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(
        "Synthetic Experiment 1: "
        f"entities={cfg.vocab.n_entities}, phase1_k={cfg.curriculum.phase1_k}, "
        f"phase2_triples={cfg.curriculum.phase2_triples}"
    )
    print(
        "Model: "
        f"{cfg.model.n_layers} layer(s), dim={cfg.model.hidden_dim}, heads={cfg.model.n_heads}"
    )
    result = run_experiment1(cfg=cfg, device=device, quiet=args.quiet)
    final = result["history"][-1]
    summary = {
        "result_path": result["result_path"],
        "plot_path": result["plot_path"],
        "report_path": result["report_path"],
        "num_parameters": result["num_parameters"],
        "phase_shift_summary": result["summary"],
        "final": final,
    }
    print(json.dumps(summary, indent=2))


def cmd_synthetic_exp2(args: argparse.Namespace) -> None:
    cfg = load_synthetic_config(args.config)
    print(
        "Synthetic Experiment 2: "
        f"entities={cfg.vocab.n_entities}, matched_train_pairs={cfg.curriculum.phase1_k}, "
        f"eval_limit={cfg.curriculum.eval_limit}"
    )
    result = run_experiment2(cfg=cfg, quiet=args.quiet)
    summary = {
        "result_path": result["result_path"],
        "matched_train_pairs": result["matched_train_pairs"],
        "matched_eval_pairs": result["matched_eval_pairs"],
        "relations": result["relations"],
    }
    print(json.dumps(summary, indent=2))


def cmd_synthetic_frequency_contrast(args: argparse.Namespace) -> None:
    cfg = load_synthetic_config(args.config)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(
        "Synthetic frequency contrast: "
        f"entities={cfg.vocab.n_entities}, train_subjects={cfg.curriculum.phase2_triples}, "
        f"low_repeats={cfg.curriculum.phase2_repeats}, high_repeats={cfg.curriculum.phase1_triple_repeats}"
    )
    result = run_frequency_contrast(cfg=cfg, device=device, quiet=args.quiet)
    summary = {
        "result_path": result["result_path"],
        "low_frequency_repeats": result["low_frequency_repeats"],
        "high_frequency_repeats": result["high_frequency_repeats"],
        "relations": result["relations"],
    }
    print(json.dumps(summary, indent=2))


def cmd_synthetic_mixed_relations(args: argparse.Namespace) -> None:
    cfg = load_synthetic_config(args.config)
    device = resolve_device(args.device)
    structured_repeats = (
        cfg.curriculum.structured_relation_repeats
        if cfg.curriculum.structured_relation_repeats is not None
        else cfg.curriculum.phase2_repeats
    )
    random_repeats = (
        cfg.curriculum.random_relation_repeats
        if cfg.curriculum.random_relation_repeats is not None
        else cfg.curriculum.phase1_triple_repeats
    )
    print(f"Device: {device}")
    print(
        "Synthetic mixed relations: "
        f"entities={cfg.vocab.n_entities}, train_subjects={cfg.curriculum.phase2_triples}, "
        f"structured_repeats={structured_repeats}, random_repeats={random_repeats}"
    )
    result = run_mixed_relations(cfg=cfg, device=device, quiet=args.quiet)
    summary = {
        "result_path": result["result_path"],
        "structured_repeats": result["structured_repeats"],
        "random_repeats": result["random_repeats"],
        "relations": result["relations"],
        "mechanistic": result["mechanistic"],
    }
    print(json.dumps(summary, indent=2))


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

    synthetic_parser = sub.add_parser("synthetic-exp1")
    synthetic_parser.add_argument("--config", type=Path, required=True)
    synthetic_parser.add_argument("--device", default="auto")
    synthetic_parser.add_argument("--quiet", action="store_true")
    synthetic_parser.set_defaults(func=cmd_synthetic_exp1)

    synthetic2_parser = sub.add_parser("synthetic-exp2")
    synthetic2_parser.add_argument("--config", type=Path, required=True)
    synthetic2_parser.add_argument("--quiet", action="store_true")
    synthetic2_parser.set_defaults(func=cmd_synthetic_exp2)

    contrast_parser = sub.add_parser("synthetic-frequency-contrast")
    contrast_parser.add_argument("--config", type=Path, required=True)
    contrast_parser.add_argument("--device", default="auto")
    contrast_parser.add_argument("--quiet", action="store_true")
    contrast_parser.set_defaults(func=cmd_synthetic_frequency_contrast)

    mixed_parser = sub.add_parser("synthetic-mixed-relations")
    mixed_parser.add_argument("--config", type=Path, required=True)
    mixed_parser.add_argument("--device", default="auto")
    mixed_parser.add_argument("--quiet", action="store_true")
    mixed_parser.set_defaults(func=cmd_synthetic_mixed_relations)

    list_parser = sub.add_parser("list-configs")
    list_parser.set_defaults(func=cmd_list_configs)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

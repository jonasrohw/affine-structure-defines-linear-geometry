"""
One-epoch relation learning — Phase 1: attribute-only pretraining.

Train on forward attribute + reverse short sentences only (no relations,
no contrastive). Save checkpoint for phase 2 to load.

Uses the 6-feature setup: k=2, 15 types, 225 entities, dim=16.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/one_epoch_phase1.py --seed 42
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.model import build_model
from src.training import build_valid_attr_table, get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_REVERSE_SHORT = 0.30
MAX_EPOCHS = 50
NUM_SEQUENCES = 62500
EVAL_EVERY = 10
N_ENTITIES = 225
HOLDOUT_FRACTION = 0.20


def make_held_out_triples(vocab: VocabConfig, fraction: float, seed: int):
    all_triples = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    rng = random.Random(seed)
    rng.shuffle(all_triples)
    n_held = int(len(all_triples) * fraction)

    held_out = []
    subject_remaining = {i: vocab.num_relations for i in range(vocab.n_entities)}
    for triple in all_triples:
        if len(held_out) >= n_held:
            break
        subj_idx, _ = triple
        if subject_remaining[subj_idx] > 1:
            held_out.append(triple)
            subject_remaining[subj_idx] -= 1
    return held_out


def eval_attr_accuracy(model, cfg, device, valid_table, n_batches=20):
    """Evaluate attribute prediction accuracy."""
    model.eval()
    loader = make_dataloader(cfg, seed_offset=9999, p_relation=0.0,
                             p_reverse_short=P_REVERSE_SHORT)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            x_entity = (x >= cfg.vocab.subject_offset) & (x < cfg.vocab.f_attr_offset)
            y_attr = (y >= cfg.vocab.f_attr_offset) & (y < cfg.vocab.relation_offset)
            attr_pos = x_entity & y_attr
            if attr_pos.sum() > 0:
                preds = logits[attr_pos].argmax(dim=-1)
                entities = x[attr_pos]
                correct += valid_table[entities, preds].sum().item()
                total += attr_pos.sum().item()
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    vocab_kwargs = dict(
        n_f_attrs=6, n_g_attrs=6, k_per_group=2,
        num_relations=2, n_entities=N_ENTITIES,
        n_train_subjects=N_ENTITIES, r0_shift=3, r1_shift=7,
    )

    held_out = make_held_out_triples(VocabConfig(**vocab_kwargs), HOLDOUT_FRACTION, seed)

    ckpt_dir = ROOT / f"checkpoints_one_epoch_p1_seed{seed}"

    cfg = Config(
        vocab=VocabConfig(**vocab_kwargs, held_out_triples=held_out),
        model=ModelConfig(n_layers=1, hidden_dim=16, n_heads=4),
        train=TrainConfig(
            max_epochs=MAX_EPOCHS,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=48,
            num_sequences=NUM_SEQUENCES,
            checkpoint_dir=ckpt_dir,
            results_dir=RESULTS_DIR,
        ),
        seed=seed,
    )

    print("=" * 70)
    print(f"ONE-EPOCH Phase 1: Attribute-only pretraining (seed={seed})")
    print(f"6 features, k=2, 15 types, {N_ENTITIES} entities, dim=16")
    print(f"p_reverse_short={P_REVERSE_SHORT}, p_attr={1.0 - P_REVERSE_SHORT:.2f}")
    print(f"{MAX_EPOCHS} epochs, {NUM_SEQUENCES} sequences/epoch")
    print(f"Held-out triples: {len(held_out)} (for phase 2 evaluation)")
    print("=" * 70)
    print(flush=True)

    torch.manual_seed(seed)
    train_loader = make_dataloader(cfg, seed_offset=0, p_relation=0.0,
                                   p_reverse_short=P_REVERSE_SHORT)
    model = build_model(cfg.model, cfg.vocab)
    model.to(device)
    model.train()
    print(f"Parameters: {model.n_parameters():,}", flush=True)

    valid_table = build_valid_attr_table(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * MAX_EPOCHS

    trajectory = []
    global_step = 0
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            lr = get_lr(global_step, cfg.train.warmup_steps, max_steps, cfg.train.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits = model(x)
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0 or epoch == MAX_EPOCHS - 1:
            attr_acc = eval_attr_accuracy(model, cfg, device, valid_table)

            snapshot = {
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 4),
                "attr_acc": round(attr_acc, 4),
            }
            trajectory.append(snapshot)

            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:4d}/{MAX_EPOCHS} | "
                  f"loss={avg_loss:.4f} | attr_acc={attr_acc:.3f} | "
                  f"lr={lr:.2e} | {elapsed:.0f}s", flush=True)
            model.train()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"
    torch.save({
        "epoch": MAX_EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "cfg_seed": seed,
        "held_out_triples": held_out,
        "vocab_kwargs": vocab_kwargs,
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"one_epoch_phase1_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "seed": seed,
            "config": {
                "n_f_attrs": 6, "n_entities": N_ENTITIES, "dim": 16,
                "p_reverse_short": P_REVERSE_SHORT,
                "max_epochs": MAX_EPOCHS, "num_sequences": NUM_SEQUENCES,
                "n_held_out": len(held_out),
            },
            "trajectory": trajectory,
        }, f, indent=2)
    print(f"Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

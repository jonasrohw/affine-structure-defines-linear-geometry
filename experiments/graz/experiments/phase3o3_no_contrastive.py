"""
Phase 3o3_no_contrastive: ablation — no contrastive sentences, 100 epochs.

Same setup as phase3o3 (5 features, k=2, 100 entities, dim=16) but
p_contrastive=0.0. The 20% contrastive budget goes to attribute sentences.

Usage (from graz/):
    CUDA_VISIBLE_DEVICES=2 python experiments/phase3o3_no_contrastive.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import random
import time

import torch
import torch.nn as nn

from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.model import build_model
from src.training import build_valid_attr_table, get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_RELATION = 0.264
P_CONTRASTIVE = 0.0
P_REVERSE_SHORT = 0.264
HOLDOUT_FRACTION = 0.20
SEED = 42
NUM_SEQUENCES = 62500
MAX_EPOCHS = 100
EVAL_EVERY = 10


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


def eval_relation_accuracy(model, vocab, device, triples):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for subj_idx, rel_k in triples:
            context = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(subj_idx)]],
                dtype=torch.long, device=device,
            )
            logits = model(context)
            pred = logits[0, -1].argmax().item()
            valid_targets = [vocab.object_id(j) for j in vocab.relation_objects(subj_idx, rel_k)]
            correct += int(pred in valid_targets)
            total += 1
    return correct / max(1, total)


def main():
    n_f_attrs = 5
    n_g_attrs = 5
    k_per_group = 2
    n_entities = 100
    r0_shift = 3
    r1_shift = 7

    vocab_kwargs = dict(
        n_f_attrs=n_f_attrs, n_g_attrs=n_g_attrs, k_per_group=k_per_group,
        num_relations=2, n_entities=n_entities,
        n_train_subjects=n_entities, r0_shift=r0_shift, r1_shift=r1_shift,
    )

    held_out = make_held_out_triples(VocabConfig(**vocab_kwargs), HOLDOUT_FRACTION, SEED)

    cfg = Config(
        vocab=VocabConfig(**vocab_kwargs, held_out_triples=held_out),
        model=ModelConfig(n_layers=1, hidden_dim=16, n_heads=4),
        train=TrainConfig(
            max_epochs=MAX_EPOCHS,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=48,
            num_sequences=NUM_SEQUENCES,
            checkpoint_dir=ROOT / "checkpoints_no_contrastive",
            results_dir=RESULTS_DIR,
        ),
        seed=SEED,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_triples = cfg.vocab.train_triples
    held_out_triples = held_out
    n_train = len(train_triples)
    n_held = len(held_out_triples)
    n_valid = len(cfg.vocab.relation_objects(0, 0))
    chance = float(n_valid) / n_entities

    p_attr = 1.0 - P_RELATION - P_CONTRASTIVE - P_REVERSE_SHORT

    print("=" * 70)
    print("GRAZ Phase 3o3 NO CONTRASTIVE: 100 epochs, seed=42")
    print(f"5 features, dim=16, k=2 (10 types, {n_entities} entities)")
    print(f"p_relation={P_RELATION}, p_contrastive={P_CONTRASTIVE}, "
          f"p_reverse_short={P_REVERSE_SHORT}, p_attr={p_attr:.2f}")
    print(f"Train: {n_train} triples, Held out: {n_held} triples")
    print(f"Chance level: {chance:.2%}")
    print("=" * 70)
    print()

    torch.manual_seed(SEED)
    train_loader = make_dataloader(cfg, seed_offset=0, p_relation=P_RELATION,
                                   train_triples=train_triples,
                                   p_reverse_short=P_REVERSE_SHORT,
                                   p_contrastive=P_CONTRASTIVE)
    model = build_model(cfg.model, cfg.vocab)
    model.to(device)
    model.train()
    print(f"Parameters: {model.n_parameters():,}")

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

        avg_train_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            train_rel_acc = eval_relation_accuracy(model, cfg.vocab, device, train_triples)
            heldout_rel_acc = eval_relation_accuracy(model, cfg.vocab, device, held_out_triples)

            snapshot = {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 4),
                "train_rel_acc": round(train_rel_acc, 4),
                "heldout_rel_acc": round(heldout_rel_acc, 4),
            }
            trajectory.append(snapshot)

            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
                  f"loss={avg_train_loss:.4f} | "
                  f"train_rel={train_rel_acc:.3f} | "
                  f"heldout_rel={heldout_rel_acc:.3f} | "
                  f"{elapsed:.0f}s")

            model.train()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase3o3_no_contrastive.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_entities": n_entities, "dim": 16, "p_contrastive": P_CONTRASTIVE,
                "n_train": n_train, "n_held": n_held, "chance": chance,
                "num_sequences": NUM_SEQUENCES, "max_epochs": MAX_EPOCHS, "seed": SEED,
            },
            "trajectory": trajectory,
        }, f, indent=2)
    print(f"\nTrajectory saved to {out_path}")

    # Markdown
    md_path = RESULTS_DIR / "phase3o3_no_contrastive_results.md"
    with open(md_path, "w") as f:
        f.write("# Phase 3o3 No Contrastive: Ablation\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Same as phase3o3 but p_contrastive=0.0\n")
        f.write(f"- p_relation={P_RELATION}, p_reverse_short={P_REVERSE_SHORT}, p_attr={p_attr:.2f}\n")
        f.write(f"- {n_entities} entities, dim=16, seed={SEED}, {MAX_EPOCHS} epochs\n")
        f.write(f"- Train: {n_train} triples, Held out: {n_held} triples\n")
        f.write(f"- Chance level: {chance:.2%}\n\n")

        f.write("## Trajectory\n\n")
        f.write("| Epoch | Train Loss | Train Rel | Held-out Rel |\n")
        f.write("|-------|------------|-----------|-------------|\n")
        for s in trajectory:
            f.write(f"| {s['epoch']:>5} | {s['train_loss']:.4f} | "
                    f"{s['train_rel_acc']:.3f} | {s['heldout_rel_acc']:.3f} |\n")

        f.write("\n## Comparison with contrastive (seed=42, first 100 epochs)\n\n")
        f.write("(To be filled after reviewing results.)\n")

    print(f"Report saved to {md_path}")


if __name__ == "__main__":
    main()

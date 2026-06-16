"""
Phase 3o3_compute: compute scaling — track generalization over training.

Same setup as phase3o3 (5 features, k=2, 100 entities, dim=16).
Evaluates held-out relation accuracy every 10 epochs to track the
memorization→generalization trajectory.

Usage (from graz/):
    python experiments/phase3o3_compute.py
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
from torch.utils.data import DataLoader

from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.model import build_model
from src.training import build_valid_attr_table, evaluate, get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_RELATION = 0.264
P_CONTRASTIVE = 0.20
P_REVERSE_SHORT = 0.264
HOLDOUT_FRACTION = 0.20
SEED = 42
NUM_SEQUENCES = 62500
EVAL_EVERY = 10


def make_held_out_triples(vocab: VocabConfig, fraction: float, seed: int):
    """Hold out a random fraction of triples, ensuring every subject keeps ≥1."""
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
    """Evaluate relation accuracy on a set of (subject_idx, rel_k) triples."""
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


def eval_contrastive_accuracy(model, vocab, device, n_eval=200, seed=456):
    """Evaluate contrastive accuracy (positive and negative)."""
    model.eval()
    rng = random.Random(seed)
    correct_pos = 0
    correct_neg = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_eval):
            subj_idx = rng.randint(0, vocab.n_entities - 1)
            rel_k = rng.randint(0, vocab.num_relations - 1)
            valid_objects = set(vocab.relation_objects(subj_idx, rel_k))

            obj_idx = rng.choice(list(valid_objects))
            context = torch.tensor(
                [[vocab.sep_id, vocab.query_id(rel_k),
                  vocab.object_id(obj_idx), vocab.subject_id(subj_idx)]],
                dtype=torch.long, device=device)
            pred = model(context)[0, -1].argmax().item()
            correct_pos += int(pred == vocab.y_id)

            invalid = [j for j in range(vocab.n_entities) if j not in valid_objects]
            obj_idx = rng.choice(invalid)
            context = torch.tensor(
                [[vocab.sep_id, vocab.query_id(rel_k),
                  vocab.object_id(obj_idx), vocab.subject_id(subj_idx)]],
                dtype=torch.long, device=device)
            pred = model(context)[0, -1].argmax().item()
            correct_neg += int(pred == vocab.n_id)

            total += 1
    return correct_pos / total, correct_neg / total


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

    held_out = make_held_out_triples(
        VocabConfig(**vocab_kwargs),
        HOLDOUT_FRACTION, SEED,
    )

    cfg = Config(
        vocab=VocabConfig(**vocab_kwargs, held_out_triples=held_out),
        model=ModelConfig(n_layers=1, hidden_dim=16, n_heads=4),
        train=TrainConfig(
            max_epochs=800,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=48,
            num_sequences=NUM_SEQUENCES,
            checkpoint_dir=ROOT / "checkpoints_phase3o3_compute",
            results_dir=RESULTS_DIR,
        ),
        seed=SEED,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_triples = cfg.vocab.train_triples
    n_total = len(cfg.vocab.all_triples)
    n_held = len(held_out)
    n_train = len(train_triples)
    n_types = cfg.vocab.n_types
    n_valid = len(cfg.vocab.relation_objects(0, 0))
    chance = float(n_valid) / n_entities

    print("=" * 70)
    print(f"GRAZ Phase 3o3_compute: generalization trajectory")
    print(f"5 features, dim=16, k=2 ({n_types} types, {n_entities} entities)")
    print(f"Total triples: {n_total} | Train: {n_train} | Held out: {n_held}")
    print(f"Evaluating held-out accuracy every {EVAL_EVERY} epochs")
    print(f"Chance level: {chance:.2%}")
    print("=" * 70)
    print()

    # Build data and model
    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, seed_offset=0, p_relation=P_RELATION,
                                   train_triples=train_triples,
                                   p_reverse_short=P_REVERSE_SHORT,
                                   p_contrastive=P_CONTRASTIVE)
    model = build_model(cfg.model, cfg.vocab)
    model.to(device)
    model.train()
    print(f"Parameters: {model.n_parameters():,}")

    valid_table = build_valid_attr_table(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * cfg.train.max_epochs

    # Trajectory log
    trajectory = []

    global_step = 0
    t0 = time.time()

    for epoch in range(cfg.train.max_epochs):
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

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0 or epoch == cfg.train.max_epochs - 1:
            train_rel_acc = eval_relation_accuracy(model, cfg.vocab, device, train_triples)
            heldout_rel_acc = eval_relation_accuracy(model, cfg.vocab, device, held_out)
            contr_pos, contr_neg = eval_contrastive_accuracy(model, cfg.vocab, device)
            val_metrics = evaluate(model, train_loader, device, cfg, valid_table, max_batches=20)

            snapshot = {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 4),
                "train_rel_acc": round(train_rel_acc, 4),
                "heldout_rel_acc": round(heldout_rel_acc, 4),
                "contrastive_pos": round(contr_pos, 4),
                "contrastive_neg": round(contr_neg, 4),
                "attr_valid": round(val_metrics["acc_attr_valid"], 4),
                "elapsed_s": round(time.time() - t0, 1),
            }
            trajectory.append(snapshot)

            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:4d}/{cfg.train.max_epochs} | "
                  f"loss={avg_train_loss:.4f} | "
                  f"train_rel={train_rel_acc:.3f} | "
                  f"heldout_rel={heldout_rel_acc:.3f} | "
                  f"contr=+{contr_pos:.2f}/-{contr_neg:.2f} | "
                  f"attr={val_metrics['acc_attr_valid']:.3f} | "
                  f"{elapsed:.0f}s")

            model.train()

    # Save trajectory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    traj_path = RESULTS_DIR / "phase3o3_compute_trajectory.json"
    with open(traj_path, "w") as f:
        json.dump({
            "config": {
                "n_entities": n_entities, "n_types": n_types, "dim": 16,
                "n_train": n_train, "n_held": n_held, "chance": chance,
                "num_sequences": NUM_SEQUENCES, "max_epochs": cfg.train.max_epochs,
            },
            "trajectory": trajectory,
        }, f, indent=2)
    print(f"\nTrajectory saved to {traj_path}")

    # Save markdown report
    results_path = RESULTS_DIR / "phase3o3_compute_results.md"
    with open(results_path, "w") as f:
        f.write("# Phase 3o3_compute: Generalization Trajectory\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Same as phase3o3: 5 features, k=2, {n_entities} entities, dim=16\n")
        f.write(f"- Train: {n_train} triples, Held out: {n_held} triples\n")
        f.write(f"- Chance level: {chance:.2%}\n")
        f.write(f"- Evaluated every {EVAL_EVERY} epochs\n\n")

        f.write("## Trajectory\n\n")
        f.write("| Epoch | Train Loss | Train Rel | Held-out Rel | Contr +/- | Attr Valid |\n")
        f.write("|-------|------------|-----------|--------------|-----------|------------|\n")
        for s in trajectory:
            f.write(f"| {s['epoch']:4d} | {s['train_loss']:.4f} | "
                    f"{s['train_rel_acc']:.3f} | {s['heldout_rel_acc']:.3f} | "
                    f"{s['contrastive_pos']:.2f}/{s['contrastive_neg']:.2f} | "
                    f"{s['attr_valid']:.3f} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("(To be filled after reviewing results.)\n")

    print(f"Report saved to {results_path}")


if __name__ == "__main__":
    main()

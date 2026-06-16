"""
Phase 3o3 with 6 features, no contrastive, per-epoch affine probe.

6 features, k=2 → C(6,2)=15 types → 225 entities.
Tracks held-out relation accuracy and affine train/test at every epoch.
Single seed per invocation for parallel execution.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/phase3o3_6feat_no_contr_affine.py --seed 42
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

from src.affine import affine_analysis
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.model import build_model
from src.training import get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_RELATION = 0.264
P_CONTRASTIVE = 0.0
P_REVERSE_SHORT = 0.264
HOLDOUT_FRACTION = 0.20
NUM_SEQUENCES = 62500
MAX_EPOCHS = 100
N_ENTITIES = 225


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


def run_seed(seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", f"CUDA not available! CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    vocab_kwargs = dict(
        n_f_attrs=6, n_g_attrs=6, k_per_group=2,
        num_relations=2, n_entities=N_ENTITIES,
        n_train_subjects=N_ENTITIES, r0_shift=3, r1_shift=7,
    )

    held_out = make_held_out_triples(VocabConfig(**vocab_kwargs), HOLDOUT_FRACTION, seed)

    cfg = Config(
        vocab=VocabConfig(**vocab_kwargs, held_out_triples=held_out),
        model=ModelConfig(n_layers=1, hidden_dim=16, n_heads=4),
        train=TrainConfig(
            max_epochs=MAX_EPOCHS,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=48,
            num_sequences=NUM_SEQUENCES,
            checkpoint_dir=ROOT / f"checkpoints_6feat_no_contr_seed{seed}",
            results_dir=RESULTS_DIR,
        ),
        seed=seed,
    )

    cfg_affine = Config(
        vocab=VocabConfig(**{**vocab_kwargs, 'n_train_subjects': N_ENTITIES // 2}),
        model=cfg.model,
        train=cfg.train,
        seed=seed,
    )

    train_triples = cfg.vocab.train_triples
    n_train = len(train_triples)
    n_held = len(held_out)
    n_valid = len(cfg.vocab.relation_objects(0, 0))
    chance = float(n_valid) / N_ENTITIES

    print(f"seed={seed} | 6 feat, no contrastive | train={n_train}, held={n_held} | "
          f"chance={chance:.2%} | {MAX_EPOCHS} epochs", flush=True)

    torch.manual_seed(seed)
    train_loader = make_dataloader(cfg, seed_offset=0, p_relation=P_RELATION,
                                   train_triples=train_triples,
                                   p_reverse_short=P_REVERSE_SHORT,
                                   p_contrastive=P_CONTRASTIVE)
    model = build_model(cfg.model, cfg.vocab)
    model.to(device)
    model.train()
    print(f"Parameters: {model.n_parameters():,}", flush=True)

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

        heldout_rel_acc = eval_relation_accuracy(model, cfg.vocab, device, held_out)

        affine_res = affine_analysis(model, cfg_affine, device, layer=-1)
        affine_train = affine_res["overall"]["train_acc"]
        affine_test = affine_res["overall"]["test_acc"]

        snapshot = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "heldout_rel_acc": round(heldout_rel_acc, 4),
            "affine_train_acc": round(affine_train, 4),
            "affine_test_acc": round(affine_test, 4),
        }
        trajectory.append(snapshot)

        elapsed = time.time() - t0
        print(f"  seed={seed} epoch {epoch+1:3d}/{MAX_EPOCHS} | "
              f"loss={avg_train_loss:.4f} | "
              f"heldout_rel={heldout_rel_acc:.3f} | "
              f"affine_train={affine_train:.3f} affine_test={affine_test:.3f} | "
              f"{elapsed:.0f}s", flush=True)

        model.train()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"phase3o3_6feat_no_contr_affine_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "seed": seed,
            "config": {
                "n_f_attrs": 6, "n_entities": N_ENTITIES, "dim": 16,
                "p_contrastive": P_CONTRASTIVE,
                "n_train": n_train, "n_held": n_held, "chance": chance,
                "num_sequences": NUM_SEQUENCES, "max_epochs": MAX_EPOCHS,
                "affine_train_subjects": N_ENTITIES // 2,
            },
            "trajectory": trajectory,
        }, f, indent=2)
    print(f"seed={seed} done — saved to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_seed(args.seed)

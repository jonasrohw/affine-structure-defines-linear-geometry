"""
One-epoch v2 — Phase 1: attribute pretraining with k relation triples.

Trains on attributes + a small number k of relation triples. k is chosen
so the model learns the relation sentence format but NOT the general
relational mapping.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v2_phase1.py --seed 42 --k 8
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v2_phase1.py --seed 42 --k 4 8 16 32
"""
import argparse
import json
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
from src.training import build_valid_attr_table, get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_REVERSE_SHORT = 0.30
P_RELATION = 0.05
MAX_EPOCHS = 50
NUM_SEQUENCES = 62500
EVAL_EVERY = 10
N_ENTITIES = 225
HOLDOUT_FRACTION = 0.20


def make_held_out_triples(vocab, fraction, seed):
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


def select_k_triples(train_triples, k, num_relations, seed):
    """Select k triples balanced across relations."""
    by_rel = {r: [] for r in range(num_relations)}
    for subj, rel in train_triples:
        by_rel[rel].append((subj, rel))
    rng = random.Random(seed + 7777)
    for r in range(num_relations):
        rng.shuffle(by_rel[r])
    selected = []
    per_rel = k // num_relations
    remainder = k % num_relations
    for r in range(num_relations):
        n = per_rel + (1 if r < remainder else 0)
        selected.extend(by_rel[r][:n])
    return selected


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


def eval_attr_accuracy(model, cfg, device, valid_table, n_batches=20):
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


def run_one_k(k, seed, device):
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
            max_epochs=MAX_EPOCHS, learning_rate=3e-4, batch_size=64,
            seq_length=48, num_sequences=NUM_SEQUENCES,
            checkpoint_dir=ROOT / f"checkpoints_v2_k{k}_seed{seed}",
            results_dir=RESULTS_DIR,
        ),
        seed=seed,
    )

    all_train_triples = cfg.vocab.train_triples
    k_triples = select_k_triples(all_train_triples, k, cfg.vocab.num_relations, seed)
    remaining_triples = [t for t in all_train_triples if t not in set(k_triples)]

    print(f"\n--- k={k}, seed={seed} ---")
    print(f"Phase 1 triples: {len(k_triples)} (k={k})")
    print(f"Remaining for phase 2: {len(remaining_triples)}")
    print(f"Held-out: {len(held_out)}")
    print(f"p_relation={P_RELATION}, p_reverse_short={P_REVERSE_SHORT}", flush=True)

    torch.manual_seed(seed)
    train_loader = make_dataloader(
        cfg, seed_offset=0, p_relation=P_RELATION,
        train_triples=k_triples, p_reverse_short=P_REVERSE_SHORT)

    model = build_model(cfg.model, cfg.vocab)
    model.to(device)
    model.train()

    valid_table = build_valid_attr_table(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * MAX_EPOCHS

    # For affine analysis: fit on first half, test on second half
    vocab_affine = VocabConfig(**{**vocab_kwargs, 'n_train_subjects': N_ENTITIES // 2})
    cfg_affine = Config(vocab=vocab_affine, model=cfg.model, train=cfg.train, seed=seed)

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
            k_rel_acc = eval_relation_accuracy(model, cfg.vocab, device, k_triples)
            remaining_rel_acc = eval_relation_accuracy(
                model, cfg.vocab, device, remaining_triples[:90])
            held_rel_acc = eval_relation_accuracy(model, cfg.vocab, device, held_out)

            snapshot = {
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 4),
                "attr_acc": round(attr_acc, 4),
                "k_rel_acc": round(k_rel_acc, 4),
                "remaining_rel_acc": round(remaining_rel_acc, 4),
                "held_rel_acc": round(held_rel_acc, 4),
            }
            trajectory.append(snapshot)

            elapsed = time.time() - t0
            print(f"  Ep {epoch+1:3d}/{MAX_EPOCHS} | loss={avg_loss:.4f} | "
                  f"attr={attr_acc:.3f} | k_rel={k_rel_acc:.3f} | "
                  f"rem_rel={remaining_rel_acc:.3f} | held_rel={held_rel_acc:.3f} | "
                  f"{elapsed:.0f}s", flush=True)
            model.train()

    # Final affine probe
    aff = affine_analysis(model, cfg_affine, device, layer=-1)
    print(f"  Affine: train={aff['overall']['train_acc']:.3f} "
          f"test={aff['overall']['test_acc']:.3f}", flush=True)

    # Save checkpoint
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"
    torch.save({
        "epoch": MAX_EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "seed": seed,
        "k": k,
        "k_triples": k_triples,
        "held_out_triples": held_out,
        "remaining_triples": remaining_triples,
        "vocab_kwargs": vocab_kwargs,
    }, ckpt_path)

    return {
        "k": k,
        "seed": seed,
        "n_k_triples": len(k_triples),
        "n_remaining": len(remaining_triples),
        "n_held_out": len(held_out),
        "trajectory": trajectory,
        "final_affine_train": round(aff["overall"]["train_acc"], 4),
        "final_affine_test": round(aff["overall"]["test_acc"], 4),
        "checkpoint": str(ckpt_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("=" * 70)
    print(f"ONE-EPOCH v2 Phase 1: k-triple sweep (seed={args.seed})")
    print(f"k values: {args.k}")
    print("=" * 70)

    all_results = {}
    for k in args.k:
        all_results[k] = run_one_k(k, args.seed, device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"one_epoch_v2_phase1_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}", flush=True)

    # Summary table
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"{'k':>4} | {'attr':>6} | {'k_rel':>6} | {'rem_rel':>7} | {'held_rel':>8} | "
          f"{'aff_train':>9} | {'aff_test':>8}")
    print("-" * 70)
    for k in args.k:
        r = all_results[k]
        final = r["trajectory"][-1]
        print(f"{k:>4} | {final['attr_acc']:>6.3f} | {final['k_rel_acc']:>6.3f} | "
              f"{final['remaining_rel_acc']:>7.3f} | {final['held_rel_acc']:>8.3f} | "
              f"{r['final_affine_train']:>9.3f} | {r['final_affine_test']:>8.3f}")


if __name__ == "__main__":
    main()

"""
One-epoch v7 — Three-way split: train / single-exposure / held-out.

Phase 1: Train on attributes + 60% of relation triples (repeated) until
well-learned. Phase 2: Present 20% single-exposure facts exactly once.
Evaluate all three groups before and after Phase 2.

The key metric: single-exposure accuracy vs held-out accuracy after Phase 2.
Both groups are unseen during Phase 1. The difference after one exposure
measures what a single fact adds beyond type-level generalization.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v7.py --seed 42
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
from torch.utils.data import TensorDataset, DataLoader

from src.affine import affine_analysis
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import (
    ATTR_SENTENCE_LEN, REL_SENTENCE_LEN, REVERSE_SHORT_SENTENCE_LEN,
    generate_attr_sentence, generate_reverse_short_sentence, make_dataloader,
)
from src.model import build_model
from src.training import build_valid_attr_table, get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_REVERSE_SHORT = 0.30
SEQ_LENGTH = 48


def precompute_relation_objects(vocab):
    f_type_to_objs = {}
    g_type_to_objs = {}
    for j in range(vocab.n_entities):
        ft = vocab.object_f_type(j)
        gt = vocab.object_g_type(j)
        f_type_to_objs.setdefault(ft, []).append(j)
        g_type_to_objs.setdefault(gt, []).append(j)
    table = {}
    for i in range(vocab.n_entities):
        table[(i, 0)] = f_type_to_objs[vocab.subject_f_type(i)]
        table[(i, 1)] = g_type_to_objs[vocab.subject_g_type(i)]
    return table


def make_three_way_split(vocab, train_frac, single_frac, seed):
    """Split (subj, rel) pairs into train/single-exposure/held-out.
    Each subject gets at least 1 train relation."""
    rng = random.Random(seed)
    n = vocab.n_entities
    n_rels = vocab.num_relations

    by_subject = {}
    for i in range(n):
        for k in range(n_rels):
            by_subject.setdefault(i, []).append((i, k))

    guaranteed_train = []
    available = []
    for subj in range(n):
        pairs = list(by_subject[subj])
        rng.shuffle(pairs)
        guaranteed_train.append(pairs[0])
        available.extend(pairs[1:])

    rng.shuffle(available)
    total = n * n_rels
    n_single = int(total * single_frac)
    n_held = total - int(total * train_frac) - n_single

    single_exposure = available[:n_single]
    held_out = available[n_single:n_single + n_held]
    extra_train = available[n_single + n_held:]
    train = guaranteed_train + extra_train

    return train, single_exposure, held_out


def enumerate_facts(triples, rel_objects_table):
    facts = []
    for subj_idx, rel_k in triples:
        for obj_idx in rel_objects_table[(subj_idx, rel_k)]:
            facts.append((subj_idx, rel_k, obj_idx))
    return facts


def build_phase1_sequences(vocab, train_triples, rel_objects_table,
                           n_sequences, seq_length, rng, p_relation=0.20):
    """Sequences with randomly sampled train relation facts + attributes."""
    all_entities = ([vocab.subject_id(i) for i in range(vocab.n_entities)] +
                    [vocab.object_id(j) for j in range(vocab.n_entities)])

    sequences = []
    for _ in range(n_sequences):
        tokens = []
        while True:
            remaining = seq_length - len(tokens)
            if remaining < ATTR_SENTENCE_LEN:
                break
            r = rng.random()
            if r < p_relation and remaining >= REL_SENTENCE_LEN:
                subj_idx, rel_k = rng.choice(train_triples)
                obj_idx = rng.choice(rel_objects_table[(subj_idx, rel_k)])
                tokens.extend([vocab.sep_id, vocab.relation_id(rel_k),
                               vocab.subject_id(subj_idx), vocab.object_id(obj_idx)])
            elif r < p_relation + P_REVERSE_SHORT and remaining >= REVERSE_SHORT_SENTENCE_LEN:
                tokens.extend(generate_reverse_short_sentence(vocab, rng))
            else:
                entity = rng.choice(all_entities)
                tokens.extend(generate_attr_sentence(entity, vocab, rng))
        while len(tokens) < seq_length:
            tokens.append(vocab.sep_id)
        sequences.append(tokens[:seq_length])
    return sequences


def build_phase2_sequences(vocab, single_facts, n_sequences, seq_length, rng):
    """Sequences with single-exposure facts (each placed once) + attributes."""
    rel_sentences = []
    for subj_idx, rel_k, obj_idx in single_facts:
        rel_sentences.append([vocab.sep_id, vocab.relation_id(rel_k),
                              vocab.subject_id(subj_idx), vocab.object_id(obj_idx)])
    rng.shuffle(rel_sentences)

    all_entities = ([vocab.subject_id(i) for i in range(vocab.n_entities)] +
                    [vocab.object_id(j) for j in range(vocab.n_entities)])

    def _attr_sentence():
        if rng.random() < P_REVERSE_SHORT:
            return generate_reverse_short_sentence(vocab, rng)
        entity = rng.choice(all_entities)
        return generate_attr_sentence(entity, vocab, rng)

    n_facts = len(rel_sentences)
    indices = list(range(n_sequences))
    rng.shuffle(indices)
    seq_fact = {}
    for i in range(min(n_facts, n_sequences)):
        seq_fact[indices[i]] = rel_sentences[i]

    sequences = []
    for seq_i in range(n_sequences):
        sentences = []
        if seq_i in seq_fact:
            sentences.append(seq_fact[seq_i])
        total_len = sum(len(s) for s in sentences)
        while total_len + ATTR_SENTENCE_LEN <= seq_length:
            s = _attr_sentence()
            sentences.append(s)
            total_len += len(s)
        rng.shuffle(sentences)
        tokens = []
        for s in sentences:
            tokens.extend(s)
        tokens = tokens[:seq_length]
        while len(tokens) < seq_length:
            tokens.append(vocab.sep_id)
        sequences.append(tokens)

    return sequences


def eval_relation_accuracy(model, vocab, device, triples, rel_objects_table, batch_size=512):
    model.eval()
    correct = 0
    total = len(triples)
    if total == 0:
        return 0.0
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_triples = triples[start:start + batch_size]
            contexts = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(subj_idx)]
                 for subj_idx, rel_k in batch_triples],
                dtype=torch.long, device=device,
            )
            logits = model(contexts)
            preds = logits[:, -1].argmax(dim=-1).cpu().tolist()
            for (subj_idx, rel_k), pred in zip(batch_triples, preds):
                valid_tokens = {vocab.object_id(j) for j in rel_objects_table[(subj_idx, rel_k)]}
                correct += int(pred in valid_tokens)
    return correct / total


def eval_attr_accuracy(model, cfg, device, valid_table, n_batches=20):
    loader = make_dataloader(cfg, seed_offset=9999, p_relation=0.0,
                             p_reverse_short=P_REVERSE_SHORT)
    model.eval()
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


def train_epoch(model, sequences, optimizer, device, batch_size, lr_fn, global_step):
    """Train one epoch, return (avg_loss, global_step)."""
    x_all = torch.stack([torch.tensor(s[:-1], dtype=torch.long) for s in sequences])
    y_all = torch.stack([torch.tensor(s[1:], dtype=torch.long) for s in sequences])
    dataset = TensorDataset(x_all, y_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        current_lr = lr_fn(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        logits = model(x)
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1
        global_step += 1

    return epoch_loss / max(1, n_batches), global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-phase1", type=int, default=10)
    parser.add_argument("--p-relation", type=float, default=0.20)
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-entities", type=int, default=2025)
    parser.add_argument("--num-sequences", type=int, default=62500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Config
    vocab_kwargs = dict(
        n_f_attrs=10, n_g_attrs=10, k_per_group=2,
        num_relations=2, n_entities=args.n_entities,
        n_train_subjects=args.n_entities, r0_shift=7, r1_shift=13,
    )

    print("Precomputing relation objects...", flush=True)
    temp_vocab = VocabConfig(**vocab_kwargs)
    rel_objects_table = precompute_relation_objects(temp_vocab)

    # Three-way split
    train_triples, single_triples, held_triples = make_three_way_split(
        temp_vocab, train_frac=0.60, single_frac=0.20, seed=args.seed)

    vocab = VocabConfig(**vocab_kwargs)
    model_cfg = ModelConfig(n_layers=args.n_layers, hidden_dim=args.dim, n_heads=4)

    _fast = lambda subj_idx, rel_k: rel_objects_table[(subj_idx, rel_k)]
    vocab.relation_objects = _fast

    n_valid = len(rel_objects_table[(0, 0)])
    chance = n_valid / args.n_entities

    # Single-exposure facts
    single_facts = enumerate_facts(single_triples, rel_objects_table)

    print(f"\n{'='*70}")
    print(f"ONE-EPOCH v7: Three-way split (seed={args.seed})")
    print(f"{'='*70}")
    print(f"Types: {vocab.n_types}, Entities: {args.n_entities}")
    print(f"Valid objects/subject: {n_valid}, Chance: {chance:.2%}")
    print(f"Model: dim={args.dim}, layers={args.n_layers}, "
          f"vocab={vocab.vocab_size}, heads=4")
    print(f"\nSplit:")
    print(f"  Train (60%):           {len(train_triples):,} (subj,rel) pairs")
    print(f"  Single-exposure (20%): {len(single_triples):,} pairs "
          f"→ {len(single_facts):,} facts")
    print(f"  Held-out (20%):        {len(held_triples):,} pairs")
    print(f"\nPhase 1: {args.n_phase1} epochs, p_relation={args.p_relation}, "
          f"lr={args.lr}")
    print(f"Phase 2: 1 epoch (single-exposure facts), lr={args.phase2_lr}")

    # Build model
    torch.manual_seed(args.seed)
    model = build_model(model_cfg, vocab)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Eval config
    cfg_eval = Config(
        vocab=vocab, model=model_cfg,
        train=TrainConfig(num_sequences=5000, seq_length=SEQ_LENGTH,
                          batch_size=args.batch_size),
        seed=args.seed,
    )
    valid_table = build_valid_attr_table(cfg_eval, device)

    # Affine config
    vocab_affine_kwargs = {**vocab_kwargs, 'n_train_subjects': args.n_entities // 2}
    vocab_affine = VocabConfig(**vocab_affine_kwargs)
    vocab_affine.relation_objects = _fast
    cfg_affine = Config(vocab=vocab_affine, model=model_cfg, train=TrainConfig(),
                        seed=args.seed)

    # Optimizer
    warmup_steps = 200
    steps_per_epoch = args.num_sequences // args.batch_size
    phase1_steps = steps_per_epoch * args.n_phase1
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def phase1_lr(step):
        return get_lr(step, warmup_steps, phase1_steps, args.lr)

    def phase2_lr(step):
        return args.phase2_lr

    print(f"\nSteps/epoch: {steps_per_epoch:,}")
    print(f"Phase 1 steps: {phase1_steps:,}, Phase 2 steps: {steps_per_epoch:,}")
    print(f"{'='*70}\n", flush=True)

    # Evaluate function for all groups
    def evaluate_all(label=""):
        attr_acc = eval_attr_accuracy(model, cfg_eval, device, valid_table)
        train_acc = eval_relation_accuracy(
            model, vocab, device, train_triples, rel_objects_table)
        single_acc = eval_relation_accuracy(
            model, vocab, device, single_triples, rel_objects_table)
        held_acc = eval_relation_accuracy(
            model, vocab, device, held_triples, rel_objects_table)
        return {
            "attr": round(attr_acc, 4),
            "train_60": round(train_acc, 4),
            "single_20": round(single_acc, 4),
            "held_20": round(held_acc, 4),
        }

    # Pre-training
    pre = evaluate_all("pre")
    print(f"Pre-training: attr={pre['attr']:.3f}  "
          f"train={pre['train_60']:.3f}  single={pre['single_20']:.3f}  "
          f"held={pre['held_20']:.3f}\n", flush=True)

    results = {
        "seed": args.seed,
        "n_entities": args.n_entities,
        "n_types": vocab.n_types,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_params": n_params,
        "n_train": len(train_triples),
        "n_single": len(single_triples),
        "n_held": len(held_triples),
        "n_single_facts": len(single_facts),
        "n_phase1": args.n_phase1,
        "p_relation": args.p_relation,
        "phase2_lr": args.phase2_lr,
        "chance": round(chance, 4),
        "vocab_size": vocab.vocab_size,
        "pre": pre,
        "phase1": {},
        "post_phase1": None,
        "post_phase2": None,
    }

    # ===== Phase 1: Train on 60% relations =====
    rng = random.Random(args.seed + 2000)
    global_step = 0
    t0 = time.time()

    print("--- Phase 1: Training on 60% relations ---", flush=True)
    for epoch in range(1, args.n_phase1 + 1):
        t_data = time.time()
        sequences = build_phase1_sequences(
            vocab, train_triples, rel_objects_table,
            args.num_sequences, SEQ_LENGTH, rng, args.p_relation)
        data_time = time.time() - t_data

        avg_loss, global_step = train_epoch(
            model, sequences, optimizer, device, args.batch_size,
            phase1_lr, global_step)

        t_eval = time.time()
        ev = evaluate_all()
        eval_time = time.time() - t_eval
        elapsed = time.time() - t0

        results["phase1"][epoch] = {
            "loss": round(avg_loss, 4), **ev,
            "data_time": round(data_time, 1), "eval_time": round(eval_time, 1),
        }

        print(f"P1 Ep {epoch:2d}/{args.n_phase1} | loss={avg_loss:.4f} | "
              f"attr={ev['attr']:.3f} | train={ev['train_60']:.3f} | "
              f"single={ev['single_20']:.3f} | held={ev['held_20']:.3f} | "
              f"{elapsed:.0f}s", flush=True)

    # Snapshot after Phase 1
    post_p1 = evaluate_all()
    results["post_phase1"] = post_p1
    print(f"\nPost-Phase 1: train={post_p1['train_60']:.3f}  "
          f"single={post_p1['single_20']:.3f}  held={post_p1['held_20']:.3f}",
          flush=True)

    # ===== Phase 2: Single-exposure facts =====
    print(f"\n--- Phase 2: Single-exposure ({len(single_facts):,} facts, "
          f"lr={args.phase2_lr}) ---", flush=True)

    t_data = time.time()
    sequences = build_phase2_sequences(
        vocab, single_facts, args.num_sequences, SEQ_LENGTH, rng)
    data_time = time.time() - t_data

    avg_loss, global_step = train_epoch(
        model, sequences, optimizer, device, args.batch_size,
        phase2_lr, global_step)

    post_p2 = evaluate_all()
    results["post_phase2"] = {
        "loss": round(avg_loss, 4), **post_p2,
        "data_time": round(data_time, 1),
    }

    elapsed = time.time() - t0
    print(f"P2 loss={avg_loss:.4f} | attr={post_p2['attr']:.3f} | "
          f"train={post_p2['train_60']:.3f} | "
          f"single={post_p2['single_20']:.3f} | "
          f"held={post_p2['held_20']:.3f} | {elapsed:.0f}s", flush=True)

    # ===== Summary =====
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'':>20} | {'Train 60%':>10} | {'Single 20%':>10} | {'Held 20%':>10}")
    print("-" * 60)
    print(f"{'Pre-training':>20} | {pre['train_60']:>10.3f} | "
          f"{pre['single_20']:>10.3f} | {pre['held_20']:>10.3f}")
    print(f"{'After Phase 1':>20} | {post_p1['train_60']:>10.3f} | "
          f"{post_p1['single_20']:>10.3f} | {post_p1['held_20']:>10.3f}")
    print(f"{'After Phase 2':>20} | {post_p2['train_60']:>10.3f} | "
          f"{post_p2['single_20']:>10.3f} | {post_p2['held_20']:>10.3f}")
    print(f"\nSingle-exposure gain: "
          f"{post_p2['single_20'] - post_p1['single_20']:+.3f} "
          f"(from {post_p1['single_20']:.3f} to {post_p2['single_20']:.3f})")
    print(f"Held-out change:      "
          f"{post_p2['held_20'] - post_p1['held_20']:+.3f} "
          f"(from {post_p1['held_20']:.3f} to {post_p2['held_20']:.3f})")
    print(f"Chance level:         {chance:.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"one_epoch_v7_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Save checkpoint
    ckpt_dir = ROOT / f"checkpoints_v7_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "seed": args.seed,
        "vocab_kwargs": vocab_kwargs,
        "train_triples": train_triples,
        "single_triples": single_triples,
        "held_triples": held_triples,
        "model_cfg": {"n_layers": args.n_layers, "hidden_dim": args.dim, "n_heads": 4},
    }, ckpt_dir / "final.pt")
    print(f"Checkpoint saved to {ckpt_dir / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()

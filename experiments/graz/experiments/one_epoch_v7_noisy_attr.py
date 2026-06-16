"""
Noisy attribute experiment: does corrupting the attribute signal degrade learning?

Same as v7 Phase 1 (60% train triples, all params trained), but each
attribute sentence has its attribute flipped to a random different attribute
from the same group (f or g) with probability p_flip. All attributes are
affected (no single-attribute targeting).

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v7_noisy_attr.py --p-flip 0.01
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v7_noisy_attr.py --p-flip 0.10
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

from src.affine import affine_analysis_centroid
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import (
    ATTR_SENTENCE_LEN, REL_SENTENCE_LEN, REVERSE_SHORT_SENTENCE_LEN,
    make_dataloader,
)
from src.model import build_model
from src.training import build_valid_attr_table, get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_REVERSE_SHORT = 0.30
SEQ_LENGTH = 48


def flip_attr(attr, n_attrs, rng):
    """Replace attr with a uniformly random different attribute from {0..n_attrs-1}."""
    new = rng.randint(0, n_attrs - 2)
    if new >= attr:
        new += 1
    return new


def generate_attr_sentence_noisy(entity_token, vocab, rng, p_flip):
    """Like generate_attr_sentence but flips the attribute with probability p_flip."""
    f_attrs, g_attrs = vocab.entity_attrs(entity_token)
    if rng.random() < 0.5:
        attr = rng.choice(f_attrs)
        if rng.random() < p_flip:
            attr = flip_attr(attr, vocab.n_f_attrs, rng)
        return [vocab.sep_id, entity_token, vocab.f_attr_id(attr)]
    else:
        attr = rng.choice(g_attrs)
        if rng.random() < p_flip:
            attr = flip_attr(attr, vocab.n_g_attrs, rng)
        return [vocab.sep_id, entity_token, vocab.g_attr_id(attr)]


def generate_reverse_short_noisy(vocab, rng, p_flip):
    """Like generate_reverse_short_sentence but flips the attribute with probability p_flip."""
    if rng.random() < 0.5:
        idx = rng.randint(0, vocab.n_entities - 1)
        entity_token = vocab.subject_id(idx)
        marker = vocab.s_marker_id
        f_attrs = list(vocab.subject_f_type(idx))
        g_attrs = list(vocab.subject_g_type(idx))
    else:
        idx = rng.randint(0, vocab.n_entities - 1)
        entity_token = vocab.object_id(idx)
        marker = vocab.o_marker_id
        f_attrs = list(vocab.object_f_type(idx))
        g_attrs = list(vocab.object_g_type(idx))

    if rng.random() < 0.5:
        attr = rng.choice(f_attrs)
        if rng.random() < p_flip:
            attr = flip_attr(attr, vocab.n_f_attrs, rng)
        attr_token = vocab.f_attr_id(attr)
    else:
        attr = rng.choice(g_attrs)
        if rng.random() < p_flip:
            attr = flip_attr(attr, vocab.n_g_attrs, rng)
        attr_token = vocab.g_attr_id(attr)

    return [vocab.sep_id, marker, attr_token, entity_token]


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


def build_phase1_sequences(vocab, train_triples, rel_objects_table,
                           n_sequences, seq_length, rng, p_relation, p_flip):
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
                tokens.extend(generate_reverse_short_noisy(vocab, rng, p_flip))
            else:
                entity = rng.choice(all_entities)
                tokens.extend(generate_attr_sentence_noisy(
                    entity, vocab, rng, p_flip))
        while len(tokens) < seq_length:
            tokens.append(vocab.sep_id)
        sequences.append(tokens[:seq_length])
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
    parser.add_argument("--p-flip", type=float, required=True,
                        help="Probability of flipping each attribute to a random other")
    parser.add_argument("--n-phase1", type=int, default=10)
    parser.add_argument("--p-relation", type=float, default=0.20)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-entities", type=int, default=2025)
    parser.add_argument("--num-sequences", type=int, default=62500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    assert 0.0 <= args.p_flip <= 1.0, "p-flip must be between 0 and 1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    vocab_kwargs = dict(
        n_f_attrs=10, n_g_attrs=10, k_per_group=2,
        num_relations=2, n_entities=args.n_entities,
        n_train_subjects=args.n_entities, r0_shift=7, r1_shift=13,
    )

    print("Precomputing relation objects...", flush=True)
    temp_vocab = VocabConfig(**vocab_kwargs)
    rel_objects_table = precompute_relation_objects(temp_vocab)

    train_triples, single_triples, held_triples = make_three_way_split(
        temp_vocab, train_frac=0.60, single_frac=0.20, seed=args.seed)

    vocab = VocabConfig(**vocab_kwargs)
    model_cfg = ModelConfig(n_layers=args.n_layers, hidden_dim=args.dim, n_heads=4)

    _fast = lambda subj_idx, rel_k: rel_objects_table[(subj_idx, rel_k)]
    vocab.relation_objects = _fast

    n_valid = len(rel_objects_table[(0, 0)])
    chance = n_valid / args.n_entities

    print(f"\n{'='*70}")
    print(f"NOISY ATTRIBUTE EXPERIMENT (p_flip={args.p_flip}, seed={args.seed})")
    print(f"{'='*70}")
    print(f"Types: {vocab.n_types}, Entities: {args.n_entities}")
    print(f"Valid objects/subject: {n_valid}, Chance: {chance:.2%}")
    print(f"Model: dim={args.dim}, layers={args.n_layers}, "
          f"vocab={vocab.vocab_size}, heads=4")
    print(f"\nNoise: p_flip={args.p_flip} (flip to random same-group attr)")
    print(f"Train triples (60%): {len(train_triples):,}")
    print(f"Phase 1: {args.n_phase1} epochs, p_relation={args.p_relation}, lr={args.lr}")

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

    print(f"\nSteps/epoch: {steps_per_epoch:,}")
    print(f"Phase 1 steps: {phase1_steps:,}")
    print(f"{'='*70}\n", flush=True)

    def evaluate_all():
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

    # Pre-training eval
    pre = evaluate_all()
    print(f"Pre-training: attr={pre['attr']:.3f}  "
          f"train={pre['train_60']:.3f}  single={pre['single_20']:.3f}  "
          f"held={pre['held_20']:.3f}\n", flush=True)

    results = {
        "experiment": "noisy_attr",
        "p_flip": args.p_flip,
        "seed": args.seed,
        "n_entities": args.n_entities,
        "n_types": vocab.n_types,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_params": n_params,
        "n_train": len(train_triples),
        "n_single": len(single_triples),
        "n_held": len(held_triples),
        "n_phase1": args.n_phase1,
        "p_relation": args.p_relation,
        "chance": round(chance, 4),
        "pre": pre,
        "phase1": {},
    }

    # Phase 1 training
    rng = random.Random(args.seed + 2000)
    global_step = 0
    t0 = time.time()

    print("--- Phase 1: Training (noisy attributes) ---", flush=True)
    for epoch in range(1, args.n_phase1 + 1):
        sequences = build_phase1_sequences(
            vocab, train_triples, rel_objects_table,
            args.num_sequences, SEQ_LENGTH, rng, args.p_relation,
            args.p_flip)

        avg_loss, global_step = train_epoch(
            model, sequences, optimizer, device, args.batch_size,
            phase1_lr, global_step)

        ev = evaluate_all()
        aff = affine_analysis_centroid(model, cfg_affine, device)
        elapsed = time.time() - t0

        results["phase1"][epoch] = {
            "loss": round(avg_loss, 4), **ev,
            "affine_r0_test": round(aff['r0']['test_acc'], 4),
            "affine_r1_test": round(aff['r1']['test_acc'], 4),
        }

        print(f"Ep {epoch:2d}/{args.n_phase1} | loss={avg_loss:.4f} | "
              f"attr={ev['attr']:.3f} | train={ev['train_60']:.3f} | "
              f"held={ev['held_20']:.3f} | "
              f"aff_r0={aff['r0']['test_acc']:.3f} aff_r1={aff['r1']['test_acc']:.3f} | "
              f"{elapsed:.0f}s", flush=True)

    # Final evaluation
    final = evaluate_all()
    final_aff = affine_analysis_centroid(model, cfg_affine, device)
    results["final"] = {
        **final,
        "affine_r0_test": round(final_aff['r0']['test_acc'], 4),
        "affine_r1_test": round(final_aff['r1']['test_acc'], 4),
    }

    # Summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (p_flip={args.p_flip})")
    print(f"{'='*70}")
    print(f"  Attribute accuracy:  {final['attr']:.3f}")
    print(f"  Train (60%) rel acc: {final['train_60']:.3f}")
    print(f"  Held-out (20%) acc:  {final['held_20']:.3f}")
    print(f"  Chance:              {chance:.3f}")
    print(f"\n  Affine centroid (Voronoi accuracy):")
    print(f"    R0: {final_aff['r0']['test_acc']:.3f}")
    print(f"    R1: {final_aff['r1']['test_acc']:.3f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p_str = f"{args.p_flip:.2f}".replace(".", "p")
    out_path = RESULTS_DIR / f"one_epoch_v7_noisy_attr_{p_str}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

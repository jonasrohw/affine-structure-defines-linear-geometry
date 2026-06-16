"""
One-epoch v6 — Graduated curriculum: pilot scaffold + single-exposure main.

Phase 1 (pilot, epochs 1-k): repeated pilot set (100 triples = 4,500 facts)
to learn attributes and the relational pattern thoroughly.
Phase 2 (main, epochs k+1 to k+8): all 145,800 single-exposure facts spread
across 8 epochs with mature representations AND a pre-learned relational map.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v6.py --seed 42
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v6.py --seed 42 --k-pilot 5 --pilot-size 100
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
HOLDOUT_FRACTION = 0.20
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


def select_pilot_set(train_triples, pilot_size, num_relations, seed):
    rng = random.Random(seed)
    by_rel = {}
    for subj, rel in train_triples:
        by_rel.setdefault(rel, []).append((subj, rel))
    for rel in by_rel:
        rng.shuffle(by_rel[rel])
    pilot = []
    per_rel = pilot_size // num_relations
    for rel in sorted(by_rel):
        pilot.extend(by_rel[rel][:per_rel])
    return pilot


def enumerate_facts(triples, rel_objects_table):
    facts = []
    for subj_idx, rel_k in triples:
        for obj_idx in rel_objects_table[(subj_idx, rel_k)]:
            facts.append((subj_idx, rel_k, obj_idx))
    return facts


def split_facts(facts, n_splits, seed):
    rng = random.Random(seed)
    facts = list(facts)
    rng.shuffle(facts)
    chunk_size = len(facts) // n_splits
    chunks = []
    for i in range(n_splits):
        start = i * chunk_size
        end = start + chunk_size if i < n_splits - 1 else len(facts)
        chunks.append(facts[start:end])
    return chunks


def build_epoch_sequences(vocab, epoch_facts, n_sequences, seq_length, rng):
    rel_sentences = []
    for subj_idx, rel_k, obj_idx in epoch_facts:
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
    return correct / max(1, total)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-pilot", type=int, default=5)
    parser.add_argument("--n-main", type=int, default=8)
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-entities", type=int, default=2025)
    parser.add_argument("--num-sequences", type=int, default=62500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    n_epochs = args.k_pilot + args.n_main

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

    held_out = make_held_out_triples(temp_vocab, HOLDOUT_FRACTION, args.seed)
    vocab = VocabConfig(**vocab_kwargs, held_out_triples=held_out)
    model_cfg = ModelConfig(n_layers=args.n_layers, hidden_dim=args.dim, n_heads=4)

    _fast = lambda subj_idx, rel_k: rel_objects_table[(subj_idx, rel_k)]
    vocab.relation_objects = _fast

    train_triples = vocab.train_triples
    n_valid = len(rel_objects_table[(0, 0)])
    chance = n_valid / args.n_entities

    # Select pilot set
    pilot_triples = select_pilot_set(
        train_triples, args.pilot_size, vocab.num_relations, args.seed + 4000)
    pilot_facts = enumerate_facts(pilot_triples, rel_objects_table)
    pilot_set = set(map(tuple, pilot_triples))

    # Non-pilot triples for evaluation
    non_pilot_train = [t for t in train_triples if tuple(t) not in pilot_set]

    # Enumerate all training facts and split across main epochs
    all_facts = enumerate_facts(train_triples, rel_objects_table)
    fact_chunks = split_facts(all_facts, args.n_main, args.seed + 1000)

    print(f"\n{'='*70}")
    print(f"ONE-EPOCH v6: Graduated curriculum (seed={args.seed})")
    print(f"{'='*70}")
    print(f"Types: {vocab.n_types}, Entities: {args.n_entities}, "
          f"Entities/type: {args.n_entities // vocab.n_types}")
    print(f"Valid objects/subject: {n_valid}, Chance: {chance:.2%}")
    print(f"Model: dim={args.dim}, layers={args.n_layers}, "
          f"vocab={vocab.vocab_size}, heads=4")
    print(f"Training (subj,rel) pairs: {len(train_triples)}, "
          f"Held-out: {len(held_out)}")
    print(f"\nCurriculum:")
    print(f"  Pilot epochs: {args.k_pilot} (pilot: {args.pilot_size} "
          f"triples → {len(pilot_facts):,} facts, repeated each epoch)")
    print(f"  Main epochs: {args.n_main} (all {len(all_facts):,} facts, "
          f"single-exposure)")
    print(f"  Facts per main epoch: {len(fact_chunks[0]):,}")
    print(f"  Total epochs: {n_epochs}")

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
    vocab_affine = VocabConfig(**vocab_affine_kwargs, held_out_triples=held_out)
    vocab_affine.relation_objects = _fast
    cfg_affine = Config(vocab=vocab_affine, model=model_cfg, train=TrainConfig(),
                        seed=args.seed)

    # Optimizer and schedule
    warmup_steps = 200
    steps_per_epoch = args.num_sequences // args.batch_size
    max_steps = steps_per_epoch * n_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    print(f"\nSteps/epoch: {steps_per_epoch:,}, Total steps: {max_steps:,}")
    print(f"LR: {args.lr}, Warmup: {warmup_steps} steps, "
          f"Cosine to {max_steps} steps")
    print(f"{'='*70}\n", flush=True)

    # Pre-training evaluation
    pre_attr = eval_attr_accuracy(model, cfg_eval, device, valid_table)
    pre_train_rel = eval_relation_accuracy(
        model, vocab, device, train_triples, rel_objects_table)
    pre_held_rel = eval_relation_accuracy(
        model, vocab, device, held_out, rel_objects_table)

    print(f"Pre-training: attr={pre_attr:.3f}  train_rel={pre_train_rel:.3f}  "
          f"held_rel={pre_held_rel:.3f}\n", flush=True)

    results = {
        "seed": args.seed,
        "n_entities": args.n_entities,
        "n_types": vocab.n_types,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_params": n_params,
        "n_train_triples": len(train_triples),
        "n_held_out": len(held_out),
        "n_total_facts": len(all_facts),
        "k_pilot": args.k_pilot,
        "n_main": args.n_main,
        "n_epochs": n_epochs,
        "pilot_size": args.pilot_size,
        "n_pilot_facts": len(pilot_facts),
        "facts_per_main_epoch": len(fact_chunks[0]),
        "chance": round(chance, 4),
        "vocab_size": vocab.vocab_size,
        "pre_training": {
            "attr": round(pre_attr, 4),
            "train_rel": round(pre_train_rel, 4),
            "held_rel": round(pre_held_rel, 4),
        },
        "epochs": {},
    }

    # Training loop
    rng = random.Random(args.seed + 2000)
    global_step = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        if epoch <= args.k_pilot:
            epoch_facts = pilot_facts
            phase = "pilot"
        else:
            main_idx = epoch - args.k_pilot - 1
            epoch_facts = fact_chunks[main_idx]
            phase = "main"

        t_data = time.time()
        sequences = build_epoch_sequences(
            vocab, epoch_facts, args.num_sequences, SEQ_LENGTH, rng)
        data_time = time.time() - t_data

        x_all = torch.stack([torch.tensor(s[:-1], dtype=torch.long) for s in sequences])
        y_all = torch.stack([torch.tensor(s[1:], dtype=torch.long) for s in sequences])
        dataset = TensorDataset(x_all, y_all)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True)

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            current_lr = get_lr(global_step, warmup_steps, max_steps, args.lr)
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

        avg_loss = epoch_loss / max(1, n_batches)

        # Evaluate
        t_eval = time.time()
        attr_acc = eval_attr_accuracy(model, cfg_eval, device, valid_table)
        pilot_rel = eval_relation_accuracy(
            model, vocab, device, pilot_triples, rel_objects_table)
        non_pilot_rel = eval_relation_accuracy(
            model, vocab, device, non_pilot_train, rel_objects_table)
        train_rel = eval_relation_accuracy(
            model, vocab, device, train_triples, rel_objects_table)
        held_rel = eval_relation_accuracy(
            model, vocab, device, held_out, rel_objects_table)
        aff = affine_analysis(model, cfg_affine, device, layer=-1)
        eval_time = time.time() - t_eval

        elapsed = time.time() - t0

        snapshot = {
            "epoch": epoch,
            "phase": phase,
            "n_facts": len(epoch_facts),
            "loss": round(avg_loss, 4),
            "attr": round(attr_acc, 4),
            "pilot_rel": round(pilot_rel, 4),
            "non_pilot_rel": round(non_pilot_rel, 4),
            "train_rel": round(train_rel, 4),
            "held_rel": round(held_rel, 4),
            "affine_train": round(aff["overall"]["train_acc"], 4),
            "affine_test": round(aff["overall"]["test_acc"], 4),
            "data_time": round(data_time, 1),
            "eval_time": round(eval_time, 1),
        }
        results["epochs"][epoch] = snapshot

        print(f"Ep {epoch:2d}/{n_epochs} [{phase:5s}] | "
              f"facts={len(epoch_facts):>6,} | loss={avg_loss:.4f} | "
              f"attr={attr_acc:.3f} | "
              f"pilot={pilot_rel:.3f} | non_pilot={non_pilot_rel:.3f} | "
              f"held={held_rel:.3f} | "
              f"aff={aff['overall']['test_acc']:.3f} | "
              f"{elapsed:.0f}s", flush=True)

        model.train()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"one_epoch_v6_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Save checkpoint
    ckpt_dir = ROOT / f"checkpoints_v6_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "seed": args.seed,
        "vocab_kwargs": vocab_kwargs,
        "held_out_triples": held_out,
        "pilot_triples": pilot_triples,
        "model_cfg": {"n_layers": args.n_layers, "hidden_dim": args.dim, "n_heads": 4},
    }, ckpt_dir / "final.pt")
    print(f"Checkpoint saved to {ckpt_dir / 'final.pt'}", flush=True)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY (seed={args.seed}, k_pilot={args.k_pilot}, "
          f"n_main={args.n_main}, pilot={args.pilot_size})")
    print(f"{'='*70}")
    print(f"{'ep':>3} {'phase':>5} | {'facts':>6} | {'loss':>6} | "
          f"{'attr':>5} | {'pilot':>5} | {'non_p':>5} | "
          f"{'held':>5} | {'aff_t':>5}")
    print("-" * 70)
    print(f"{'pre':>3} {'':>5} | {'':>6} | {'—':>6} | {pre_attr:>5.3f} | "
          f"{'—':>5} | {'—':>5} | {pre_held_rel:>5.3f} | {'—':>5}")
    for ep in range(1, n_epochs + 1):
        s = results["epochs"][ep]
        print(f"{ep:>3} {s['phase']:>5} | {s['n_facts']:>6,} | "
              f"{s['loss']:>6.3f} | {s['attr']:>5.3f} | "
              f"{s['pilot_rel']:>5.3f} | {s['non_pilot_rel']:>5.3f} | "
              f"{s['held_rel']:>5.3f} | {s['affine_test']:>5.3f}")


if __name__ == "__main__":
    main()

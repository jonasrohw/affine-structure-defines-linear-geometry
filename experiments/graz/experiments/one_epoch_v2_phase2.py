"""
One-epoch v2 — Phase 2: single-epoch relation learning from phase 1 checkpoint.

Loads a v2 phase 1 checkpoint (trained with k relation triples) and exposes
the model to the remaining train triples. Each relation epoch sees every
remaining triple exactly once, plus the k phase 1 triples interleaved
for stability.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v2_phase2.py --seed 42 --k 8
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v2_phase2.py --seed 42 --k 8 --rel-epochs 1 2 5 10 20 50
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
from src.data_generation import (
    ATTR_SENTENCE_LEN, REL_SENTENCE_LEN, REVERSE_SHORT_SENTENCE_LEN,
    generate_attr_sentence, generate_rel_sentence, generate_reverse_short_sentence,
)
from src.model import build_model
from src.training import build_valid_attr_table, get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_REVERSE_SHORT = 0.30
N_ENTITIES = 225
SEQ_LENGTH = 48
BATCH_SIZE = 16


def build_phase2_sequences(vocab, remaining_triples, k_triples, seq_length, rng):
    """Pack every remaining triple exactly once, with k triples and attrs interleaved."""
    # Build the exhaustive relation sentences for remaining triples
    rel_sentences = []
    for subj_idx, rel_k in remaining_triples:
        objects = vocab.relation_objects(subj_idx, rel_k)
        obj_idx = objects[0] if len(objects) == 1 else rng.choice(objects)
        rel_sentences.append(
            [vocab.sep_id, vocab.relation_id(rel_k),
             vocab.subject_id(subj_idx), vocab.object_id(obj_idx)]
        )
    rng.shuffle(rel_sentences)

    # Build k-triple sentences (sampled randomly, not exhaustive)
    k_rel_sentences = []
    for subj_idx, rel_k in k_triples:
        objects = vocab.relation_objects(subj_idx, rel_k)
        obj_idx = objects[0] if len(objects) == 1 else rng.choice(objects)
        k_rel_sentences.append(
            [vocab.sep_id, vocab.relation_id(rel_k),
             vocab.subject_id(subj_idx), vocab.object_id(obj_idx)]
        )

    all_entities = ([vocab.subject_id(i) for i in range(vocab.n_entities)] +
                    [vocab.object_id(j) for j in range(vocab.n_entities)])

    def _attr_sentence():
        r = rng.random()
        if r < P_REVERSE_SHORT:
            return generate_reverse_short_sentence(vocab, rng)
        entity = rng.choice(all_entities)
        return generate_attr_sentence(entity, vocab, rng)

    sequences = []
    rel_idx = 0
    k_counter = 0
    while rel_idx < len(rel_sentences):
        tokens = []
        while len(tokens) + REL_SENTENCE_LEN <= seq_length and rel_idx < len(rel_sentences):
            # Add one remaining triple
            tokens.extend(rel_sentences[rel_idx])
            rel_idx += 1
            remaining = seq_length - len(tokens)
            # Interleave: sometimes a k-triple, sometimes an attr sentence
            if remaining >= REL_SENTENCE_LEN and k_rel_sentences and rng.random() < 0.15:
                tokens.extend(k_rel_sentences[k_counter % len(k_rel_sentences)])
                k_counter += 1
            elif remaining >= ATTR_SENTENCE_LEN:
                tokens.extend(_attr_sentence())
        while len(tokens) + ATTR_SENTENCE_LEN <= seq_length:
            tokens.extend(_attr_sentence())
        tokens = tokens[:seq_length]
        while len(tokens) < seq_length:
            tokens.append(vocab.sep_id)
        sequences.append(tokens)
    return sequences


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


def eval_attr_accuracy(model, vocab, device, valid_table):
    from src.data_generation import make_dataloader
    cfg_eval = Config(
        vocab=vocab,
        model=ModelConfig(n_layers=1, hidden_dim=16, n_heads=4),
        train=TrainConfig(num_sequences=5000, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE),
    )
    loader = make_dataloader(cfg_eval, seed_offset=9999, p_relation=0.0,
                             p_reverse_short=P_REVERSE_SHORT)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= 20:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            x_entity = (x >= vocab.subject_offset) & (x < vocab.f_attr_offset)
            y_attr = (y >= vocab.f_attr_offset) & (y < vocab.relation_offset)
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
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--rel-epochs", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load phase 1 checkpoint
    ckpt_path = ROOT / f"checkpoints_v2_k{args.k}_seed{args.seed}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    vocab_kwargs = ckpt["vocab_kwargs"]
    held_out = ckpt["held_out_triples"]
    k_triples = ckpt["k_triples"]
    remaining_triples = ckpt["remaining_triples"]

    vocab_cfg = VocabConfig(**vocab_kwargs, held_out_triples=held_out)
    model_cfg = ModelConfig(n_layers=1, hidden_dim=16, n_heads=4)

    n_valid = len(vocab_cfg.relation_objects(0, 0))
    chance = float(n_valid) / N_ENTITIES
    rel_epochs_list = sorted(args.rel_epochs)
    max_rel_epochs = max(rel_epochs_list)

    print("=" * 70)
    print(f"ONE-EPOCH v2 Phase 2 (seed={args.seed}, k={args.k})")
    print(f"Phase 1 triples (k): {len(k_triples)}")
    print(f"Remaining triples: {len(remaining_triples)}")
    print(f"Held-out triples: {len(held_out)}")
    print(f"Relation epochs: {rel_epochs_list}, Chance: {chance:.2%}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    model = build_model(model_cfg, vocab_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"Loaded phase 1 checkpoint (k={args.k}, {ckpt['epoch']} epochs)", flush=True)

    valid_table = build_valid_attr_table(
        Config(vocab=vocab_cfg, model=model_cfg, train=TrainConfig()), device)

    vocab_affine = VocabConfig(**{**vocab_kwargs, 'n_train_subjects': N_ENTITIES // 2})
    cfg_affine = Config(vocab=vocab_affine, model=model_cfg, train=TrainConfig(), seed=args.seed)

    # Evaluate before phase 2
    pre_k_rel = eval_relation_accuracy(model, vocab_cfg, device, k_triples)
    pre_rem_rel = eval_relation_accuracy(model, vocab_cfg, device, remaining_triples[:90])
    pre_held_rel = eval_relation_accuracy(model, vocab_cfg, device, held_out)
    pre_attr = eval_attr_accuracy(model, vocab_cfg, device, valid_table)
    pre_aff = affine_analysis(model, cfg_affine, device, layer=-1)

    print(f"\nBefore phase 2:")
    print(f"  k_rel={pre_k_rel:.3f}  rem_rel={pre_rem_rel:.3f}  "
          f"held_rel={pre_held_rel:.3f}  attr={pre_attr:.3f}")
    print(f"  affine_train={pre_aff['overall']['train_acc']:.3f}  "
          f"affine_test={pre_aff['overall']['test_acc']:.3f}")
    print(flush=True)

    results = {
        "seed": args.seed,
        "k": args.k,
        "n_k_triples": len(k_triples),
        "n_remaining": len(remaining_triples),
        "n_held_out": len(held_out),
        "chance": chance,
        "pre_phase2": {
            "k_rel": round(pre_k_rel, 4),
            "remaining_rel": round(pre_rem_rel, 4),
            "held_rel": round(pre_held_rel, 4),
            "attr_acc": round(pre_attr, 4),
            "affine_train": round(pre_aff["overall"]["train_acc"], 4),
            "affine_test": round(pre_aff["overall"]["test_acc"], 4),
        },
        "after_epochs": {},
    }

    # Optimizer — same lr as phase 1 (model is adapted to relation tokens)
    lr = 3e-4
    warmup_steps = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Estimate steps for lr schedule
    test_seqs = build_phase2_sequences(
        vocab_cfg, remaining_triples, k_triples, SEQ_LENGTH, random.Random(args.seed + 5000))
    steps_per_rel_epoch = max(1, len(test_seqs) // BATCH_SIZE)
    max_steps = steps_per_rel_epoch * max_rel_epochs
    print(f"Sequences per rel epoch: {len(test_seqs)}, "
          f"batches: {steps_per_rel_epoch}, "
          f"total steps: {max_steps}", flush=True)
    del test_seqs

    rng = random.Random(args.seed + 2000)
    t0 = time.time()
    global_step = 0

    model.train()
    for rel_epoch in range(1, max_rel_epochs + 1):
        sequences = build_phase2_sequences(
            vocab_cfg, remaining_triples, k_triples, SEQ_LENGTH, rng)
        random.Random(args.seed + rel_epoch).shuffle(sequences)

        epoch_loss = 0.0
        n_batches = 0
        for batch_start in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[batch_start:batch_start + BATCH_SIZE]
            if len(batch_seqs) < 2:
                continue

            x = torch.stack([torch.tensor(s[:-1], dtype=torch.long) for s in batch_seqs]).to(device)
            y = torch.stack([torch.tensor(s[1:], dtype=torch.long) for s in batch_seqs]).to(device)

            current_lr = get_lr(global_step, warmup_steps, max_steps, lr)
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

        if rel_epoch in rel_epochs_list or rel_epoch == max_rel_epochs:
            k_rel = eval_relation_accuracy(model, vocab_cfg, device, k_triples)
            rem_rel = eval_relation_accuracy(model, vocab_cfg, device, remaining_triples[:90])
            held_rel = eval_relation_accuracy(model, vocab_cfg, device, held_out)
            attr_acc = eval_attr_accuracy(model, vocab_cfg, device, valid_table)
            aff = affine_analysis(model, cfg_affine, device, layer=-1)

            snapshot = {
                "rel_epoch": rel_epoch,
                "train_loss": round(avg_loss, 4),
                "k_rel": round(k_rel, 4),
                "remaining_rel": round(rem_rel, 4),
                "held_rel": round(held_rel, 4),
                "attr_acc": round(attr_acc, 4),
                "affine_train": round(aff["overall"]["train_acc"], 4),
                "affine_test": round(aff["overall"]["test_acc"], 4),
            }
            results["after_epochs"][rel_epoch] = snapshot

            elapsed = time.time() - t0
            print(f"Rel ep {rel_epoch:3d} | loss={avg_loss:.4f} | "
                  f"k={k_rel:.3f} rem={rem_rel:.3f} held={held_rel:.3f} | "
                  f"attr={attr_acc:.3f} | "
                  f"aff_tr={aff['overall']['train_acc']:.3f} "
                  f"aff_te={aff['overall']['test_acc']:.3f} | "
                  f"{elapsed:.0f}s", flush=True)
            model.train()
        else:
            elapsed = time.time() - t0
            print(f"Rel ep {rel_epoch:3d} | loss={avg_loss:.4f} | {elapsed:.0f}s", flush=True)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"one_epoch_v2_phase2_k{args.k}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: k={args.k}, seed={args.seed}")
    print("=" * 70)
    print(f"{'epoch':>6} | {'k_rel':>6} | {'rem_rel':>7} | {'held_rel':>8} | "
          f"{'attr':>5} | {'aff_test':>8}")
    print("-" * 55)
    print(f"{'pre':>6} | {pre_k_rel:>6.3f} | {pre_rem_rel:>7.3f} | "
          f"{pre_held_rel:>8.3f} | {pre_attr:>5.3f} | "
          f"{pre_aff['overall']['test_acc']:>8.3f}")
    for ep in rel_epochs_list:
        s = results["after_epochs"].get(ep, {})
        if s:
            print(f"{ep:>6} | {s['k_rel']:>6.3f} | {s['remaining_rel']:>7.3f} | "
                  f"{s['held_rel']:>8.3f} | {s['attr_acc']:>5.3f} | "
                  f"{s['affine_test']:>8.3f}")


if __name__ == "__main__":
    main()

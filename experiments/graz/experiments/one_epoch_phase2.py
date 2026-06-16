"""
One-epoch relation learning — Phase 2: relation exposure from phase 1 checkpoint.

Loads a phase 1 (attribute-only) checkpoint and trains for N relation epochs.
Each relation epoch exposes every train triple exactly once, packed into
sequences alongside continued attribute sentences.

Also runs the no-attribute control (random init + same relation epochs).

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/one_epoch_phase2.py --seed 42
    CUDA_VISIBLE_DEVICES=2 python experiments/one_epoch_phase2.py --seed 42 --rel-epochs 1
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
from src.data_generation import (
    ATTR_SENTENCE_LEN, REL_SENTENCE_LEN, REVERSE_SHORT_SENTENCE_LEN,
    generate_attr_sentence, generate_reverse_short_sentence,
)
from src.model import build_model
from src.training import build_valid_attr_table, get_lr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_REVERSE_SHORT = 0.30
N_ENTITIES = 225
HOLDOUT_FRACTION = 0.20
SEQ_LENGTH = 48
BATCH_SIZE = 16


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


def build_exhaustive_relation_sequences(vocab, train_triples, seq_length, rng):
    """Pack every train triple exactly once into sequences, interleaved with attribute sentences.

    Alternates: one relation sentence, one attribute sentence, one relation, ...
    This gives roughly 50/50 relation/attribute mix by sentence count.
    """
    rel_sentences = []
    for subj_idx, rel_k in train_triples:
        objects = vocab.relation_objects(subj_idx, rel_k)
        obj_idx = objects[0] if len(objects) == 1 else rng.choice(objects)
        rel_sentences.append(
            [vocab.sep_id, vocab.relation_id(rel_k),
             vocab.subject_id(subj_idx), vocab.object_id(obj_idx)]
        )
    rng.shuffle(rel_sentences)

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
    while rel_idx < len(rel_sentences):
        tokens = []
        while len(tokens) + REL_SENTENCE_LEN <= seq_length and rel_idx < len(rel_sentences):
            tokens.extend(rel_sentences[rel_idx])
            rel_idx += 1
            remaining = seq_length - len(tokens)
            if remaining >= ATTR_SENTENCE_LEN:
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


def eval_attr_accuracy(model, vocab, device, valid_table, rng):
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


def run_phase2(seed, rel_epochs_list, from_pretrained=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    vocab_kwargs = dict(
        n_f_attrs=6, n_g_attrs=6, k_per_group=2,
        num_relations=2, n_entities=N_ENTITIES,
        n_train_subjects=N_ENTITIES, r0_shift=3, r1_shift=7,
    )

    held_out = make_held_out_triples(VocabConfig(**vocab_kwargs), HOLDOUT_FRACTION, seed)

    vocab_cfg = VocabConfig(**vocab_kwargs, held_out_triples=held_out)
    model_cfg = ModelConfig(n_layers=1, hidden_dim=16, n_heads=4)

    train_triples = vocab_cfg.train_triples
    held_out_triples = held_out
    n_valid = len(vocab_cfg.relation_objects(0, 0))
    chance = float(n_valid) / N_ENTITIES

    # For affine analysis: fit on first half, test on second half
    vocab_affine = VocabConfig(**{**vocab_kwargs, 'n_train_subjects': N_ENTITIES // 2})

    max_rel_epochs = max(rel_epochs_list)
    tag = "pretrained" if from_pretrained else "random_init"

    print("=" * 70)
    print(f"ONE-EPOCH Phase 2: {tag} (seed={seed})")
    print(f"Train triples: {len(train_triples)}, Held-out: {len(held_out_triples)}")
    print(f"Relation epochs: {rel_epochs_list}, Chance: {chance:.2%}")
    print("=" * 70)
    print(flush=True)

    torch.manual_seed(seed)
    model = build_model(model_cfg, vocab_cfg)

    if from_pretrained:
        ckpt_path = ROOT / f"checkpoints_one_epoch_p1_seed{seed}" / "best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded phase 1 checkpoint from {ckpt_path}", flush=True)
        print(f"Phase 1 trained for {ckpt['epoch']} epochs", flush=True)
    else:
        print("Starting from random initialization", flush=True)

    model.to(device)
    model.train()

    valid_table = build_valid_attr_table(
        Config(vocab=vocab_cfg, model=model_cfg, train=TrainConfig()), device)

    # Evaluate before any relation training
    pre_train_rel = eval_relation_accuracy(model, vocab_cfg, device, train_triples)
    pre_held_rel = eval_relation_accuracy(model, vocab_cfg, device, held_out_triples)
    pre_attr = eval_attr_accuracy(model, vocab_cfg, device, valid_table, None)

    cfg_affine = Config(
        vocab=vocab_affine,
        model=model_cfg,
        train=TrainConfig(),
        seed=seed,
    )
    pre_affine = affine_analysis(model, cfg_affine, device, layer=-1)

    print(f"Before phase 2:")
    print(f"  train_rel={pre_train_rel:.3f}  held_rel={pre_held_rel:.3f}  attr={pre_attr:.3f}")
    print(f"  affine_train={pre_affine['overall']['train_acc']:.3f}  "
          f"affine_test={pre_affine['overall']['test_acc']:.3f}")
    print(flush=True)

    results = {
        "seed": seed,
        "from_pretrained": from_pretrained,
        "n_train_triples": len(train_triples),
        "n_held_triples": len(held_out_triples),
        "chance": chance,
        "pre_phase2": {
            "train_rel": round(pre_train_rel, 4),
            "held_rel": round(pre_held_rel, 4),
            "attr_acc": round(pre_attr, 4),
            "affine_train": round(pre_affine["overall"]["train_acc"], 4),
            "affine_test": round(pre_affine["overall"]["test_acc"], 4),
        },
        "after_epochs": {},
    }

    # Set up optimizer — lower lr than phase 1 to avoid catastrophic forgetting
    lr = 3e-5
    warmup_steps = 10
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=1e-2,
    )

    rng = random.Random(seed + 2000)
    t0 = time.time()
    global_step = 0

    # Estimate total steps for lr schedule
    test_seqs = build_exhaustive_relation_sequences(
        vocab_cfg, train_triples, SEQ_LENGTH, random.Random(seed + 3000))
    steps_per_rel_epoch = max(1, len(test_seqs) // BATCH_SIZE)
    max_steps = steps_per_rel_epoch * max_rel_epochs
    del test_seqs

    for rel_epoch in range(1, max_rel_epochs + 1):
        sequences = build_exhaustive_relation_sequences(
            vocab_cfg, train_triples, SEQ_LENGTH, rng)

        random.Random(seed + rel_epoch).shuffle(sequences)

        epoch_loss = 0.0
        n_batches = 0
        for batch_start in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[batch_start:batch_start + BATCH_SIZE]
            if len(batch_seqs) < 2:
                continue

            x_list = [torch.tensor(s[:-1], dtype=torch.long) for s in batch_seqs]
            y_list = [torch.tensor(s[1:], dtype=torch.long) for s in batch_seqs]
            x = torch.stack(x_list).to(device)
            y = torch.stack(y_list).to(device)

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
            train_rel = eval_relation_accuracy(model, vocab_cfg, device, train_triples)
            held_rel = eval_relation_accuracy(model, vocab_cfg, device, held_out_triples)
            attr_acc = eval_attr_accuracy(model, vocab_cfg, device, valid_table, None)
            aff = affine_analysis(model, cfg_affine, device, layer=-1)

            snapshot = {
                "rel_epoch": rel_epoch,
                "train_loss": round(avg_loss, 4),
                "train_rel": round(train_rel, 4),
                "held_rel": round(held_rel, 4),
                "attr_acc": round(attr_acc, 4),
                "affine_train": round(aff["overall"]["train_acc"], 4),
                "affine_test": round(aff["overall"]["test_acc"], 4),
            }
            results["after_epochs"][rel_epoch] = snapshot

            elapsed = time.time() - t0
            print(f"Rel epoch {rel_epoch:2d} | loss={avg_loss:.4f} | "
                  f"train_rel={train_rel:.3f} held_rel={held_rel:.3f} | "
                  f"attr={attr_acc:.3f} | "
                  f"aff_train={aff['overall']['train_acc']:.3f} "
                  f"aff_test={aff['overall']['test_acc']:.3f} | "
                  f"{elapsed:.0f}s", flush=True)

            model.train()
        else:
            elapsed = time.time() - t0
            print(f"Rel epoch {rel_epoch:2d} | loss={avg_loss:.4f} | {elapsed:.0f}s",
                  flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rel-epochs", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--skip-control", action="store_true",
                        help="Skip the no-attribute control run")
    args = parser.parse_args()

    rel_epochs_list = sorted(args.rel_epochs)

    # --- Run with pretrained phase 1 ---
    print("\n" + "=" * 70)
    print("CONDITION 1: From pretrained (attribute-only) checkpoint")
    print("=" * 70 + "\n")
    pretrained_results = run_phase2(args.seed, rel_epochs_list, from_pretrained=True)

    # --- Run from random init (no-attribute control) ---
    control_results = None
    if not args.skip_control:
        print("\n" + "=" * 70)
        print("CONDITION 2: From random initialization (no attribute pretraining)")
        print("=" * 70 + "\n")
        control_results = run_phase2(args.seed, rel_epochs_list, from_pretrained=False)

    # --- Save all results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"one_epoch_phase2_seed{args.seed}.json"
    all_results = {
        "seed": args.seed,
        "rel_epochs_evaluated": rel_epochs_list,
        "pretrained": pretrained_results,
        "control": control_results,
    }
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}", flush=True)

    # --- Print comparison table ---
    print("\n" + "=" * 70)
    print("COMPARISON: Pretrained vs Random Init")
    print("=" * 70)
    print(f"{'Epoch':>6} | {'Pretrained':^30} | {'Random Init':^30}")
    print(f"{'':>6} | {'train_rel':>9} {'held_rel':>9} {'affine_t':>9} | "
          f"{'train_rel':>9} {'held_rel':>9} {'affine_t':>9}")
    print("-" * 70)

    print(f"{'pre':>6} | "
          f"{pretrained_results['pre_phase2']['train_rel']:>9.3f} "
          f"{pretrained_results['pre_phase2']['held_rel']:>9.3f} "
          f"{pretrained_results['pre_phase2']['affine_test']:>9.3f} | ", end="")
    if control_results:
        print(f"{control_results['pre_phase2']['train_rel']:>9.3f} "
              f"{control_results['pre_phase2']['held_rel']:>9.3f} "
              f"{control_results['pre_phase2']['affine_test']:>9.3f}")
    else:
        print("(skipped)")

    for ep in rel_epochs_list:
        ep_key = ep
        p = pretrained_results["after_epochs"].get(ep_key, {})
        print(f"{ep:>6} | "
              f"{p.get('train_rel', 0):>9.3f} "
              f"{p.get('held_rel', 0):>9.3f} "
              f"{p.get('affine_test', 0):>9.3f} | ", end="")
        if control_results:
            c = control_results["after_epochs"].get(ep_key, {})
            print(f"{c.get('train_rel', 0):>9.3f} "
                  f"{c.get('held_rel', 0):>9.3f} "
                  f"{c.get('affine_test', 0):>9.3f}")
        else:
            print("(skipped)")


if __name__ == "__main__":
    main()

"""
Random control experiment: original set (informative attributes) vs
control set (random attributes, random relations).

Both sets have 2025 subjects/objects, 10+10 attributes, 2 relations,
45 objects per (subject, relation). Trained together in the same model.

Original set: attributes define types, types determine relation targets.
Control set: attributes are random, relation targets are random.
The control set requires pure memorization; the original can generalize.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v7_random_control.py --seed 42
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v7_random_control.py --seed 42 --p-flip 0.10
"""
import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.affine import affine_analysis_centroid
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import (
    ATTR_SENTENCE_LEN, REL_SENTENCE_LEN, REVERSE_SHORT_SENTENCE_LEN,
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


class ControlSet:
    """Random control set: same structure, no attribute-relation correlation."""

    def __init__(self, n_entities, n_f_attrs, n_g_attrs, k_per_group,
                 n_objs_per_rel, orig_vocab_size, seed):
        self.n_entities = n_entities
        self.n_f_attrs = n_f_attrs
        self.n_g_attrs = n_g_attrs
        self.k_per_group = k_per_group
        self.n_objs_per_rel = n_objs_per_rel

        self.subject_offset = orig_vocab_size
        self.object_offset = self.subject_offset + n_entities
        self.f_attr_offset = self.object_offset + n_entities
        self.g_attr_offset = self.f_attr_offset + n_f_attrs
        self.rel_offset = self.g_attr_offset + n_g_attrs
        self.total_new_tokens = 2 * n_entities + n_f_attrs + n_g_attrs + 2

        rng = random.Random(seed)
        combos = list(combinations(range(n_f_attrs), k_per_group))

        self.entity_f_types = [combos[rng.randint(0, len(combos) - 1)]
                               for _ in range(n_entities)]
        self.entity_g_types = [combos[rng.randint(0, len(combos) - 1)]
                               for _ in range(n_entities)]

        all_objs = list(range(n_entities))
        self.rel_objects = {}
        for i in range(n_entities):
            for k in range(2):
                self.rel_objects[(i, k)] = rng.sample(all_objs, n_objs_per_rel)

    def subject_id(self, i):
        return self.subject_offset + i

    def object_id(self, j):
        return self.object_offset + j

    def f_attr_id(self, a):
        return self.f_attr_offset + a

    def g_attr_id(self, a):
        return self.g_attr_offset + a

    def relation_id(self, k):
        return self.rel_offset + k

    def entity_f_type(self, i):
        return self.entity_f_types[i]

    def entity_g_type(self, i):
        return self.entity_g_types[i]


class ExtendedVocabConfig(VocabConfig):
    """VocabConfig with extended vocab_size to include control set tokens."""

    def __init__(self, extra_tokens, **kwargs):
        super().__init__(**kwargs)
        self._extra_tokens = extra_tokens

    @property
    def vocab_size(self):
        return super().vocab_size + self._extra_tokens


def precompute_relation_objects(vocab, n_objs=None, seed=None):
    f_type_to_objs = {}
    g_type_to_objs = {}
    for j in range(vocab.n_entities):
        ft = vocab.object_f_type(j)
        gt = vocab.object_g_type(j)
        f_type_to_objs.setdefault(ft, []).append(j)
        g_type_to_objs.setdefault(gt, []).append(j)
    if n_objs is not None:
        rng = random.Random(seed)
        for ft in f_type_to_objs:
            if len(f_type_to_objs[ft]) > n_objs:
                f_type_to_objs[ft] = rng.sample(f_type_to_objs[ft], n_objs)
        for gt in g_type_to_objs:
            if len(g_type_to_objs[gt]) > n_objs:
                g_type_to_objs[gt] = rng.sample(g_type_to_objs[gt], n_objs)
    table = {}
    for i in range(vocab.n_entities):
        table[(i, 0)] = f_type_to_objs[vocab.subject_f_type(i)]
        table[(i, 1)] = g_type_to_objs[vocab.subject_g_type(i)]
    return table


def make_split(n_entities, n_rels, train_frac, seed):
    """Simple train/held-out split without per-subject guarantees."""
    rng = random.Random(seed)
    all_pairs = [(i, k) for i in range(n_entities) for k in range(n_rels)]
    rng.shuffle(all_pairs)
    n_train = int(len(all_pairs) * train_frac)
    return all_pairs[:n_train], all_pairs[n_train:]


def generate_attr_sentence_orig(entity_token, vocab, rng, p_flip=0.0):
    f_attrs, g_attrs = vocab.entity_attrs(entity_token)
    if rng.random() < 0.5:
        attr = rng.choice(f_attrs)
        if p_flip > 0 and rng.random() < p_flip:
            attr = flip_attr(attr, vocab.n_f_attrs, rng)
        return [vocab.sep_id, entity_token, vocab.f_attr_id(attr)]
    else:
        attr = rng.choice(g_attrs)
        if p_flip > 0 and rng.random() < p_flip:
            attr = flip_attr(attr, vocab.n_g_attrs, rng)
        return [vocab.sep_id, entity_token, vocab.g_attr_id(attr)]


def generate_attr_sentence_ctrl(entity_idx, is_subject, ctrl, vocab, rng):
    token = ctrl.subject_id(entity_idx) if is_subject else ctrl.object_id(entity_idx)
    f_type = ctrl.entity_f_type(entity_idx)
    g_type = ctrl.entity_g_type(entity_idx)
    if rng.random() < 0.5:
        attr = rng.choice(f_type)
        return [vocab.sep_id, token, ctrl.f_attr_id(attr)]
    else:
        attr = rng.choice(g_type)
        return [vocab.sep_id, token, ctrl.g_attr_id(attr)]


def generate_reverse_short_orig(vocab, rng, p_flip=0.0):
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
        if p_flip > 0 and rng.random() < p_flip:
            attr = flip_attr(attr, vocab.n_f_attrs, rng)
        attr_token = vocab.f_attr_id(attr)
    else:
        attr = rng.choice(g_attrs)
        if p_flip > 0 and rng.random() < p_flip:
            attr = flip_attr(attr, vocab.n_g_attrs, rng)
        attr_token = vocab.g_attr_id(attr)
    return [vocab.sep_id, marker, attr_token, entity_token]


def generate_reverse_short_ctrl(ctrl, vocab, rng):
    idx = rng.randint(0, ctrl.n_entities - 1)
    if rng.random() < 0.5:
        entity_token = ctrl.subject_id(idx)
        marker = vocab.s_marker_id
    else:
        entity_token = ctrl.object_id(idx)
        marker = vocab.o_marker_id
    f_type = ctrl.entity_f_type(idx)
    g_type = ctrl.entity_g_type(idx)
    if rng.random() < 0.5:
        attr_token = ctrl.f_attr_id(rng.choice(f_type))
    else:
        attr_token = ctrl.g_attr_id(rng.choice(g_type))
    return [vocab.sep_id, marker, attr_token, entity_token]


def build_phase1_sequences(vocab, ctrl,
                           orig_train, orig_rel_table,
                           ctrl_train, ctrl_rel_table,
                           n_sequences, seq_length, rng, p_relation,
                           p_flip=0.0, no_attr=False):
    orig_entities = ([vocab.subject_id(i) for i in range(vocab.n_entities)] +
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
                if rng.random() < 0.5:
                    subj_idx, rel_k = rng.choice(orig_train)
                    obj_idx = rng.choice(orig_rel_table[(subj_idx, rel_k)])
                    tokens.extend([vocab.sep_id, vocab.relation_id(rel_k),
                                   vocab.subject_id(subj_idx),
                                   vocab.object_id(obj_idx)])
                else:
                    subj_idx, rel_k = rng.choice(ctrl_train)
                    obj_idx = rng.choice(ctrl_rel_table[(subj_idx, rel_k)])
                    tokens.extend([vocab.sep_id, ctrl.relation_id(rel_k),
                                   ctrl.subject_id(subj_idx),
                                   ctrl.object_id(obj_idx)])
            elif no_attr:
                break
            elif r < p_relation + P_REVERSE_SHORT and remaining >= REVERSE_SHORT_SENTENCE_LEN:
                if rng.random() < 0.5:
                    tokens.extend(generate_reverse_short_orig(vocab, rng, p_flip))
                else:
                    tokens.extend(generate_reverse_short_ctrl(ctrl, vocab, rng))
            else:
                if rng.random() < 0.5:
                    entity = rng.choice(orig_entities)
                    tokens.extend(generate_attr_sentence_orig(entity, vocab, rng, p_flip))
                else:
                    idx = rng.randint(0, ctrl.n_entities - 1)
                    is_subj = rng.random() < 0.5
                    tokens.extend(generate_attr_sentence_ctrl(
                        idx, is_subj, ctrl, vocab, rng))
        while len(tokens) < seq_length:
            tokens.append(vocab.sep_id)
        sequences.append(tokens[:seq_length])
    return sequences


def eval_relation_accuracy_orig(model, vocab, device, triples, rel_objects_table,
                                batch_size=512):
    model.eval()
    correct = 0
    total = len(triples)
    if total == 0:
        return 0.0
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = triples[start:start + batch_size]
            contexts = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(subj_idx)]
                 for subj_idx, rel_k in batch],
                dtype=torch.long, device=device,
            )
            logits = model(contexts)
            preds = logits[:, -1].argmax(dim=-1).cpu().tolist()
            for (subj_idx, rel_k), pred in zip(batch, preds):
                valid = {vocab.object_id(j) for j in rel_objects_table[(subj_idx, rel_k)]}
                correct += int(pred in valid)
    return correct / total


def eval_relation_accuracy_ctrl(model, vocab, ctrl, device, triples, batch_size=512):
    model.eval()
    correct = 0
    total = len(triples)
    if total == 0:
        return 0.0
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = triples[start:start + batch_size]
            contexts = torch.tensor(
                [[vocab.sep_id, ctrl.relation_id(rel_k), ctrl.subject_id(subj_idx)]
                 for subj_idx, rel_k in batch],
                dtype=torch.long, device=device,
            )
            logits = model(contexts)
            preds = logits[:, -1].argmax(dim=-1).cpu().tolist()
            for (subj_idx, rel_k), pred in zip(batch, preds):
                valid = {ctrl.object_id(j) for j in ctrl.rel_objects[(subj_idx, rel_k)]}
                correct += int(pred in valid)
    return correct / total


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
    parser.add_argument("--n-phase1", type=int, default=10)
    parser.add_argument("--p-relation", type=float, default=0.20)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-entities", type=int, default=2025)
    parser.add_argument("--num-sequences", type=int, default=62500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-frac", type=float, default=0.60,
                        help="Fraction of (subj,rel) pairs used for training")
    parser.add_argument("--p-flip", type=float, default=0.0,
                        help="Probability of flipping original set attributes")
    parser.add_argument("--no-attr", action="store_true",
                        help="Remove all attribute/reverse_short sentences")
    parser.add_argument("--n-objs", type=int, default=None,
                        help="Valid objects per (subject,relation), default=n_entities/n_types")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Original set
    vocab_kwargs = dict(
        n_f_attrs=10, n_g_attrs=10, k_per_group=2,
        num_relations=2, n_entities=args.n_entities,
        n_train_subjects=args.n_entities, r0_shift=7, r1_shift=13,
    )
    temp_vocab = VocabConfig(**vocab_kwargs)
    orig_vocab_size = temp_vocab.vocab_size

    # Determine n_objs
    temp_n_types = len(list(combinations(range(10), 2)))  # 45
    default_n_objs = args.n_entities // temp_n_types  # 45
    n_objs = args.n_objs if args.n_objs is not None else default_n_objs

    # Control set
    ctrl = ControlSet(
        n_entities=args.n_entities,
        n_f_attrs=10, n_g_attrs=10, k_per_group=2,
        n_objs_per_rel=n_objs,
        orig_vocab_size=orig_vocab_size,
        seed=args.seed + 7777,
    )
    total_vocab_size = orig_vocab_size + ctrl.total_new_tokens

    # Build vocab with extended size
    vocab = ExtendedVocabConfig(extra_tokens=ctrl.total_new_tokens, **vocab_kwargs)

    print("Precomputing relation objects...", flush=True)
    orig_rel_table = precompute_relation_objects(
        vocab, n_objs=args.n_objs, seed=args.seed + 3333)
    ctrl_rel_table = ctrl.rel_objects

    _fast = lambda subj_idx, rel_k: orig_rel_table[(subj_idx, rel_k)]
    vocab.relation_objects = _fast

    n_valid = len(orig_rel_table[(0, 0)])
    chance = n_valid / args.n_entities

    # Train/held-out splits (independent for each set)
    orig_train, orig_held = make_split(
        args.n_entities, 2, train_frac=args.train_frac, seed=args.seed)
    ctrl_train, ctrl_held = make_split(
        args.n_entities, 2, train_frac=args.train_frac, seed=args.seed + 5555)

    model_cfg = ModelConfig(n_layers=args.n_layers, hidden_dim=args.dim, n_heads=4)

    print(f"\n{'='*70}")
    print(f"RANDOM CONTROL EXPERIMENT (seed={args.seed})")
    print(f"{'='*70}")
    print(f"Types: {vocab.n_types}, Entities: {args.n_entities} per set")
    print(f"Valid objects/subject: {n_valid} (n_objs={n_objs}), Chance: {chance:.2%}")
    print(f"Model: dim={args.dim}, layers={args.n_layers}, heads=4")
    print(f"\nOriginal vocab: {orig_vocab_size} tokens")
    print(f"Control tokens: {ctrl.total_new_tokens}")
    print(f"Total vocab:    {total_vocab_size}")
    print(f"\nOriginal set: attributes define types → types determine relations")
    if args.no_attr:
        print(f"  No attribute sentences (relation sentences only)")
    elif args.p_flip > 0:
        print(f"  Attribute noise: p_flip={args.p_flip:.2f} (original set only)")
    print(f"Control set:  random attributes, random relation targets")
    print(f"\nSplit (each set independently, train_frac={args.train_frac}):")
    print(f"  Train:    {len(orig_train):,} / {len(ctrl_train):,} (subj,rel) pairs")
    print(f"  Held-out: {len(orig_held):,} / {len(ctrl_held):,} pairs")
    print(f"\nPhase 1: {args.n_phase1} epochs, p_relation={args.p_relation}, lr={args.lr}")

    # Build model
    torch.manual_seed(args.seed)
    model = build_model(model_cfg, vocab)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Affine config (original set only)
    vocab_affine_kwargs = {**vocab_kwargs, 'n_train_subjects': args.n_entities // 2}
    vocab_affine = ExtendedVocabConfig(extra_tokens=ctrl.total_new_tokens,
                                       **vocab_affine_kwargs)
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
        orig_train_acc = eval_relation_accuracy_orig(
            model, vocab, device, orig_train, orig_rel_table)
        orig_held_acc = eval_relation_accuracy_orig(
            model, vocab, device, orig_held, orig_rel_table)
        ctrl_train_acc = eval_relation_accuracy_ctrl(
            model, vocab, ctrl, device, ctrl_train)
        ctrl_held_acc = eval_relation_accuracy_ctrl(
            model, vocab, ctrl, device, ctrl_held)
        return {
            "orig_train": round(orig_train_acc, 4),
            "orig_held": round(orig_held_acc, 4),
            "ctrl_train": round(ctrl_train_acc, 4),
            "ctrl_held": round(ctrl_held_acc, 4),
        }

    # Pre-training
    pre = evaluate_all()
    print(f"Pre-training: orig_train={pre['orig_train']:.3f} orig_held={pre['orig_held']:.3f} | "
          f"ctrl_train={pre['ctrl_train']:.3f} ctrl_held={pre['ctrl_held']:.3f}\n", flush=True)

    results = {
        "experiment": "random_control",
        "train_frac": args.train_frac,
        "p_flip": args.p_flip,
        "no_attr": args.no_attr,
        "n_objs": n_objs,
        "seed": args.seed,
        "n_entities": args.n_entities,
        "n_types": vocab.n_types,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_params": n_params,
        "orig_vocab_size": orig_vocab_size,
        "ctrl_new_tokens": ctrl.total_new_tokens,
        "total_vocab_size": total_vocab_size,
        "n_orig_train": len(orig_train),
        "n_orig_held": len(orig_held),
        "n_ctrl_train": len(ctrl_train),
        "n_ctrl_held": len(ctrl_held),
        "n_phase1": args.n_phase1,
        "p_relation": args.p_relation,
        "chance": round(chance, 4),
        "pre": pre,
        "phase1": {},
    }

    # Training
    rng = random.Random(args.seed + 2000)
    global_step = 0
    t0 = time.time()

    print("--- Phase 1: Training (original + control) ---", flush=True)
    for epoch in range(1, args.n_phase1 + 1):
        sequences = build_phase1_sequences(
            vocab, ctrl,
            orig_train, orig_rel_table,
            ctrl_train, ctrl_rel_table,
            args.num_sequences, SEQ_LENGTH, rng, args.p_relation,
            p_flip=args.p_flip, no_attr=args.no_attr)

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
              f"orig: train={ev['orig_train']:.3f} held={ev['orig_held']:.3f} | "
              f"ctrl: train={ev['ctrl_train']:.3f} held={ev['ctrl_held']:.3f} | "
              f"aff={aff['r0']['test_acc']:.3f}/{aff['r1']['test_acc']:.3f} | "
              f"{elapsed:.0f}s", flush=True)

    # Final
    final = evaluate_all()
    final_aff = affine_analysis_centroid(model, cfg_affine, device)
    results["final"] = {
        **final,
        "affine_r0_test": round(final_aff['r0']['test_acc'], 4),
        "affine_r1_test": round(final_aff['r1']['test_acc'], 4),
    }

    # Summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Original set (informative attributes):")
    print(f"    Train:   {final['orig_train']:.3f}")
    print(f"    Held-out:     {final['orig_held']:.3f}")
    print(f"    Affine R0:    {final_aff['r0']['test_acc']:.3f}")
    print(f"    Affine R1:    {final_aff['r1']['test_acc']:.3f}")
    print(f"  Control set (random attributes + relations):")
    print(f"    Train:   {final['ctrl_train']:.3f}")
    print(f"    Held-out:     {final['ctrl_held']:.3f}")
    print(f"  Chance:         {chance:.3f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frac_str = f"{args.train_frac:.2f}".replace(".", "p")
    flip_str = f"_flip{args.p_flip:.2f}".replace(".", "p") if args.p_flip > 0 else ""
    attr_str = "_noattr" if args.no_attr else ""
    objs_str = f"_objs{n_objs}" if args.n_objs is not None else ""
    out_path = RESULTS_DIR / f"one_epoch_v7_random_control_{frac_str}{flip_str}{attr_str}{objs_str}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

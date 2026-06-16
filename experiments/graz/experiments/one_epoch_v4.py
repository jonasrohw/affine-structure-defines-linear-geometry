"""
One-epoch v4 — Reduced object pool for easier prediction.

Like v3 (C(10,2) = 45 types, 2025 subjects) but with only 675 objects:
exactly 15 per f-type and 15 per g-type (vs 45 in v3).

The object subset is selected via iterative perfect matching decomposition
of the bipartite f-type × g-type graph (45-regular → extract 15 matchings).

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v4.py --seed 42
    CUDA_VISIBLE_DEVICES=0 python experiments/one_epoch_v4.py --seed 42 --keep-per-type 15
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.affine import (
    extract_subject_hidden, fit_affine, apply_affine,
    reconstruction_accuracy, stack_pairs,
)
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


# ---------- Object subset selection via perfect matching ----------

def _find_matching(adj, n):
    """Perfect matching in bipartite graph via augmenting paths.
    adj[u] = list of v's that u can be matched to.
    Returns {u: v} dict of size n, or None if no perfect matching exists.
    """
    match_v = [-1] * n

    def dfs(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_v[v] == -1 or dfs(match_v[v], visited):
                    match_v[v] = u
                    return True
        return False

    matched = 0
    for u in range(n):
        visited = [False] * n
        if dfs(u, visited):
            matched += 1

    if matched < n:
        return None
    result = {}
    for v, u in enumerate(match_v):
        if u != -1:
            result[u] = v
    return result


def select_object_subset(vocab, n_keep_per_type, seed):
    """Select objects: exactly n_keep_per_type per f-type AND per g-type.

    The 2025 objects form a 45-regular bipartite graph (f-types × g-types).
    We iteratively extract perfect matchings (each selects 1 per f-type and
    1 per g-type) and take n_keep_per_type of them → uniform counts on both sides.
    """
    rng = random.Random(seed)
    n_types = vocab.n_types
    combos = vocab.combos
    type_to_idx = {t: i for i, t in enumerate(combos)}

    cells = {}
    for j in range(vocab.n_entities):
        fi = type_to_idx[vocab.object_f_type(j)]
        gi = type_to_idx[vocab.object_g_type(j)]
        cells.setdefault((fi, gi), []).append(j)

    for key in cells:
        rng.shuffle(cells[key])

    cell_ptrs = {key: 0 for key in cells}
    selected = []

    for m in range(n_keep_per_type):
        adj = [[] for _ in range(n_types)]
        for (fi, gi), objs in cells.items():
            if cell_ptrs[(fi, gi)] < len(objs):
                adj[fi].append(gi)
        for fi in range(n_types):
            rng.shuffle(adj[fi])

        matching = _find_matching(adj, n_types)
        assert matching is not None, f"Failed to find matching {m+1}/{n_keep_per_type}"

        for fi, gi in matching.items():
            obj = cells[(fi, gi)][cell_ptrs[(fi, gi)]]
            cell_ptrs[(fi, gi)] += 1
            selected.append(obj)

    return set(selected)


# ---------- Relation objects with subset filtering ----------

def precompute_relation_objects(vocab, object_subset):
    """Build (subj_idx, rel_k) -> [obj_indices] lookup, filtered to subset."""
    f_type_to_objs = {}
    g_type_to_objs = {}
    for j in object_subset:
        ft = vocab.object_f_type(j)
        gt = vocab.object_g_type(j)
        f_type_to_objs.setdefault(ft, []).append(j)
        g_type_to_objs.setdefault(gt, []).append(j)
    table = {}
    for i in range(vocab.n_entities):
        table[(i, 0)] = f_type_to_objs.get(vocab.subject_f_type(i), [])
        table[(i, 1)] = g_type_to_objs.get(vocab.subject_g_type(i), [])
    return table


# ---------- Held-out / fact generation (same as v3) ----------

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


# ---------- Evaluation ----------

def eval_relation_accuracy(model, vocab, device, triples, rel_objects_table,
                           object_subset, batch_size=512):
    """Relation accuracy with logits masked to subset objects only."""
    model.eval()
    mask = torch.full((vocab.vocab_size,), float('-inf'), device=device)
    for j in object_subset:
        mask[vocab.object_id(j)] = 0.0

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
            masked_logits = logits[:, -1] + mask
            preds = masked_logits.argmax(dim=-1).cpu().tolist()
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


# ---------- Affine analysis with subset ----------

def affine_analysis_subset(model, cfg, device, object_subset, layer=-1):
    """Affine probe: fit on train subjects, evaluate on test, search among subset objects."""
    vocab = cfg.vocab
    all_subjects = list(range(vocab.n_entities))
    h_all = extract_subject_hidden(model, cfg, all_subjects, device, layer=layer)

    weight = model.lm_head.weight.detach().cpu()
    t_all = {}
    for i in all_subjects:
        for k in range(vocab.num_relations):
            objs = vocab.relation_objects(i, k)
            t_all[(i, k)] = weight[vocab.object_id(objs[0])]

    subset_list = sorted(object_subset)
    O_all = weight[[vocab.object_id(j) for j in subset_list]]

    train_subjs = vocab.train_subject_indices
    test_subjs = vocab.test_subject_indices

    results = {}
    for k in range(vocab.num_relations):
        S_train, O_train = stack_pairs(h_all, t_all, train_subjs, k)
        S_test, O_test = stack_pairs(h_all, t_all, test_subjs, k)
        W, b = fit_affine(S_train, O_train, bias=True)
        train_acc = reconstruction_accuracy(S_train, O_train, W, b, O_all)
        test_acc = reconstruction_accuracy(S_test, O_test, W, b, O_all)
        results[f"r{k}"] = {"train_acc": train_acc, "test_acc": test_acc}

    all_train = np.mean([results[f"r{k}"]["train_acc"] for k in range(vocab.num_relations)])
    all_test = np.mean([results[f"r{k}"]["test_acc"] for k in range(vocab.num_relations)])
    results["overall"] = {"train_acc": float(all_train), "test_acc": float(all_test)}
    return results


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-entities", type=int, default=2025)
    parser.add_argument("--keep-per-type", type=int, default=15)
    parser.add_argument("--num-sequences", type=int, default=62500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Config — all 2025 entities in the vocab (subjects + objects)
    vocab_kwargs = dict(
        n_f_attrs=10, n_g_attrs=10, k_per_group=2,
        num_relations=2, n_entities=args.n_entities,
        n_train_subjects=args.n_entities, r0_shift=7, r1_shift=13,
    )

    temp_vocab = VocabConfig(**vocab_kwargs)

    # Select object subset: exactly keep_per_type per f-type AND per g-type
    print(f"Selecting {args.keep_per_type} objects per type "
          f"(of {args.n_entities // temp_vocab.n_types} per type)...", flush=True)
    object_subset = select_object_subset(temp_vocab, args.keep_per_type, args.seed + 3000)
    n_objects = len(object_subset)
    n_valid = args.keep_per_type
    print(f"Object subset: {n_objects} objects ({n_valid} per type)", flush=True)

    # Verify uniform type counts
    combos = temp_vocab.combos
    type_to_idx = {t: i for i, t in enumerate(combos)}
    f_counts = [0] * temp_vocab.n_types
    g_counts = [0] * temp_vocab.n_types
    for j in object_subset:
        f_counts[type_to_idx[temp_vocab.object_f_type(j)]] += 1
        g_counts[type_to_idx[temp_vocab.object_g_type(j)]] += 1
    assert all(c == n_valid for c in f_counts), f"f-type counts not uniform: {set(f_counts)}"
    assert all(c == n_valid for c in g_counts), f"g-type counts not uniform: {set(g_counts)}"
    print("Verified: uniform type counts on both f and g sides", flush=True)

    # Precompute relation objects (subset only)
    print("Precomputing relation objects (subset)...", flush=True)
    rel_objects_table = precompute_relation_objects(temp_vocab, object_subset)

    held_out = make_held_out_triples(temp_vocab, HOLDOUT_FRACTION, args.seed)
    vocab = VocabConfig(**vocab_kwargs, held_out_triples=held_out)
    model_cfg = ModelConfig(n_layers=args.n_layers, hidden_dim=args.dim, n_heads=4)

    _fast = lambda subj_idx, rel_k: rel_objects_table[(subj_idx, rel_k)]
    vocab.relation_objects = _fast

    train_triples = vocab.train_triples
    chance = n_valid / n_objects

    print(f"\n{'='*70}")
    print(f"ONE-EPOCH v4: Reduced object pool (seed={args.seed})")
    print(f"{'='*70}")
    print(f"Types: {vocab.n_types}, Subjects: {args.n_entities}, "
          f"Objects: {n_objects} (of {args.n_entities})")
    print(f"Valid objects/subject: {n_valid}, Chance: {chance:.2%}")
    print(f"Model: dim={args.dim}, layers={args.n_layers}, "
          f"vocab={vocab.vocab_size}, heads=4")
    print(f"Training (subj,rel) pairs: {len(train_triples)}, "
          f"Held-out: {len(held_out)}")

    # Enumerate and split facts
    print("Enumerating training facts...", flush=True)
    all_facts = enumerate_facts(train_triples, rel_objects_table)
    print(f"Total training facts: {len(all_facts):,}")

    fact_chunks = split_facts(all_facts, args.n_epochs, args.seed + 1000)
    print(f"Facts per epoch: {len(fact_chunks[0]):,}")

    # Build model (vocab size unchanged — includes all 2025 object tokens)
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

    # Affine config: train on first half, test on second half
    vocab_affine_kwargs = {**vocab_kwargs, 'n_train_subjects': args.n_entities // 2}
    vocab_affine = VocabConfig(**vocab_affine_kwargs, held_out_triples=held_out)
    vocab_affine.relation_objects = _fast
    cfg_affine = Config(vocab=vocab_affine, model=model_cfg, train=TrainConfig(),
                        seed=args.seed)

    # Optimizer and schedule
    warmup_steps = 200
    steps_per_epoch = args.num_sequences // args.batch_size
    max_steps = steps_per_epoch * args.n_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    fact_density = len(fact_chunks[0]) / args.num_sequences
    print(f"\nSteps/epoch: {steps_per_epoch:,}, Total steps: {max_steps:,}")
    print(f"Fact density: {fact_density:.1%} of sequences contain a relation fact")
    print(f"LR: {args.lr}, Warmup: {warmup_steps} steps, "
          f"Cosine to {max_steps} steps")
    print(f"{'='*70}\n", flush=True)

    # Pre-training evaluation
    pre_attr = eval_attr_accuracy(model, cfg_eval, device, valid_table)
    pre_train_rel = eval_relation_accuracy(
        model, vocab, device, train_triples, rel_objects_table, object_subset)
    pre_held_rel = eval_relation_accuracy(
        model, vocab, device, held_out, rel_objects_table, object_subset)

    print(f"Pre-training: attr={pre_attr:.3f}  train_rel={pre_train_rel:.3f}  "
          f"held_rel={pre_held_rel:.3f}\n", flush=True)

    results = {
        "seed": args.seed,
        "n_entities": args.n_entities,
        "n_objects": n_objects,
        "keep_per_type": args.keep_per_type,
        "n_types": vocab.n_types,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_params": n_params,
        "n_train_triples": len(train_triples),
        "n_held_out": len(held_out),
        "n_total_facts": len(all_facts),
        "n_epochs": args.n_epochs,
        "chance": round(chance, 4),
        "vocab_size": vocab.vocab_size,
        "fact_density": round(fact_density, 4),
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

    for epoch in range(1, args.n_epochs + 1):
        t_data = time.time()
        sequences = build_epoch_sequences(
            vocab, fact_chunks[epoch - 1], args.num_sequences, SEQ_LENGTH, rng)
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
        train_rel = eval_relation_accuracy(
            model, vocab, device, train_triples, rel_objects_table, object_subset)
        held_rel = eval_relation_accuracy(
            model, vocab, device, held_out, rel_objects_table, object_subset)
        aff = affine_analysis_subset(model, cfg_affine, device, object_subset, layer=-1)
        eval_time = time.time() - t_eval

        elapsed = time.time() - t0

        snapshot = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "attr": round(attr_acc, 4),
            "train_rel": round(train_rel, 4),
            "held_rel": round(held_rel, 4),
            "affine_train": round(aff["overall"]["train_acc"], 4),
            "affine_test": round(aff["overall"]["test_acc"], 4),
            "data_time": round(data_time, 1),
            "eval_time": round(eval_time, 1),
        }
        results["epochs"][epoch] = snapshot

        print(f"Ep {epoch:2d}/{args.n_epochs} | loss={avg_loss:.4f} | "
              f"attr={attr_acc:.3f} | tr_rel={train_rel:.3f} | "
              f"hd_rel={held_rel:.3f} | "
              f"aff={aff['overall']['train_acc']:.3f}/"
              f"{aff['overall']['test_acc']:.3f} | "
              f"data={data_time:.0f}s eval={eval_time:.0f}s | "
              f"{elapsed:.0f}s", flush=True)

        model.train()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"one_epoch_v4_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Save checkpoint
    ckpt_dir = ROOT / f"checkpoints_v4_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "seed": args.seed,
        "vocab_kwargs": vocab_kwargs,
        "held_out_triples": held_out,
        "object_subset": sorted(object_subset),
        "keep_per_type": args.keep_per_type,
        "model_cfg": {"n_layers": args.n_layers, "hidden_dim": args.dim, "n_heads": 4},
    }, ckpt_dir / "final.pt")
    print(f"Checkpoint saved to {ckpt_dir / 'final.pt'}", flush=True)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY (seed={args.seed}, dim={args.dim}, layers={args.n_layers}, "
          f"objects={n_objects})")
    print(f"{'='*70}")
    print(f"{'epoch':>6} | {'loss':>6} | {'attr':>5} | {'tr_rel':>6} | "
          f"{'hd_rel':>6} | {'aff_te':>6}")
    print("-" * 55)
    print(f"{'pre':>6} | {'—':>6} | {pre_attr:>5.3f} | "
          f"{pre_train_rel:>6.3f} | {pre_held_rel:>6.3f} | {'—':>6}")
    for ep in range(1, args.n_epochs + 1):
        s = results["epochs"][ep]
        print(f"{ep:>6} | {s['loss']:>6.3f} | {s['attr']:>5.3f} | "
              f"{s['train_rel']:>6.3f} | {s['held_rel']:>6.3f} | "
              f"{s['affine_test']:>6.3f}")


if __name__ == "__main__":
    main()

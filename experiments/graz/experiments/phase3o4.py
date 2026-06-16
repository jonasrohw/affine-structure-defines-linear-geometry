"""
Phase 3o4: 5 features with dim=8.

C(5,2) = 10 types, 10² = 100 entities.
Shifts coprime to 10: r0_shift=3, r1_shift=7.

Usage (from graz/):
    python experiments/phase3o4.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random

import torch

from src.affine import affine_analysis_all_layers
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import (make_dataloader, generate_reverse_short_sentence,
                                  generate_contrastive_sentence)
from src.evaluation import similarity_report
from src.model import build_model
from src.training import evaluate, build_valid_attr_table, train

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_RELATION = 0.264
P_CONTRASTIVE = 0.20
P_REVERSE_SHORT = 0.264
HOLDOUT_FRACTION = 0.20
SEED = 42
NUM_SEQUENCES = 62500


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


def main():
    n_f_attrs = 5
    n_g_attrs = 5
    k_per_group = 2
    n_entities = 100  # 10² = 100
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
        model=ModelConfig(n_layers=1, hidden_dim=8, n_heads=4),
        train=TrainConfig(
            max_epochs=800,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=48,
            num_sequences=NUM_SEQUENCES,
            checkpoint_dir=ROOT / "checkpoints_phase3o4",
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
    n_per_type = n_entities // n_types

    p_attr = 1.0 - P_RELATION - P_CONTRASTIVE - P_REVERSE_SHORT
    n_valid = len(cfg.vocab.relation_objects(0, 0))
    chance = float(n_valid) / n_entities

    print("=" * 70)
    print(f"GRAZ Phase 3o4: 5 features, dim=8, k=2 ({n_types} types, {n_entities} entities)")
    print(f"Entities: {n_entities} subjects, {n_entities} objects ({n_per_type} per type)")
    print(f"Types: C({n_f_attrs},{k_per_group}) = {n_types} | k_per_group={k_per_group}")
    print(f"Total triples: {n_total} | Train: {n_train} | Held out: {n_held}")
    print(f"p_relation={P_RELATION}, p_contrastive={P_CONTRASTIVE}, "
          f"p_reverse_short={P_REVERSE_SHORT}, p_attr={p_attr:.2f}")
    print(f"Each relation has {n_valid} valid targets per subject")
    print(f"r0_shift={r0_shift}, r1_shift={r1_shift}, dim=64, 1 layer, 4 heads")
    print(f"Sequences: {NUM_SEQUENCES}")
    print("=" * 70)
    print()

    # Show example contrastive sentences
    rng = random.Random(42)
    print("Example contrastive sentences:")
    for _ in range(4):
        s = generate_contrastive_sentence(cfg.vocab, rng, train_triples)
        print(f"  {' '.join(cfg.vocab.token_to_str(t) for t in s)}")
    print()

    # Verify relation structure
    print("Relation structure check:")
    print(f"  R0 targets for subject 0: {cfg.vocab.relation_objects(0, 0)[:5]}...")
    print(f"  R1 targets for subject 0: {cfg.vocab.relation_objects(0, 1)[:5]}...")
    print()

    # Train
    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, seed_offset=0, p_relation=P_RELATION,
                                   train_triples=train_triples,
                                   p_reverse_short=P_REVERSE_SHORT,
                                   p_contrastive=P_CONTRASTIVE)
    model = build_model(cfg.model, cfg.vocab)
    print(f"Parameters: {model.n_parameters():,}")
    print(f"Vocab size: {cfg.vocab.vocab_size}")

    train(model=model, train_loader=train_loader, cfg=cfg, device=device,
          val_loader=train_loader, verbose=True)

    model.to(device)
    model.eval()

    # Evaluate attribute validity
    valid_table = build_valid_attr_table(cfg, device)
    final_metrics = evaluate(model, train_loader, device, cfg, valid_table, max_batches=50)
    print(f"\nFinal: loss={final_metrics['loss']:.4f} | "
          f"loss_attr={final_metrics['loss_attr']:.4f} | "
          f"attr_valid={final_metrics['acc_attr_valid']:.4f}")

    # Evaluate contrastive accuracy
    print("\nContrastive accuracy (does model predict y/n correctly?):")
    n_eval = 500
    rng_eval = random.Random(456)
    correct_pos = 0
    correct_neg = 0
    total_pos = 0
    total_neg = 0
    with torch.no_grad():
        for _ in range(n_eval):
            subj_idx = rng_eval.randint(0, cfg.vocab.n_entities - 1)
            rel_k = rng_eval.randint(0, cfg.vocab.num_relations - 1)
            valid_objects = set(cfg.vocab.relation_objects(subj_idx, rel_k))

            # Positive
            obj_idx = rng_eval.choice(list(valid_objects))
            context = torch.tensor(
                [[cfg.vocab.sep_id, cfg.vocab.query_id(rel_k),
                  cfg.vocab.object_id(obj_idx), cfg.vocab.subject_id(subj_idx)]],
                dtype=torch.long, device=device)
            pred = model(context)[0, -1].argmax().item()
            correct_pos += int(pred == cfg.vocab.y_id)
            total_pos += 1

            # Negative
            invalid = [j for j in range(cfg.vocab.n_entities) if j not in valid_objects]
            obj_idx = rng_eval.choice(invalid)
            context = torch.tensor(
                [[cfg.vocab.sep_id, cfg.vocab.query_id(rel_k),
                  cfg.vocab.object_id(obj_idx), cfg.vocab.subject_id(subj_idx)]],
                dtype=torch.long, device=device)
            pred = model(context)[0, -1].argmax().item()
            correct_neg += int(pred == cfg.vocab.n_id)
            total_neg += 1

    print(f"  Positive (valid triple → y): {correct_pos}/{total_pos} = {correct_pos/total_pos:.4f}")
    print(f"  Negative (invalid triple → n): {correct_neg}/{total_neg} = {correct_neg/total_neg:.4f}")

    # Evaluate short reverse lookup validity
    print("\nShort reverse lookup (is prediction valid for the given attr?):")
    rng_eval = random.Random(123)
    s_valid = 0
    o_valid = 0
    s_total = 0
    o_total = 0
    with torch.no_grad():
        for _ in range(n_eval):
            idx = rng_eval.randint(0, cfg.vocab.n_entities - 1)
            f_attrs = list(cfg.vocab.subject_f_type(idx))
            g_attrs = list(cfg.vocab.subject_g_type(idx))
            if rng_eval.random() < 0.5:
                attr_token = cfg.vocab.f_attr_id(rng_eval.choice(f_attrs))
                valid_subjects = [i for i in range(cfg.vocab.n_entities)
                                  if (attr_token - cfg.vocab.f_attr_offset) in cfg.vocab.subject_f_type(i)]
            else:
                attr_token = cfg.vocab.g_attr_id(rng_eval.choice(g_attrs))
                valid_subjects = [i for i in range(cfg.vocab.n_entities)
                                  if (attr_token - cfg.vocab.g_attr_offset) in cfg.vocab.subject_g_type(i)]
            context = torch.tensor([[cfg.vocab.sep_id, cfg.vocab.s_marker_id, attr_token]],
                                   dtype=torch.long, device=device)
            pred = model(context)[0, -1].argmax().item()
            valid_ids = [cfg.vocab.subject_id(i) for i in valid_subjects]
            s_valid += int(pred in valid_ids)
            s_total += 1

            idx = rng_eval.randint(0, cfg.vocab.n_entities - 1)
            f_attrs = list(cfg.vocab.object_f_type(idx))
            g_attrs = list(cfg.vocab.object_g_type(idx))
            if rng_eval.random() < 0.5:
                attr_token = cfg.vocab.f_attr_id(rng_eval.choice(f_attrs))
                valid_objects = [j for j in range(cfg.vocab.n_entities)
                                 if (attr_token - cfg.vocab.f_attr_offset) in cfg.vocab.object_f_type(j)]
            else:
                attr_token = cfg.vocab.g_attr_id(rng_eval.choice(g_attrs))
                valid_objects = [j for j in range(cfg.vocab.n_entities)
                                 if (attr_token - cfg.vocab.g_attr_offset) in cfg.vocab.object_g_type(j)]
            context = torch.tensor([[cfg.vocab.sep_id, cfg.vocab.o_marker_id, attr_token]],
                                   dtype=torch.long, device=device)
            pred = model(context)[0, -1].argmax().item()
            valid_ids = [cfg.vocab.object_id(j) for j in valid_objects]
            o_valid += int(pred in valid_ids)
            o_total += 1

    print(f"  S + attr → valid subject: {s_valid}/{s_total} = {s_valid/s_total:.4f}")
    print(f"  O + attr → valid object: {o_valid}/{o_total} = {o_valid/o_total:.4f}")

    # Evaluate relation accuracy (accept any valid target)
    print("\nRelation accuracy (any valid target counts as correct):")
    train_correct = 0
    test_correct = 0
    train_set = set(train_triples)

    train_by_rel = {k: {"correct": 0, "total": 0} for k in range(cfg.vocab.num_relations)}
    test_by_rel = {k: {"correct": 0, "total": 0} for k in range(cfg.vocab.num_relations)}

    with torch.no_grad():
        for subj_idx in range(cfg.vocab.n_entities):
            for rel_k in range(cfg.vocab.num_relations):
                context = torch.tensor(
                    [[cfg.vocab.sep_id, cfg.vocab.relation_id(rel_k), cfg.vocab.subject_id(subj_idx)]],
                    dtype=torch.long, device=device,
                )
                logits = model(context)
                pred = logits[0, -1].argmax().item()
                valid_targets = [cfg.vocab.object_id(j) for j in cfg.vocab.relation_objects(subj_idx, rel_k)]
                correct = int(pred in valid_targets)
                if (subj_idx, rel_k) in train_set:
                    train_correct += correct
                    train_by_rel[rel_k]["correct"] += correct
                    train_by_rel[rel_k]["total"] += 1
                else:
                    test_correct += correct
                    test_by_rel[rel_k]["correct"] += correct
                    test_by_rel[rel_k]["total"] += 1

    train_acc = train_correct / n_train
    test_acc = test_correct / n_held
    print(f"  Train triples ({n_train}): {train_acc:.4f}")
    print(f"  Held-out triples ({n_held}): {test_acc:.4f}")

    print("\n  Per-relation breakdown:")
    for k in range(cfg.vocab.num_relations):
        tr = train_by_rel[k]
        te = test_by_rel[k]
        tr_acc = tr["correct"] / max(1, tr["total"])
        te_acc = te["correct"] / max(1, te["total"])
        print(f"    R{k}: train={tr_acc:.4f} ({tr['total']} triples) | "
              f"test={te_acc:.4f} ({te['total']} triples)")

    print(f"\n  Chance level ({n_valid}/{n_entities}): {chance:.4f}")

    # Affine analysis (half/half subject split)
    affine_train_subjs = n_entities // 2
    print(f"\nAffine analysis ({affine_train_subjs}/{n_entities} subject split)...")
    cfg_affine = Config(
        vocab=VocabConfig(**{**vocab_kwargs, 'n_train_subjects': affine_train_subjs}),
        model=cfg.model,
        train=cfg.train,
        seed=cfg.seed,
    )
    affine_all = affine_analysis_all_layers(model, cfg_affine, device)
    for label, layer_results in affine_all.items():
        print(f"  {label}: train={layer_results['overall']['train_acc']:.4f} "
              f"test={layer_results['overall']['test_acc']:.4f}")

    # Embedding structure
    report = similarity_report(model, cfg.vocab)
    print(f"\nEmbedding structure:")
    max_ov = k_per_group
    for ov in range(max_ov):
        print(f"  Subj {ov}/{max_ov} overlap: {report['subj_by_f_overlap'][ov]:+.4f}")
    print(f"  R0 partner sim: {report['r0_partner_sim']:+.4f}")
    print(f"  R1 partner sim: {report['r1_partner_sim']:+.4f}")

    # Save results
    n_seqs = NUM_SEQUENCES
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "phase3o4_results.md"
    with open(results_path, "w") as f:
        f.write(f"# Phase 3o4: 5 features, dim=8, k=2 ({n_types} types, {n_entities} entities)\n\n")
        f.write("## Setup\n\n")
        f.write(f"- {n_entities} subjects × 2 relations = {n_total} total triples\n")
        f.write(f"- k_per_group=2: C({n_f_attrs},{k_per_group}) = {n_types} types, "
                f"{n_per_type} entities per type\n")
        f.write(f"- {n_valid} valid targets per relation per subject\n")
        f.write(f"- Train: {n_train} triples, Held out: {n_held} triples\n")
        f.write(f"- p_relation={P_RELATION}, p_contrastive={P_CONTRASTIVE}, "
                f"p_reverse_short={P_REVERSE_SHORT}, p_attr={p_attr:.2f}\n")
        f.write(f"- Contrastive sentences: [x, Qk, object, subject, y/n]\n")
        f.write(f"- Architecture: dim=64, 1 layer, 4 heads\n")
        f.write(f"- Parameters: {model.n_parameters():,}\n")
        f.write(f"- r0_shift={r0_shift}, r1_shift={r1_shift}, seq_length=48, {cfg.train.max_epochs} epochs\n")
        f.write(f"- Sequences: {n_seqs}\n\n")

        f.write("## Training\n\n")
        f.write(f"- Final loss: {final_metrics['loss']:.4f}\n")
        f.write(f"- Attr validity: {final_metrics['acc_attr_valid']:.4f}\n\n")

        f.write("## Contrastive Accuracy\n\n")
        f.write("| Direction | Accuracy |\n")
        f.write("|-----------|----------|\n")
        f.write(f"| Valid triple → y | {correct_pos/total_pos:.4f} |\n")
        f.write(f"| Invalid triple → n | {correct_neg/total_neg:.4f} |\n\n")

        f.write("## Short Reverse Lookup (is prediction valid?)\n\n")
        f.write("| Direction | Valid Rate |\n")
        f.write("|-----------|------------|\n")
        f.write(f"| S + attr → valid subject | {s_valid/s_total:.4f} |\n")
        f.write(f"| O + attr → valid object | {o_valid/o_total:.4f} |\n\n")

        f.write("## Relation Accuracy (any valid target = correct)\n\n")
        f.write("| Split | R0 | R1 | Overall |\n")
        f.write("|-------|-----|-----|--------|\n")
        tr0 = train_by_rel[0]["correct"] / max(1, train_by_rel[0]["total"])
        tr1 = train_by_rel[1]["correct"] / max(1, train_by_rel[1]["total"])
        te0 = test_by_rel[0]["correct"] / max(1, test_by_rel[0]["total"])
        te1 = test_by_rel[1]["correct"] / max(1, test_by_rel[1]["total"])
        f.write(f"| Train ({n_train}) | {tr0:.4f} | {tr1:.4f} | {train_acc:.4f} |\n")
        f.write(f"| Held-out ({n_held}) | {te0:.4f} | {te1:.4f} | {test_acc:.4f} |\n")
        f.write(f"| Chance ({n_valid}/{n_entities}) | {chance:.4f} | {chance:.4f} | {chance:.4f} |\n\n")

        f.write(f"## Affine Analysis ({affine_train_subjs}/{n_entities} subject split)\n\n")
        f.write("| Layer | Train Acc | Test Acc |\n")
        f.write("|-------|-----------|----------|\n")
        for label, layer_results in affine_all.items():
            f.write(f"| {label} | {layer_results['overall']['train_acc']:.4f} | "
                    f"{layer_results['overall']['test_acc']:.4f} |\n")

        f.write("\n## Embedding Structure\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for ov in range(max_ov):
            f.write(f"| Subj {ov}/{max_ov} overlap | {report['subj_by_f_overlap'][ov]:+.4f} |\n")
        f.write(f"| R0 partner sim | {report['r0_partner_sim']:+.4f} |\n")
        f.write(f"| R1 partner sim | {report['r1_partner_sim']:+.4f} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("(To be filled after reviewing results.)\n")

    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()

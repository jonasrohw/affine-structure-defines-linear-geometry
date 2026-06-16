"""
Phase 3: Matrix completion with dense relational exposure.

All 120 subjects appear in relation triples. A random 20% of (subject, relation)
pairs are held out for testing. p_relation=0.5 for stronger relational signal.

Usage (from graz/):
    python experiments/phase3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random

import torch

from src.affine import affine_analysis_all_layers
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.evaluation import similarity_report
from src.model import build_model
from src.training import evaluate, build_valid_attr_table, train

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

P_RELATION = 0.50
HOLDOUT_FRACTION = 0.20
SEED = 42


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
    held_out = make_held_out_triples(
        VocabConfig(num_relations=2, n_train_subjects=120),
        HOLDOUT_FRACTION, SEED,
    )

    cfg = Config(
        vocab=VocabConfig(
            num_relations=2,
            n_train_subjects=120,
            held_out_triples=held_out,
        ),
        model=ModelConfig(n_layers=1, hidden_dim=32, n_heads=4),
        train=TrainConfig(
            max_epochs=800,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=39,
            num_sequences=5000,
            checkpoint_dir=ROOT / "checkpoints_phase3",
            results_dir=RESULTS_DIR,
        ),
        seed=SEED,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_triples = cfg.vocab.train_triples
    n_total = len(cfg.vocab.all_triples)
    n_held = len(held_out)
    n_train = len(train_triples)

    print("=" * 70)
    print("GRAZ Phase 3: Matrix Completion with Dense Relational Exposure")
    print(f"Total triples: {n_total} | Train: {n_train} | Held out: {n_held}")
    print(f"p_relation: {P_RELATION}, dim=32, 1 layer, 4 heads")
    print("=" * 70)
    print()

    # Verify every subject appears in at least one train triple
    subjects_in_train = set(s for s, k in train_triples)
    print(f"Subjects with ≥1 train triple: {len(subjects_in_train)}/120")
    assert len(subjects_in_train) == 120, "Some subjects have no train triples!"
    print()

    # Train
    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, seed_offset=0, p_relation=P_RELATION,
                                   train_triples=train_triples)
    model = build_model(cfg.model, cfg.vocab)
    print(f"Parameters: {model.n_parameters():,}")

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

    # Evaluate relation accuracy on train vs held-out triples
    print("\nRelation accuracy (per triple):")
    train_correct = 0
    test_correct = 0
    train_set = set(train_triples)
    held_set = set(held_out)

    with torch.no_grad():
        for subj_idx in range(cfg.vocab.n_entities):
            for rel_k in range(cfg.vocab.num_relations):
                context = torch.tensor(
                    [[cfg.vocab.sep_id, cfg.vocab.relation_id(rel_k), cfg.vocab.subject_id(subj_idx)]],
                    dtype=torch.long, device=device,
                )
                logits = model(context)
                pred = logits[0, -1].argmax().item()
                target = cfg.vocab.object_id(cfg.vocab.relation_object(subj_idx, rel_k))
                if (subj_idx, rel_k) in train_set:
                    train_correct += int(pred == target)
                else:
                    test_correct += int(pred == target)

    train_acc = train_correct / n_train
    test_acc = test_correct / n_held
    print(f"  Train triples ({n_train}): {train_acc:.4f}")
    print(f"  Held-out triples ({n_held}): {test_acc:.4f}")

    # Breakdown by relation
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
                target = cfg.vocab.object_id(cfg.vocab.relation_object(subj_idx, rel_k))
                correct = int(pred == target)
                if (subj_idx, rel_k) in train_set:
                    train_by_rel[rel_k]["correct"] += correct
                    train_by_rel[rel_k]["total"] += 1
                else:
                    test_by_rel[rel_k]["correct"] += correct
                    test_by_rel[rel_k]["total"] += 1

    print("\n  Per-relation breakdown:")
    for k in range(cfg.vocab.num_relations):
        tr = train_by_rel[k]
        te = test_by_rel[k]
        tr_acc = tr["correct"] / max(1, tr["total"])
        te_acc = te["correct"] / max(1, te["total"])
        print(f"    R{k}: train={tr_acc:.4f} ({tr['total']} triples) | "
              f"test={te_acc:.4f} ({te['total']} triples)")

    # Affine analysis (fit on 80/120 split for comparability with Phase 2)
    print("\nAffine analysis (80/120 subject split for comparability)...")
    cfg_affine = Config(
        vocab=VocabConfig(num_relations=2, n_train_subjects=80),
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
    for ov in [0, 1, 2]:
        print(f"  Subj {ov}/3 overlap: {report['subj_by_f_overlap'][ov]:+.4f}")
    print(f"  R0 partner sim: {report['r0_partner_sim']:+.4f}")
    print(f"  R1 partner sim: {report['r1_partner_sim']:+.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "phase3_results.md"
    with open(results_path, "w") as f:
        f.write("# Phase 3: Matrix Completion Results\n\n")
        f.write("## Setup\n\n")
        f.write(f"- 120 subjects × 2 relations = 240 total triples\n")
        f.write(f"- Train: {n_train} triples ({100*(1-HOLDOUT_FRACTION):.0f}%)\n")
        f.write(f"- Held out: {n_held} triples ({100*HOLDOUT_FRACTION:.0f}%)\n")
        f.write(f"- p_relation: {P_RELATION}\n")
        f.write(f"- Architecture: dim=32, 1 layer, 4 heads\n")
        f.write(f"- Parameters: {model.n_parameters():,}\n\n")

        f.write("## Training\n\n")
        f.write(f"- Final loss: {final_metrics['loss']:.4f}\n")
        f.write(f"- Attr validity: {final_metrics['acc_attr_valid']:.4f}\n\n")

        f.write("## Relation Accuracy\n\n")
        f.write("| Split | R0 | R1 | Overall |\n")
        f.write("|-------|-----|-----|--------|\n")
        tr0 = train_by_rel[0]["correct"] / max(1, train_by_rel[0]["total"])
        tr1 = train_by_rel[1]["correct"] / max(1, train_by_rel[1]["total"])
        te0 = test_by_rel[0]["correct"] / max(1, test_by_rel[0]["total"])
        te1 = test_by_rel[1]["correct"] / max(1, test_by_rel[1]["total"])
        f.write(f"| Train ({n_train}) | {tr0:.4f} | {tr1:.4f} | {train_acc:.4f} |\n")
        f.write(f"| Held-out ({n_held}) | {te0:.4f} | {te1:.4f} | {test_acc:.4f} |\n\n")

        f.write("## Affine Analysis (80/120 subject split)\n\n")
        f.write("Fit on subjects 0-79, test on 80-119 (same split as Phase 2 for comparability).\n\n")
        f.write("| Layer | Train Acc | Test Acc |\n")
        f.write("|-------|-----------|----------|\n")
        for label, layer_results in affine_all.items():
            f.write(f"| {label} | {layer_results['overall']['train_acc']:.4f} | "
                    f"{layer_results['overall']['test_acc']:.4f} |\n")

        f.write("\n## Embedding Structure\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for ov in [0, 1, 2]:
            f.write(f"| Subj {ov}/3 overlap | {report['subj_by_f_overlap'][ov]:+.4f} |\n")
        f.write(f"| R0 partner sim | {report['r0_partner_sim']:+.4f} |\n")
        f.write(f"| R1 partner sim | {report['r1_partner_sim']:+.4f} |\n")

        f.write("\n## Interpretation\n\n")
        if test_acc > 0.9:
            f.write("The model generalizes to held-out triples — matrix completion succeeds.\n")
            f.write("Dense relational exposure + higher p_relation enables generalization.\n")
        elif test_acc > 0.5:
            f.write("Partial generalization. The model fills in some held-out triples but not all.\n")
        else:
            f.write("No generalization. Even with all subjects seen in relational context and "
                    "80% of triples observed, the model memorizes without abstracting.\n")

    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()

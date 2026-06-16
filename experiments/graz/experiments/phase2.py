"""
Phase 2: Train on attribute associations + relation triples, test generalization.

80 train subjects have relation triples in the data; 40 test subjects do not.
Does the model generalize relations to unseen subjects based on embedding
similarity?

Usage (from graz/):
    python experiments/phase2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.affine import affine_analysis_all_layers
from src.config import COMBOS, Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.evaluation import extract_embeddings, print_report, similarity_report
from src.model import build_model
from src.training import evaluate, evaluate_relations, build_valid_attr_table, load_checkpoint, train

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def make_cfg():
    return Config(
        vocab=VocabConfig(num_relations=2, n_train_subjects=80),
        model=ModelConfig(n_layers=1, hidden_dim=32, n_heads=4),
        train=TrainConfig(
            max_epochs=800,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=39,
            num_sequences=5000,
            checkpoint_dir=ROOT / "checkpoints_phase2",
            results_dir=RESULTS_DIR,
        ),
        seed=42,
    )


P_RELATION = 0.25


def main():
    cfg = make_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("GRAZ Phase 2: Attribute Associations + Relation Triples")
    print(f"120 subjects, 120 objects, vocab size: {cfg.vocab.vocab_size}")
    print(f"Hidden dim: {cfg.model.hidden_dim}, 1 layer, 4 heads")
    print(f"Train subjects: {cfg.vocab.n_train_subjects}, "
          f"Test subjects: {cfg.vocab.n_test_subjects}")
    print(f"Relations: {cfg.vocab.num_relations}, p_relation: {P_RELATION}")
    print(f"Sentence types: attribute (3 tokens) + relation (4 tokens)")
    print("=" * 70)
    print()

    # Show example mixed sentences
    from src.data_generation import generate_mixed_sequence
    import random
    rng = random.Random(42)
    seq = generate_mixed_sequence(39, cfg.vocab, rng, P_RELATION)
    print(f"Example sequence ({len(seq)} tokens):")
    pos = 0
    shown = 0
    while pos < len(seq) and shown < 6:
        tok = seq[pos]
        if tok == cfg.vocab.sep_id and pos + 1 < len(seq):
            next_tok = seq[pos + 1]
            if cfg.vocab.is_relation(next_tok) and pos + 3 < len(seq):
                sentence = seq[pos:pos + 4]
                print(f"  {' '.join(cfg.vocab.token_to_str(t) for t in sentence)}")
                pos += 4
            else:
                sentence = seq[pos:pos + 3]
                print(f"  {' '.join(cfg.vocab.token_to_str(t) for t in sentence)}")
                pos += 3
            shown += 1
        else:
            pos += 1
    print()

    # Train
    print("PHASE 2a: Training")
    print("-" * 70)
    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, seed_offset=0, p_relation=P_RELATION)
    model = build_model(cfg.model, cfg.vocab)
    print(f"Parameters: {model.n_parameters():,}")
    print()

    train(model=model, train_loader=train_loader, cfg=cfg, device=device,
          val_loader=train_loader, verbose=True)

    # Use the final model, not the early best-attr-valid checkpoint
    model.to(device)
    model.eval()

    # Evaluate attribute validity
    valid_table = build_valid_attr_table(cfg, device)
    final_metrics = evaluate(model, train_loader, device, cfg, valid_table, max_batches=50)
    print(f"\nFinal: loss={final_metrics['loss']:.4f} | "
          f"loss_attr={final_metrics['loss_attr']:.4f} | "
          f"attr_valid={final_metrics['acc_attr_valid']:.4f}")
    print()

    # Evaluate relation accuracy
    print("PHASE 2b: Relation Generalization")
    print("-" * 70)

    train_rel = evaluate_relations(model, cfg, device, cfg.vocab.train_subject_indices)
    test_rel = evaluate_relations(model, cfg, device, cfg.vocab.test_subject_indices)

    print(f"Train subjects (n={cfg.vocab.n_train_subjects}):")
    for k in range(cfg.vocab.num_relations):
        print(f"  R{k} accuracy: {train_rel[f'r{k}_acc']:.4f}")
    print(f"  Overall:      {train_rel['overall_acc']:.4f}")
    print()
    print(f"Test subjects (n={cfg.vocab.n_test_subjects}):")
    for k in range(cfg.vocab.num_relations):
        print(f"  R{k} accuracy: {test_rel[f'r{k}_acc']:.4f}")
    print(f"  Overall:      {test_rel['overall_acc']:.4f}")
    print()

    # Affine analysis
    print("PHASE 2c: Affine Transformation Analysis")
    print("-" * 70)
    print("Fitting affine map on train subjects, evaluating on train + test...")

    affine_all = affine_analysis_all_layers(model, cfg, device)
    affine_emb = affine_all["layer0"]
    affine_final = affine_all["layer1"]

    print("\nLayer 0 (post-embedding):")
    for k in range(cfg.vocab.num_relations):
        r = affine_emb[f"r{k}"]
        print(f"  R{k}: train_acc={r['train_acc']:.4f}  test_acc={r['test_acc']:.4f}"
              f"  train_cos={r['train_cos']:.4f}  test_cos={r['test_cos']:.4f}")
    print(f"  Overall: train={affine_emb['overall']['train_acc']:.4f}"
          f"  test={affine_emb['overall']['test_acc']:.4f}")

    print("\nLayer 1 (post-transformer, pre-ln_f):")
    for k in range(cfg.vocab.num_relations):
        r = affine_final[f"r{k}"]
        print(f"  R{k}: train_acc={r['train_acc']:.4f}  test_acc={r['test_acc']:.4f}"
              f"  train_cos={r['train_cos']:.4f}  test_cos={r['test_cos']:.4f}")
    print(f"  Overall: train={affine_final['overall']['train_acc']:.4f}"
          f"  test={affine_final['overall']['test_acc']:.4f}")
    print()

    # Embedding structure
    print("PHASE 2d: Embedding Similarity Analysis")
    print("-" * 70)
    report = similarity_report(model, cfg.vocab)
    print_report(report)
    print()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "phase2_results.md"
    with open(results_path, "w") as f:
        f.write("# Graz Phase 2: Relation Generalization Results\n\n")
        f.write("## Setup\n\n")
        f.write(f"- 120 subjects, 120 objects, vocab size {cfg.vocab.vocab_size}\n")
        f.write(f"- Hidden dim: {cfg.model.hidden_dim}, 1 layer, 4 heads\n")
        f.write(f"- Parameters: {model.n_parameters():,}\n")
        f.write(f"- Training: {cfg.train.max_epochs} epochs, "
                f"{cfg.train.num_sequences} sequences\n")
        f.write(f"- Train subjects: {cfg.vocab.n_train_subjects}, "
                f"Test subjects: {cfg.vocab.n_test_subjects}\n")
        f.write(f"- p_relation: {P_RELATION}\n")
        f.write(f"- Relations: R0 (shared f-type), R1 (shared g-type)\n\n")

        f.write("## Training\n\n")
        f.write(f"- Final loss: {final_metrics['loss']:.4f}\n")
        f.write(f"- Final attr loss: {final_metrics['loss_attr']:.4f} "
                f"(theoretical min: {np.log(6):.4f})\n")
        f.write(f"- Attr validity: {final_metrics['acc_attr_valid']:.4f}\n\n")

        f.write("## Relation Accuracy\n\n")
        f.write("### Primary metric: object prediction\n\n")
        f.write("| Split | R0 | R1 | Overall |\n")
        f.write("|-------|-----|-----|--------|\n")
        f.write(f"| Train (n={cfg.vocab.n_train_subjects}) | "
                f"{train_rel['r0_acc']:.4f} | {train_rel['r1_acc']:.4f} | "
                f"{train_rel['overall_acc']:.4f} |\n")
        f.write(f"| Test (n={cfg.vocab.n_test_subjects}) | "
                f"{test_rel['r0_acc']:.4f} | {test_rel['r1_acc']:.4f} | "
                f"{test_rel['overall_acc']:.4f} |\n\n")

        f.write("## Affine Transformation Analysis\n\n")
        f.write("Affine map fitted on train subjects (80), evaluated on train + test.\n\n")
        f.write("### Layer 0 (post-embedding)\n\n")
        f.write("| Relation | Train Acc | Test Acc | Train cos | Test cos |\n")
        f.write("|----------|-----------|----------|-----------|----------|\n")
        for k in range(cfg.vocab.num_relations):
            r = affine_emb[f"r{k}"]
            f.write(f"| R{k} | {r['train_acc']:.4f} | {r['test_acc']:.4f} | "
                    f"{r['train_cos']:.4f} | {r['test_cos']:.4f} |\n")
        f.write(f"| Overall | {affine_emb['overall']['train_acc']:.4f} | "
                f"{affine_emb['overall']['test_acc']:.4f} | | |\n\n")
        f.write("### Layer 1 (post-transformer)\n\n")
        f.write("| Relation | Train Acc | Test Acc | Train cos | Test cos |\n")
        f.write("|----------|-----------|----------|-----------|----------|\n")
        for k in range(cfg.vocab.num_relations):
            r = affine_final[f"r{k}"]
            f.write(f"| R{k} | {r['train_acc']:.4f} | {r['test_acc']:.4f} | "
                    f"{r['train_cos']:.4f} | {r['test_cos']:.4f} |\n")
        f.write(f"| Overall | {affine_final['overall']['train_acc']:.4f} | "
                f"{affine_final['overall']['test_acc']:.4f} | | |\n\n")

        f.write("## Embedding Similarity\n\n")
        f.write("### Subject-subject by f-overlap\n\n")
        f.write("| Overlap | cos sim |\n")
        f.write("|---------|--------|\n")
        for ov in [0, 1, 2, 3]:
            sim = report["subj_by_f_overlap"][ov]
            f.write(f"| {ov}/3 | {sim:+.4f} |\n")
        f.write("\n### Subject-object partner similarity\n\n")
        f.write(f"| Pair type | cos sim |\n")
        f.write(f"|-----------|--------|\n")
        f.write(f"| R0 partners (shared f) | {report['r0_partner_sim']:+.4f} |\n")
        f.write(f"| R1 partners (shared g) | {report['r1_partner_sim']:+.4f} |\n")
        f.write(f"| Random non-partner | {report['random_cross_sim']:+.4f} |\n\n")

        f.write("## Interpretation\n\n")
        if test_rel['overall_acc'] > 0.9:
            f.write("The model generalizes relations to unseen test subjects.\n")
            f.write("Attribute-structured embeddings enable relational transfer.\n")
        elif test_rel['overall_acc'] > 0.5:
            f.write("Partial generalization — the model captures some relational "
                    "structure but does not fully generalize.\n")
        else:
            f.write("The model does not generalize to test subjects — it memorized "
                    "train pairs without abstracting the relation.\n")

    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()

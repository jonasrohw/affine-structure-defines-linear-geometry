"""
Phase 2 dimension sweep: test whether lower dimensionality forces generalization.

With dim=32 and 80 train pairs, memorization is cheap (1,056 affine params
for 80 data points). Lower dims create a bottleneck:
  dim=16: 272 affine params for 80 points (3.4x underdetermined)
  dim=8:  72 affine params for 80 points (0.9x — overdetermined!)

Usage (from graz/):
    python experiments/phase2_dim_sweep.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.affine import affine_analysis_all_layers
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.evaluation import similarity_report
from src.model import build_model
from src.training import evaluate, evaluate_relations, build_valid_attr_table, train

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
P_RELATION = 0.25

DIMS = [
    (16, 4),  # (hidden_dim, n_heads)
    (8, 2),
]


def run_one(hidden_dim: int, n_heads: int) -> dict:
    cfg = Config(
        vocab=VocabConfig(num_relations=2, n_train_subjects=80),
        model=ModelConfig(n_layers=1, hidden_dim=hidden_dim, n_heads=n_heads),
        train=TrainConfig(
            max_epochs=800,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=39,
            num_sequences=5000,
            checkpoint_dir=ROOT / f"checkpoints_phase2_dim{hidden_dim}",
            results_dir=RESULTS_DIR,
        ),
        seed=42,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"DIMENSION SWEEP: dim={hidden_dim}, heads={n_heads}")
    print(f"Affine params per relation: {hidden_dim*hidden_dim + hidden_dim} "
          f"for 80 train points")
    print("=" * 70)
    print()

    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, seed_offset=0, p_relation=P_RELATION)
    model = build_model(cfg.model, cfg.vocab)
    print(f"Parameters: {model.n_parameters():,}")

    train(model=model, train_loader=train_loader, cfg=cfg, device=device,
          val_loader=train_loader, verbose=True)

    model.to(device)
    model.eval()

    # Evaluate
    valid_table = build_valid_attr_table(cfg, device)
    final_metrics = evaluate(model, train_loader, device, cfg, valid_table, max_batches=50)
    print(f"\nFinal: loss={final_metrics['loss']:.4f} | "
          f"loss_attr={final_metrics['loss_attr']:.4f} | "
          f"attr_valid={final_metrics['acc_attr_valid']:.4f}")

    train_rel = evaluate_relations(model, cfg, device, cfg.vocab.train_subject_indices)
    test_rel = evaluate_relations(model, cfg, device, cfg.vocab.test_subject_indices)
    print(f"\nRelation accuracy:")
    print(f"  Train: R0={train_rel['r0_acc']:.4f} R1={train_rel['r1_acc']:.4f} "
          f"overall={train_rel['overall_acc']:.4f}")
    print(f"  Test:  R0={test_rel['r0_acc']:.4f} R1={test_rel['r1_acc']:.4f} "
          f"overall={test_rel['overall_acc']:.4f}")

    # Affine analysis (single batched forward pass for all layers)
    print("\nAffine analysis...")
    affine_all = affine_analysis_all_layers(model, cfg, device)
    affine_emb = affine_all["layer0"]
    affine_final = affine_all["layer1"]

    print(f"  Layer 0: train={affine_emb['overall']['train_acc']:.4f} "
          f"test={affine_emb['overall']['test_acc']:.4f}")
    print(f"  Layer 1: train={affine_final['overall']['train_acc']:.4f} "
          f"test={affine_final['overall']['test_acc']:.4f}")

    # Embedding structure
    report = similarity_report(model, cfg.vocab)
    print(f"\nEmbedding structure:")
    for ov in [0, 1, 2]:
        print(f"  Subj {ov}/3 overlap: {report['subj_by_f_overlap'][ov]:+.4f}")
    print(f"  R0 partner sim: {report['r0_partner_sim']:+.4f}")
    print(f"  R1 partner sim: {report['r1_partner_sim']:+.4f}")
    print()

    return {
        "dim": hidden_dim,
        "n_heads": n_heads,
        "params": model.n_parameters(),
        "final_loss": final_metrics["loss"],
        "attr_valid": final_metrics["acc_attr_valid"],
        "train_rel": train_rel,
        "test_rel": test_rel,
        "affine_emb": affine_emb,
        "affine_final": affine_final,
        "sim_report": report,
    }


def main():
    all_results = []
    for hidden_dim, n_heads in DIMS:
        result = run_one(hidden_dim, n_heads)
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("DIMENSION SWEEP SUMMARY")
    print("=" * 70)

    print("\n### Relation prediction accuracy\n")
    print("| dim | params | train obj | test obj |")
    print("|-----|--------|-----------|----------|")
    # Include dim=32 from the main Phase 2 run
    print("| 32  | 33,568 | 1.0000    | 0.0000   |")
    for r in all_results:
        print(f"| {r['dim']:<3d} | {r['params']:,} | "
              f"{r['train_rel']['overall_acc']:.4f}    | "
              f"{r['test_rel']['overall_acc']:.4f}   |")

    print("\n### Affine accuracy (layer 0 / layer 1)\n")
    print("| dim | L0 train | L0 test | L1 train | L1 test |")
    print("|-----|----------|---------|----------|---------|")
    print("| 32  | 1.0000   | 0.0000  | 1.0000   | 0.0000  |")
    for r in all_results:
        ae = r["affine_emb"]
        af = r["affine_final"]
        print(f"| {r['dim']:<3d} | {ae['overall']['train_acc']:.4f}   | "
              f"{ae['overall']['test_acc']:.4f}  | "
              f"{af['overall']['train_acc']:.4f}   | "
              f"{af['overall']['test_acc']:.4f}  |")

    print("\n### Embedding structure\n")
    print("| dim | 0/3 sim | 2/3 sim | R0 partner | R1 partner |")
    print("|-----|---------|---------|------------|------------|")
    print("| 32  | -0.1062 | +0.2984 | +0.2128    | +0.2062    |")
    for r in all_results:
        sr = r["sim_report"]
        print(f"| {r['dim']:<3d} | {sr['subj_by_f_overlap'][0]:+.4f} | "
              f"{sr['subj_by_f_overlap'][2]:+.4f} | "
              f"{sr['r0_partner_sim']:+.4f}    | "
              f"{sr['r1_partner_sim']:+.4f}    |")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "phase2_dim_sweep.md"
    with open(results_path, "w") as f:
        f.write("# Phase 2 Dimension Sweep\n\n")
        f.write("Does lower dimensionality force relational generalization?\n\n")
        f.write("## Setup\n\n")
        f.write("- 80 train / 40 test subjects, 2 relations, p_relation=0.25\n")
        f.write("- 800 epochs, 5000 sequences, 1 layer\n\n")

        f.write("## Relation Prediction Accuracy\n\n")
        f.write("| dim | params | train obj | test obj |\n")
        f.write("|-----|--------|-----------|----------|\n")
        f.write("| 32  | 33,568 | 1.0000    | 0.0000   |\n")
        for r in all_results:
            f.write(f"| {r['dim']} | {r['params']:,} | "
                    f"{r['train_rel']['overall_acc']:.4f} | "
                    f"{r['test_rel']['overall_acc']:.4f} |\n")

        f.write("\n## Affine Accuracy\n\n")
        f.write("| dim | L0 train | L0 test | L1 train | L1 test |\n")
        f.write("|-----|----------|---------|----------|----------|\n")
        f.write("| 32  | 1.0000   | 0.0000  | 1.0000   | 0.0000   |\n")
        for r in all_results:
            ae = r["affine_emb"]
            af = r["affine_final"]
            f.write(f"| {r['dim']} | {ae['overall']['train_acc']:.4f} | "
                    f"{ae['overall']['test_acc']:.4f} | "
                    f"{af['overall']['train_acc']:.4f} | "
                    f"{af['overall']['test_acc']:.4f} |\n")

        f.write("\n## Embedding Structure\n\n")
        f.write("| dim | 0/3 sim | 2/3 sim | R0 partner | R1 partner |\n")
        f.write("|-----|---------|---------|------------|------------|\n")
        f.write("| 32  | -0.1062 | +0.2984 | +0.2128    | +0.2062    |\n")
        for r in all_results:
            sr = r["sim_report"]
            f.write(f"| {r['dim']} | {sr['subj_by_f_overlap'][0]:+.4f} | "
                    f"{sr['subj_by_f_overlap'][2]:+.4f} | "
                    f"{sr['r0_partner_sim']:+.4f} | "
                    f"{sr['r1_partner_sim']:+.4f} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("(To be filled after reviewing results.)\n")

    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()

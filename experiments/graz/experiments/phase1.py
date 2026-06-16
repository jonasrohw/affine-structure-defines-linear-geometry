"""
Phase 1: Train on attribute associations, evaluate embedding structure.

No explicit relation tokens — the model only learns which attributes each
entity has. We then test whether the learned embeddings reflect the implicit
relational structure (shared f-type → similar embeddings, etc.).

Usage (from graz/):
    python experiments/phase1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.config import COMBOS, Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.evaluation import extract_embeddings, print_report, similarity_report
from src.model import build_model
from src.training import evaluate, build_valid_attr_table, load_checkpoint, train

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def make_cfg():
    return Config(
        vocab=VocabConfig(),
        model=ModelConfig(n_layers=1, hidden_dim=32, n_heads=4),
        train=TrainConfig(
            max_epochs=800,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=39,
            num_sequences=5000,
            checkpoint_dir=ROOT / "checkpoints_phase1",
            results_dir=RESULTS_DIR,
        ),
        seed=42,
    )


def main():
    cfg = make_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("GRAZ Phase 1: Attribute Association Training")
    print(f"120 subjects, 120 objects, 10 f-attrs, 10 g-attrs")
    print(f"Vocab size: {cfg.vocab.vocab_size}, Hidden dim: {cfg.model.hidden_dim}")
    print(f"Sentences: x entity attribute (3 tokens)")
    print(f"Unique facts: 240 entities x 6 attrs = 1,440")
    print("=" * 70)
    print()

    # Verify dataset construction
    print("Dataset verification:")
    v = cfg.vocab
    print(f"  Subject 0:  f-type={COMBOS[0]}, g-type={COMBOS[0]}")
    print(f"  Object 0:   f-type={COMBOS[0]}, g-type={COMBOS[v.r1_shift]}")
    print(f"  R0: subject 0 ↔ object 0 (shared f-type {COMBOS[0]})")
    r1_obj = v.r1_partner(0)
    print(f"  R1: subject 0 ↔ object {r1_obj} (shared g-type {COMBOS[0]})")
    print(f"  Subject 5:  f-type={COMBOS[5]}, g-type={COMBOS[5]}")
    print(f"  Object 5:   f-type={COMBOS[5]}, g-type={COMBOS[(5 + v.r1_shift) % v.n_entities]}")
    print()

    # Show a few example sentences
    from src.data_generation import generate_sequence
    import random
    rng = random.Random(42)
    seq = generate_sequence(12, v, rng)
    print("Example sequence (4 sentences):")
    for s in range(4):
        tokens = seq[s * 3:(s + 1) * 3]
        print(f"  {' '.join(v.token_to_str(t) for t in tokens)}")
    print()

    # Train
    print("PHASE 1a: Training")
    print("-" * 70)
    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, seed_offset=0)
    model = build_model(cfg.model, cfg.vocab)
    print(f"Parameters: {model.n_parameters():,}")
    print()

    train(model=model, train_loader=train_loader, cfg=cfg, device=device,
          val_loader=train_loader, verbose=True)

    best_ckpt = cfg.train.checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, device)
    model.to(device)
    model.eval()

    valid_table = build_valid_attr_table(cfg, device)
    final_metrics = evaluate(model, train_loader, device, cfg, valid_table, max_batches=50)
    print(f"\nFinal: loss={final_metrics['loss']:.4f} | "
          f"loss_attr={final_metrics['loss_attr']:.4f} | "
          f"attr_valid={final_metrics['acc_attr_valid']:.4f}")
    print(f"Theoretical minimum attr loss: -log(1/6) = {np.log(6):.4f}")
    print()

    # Evaluate embedding structure
    print("PHASE 1b: Embedding Similarity Analysis")
    print("-" * 70)
    report = similarity_report(model, cfg.vocab)
    print_report(report)
    print()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "phase1_results.md"
    with open(results_path, "w") as f:
        f.write("# Graz Phase 1: Attribute Association Results\n\n")
        f.write("## Setup\n\n")
        f.write(f"- 120 subjects, 120 objects, vocab size {cfg.vocab.vocab_size}\n")
        f.write(f"- Hidden dim: {cfg.model.hidden_dim}, 1 layer, 4 heads\n")
        f.write(f"- Parameters: {model.n_parameters():,}\n")
        f.write(f"- Training: {cfg.train.max_epochs} epochs, {cfg.train.num_sequences} sequences\n")
        f.write(f"- Unique facts: 1,440 (240 entities x 6 attributes)\n\n")
        f.write("## Training\n\n")
        f.write(f"- Final loss: {final_metrics['loss']:.4f}\n")
        f.write(f"- Final attr loss: {final_metrics['loss_attr']:.4f} ")
        f.write(f"(theoretical min: {np.log(6):.4f})\n")
        f.write(f"- Attr validity: {final_metrics['acc_attr_valid']:.4f}\n\n")
        f.write("## Embedding Similarity by Attribute Overlap\n\n")
        f.write("### Subject-subject\n\n")
        f.write("| f-overlap | cos sim | g-overlap | cos sim |\n")
        f.write("|-----------|---------|-----------|--------|\n")
        for ov in [0, 1, 2, 3]:
            sf = report["subj_by_f_overlap"][ov]
            sg = report["subj_by_g_overlap"][ov]
            f.write(f"| {ov}/3 | {sf:+.4f} | {ov}/3 | {sg:+.4f} |\n")
        f.write("\n### Object-object\n\n")
        f.write("| f-overlap | cos sim | g-overlap | cos sim |\n")
        f.write("|-----------|---------|-----------|--------|\n")
        for ov in [0, 1, 2, 3]:
            of_ = report["obj_by_f_overlap"][ov]
            og = report["obj_by_g_overlap"][ov]
            f.write(f"| {ov}/3 | {of_:+.4f} | {ov}/3 | {og:+.4f} |\n")
        f.write("\n### Subject-object partner similarity\n\n")
        f.write(f"| Pair type | cos sim |\n")
        f.write(f"|-----------|--------|\n")
        f.write(f"| R0 partners (shared f) | {report['r0_partner_sim']:+.4f} |\n")
        f.write(f"| R1 partners (shared g) | {report['r1_partner_sim']:+.4f} |\n")
        f.write(f"| Random non-partner | {report['random_cross_sim']:+.4f} |\n\n")
        f.write("## Interpretation\n\n")
        f.write("(To be filled in after reviewing results.)\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()

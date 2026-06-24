"""
Graz-style sweep over hidden attributes per group.

This keeps the core Graz setup fixed:
  - entities have f/g hidden attribute sets
  - R0 selects objects with the same f-type
  - R1 selects objects with the same g-type
  - training mixes attribute, relation, contrastive, and reverse-short sentences

The swept parameter is k_per_group: how many hidden attributes define each
f-type and g-type. The goal is to see where affine relation maps stop
generalizing.

Usage from experiments/graz:
    python experiments/hidden_attr_sweep.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.affine import affine_analysis_all_layers, affine_analysis_centroid
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader
from src.model import build_model
from src.training import build_valid_attr_table, evaluate, train

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "hidden_attr_sweep"

P_RELATION = 0.264
P_CONTRASTIVE = 0.20
P_REVERSE_SHORT = 0.264
HOLDOUT_FRACTION = 0.20


def make_held_out_triples(vocab: VocabConfig, fraction: float, seed: int) -> list[tuple[int, int]]:
    """Hold out triples while leaving each subject at least one relation."""
    all_triples = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    rng = random.Random(seed)
    rng.shuffle(all_triples)
    n_held = int(len(all_triples) * fraction)
    held_out: list[tuple[int, int]] = []
    subject_remaining = {i: vocab.num_relations for i in range(vocab.n_entities)}
    for subj_idx, rel_k in all_triples:
        if len(held_out) >= n_held:
            break
        if subject_remaining[subj_idx] > 1:
            held_out.append((subj_idx, rel_k))
            subject_remaining[subj_idx] -= 1
    return held_out


@torch.no_grad()
def relation_accuracy_any_valid(
    model,
    cfg: Config,
    device: torch.device,
) -> dict:
    """Relation accuracy where any valid object for the f/g type counts."""
    model.eval()
    train_triples = set(cfg.vocab.train_triples)
    rows = {
        "train": {k: {"correct": 0, "total": 0} for k in range(cfg.vocab.num_relations)},
        "heldout": {k: {"correct": 0, "total": 0} for k in range(cfg.vocab.num_relations)},
    }
    for subj_idx in range(cfg.vocab.n_entities):
        for rel_k in range(cfg.vocab.num_relations):
            context = torch.tensor(
                [[cfg.vocab.sep_id, cfg.vocab.relation_id(rel_k), cfg.vocab.subject_id(subj_idx)]],
                dtype=torch.long,
                device=device,
            )
            pred = model(context)[0, -1].argmax().item()
            valid_targets = {cfg.vocab.object_id(j) for j in cfg.vocab.relation_objects(subj_idx, rel_k)}
            split = "train" if (subj_idx, rel_k) in train_triples else "heldout"
            rows[split][rel_k]["correct"] += int(pred in valid_targets)
            rows[split][rel_k]["total"] += 1

    result: dict[str, float] = {}
    for split, rel_rows in rows.items():
        total_correct = sum(row["correct"] for row in rel_rows.values())
        total = sum(row["total"] for row in rel_rows.values())
        result[f"{split}_overall"] = total_correct / max(1, total)
        for rel_k, row in rel_rows.items():
            result[f"{split}_r{rel_k}"] = row["correct"] / max(1, row["total"])
    return result


def build_cfg(
    *,
    n_attrs: int,
    k_per_group: int,
    copies_per_type: int,
    hidden_dim: int,
    n_heads: int,
    max_epochs: int,
    num_sequences: int,
    batch_size: int,
    seed: int,
) -> Config:
    vocab_probe = VocabConfig(
        n_f_attrs=n_attrs,
        n_g_attrs=n_attrs,
        k_per_group=k_per_group,
        num_relations=2,
        n_entities=1,
        n_train_subjects=1,
        r0_shift=1,
        r1_shift=1,
    )
    n_types = vocab_probe.n_types
    n_entities = n_types * copies_per_type
    vocab_kwargs = dict(
        n_f_attrs=n_attrs,
        n_g_attrs=n_attrs,
        k_per_group=k_per_group,
        num_relations=2,
        n_entities=n_entities,
        n_train_subjects=n_entities,
        r0_shift=1,
        r1_shift=1,
    )
    held_out = make_held_out_triples(VocabConfig(**vocab_kwargs), HOLDOUT_FRACTION, seed)
    return Config(
        vocab=VocabConfig(**vocab_kwargs, held_out_triples=held_out),
        model=ModelConfig(n_layers=1, hidden_dim=hidden_dim, n_heads=n_heads),
        train=TrainConfig(
            max_epochs=max_epochs,
            learning_rate=3e-4,
            batch_size=batch_size,
            seq_length=48,
            num_sequences=num_sequences,
            checkpoint_dir=ROOT / "checkpoints_hidden_attr_sweep" / f"k{k_per_group}",
            results_dir=RESULTS_DIR,
            eval_every=max(1, max_epochs),
        ),
        seed=seed,
    )


def run_one(args: argparse.Namespace, k_per_group: int) -> dict:
    cfg = build_cfg(
        n_attrs=args.n_attrs,
        k_per_group=k_per_group,
        copies_per_type=args.copies_per_type,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        max_epochs=args.max_epochs,
        num_sequences=args.num_sequences,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    device = torch.device(args.device)
    torch.manual_seed(cfg.seed)

    train_loader = make_dataloader(
        cfg,
        seed_offset=0,
        p_relation=P_RELATION,
        train_triples=cfg.vocab.train_triples,
        p_reverse_short=P_REVERSE_SHORT,
        p_contrastive=P_CONTRASTIVE,
    )
    model = build_model(cfg.model, cfg.vocab)
    t0 = time.time()
    train(
        model=model,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        val_loader=train_loader,
        verbose=args.verbose,
    )
    model.to(device)
    model.eval()

    valid_table = build_valid_attr_table(cfg, device)
    final_metrics = evaluate(model, train_loader, device, cfg, valid_table, max_batches=20)
    rel_acc = relation_accuracy_any_valid(model, cfg, device)
    cfg_affine = Config(
        vocab=VocabConfig(
            n_f_attrs=cfg.vocab.n_f_attrs,
            n_g_attrs=cfg.vocab.n_g_attrs,
            k_per_group=cfg.vocab.k_per_group,
            n_entities=cfg.vocab.n_entities,
            r0_shift=cfg.vocab.r0_shift,
            r1_shift=cfg.vocab.r1_shift,
            num_relations=cfg.vocab.num_relations,
            n_train_subjects=cfg.vocab.n_entities // 2,
        ),
        model=cfg.model,
        train=cfg.train,
        seed=cfg.seed,
    )
    affine_all = affine_analysis_all_layers(model, cfg_affine, device)
    centroid_final = affine_analysis_centroid(model, cfg_affine, device, layer=-1)
    final_layer = affine_all[f"layer{cfg.model.n_layers}"]

    n_valid = len(cfg.vocab.relation_objects(0, 0))
    result = {
        "k_per_group": k_per_group,
        "n_attrs": args.n_attrs,
        "n_types": cfg.vocab.n_types,
        "copies_per_type": args.copies_per_type,
        "n_entities": cfg.vocab.n_entities,
        "n_valid_targets": n_valid,
        "chance": n_valid / cfg.vocab.n_entities,
        "num_train_triples": len(cfg.vocab.train_triples),
        "num_heldout_triples": len(cfg.vocab.held_out_triples),
        "num_parameters": model.n_parameters(),
        "final_loss": final_metrics["loss"],
        "attr_valid": final_metrics["acc_attr_valid"],
        "relation_accuracy": rel_acc,
        "affine_object_final": final_layer,
        "affine_centroid_final": centroid_final,
        "elapsed_sec": time.time() - t0,
        "config": {
            "vocab": asdict(cfg.vocab),
            "model": asdict(cfg.model),
            "train": {
                **asdict(cfg.train),
                "checkpoint_dir": str(cfg.train.checkpoint_dir),
                "results_dir": str(cfg.train.results_dir),
            },
            "seed": cfg.seed,
        },
    }
    return result


def write_report(results: list[dict], output_dir: Path) -> Path:
    report_path = output_dir / "hidden_attr_sweep.md"
    lines = [
        "# Graz Hidden-Attribute Sweep",
        "",
        "Swept `k_per_group`, the number of hidden f attributes and hidden g attributes assigned to each entity type.",
        "The data/model/training recipe is Graz-style: attribute sentences, relation triples, reverse-short lookup, and contrastive y/n sentences.",
        "",
        "| k | entities | chance | attr valid | rel train | rel heldout | object affine test | centroid affine test | centroid R0 | centroid R1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        rel = row["relation_accuracy"]
        obj_aff = row["affine_object_final"]["overall"]["test_acc"]
        cent = row["affine_centroid_final"]
        lines.append(
            f"| {row['k_per_group']} | {row['n_entities']} | {row['chance']:.3f} | "
            f"{row['attr_valid']:.3f} | {rel['train_overall']:.3f} | {rel['heldout_overall']:.3f} | "
            f"{obj_aff:.3f} | {cent['overall']['test_acc']:.3f} | "
            f"{cent['r0']['test_acc']:.3f} | {cent['r1']['test_acc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide: object affine test uses the first valid object target, while centroid affine test uses the type centroid of all valid objects. In Graz-style many-target relations, centroid affine accuracy is the cleaner signal for whether the type-level relation is affine.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", default="1,2,3,4")
    parser.add_argument("--n-attrs", type=int, default=6)
    parser.add_argument("--copies-per-type", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--num-sequences", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    results = []
    for k_per_group in k_values:
        print(f"Running Graz hidden-attribute sweep k={k_per_group}...", flush=True)
        row = run_one(args, k_per_group)
        results.append(row)
        rel = row["relation_accuracy"]
        print(
            f"  k={k_per_group} attr_valid={row['attr_valid']:.3f} "
            f"heldout_rel={rel['heldout_overall']:.3f} "
            f"centroid_affine={row['affine_centroid_final']['overall']['test_acc']:.3f}",
            flush=True,
        )

    result_path = args.output_dir / "hidden_attr_sweep.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = write_report(results, args.output_dir)
    print(f"Saved JSON: {result_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

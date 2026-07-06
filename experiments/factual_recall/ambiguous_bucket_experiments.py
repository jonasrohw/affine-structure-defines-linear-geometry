"""Ambiguous-bucket factual recall experiments.

These runs test two stricter claims than the original synthetic curriculum:

1. A single exposure must improve exact-token recall over matched held-out
   controls, not merely induce affine geometry.
2. High-frequency exact recall should be separable from affine structure for both
   semantic and random relations.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT, TransformerConfig
from .model import CausalTransformer, build_model
from .synthetic_curriculum import (
    SequenceListDataset,
    affine_metrics,
    affine_nearest_metrics,
    fit_affine,
    get_lr,
    train_one_batch,
)


@dataclass
class AmbiguousBucketConfig:
    seed: int = 23
    n_buckets: int = 24
    subjects_per_bucket: int = 4
    objects_per_bucket: int = 4
    n_f: int = 24
    n_g: int = 24
    delta_f: int = 5
    delta_g: int = 7
    attr_repeats: int = 8
    pilot_repeats: int = 160
    single_repeats: int = 1
    phase2_attr_repeats: int = 0
    phase2_pilot_repeats: int = 0
    batch_size: int = 32
    learning_rate: float = 2e-3
    phase2_learning_rate: float = 5e-3
    phase2_update_mode: str = "all"
    warmup_steps: int = 10
    grad_clip: float = 1.0
    object_loss_weight: float = 16.0
    init_scale: float = 0.25
    hidden_dim: int = 128
    n_layers: int = 2
    output_dir: Path = PROJECT_ROOT / "factual_recall" / "results" / "ambiguous_bucket"

    @property
    def n_subjects(self) -> int:
        return self.n_buckets * self.subjects_per_bucket

    @property
    def n_objects(self) -> int:
        return self.n_buckets * self.objects_per_bucket


class AmbiguousBucketVocab:
    def __init__(self, cfg: AmbiguousBucketConfig) -> None:
        if cfg.objects_per_bucket < cfg.subjects_per_bucket:
            raise ValueError("objects_per_bucket must be >= subjects_per_bucket for no-reuse targets")
        if cfg.n_buckets > cfg.n_f * cfg.n_g:
            raise ValueError("Need enough f/g pairs for buckets")
        self.cfg = cfg
        self.sep_id = 0
        self.relation_id = 1
        self.n_relations = 2
        self.subject_offset = self.relation_id + self.n_relations
        self.object_offset = self.subject_offset + cfg.n_subjects
        self.f_offset = self.object_offset + cfg.n_objects
        self.g_offset = self.f_offset + cfg.n_f
        self.vocab_size = self.g_offset + cfg.n_g

        rng = np.random.default_rng(cfg.seed)
        pairs = [(f, g) for f in range(cfg.n_f) for g in range(cfg.n_g)]
        rng.shuffle(pairs)
        self.bucket_attrs = pairs[: cfg.n_buckets]
        self.bucket_target_attrs = [
            ((f + cfg.delta_f) % cfg.n_f, (g + cfg.delta_g) % cfg.n_g)
            for f, g in self.bucket_attrs
        ]
        self.subject_attrs = tuple(
            self.bucket_attrs[b]
            for b in range(cfg.n_buckets)
            for _ in range(cfg.subjects_per_bucket)
        )
        self.object_attrs = tuple(
            self.bucket_target_attrs[b]
            for b in range(cfg.n_buckets)
            for _ in range(cfg.objects_per_bucket)
        )
        semantic_targets: list[int] = []
        for bucket in range(cfg.n_buckets):
            candidates = np.arange(
                bucket * cfg.objects_per_bucket,
                (bucket + 1) * cfg.objects_per_bucket,
                dtype=np.int64,
            )
            perm = rng.permutation(candidates)
            semantic_targets.extend(int(perm[pos]) for pos in range(cfg.subjects_per_bucket))
        self.semantic_targets = np.asarray(semantic_targets, dtype=np.int64)
        self.random_targets = rng.choice(cfg.n_objects, size=cfg.n_subjects, replace=False).astype(np.int64)

    def subject_id(self, index: int) -> int:
        return self.subject_offset + index

    def object_id(self, index: int) -> int:
        return self.object_offset + index

    def relation_token(self, relation_idx: int) -> int:
        return self.relation_id + relation_idx

    def f_id(self, index: int) -> int:
        return self.f_offset + index

    def g_id(self, index: int) -> int:
        return self.g_offset + index

    def subject_bucket(self, subject_idx: int) -> int:
        return subject_idx // self.cfg.subjects_per_bucket

    def object_bucket(self, object_idx: int) -> int:
        return object_idx // self.cfg.objects_per_bucket

    def bucket_object_indices(self, bucket: int) -> list[int]:
        start = bucket * self.cfg.objects_per_bucket
        return list(range(start, start + self.cfg.objects_per_bucket))


def subject_attr_sequence(vocab: AmbiguousBucketVocab, subject_idx: int) -> list[int]:
    f, g = vocab.subject_attrs[subject_idx]
    return [vocab.sep_id, vocab.subject_id(subject_idx), vocab.f_id(f), vocab.g_id(g), vocab.sep_id]


def object_attr_sequence(vocab: AmbiguousBucketVocab, object_idx: int) -> list[int]:
    f, g = vocab.object_attrs[object_idx]
    return [vocab.sep_id, vocab.object_id(object_idx), vocab.f_id(f), vocab.g_id(g), vocab.sep_id]


def triple_sequence(
    vocab: AmbiguousBucketVocab,
    subject_idx: int,
    object_idx: int,
    relation_idx: int,
) -> list[int]:
    return [
        vocab.sep_id,
        vocab.subject_id(subject_idx),
        vocab.relation_token(relation_idx),
        vocab.object_id(object_idx),
        vocab.sep_id,
    ]


def make_loader(sequences: list[list[int]], batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        SequenceListDataset(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def split_subjects(vocab: AmbiguousBucketVocab) -> tuple[list[int], list[int], list[int]]:
    pilot: list[int] = []
    single: list[int] = []
    heldout: list[int] = []
    spb = vocab.cfg.subjects_per_bucket
    for bucket in range(vocab.cfg.n_buckets):
        base = bucket * spb
        pilot.append(base)
        heldout.append(base + spb - 1)
        single.extend(base + pos for pos in range(1, spb - 1))
    return pilot, single, heldout


def init_embeddings(
    model: CausalTransformer,
    vocab: AmbiguousBucketVocab,
    cfg: AmbiguousBucketConfig,
    device: torch.device,
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    dim = model.token_emb.weight.shape[1]
    f_codes = nn.functional.normalize(torch.randn(cfg.n_f, dim, generator=generator, device=device), dim=1)
    g_codes = nn.functional.normalize(torch.randn(cfg.n_g, dim, generator=generator, device=device), dim=1)
    role_subject = nn.functional.normalize(torch.randn(dim, generator=generator, device=device), dim=0)
    role_object = nn.functional.normalize(torch.randn(dim, generator=generator, device=device), dim=0)
    object_noise = nn.functional.normalize(torch.randn(cfg.n_objects, dim, generator=generator, device=device), dim=1)
    subject_noise = nn.functional.normalize(torch.randn(cfg.n_subjects, dim, generator=generator, device=device), dim=1)
    with torch.no_grad():
        for f in range(cfg.n_f):
            vec = cfg.init_scale * f_codes[f]
            model.token_emb.weight[vocab.f_id(f)].copy_(vec)
            model.lm_head.weight[vocab.f_id(f)].copy_(vec)
        for g in range(cfg.n_g):
            vec = cfg.init_scale * g_codes[g]
            model.token_emb.weight[vocab.g_id(g)].copy_(vec)
            model.lm_head.weight[vocab.g_id(g)].copy_(vec)
        for i, (f, g) in enumerate(vocab.subject_attrs):
            vec = cfg.init_scale * (f_codes[f] + g_codes[g] + 0.5 * role_subject + 0.2 * subject_noise[i])
            model.token_emb.weight[vocab.subject_id(i)].copy_(vec)
        for j, (f, g) in enumerate(vocab.object_attrs):
            vec = cfg.init_scale * (f_codes[f] + g_codes[g] + 0.5 * role_object + 0.35 * object_noise[j])
            model.token_emb.weight[vocab.object_id(j)].copy_(vec)
            model.lm_head.weight[vocab.object_id(j)].copy_(vec)


def build_model_for_vocab(
    cfg: AmbiguousBucketConfig,
    vocab: AmbiguousBucketVocab,
    device: torch.device,
    seed_offset: int = 0,
) -> CausalTransformer:
    torch.manual_seed(cfg.seed + seed_offset)
    model_cfg = TransformerConfig(
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
        n_heads=4,
        context_len=8,
        tie_embeddings=False,
    )
    model = build_model(model_cfg, vocab).to(device)
    init_embeddings(model, vocab, cfg, device)
    return model


def attr_sequences(vocab: AmbiguousBucketVocab) -> list[list[int]]:
    return [
        *(subject_attr_sequence(vocab, i) for i in range(vocab.cfg.n_subjects)),
        *(object_attr_sequence(vocab, j) for j in range(vocab.cfg.n_objects)),
    ]


def train_sequences(
    model: CausalTransformer,
    vocab: AmbiguousBucketVocab,
    cfg: AmbiguousBucketConfig,
    sequences: list[list[int]],
    device: torch.device,
    lr_max: float,
    seed: int,
    update_mode: str = "all",
) -> tuple[int, float]:
    loader = make_loader(sequences, cfg.batch_size, seed=seed, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.0)
    last_loss = float("nan")
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        lr = get_lr(step, cfg.warmup_steps, len(loader), lr_max)
        if update_mode == "all":
            last_loss = train_one_batch(
                model,
                optimizer,
                x,
                y,
                lr,
                cfg.grad_clip,
                object_token_range=(vocab.object_offset, vocab.f_offset),
                object_loss_weight=cfg.object_loss_weight,
            )
        elif update_mode == "target_lm_rows":
            for group in optimizer.param_groups:
                group["lr"] = lr
            logits = model(x)
            batch_size, seq_len, vocab_size = logits.shape
            flat_logits = logits.reshape(batch_size * seq_len, vocab_size)
            flat_y = y.reshape(batch_size * seq_len)
            losses = nn.functional.cross_entropy(flat_logits, flat_y, reduction="none")
            lo, hi = vocab.object_offset, vocab.f_offset
            object_targets = (flat_y >= lo) & (flat_y < hi)
            weights = torch.where(
                object_targets,
                torch.full_like(losses, cfg.object_loss_weight),
                torch.ones_like(losses),
            )
            loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            target_rows = flat_y[object_targets].unique()
            for param in model.parameters():
                if param is not model.lm_head.weight and param.grad is not None:
                    param.grad.zero_()
            if model.lm_head.weight.grad is not None:
                mask = torch.zeros_like(model.lm_head.weight.grad)
                mask[target_rows] = 1.0
                model.lm_head.weight.grad.mul_(mask)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            last_loss = float(loss.item())
        else:
            raise ValueError(f"Unknown update_mode: {update_mode}")
    return len(loader), last_loss


def prompt_tensor(vocab: AmbiguousBucketVocab, subjects: Iterable[int], relation_idx: int) -> torch.Tensor:
    return torch.tensor(
        [[vocab.sep_id, vocab.subject_id(i), vocab.relation_token(relation_idx)] for i in subjects],
        dtype=torch.long,
    )


def evaluate_relation(
    model: CausalTransformer,
    vocab: AmbiguousBucketVocab,
    subjects: list[int],
    targets: np.ndarray,
    relation_idx: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    object_ids = torch.tensor([vocab.object_id(i) for i in range(vocab.cfg.n_objects)], device=device)
    totals = {
        "exact": 0,
        "object_exact": 0,
        "top5": 0,
        "top10": 0,
        "coarse": 0,
    }
    ranks: list[int] = []
    bucket_ranks: list[int] = []
    losses = 0.0
    with torch.no_grad():
        for start in range(0, len(subjects), 256):
            batch_subjects = subjects[start : start + 256]
            x = prompt_tensor(vocab, batch_subjects, relation_idx).to(device)
            target_obj = [int(targets[i]) for i in batch_subjects]
            y = torch.tensor([vocab.object_id(j) for j in target_obj], device=device)
            logits = model(x)[:, -1, :]
            losses += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            totals["exact"] += int((logits.argmax(dim=-1) == y).sum().item())
            object_logits = logits.index_select(dim=-1, index=object_ids)
            object_pred_local = object_logits.argmax(dim=-1).detach().cpu().numpy()
            totals["object_exact"] += int(np.sum(object_pred_local == np.asarray(target_obj)))
            target_logits = logits.gather(1, y[:, None])
            ranks.extend((1 + (logits > target_logits).sum(dim=1)).detach().cpu().numpy().astype(int).tolist())
            topk = logits.topk(k=min(10, logits.shape[-1]), dim=-1).indices
            totals["top5"] += int((topk[:, : min(5, topk.shape[1])] == y[:, None]).any(dim=1).sum().item())
            totals["top10"] += int((topk == y[:, None]).any(dim=1).sum().item())
            for row, subject_idx in enumerate(batch_subjects):
                target = target_obj[row]
                target_bucket = vocab.object_bucket(target)
                pred_bucket = vocab.object_bucket(int(object_pred_local[row]))
                totals["coarse"] += int(pred_bucket == target_bucket)
                bucket_ids = torch.tensor(
                    [vocab.object_id(j) for j in vocab.bucket_object_indices(target_bucket)],
                    device=device,
                )
                bucket_logits = logits[row].index_select(dim=0, index=bucket_ids)
                target_pos = vocab.bucket_object_indices(target_bucket).index(target)
                bucket_ranks.append(int(1 + (bucket_logits > bucket_logits[target_pos]).sum().item()))
    n = max(1, len(subjects))
    model.train()
    return {
        "accuracy": totals["exact"] / n,
        "object_accuracy": totals["object_exact"] / n,
        "top5_accuracy": totals["top5"] / n,
        "top10_accuracy": totals["top10"] / n,
        "coarse_bucket_accuracy": totals["coarse"] / n,
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "mean_bucket_rank": float(np.mean(bucket_ranks)) if bucket_ranks else float("nan"),
        "bucket_top1_accuracy": float(np.mean(np.asarray(bucket_ranks) == 1)) if bucket_ranks else float("nan"),
        "loss": losses / n,
    }


def prompt_hidden(
    model: CausalTransformer,
    vocab: AmbiguousBucketVocab,
    subjects: list[int],
    relation_idx: int,
    device: torch.device,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(subjects), 256):
            batch = subjects[start : start + 256]
            x = prompt_tensor(vocab, batch, relation_idx).to(device)
            with model.capture_residuals() as residuals:
                model(x)
            rows.append(model.ln_f(residuals[-1])[:, -1, :].detach().cpu().numpy().astype(np.float64))
    model.train()
    return np.concatenate(rows, axis=0)


def prompt_affine_metrics(
    model: CausalTransformer,
    vocab: AmbiguousBucketVocab,
    train_subjects: list[int],
    eval_subjects: list[int],
    targets: np.ndarray,
    relation_idx: int,
    device: torch.device,
) -> dict[str, float]:
    train_h = prompt_hidden(model, vocab, train_subjects, relation_idx, device)
    eval_h = prompt_hidden(model, vocab, eval_subjects, relation_idx, device)
    unembed = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    object_unembed = unembed[[vocab.object_id(i) for i in range(vocab.cfg.n_objects)]]
    train_targets = [int(targets[i]) for i in train_subjects]
    eval_targets = [int(targets[i]) for i in eval_subjects]
    train_y = object_unembed[train_targets]
    eval_y = object_unembed[eval_targets]
    affine = affine_metrics(train_h, train_y, eval_h, eval_y)
    nearest = affine_nearest_metrics(train_h, train_y, eval_h, eval_targets, object_unembed)
    weights = fit_affine(train_h, train_y)
    return {
        "prompt_affine_r2": affine["r2"],
        "prompt_affine_cosine": affine["cosine"],
        "prompt_affine_nn_accuracy": nearest["accuracy"],
        "prompt_affine_nn_mean_rank": nearest["mean_rank"],
        "prompt_affine_weight_norm": float(np.linalg.norm(weights)),
    }


def run_single_exposure(cfg: AmbiguousBucketConfig, device: torch.device) -> dict:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    vocab = AmbiguousBucketVocab(cfg)
    pilot_subjects, single_subjects, heldout_subjects = split_subjects(vocab)
    model = build_model_for_vocab(cfg, vocab, device)
    phase1 = attr_sequences(vocab) * cfg.attr_repeats
    phase1.extend(
        triple_sequence(vocab, i, int(vocab.semantic_targets[i]), relation_idx=0)
        for _ in range(cfg.pilot_repeats)
        for i in pilot_subjects
    )
    t0 = time.time()
    phase1_steps, phase1_loss = train_sequences(model, vocab, cfg, phase1, device, cfg.learning_rate, cfg.seed)
    before_single = evaluate_relation(model, vocab, single_subjects, vocab.semantic_targets, 0, device)
    before_heldout = evaluate_relation(model, vocab, heldout_subjects, vocab.semantic_targets, 0, device)
    phase2 = [
        *(attr_sequences(vocab) * cfg.phase2_attr_repeats),
        *(
            triple_sequence(vocab, i, int(vocab.semantic_targets[i]), relation_idx=0)
            for _ in range(cfg.phase2_pilot_repeats)
            for i in pilot_subjects
        ),
        *(
        triple_sequence(vocab, i, int(vocab.semantic_targets[i]), relation_idx=0)
        for _ in range(cfg.single_repeats)
        for i in single_subjects
        ),
    ]
    phase2_steps, phase2_loss = train_sequences(
        model,
        vocab,
        cfg,
        phase2,
        device,
        cfg.phase2_learning_rate,
        cfg.seed + 1,
        update_mode=cfg.phase2_update_mode,
    )
    after_pilot = evaluate_relation(model, vocab, pilot_subjects, vocab.semantic_targets, 0, device)
    after_single = evaluate_relation(model, vocab, single_subjects, vocab.semantic_targets, 0, device)
    after_heldout = evaluate_relation(model, vocab, heldout_subjects, vocab.semantic_targets, 0, device)
    affine = prompt_affine_metrics(model, vocab, pilot_subjects + single_subjects, heldout_subjects, vocab.semantic_targets, 0, device)
    return {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "phase1_steps": phase1_steps,
        "phase2_steps": phase2_steps,
        "phase1_loss": phase1_loss,
        "phase2_loss": phase2_loss,
        "pilot_subjects": len(pilot_subjects),
        "single_subjects": len(single_subjects),
        "heldout_subjects": len(heldout_subjects),
        "before": {"single": before_single, "heldout": before_heldout},
        "after": {"pilot": after_pilot, "single": after_single, "heldout": after_heldout},
        "single_exposure_advantage": {
            "object_accuracy": after_single["object_accuracy"] - after_heldout["object_accuracy"],
            "bucket_top1_accuracy": after_single["bucket_top1_accuracy"] - after_heldout["bucket_top1_accuracy"],
            "mean_bucket_rank_improvement": after_heldout["mean_bucket_rank"] - after_single["mean_bucket_rank"],
            "loss_reduction_advantage": (before_single["loss"] - after_single["loss"]) - (before_heldout["loss"] - after_heldout["loss"]),
        },
        "prompt_affine_to_heldout": affine,
        "elapsed_sec": time.time() - t0,
    }


def train_frequency_model(
    cfg: AmbiguousBucketConfig,
    vocab: AmbiguousBucketVocab,
    train_subjects: list[int],
    repeats: int,
    device: torch.device,
    seed_offset: int,
) -> CausalTransformer:
    model = build_model_for_vocab(cfg, vocab, device, seed_offset=seed_offset)
    sequences = attr_sequences(vocab) * cfg.attr_repeats
    sequences.extend(
        triple_sequence(vocab, i, int(vocab.semantic_targets[i]), relation_idx=0)
        for _ in range(repeats)
        for i in train_subjects
    )
    sequences.extend(
        triple_sequence(vocab, i, int(vocab.random_targets[i]), relation_idx=1)
        for _ in range(repeats)
        for i in train_subjects
    )
    train_sequences(model, vocab, cfg, sequences, device, cfg.learning_rate, cfg.seed + seed_offset)
    return model


def run_frequency_sweep(cfg: AmbiguousBucketConfig, repeats: list[int], device: torch.device) -> dict:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    vocab = AmbiguousBucketVocab(cfg)
    pilot_subjects, single_subjects, heldout_subjects = split_subjects(vocab)
    train_subjects = pilot_subjects + single_subjects
    rows = []
    t0 = time.time()
    for rep in repeats:
        model = train_frequency_model(cfg, vocab, train_subjects, rep, device, seed_offset=100 + rep)
        semantic_train = evaluate_relation(model, vocab, train_subjects, vocab.semantic_targets, 0, device)
        semantic_eval = evaluate_relation(model, vocab, heldout_subjects, vocab.semantic_targets, 0, device)
        random_train = evaluate_relation(model, vocab, train_subjects, vocab.random_targets, 1, device)
        random_eval = evaluate_relation(model, vocab, heldout_subjects, vocab.random_targets, 1, device)
        semantic_affine = prompt_affine_metrics(model, vocab, train_subjects, heldout_subjects, vocab.semantic_targets, 0, device)
        random_affine = prompt_affine_metrics(model, vocab, train_subjects, heldout_subjects, vocab.random_targets, 1, device)
        rows.append(
            {
                "repeats": rep,
                "semantic": {
                    "train": semantic_train,
                    "heldout": semantic_eval,
                    "affine": semantic_affine,
                },
                "random": {
                    "train": random_train,
                    "heldout": random_eval,
                    "affine": random_affine,
                },
            }
        )
        print(
            f"F={rep:3d} | sem train obj={semantic_train['object_accuracy']:.3f} "
            f"held obj={semantic_eval['object_accuracy']:.3f} affNN={semantic_affine['prompt_affine_nn_accuracy']:.3f} | "
            f"rand train obj={random_train['object_accuracy']:.3f} "
            f"held obj={random_eval['object_accuracy']:.3f} affNN={random_affine['prompt_affine_nn_accuracy']:.3f}"
        )
    return {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "train_subjects": len(train_subjects),
        "heldout_subjects": len(heldout_subjects),
        "rows": rows,
        "elapsed_sec": time.time() - t0,
    }


def write_report(result: dict, output_dir: Path) -> Path:
    one = result["single_exposure"]
    lines = [
        "# Ambiguous-Bucket Experiments",
        "",
        "## Experiment 1: single exposure",
        "",
        f"- Single subjects: {one['single_subjects']}",
        f"- Held-out subjects: {one['heldout_subjects']}",
        f"- Single object accuracy: {one['after']['single']['object_accuracy']:.3f}",
        f"- Held-out object accuracy: {one['after']['heldout']['object_accuracy']:.3f}",
        f"- Net object-accuracy advantage: {one['single_exposure_advantage']['object_accuracy']:.3f}",
        f"- Single bucket top-1: {one['after']['single']['bucket_top1_accuracy']:.3f}",
        f"- Held-out bucket top-1: {one['after']['heldout']['bucket_top1_accuracy']:.3f}",
        f"- Mean bucket-rank improvement: {one['single_exposure_advantage']['mean_bucket_rank_improvement']:.3f}",
        "",
        "## Experiment 2: frequency sweep",
        "",
        "| repeats | sem train | sem held | sem coarse | sem affine NN | rand train | rand held | rand affine NN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["frequency_sweep"]["rows"]:
        lines.append(
            f"| {row['repeats']} "
            f"| {row['semantic']['train']['object_accuracy']:.3f} "
            f"| {row['semantic']['heldout']['object_accuracy']:.3f} "
            f"| {row['semantic']['heldout']['coarse_bucket_accuracy']:.3f} "
            f"| {row['semantic']['affine']['prompt_affine_nn_accuracy']:.3f} "
            f"| {row['random']['train']['object_accuracy']:.3f} "
            f"| {row['random']['heldout']['object_accuracy']:.3f} "
            f"| {row['random']['affine']['prompt_affine_nn_accuracy']:.3f} |"
        )
    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repeats", default="1,4,16,64")
    parser.add_argument("--n-buckets", type=int, default=None)
    parser.add_argument("--subjects-per-bucket", type=int, default=None)
    parser.add_argument("--objects-per-bucket", type=int, default=None)
    parser.add_argument("--attr-repeats", type=int, default=None)
    parser.add_argument("--pilot-repeats", type=int, default=None)
    parser.add_argument("--phase2-attr-repeats", type=int, default=None)
    parser.add_argument("--phase2-pilot-repeats", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--phase2-learning-rate", type=float, default=None)
    parser.add_argument("--phase2-update-mode", default=None)
    parser.add_argument("--object-loss-weight", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    cfg = AmbiguousBucketConfig()
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    for arg_name, field_name in [
        ("n_buckets", "n_buckets"),
        ("subjects_per_bucket", "subjects_per_bucket"),
        ("objects_per_bucket", "objects_per_bucket"),
        ("attr_repeats", "attr_repeats"),
        ("pilot_repeats", "pilot_repeats"),
        ("phase2_attr_repeats", "phase2_attr_repeats"),
        ("phase2_pilot_repeats", "phase2_pilot_repeats"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("phase2_learning_rate", "phase2_learning_rate"),
        ("phase2_update_mode", "phase2_update_mode"),
        ("object_loss_weight", "object_loss_weight"),
        ("hidden_dim", "hidden_dim"),
        ("n_layers", "n_layers"),
        ("seed", "seed"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, field_name, value)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    repeats = [int(part) for part in args.repeats.split(",") if part]

    print(f"Device: {device}")
    print(f"Output: {cfg.output_dir}")
    single = run_single_exposure(cfg, device)
    print(
        "single exposure | "
        f"single obj={single['after']['single']['object_accuracy']:.3f} "
        f"held obj={single['after']['heldout']['object_accuracy']:.3f} "
        f"bucket rank improvement={single['single_exposure_advantage']['mean_bucket_rank_improvement']:.3f}"
    )
    sweep = run_frequency_sweep(cfg, repeats, device)
    result = {"single_exposure": single, "frequency_sweep": sweep}
    result_path = cfg.output_dir / "ambiguous_bucket_metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = write_report(result, cfg.output_dir)
    print(json.dumps({"result_path": str(result_path), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()

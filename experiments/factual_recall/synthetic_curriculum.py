"""Structured synthetic factual-recall curricula and affine analyses."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import PROJECT_ROOT, TransformerConfig
from .model import CausalTransformer, build_model
from .training import get_lr


@dataclass
class StructuredVocab:
    """Token layout for f/g-scaffolded subject-object entities."""

    n_entities: int = 1024
    n_f: int = 64
    n_g: int = 64
    delta_f: int = 7
    delta_g: int = 11
    seed: int = 0
    sep_id: int = 0
    relation_id: int = 1
    n_relations: int = 1
    attr_mode: str = "disjoint_shift"
    attr_buckets: int = 0
    subject_attrs: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    object_attrs: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.subject_attrs or not self.object_attrs:
            if self.attr_mode == "disjoint_shift":
                subject_attrs, object_attrs = make_disjoint_relation_attrs(
                    n_entities=self.n_entities,
                    n_f=self.n_f,
                    n_g=self.n_g,
                    delta_f=self.delta_f,
                    delta_g=self.delta_g,
                    seed=self.seed,
                )
            elif self.attr_mode == "f_bucket":
                buckets = self.attr_buckets or self.n_f
                if buckets > self.n_f or buckets > self.n_entities:
                    raise ValueError("attr_buckets must be <= n_f and <= n_entities")
                subject_attrs = [
                    (i % buckets, (i // buckets) % self.n_g)
                    for i in range(self.n_entities)
                ]
                object_attrs = []
                for i in range(self.n_entities):
                    if i < buckets:
                        object_attrs.append(((i + self.delta_f) % self.n_f, 0))
                    else:
                        object_attrs.append((i % self.n_f, (i // self.n_f + self.delta_g) % self.n_g))
            else:
                raise ValueError(f"Unknown attr_mode: {self.attr_mode}")
            self.subject_attrs = tuple(subject_attrs)
            self.object_attrs = tuple(object_attrs)

    @property
    def subject_offset(self) -> int:
        return self.relation_id + self.n_relations

    @property
    def object_offset(self) -> int:
        return self.subject_offset + self.n_entities

    @property
    def f_offset(self) -> int:
        return self.object_offset + self.n_entities

    @property
    def g_offset(self) -> int:
        return self.f_offset + self.n_f

    @property
    def vocab_size(self) -> int:
        return self.g_offset + self.n_g

    def subject_id(self, index: int) -> int:
        return self.subject_offset + index

    def object_id(self, index: int) -> int:
        return self.object_offset + index

    def relation_token(self, relation_idx: int = 0) -> int:
        if relation_idx < 0 or relation_idx >= self.n_relations:
            raise ValueError(f"relation_idx must be in [0, {self.n_relations})")
        return self.relation_id + relation_idx

    def f_id(self, index: int) -> int:
        return self.f_offset + index

    def g_id(self, index: int) -> int:
        return self.g_offset + index


@dataclass
class CurriculumConfig:
    """Runtime settings for Experiment 1."""

    seed: int = 7
    n_entities: int = 1024
    n_f: int = 64
    n_g: int = 64
    delta_f: int = 7
    delta_g: int = 11
    phase1_k: int = 10
    phase1_attr_repeats: int = 8
    reverse_attr_repeats: int = 0
    phase1_triple_repeats: int = 256
    phase2_triples: int = 1014
    phase2_repeats: int = 1
    structured_relation_repeats: int | None = None
    random_relation_repeats: int | None = None
    structured_target_mode: str = "identity"
    structured_target_buckets: int = 0
    batch_size: int = 64
    learning_rate: float = 3e-4
    phase2_learning_rate: float | None = None
    phase2_warmup_steps: int = 0
    phase2_decay: bool = False
    weight_decay: float = 1e-2
    warmup_steps: int = 25
    grad_clip: float = 1.0
    object_loss_weight: float = 1.0
    structured_embedding_init: bool = True
    init_scale: float = 0.2
    snapshot_every_triples: int = 128
    eval_limit: int = 512
    affine_train_limit: int = 512
    output_dir: Path = PROJECT_ROOT / "factual_recall" / "results" / "synthetic_exp1"


@dataclass
class SyntheticExperimentConfig:
    vocab: StructuredVocab = field(default_factory=StructuredVocab)
    model: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            n_layers=2,
            hidden_dim=128,
            n_heads=4,
            context_len=8,
            tie_embeddings=False,
        )
    )
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


class SequenceListDataset(Dataset):
    """Precomputed fixed-length language-modeling sequences."""

    def __init__(self, sequences: list[list[int]]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.sequences[idx]
        return (
            torch.tensor(tokens[:-1], dtype=torch.long),
            torch.tensor(tokens[1:], dtype=torch.long),
        )


def make_disjoint_relation_attrs(
    n_entities: int,
    n_f: int,
    n_g: int,
    delta_f: int,
    delta_g: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Sample unique source and target f/g combinations for an affine relation."""
    capacity = n_f * n_g
    if 2 * n_entities > capacity:
        raise ValueError(
            "Need at least two f/g combinations per entity so subject and object "
            "attribute pairs are globally unique."
        )

    rng = random.Random(seed)
    candidates = [(f, g) for f in range(n_f) for g in range(n_g)]
    rng.shuffle(candidates)

    sources: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for f, g in candidates:
        target = ((f + delta_f) % n_f, (g + delta_g) % n_g)
        if (f, g) in used or target in used:
            continue
        sources.append((f, g))
        targets.append(target)
        used.add((f, g))
        used.add(target)
        if len(sources) == n_entities:
            return sources, targets

    raise ValueError(
        "Could not sample enough disjoint f/g pairs. Increase n_f*n_g or change deltas."
    )


def subject_attr_sequence(vocab: StructuredVocab, entity_idx: int) -> list[int]:
    f, g = vocab.subject_attrs[entity_idx]
    return [vocab.sep_id, vocab.subject_id(entity_idx), vocab.f_id(f), vocab.g_id(g), vocab.sep_id]


def object_attr_sequence(vocab: StructuredVocab, entity_idx: int) -> list[int]:
    f, g = vocab.object_attrs[entity_idx]
    return [vocab.sep_id, vocab.object_id(entity_idx), vocab.f_id(f), vocab.g_id(g), vocab.sep_id]


def reverse_object_attr_sequence(vocab: StructuredVocab, entity_idx: int) -> list[int]:
    f, g = vocab.object_attrs[entity_idx]
    return [vocab.sep_id, vocab.f_id(f), vocab.g_id(g), vocab.object_id(entity_idx), vocab.sep_id]


def triple_sequence(vocab: StructuredVocab, entity_idx: int) -> list[int]:
    return [
        vocab.sep_id,
        vocab.subject_id(entity_idx),
        vocab.relation_id,
        vocab.object_id(entity_idx),
        vocab.sep_id,
    ]


def mapped_triple_sequence(
    vocab: StructuredVocab,
    subject_idx: int,
    object_idx: int,
    relation_idx: int = 0,
) -> list[int]:
    return [
        vocab.sep_id,
        vocab.subject_id(subject_idx),
        vocab.relation_token(relation_idx),
        vocab.object_id(object_idx),
        vocab.sep_id,
    ]


def build_curriculum_sequences(
    vocab: StructuredVocab,
    cfg: CurriculumConfig,
) -> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
    """Return phase 1 scaffold/triples and phase 2 novel triples."""
    if cfg.phase1_k >= vocab.n_entities:
        raise ValueError("phase1_k must leave at least one novel phase-2 triple")

    phase1_indices = list(range(cfg.phase1_k))
    max_phase2 = vocab.n_entities - cfg.phase1_k
    phase2_n = min(cfg.phase2_triples, max_phase2)
    phase2_indices = list(range(cfg.phase1_k, cfg.phase1_k + phase2_n))

    attr_sequences = [
        seq
        for entity_idx in range(vocab.n_entities)
        for seq in (subject_attr_sequence(vocab, entity_idx), object_attr_sequence(vocab, entity_idx))
    ]
    phase1_sequences = attr_sequences * cfg.phase1_attr_repeats
    if cfg.reverse_attr_repeats > 0:
        reverse_sequences = [reverse_object_attr_sequence(vocab, entity_idx) for entity_idx in range(vocab.n_entities)]
        phase1_sequences.extend(reverse_sequences * cfg.reverse_attr_repeats)
    phase1_sequences.extend(
        triple_sequence(vocab, entity_idx)
        for entity_idx in phase1_indices
        for _ in range(cfg.phase1_triple_repeats)
    )
    phase2_sequences = [
        triple_sequence(vocab, entity_idx)
        for _ in range(cfg.phase2_repeats)
        for entity_idx in phase2_indices
    ]
    return phase1_sequences, phase2_sequences, phase1_indices, phase2_indices


def init_entity_embeddings_from_attrs(
    model: CausalTransformer,
    vocab: StructuredVocab,
    scale: float,
    seed: int,
) -> None:
    """Seed entity and output rows with additive f/g attribute geometry."""
    generator = torch.Generator(device=model.token_emb.weight.device)
    generator.manual_seed(seed)
    hidden_dim = model.token_emb.weight.shape[1]
    device = model.token_emb.weight.device
    f_codes = torch.randn(vocab.n_f, hidden_dim, generator=generator, device=device)
    g_codes = torch.randn(vocab.n_g, hidden_dim, generator=generator, device=device)
    role_subject = torch.randn(hidden_dim, generator=generator, device=device)
    role_object = torch.randn(hidden_dim, generator=generator, device=device)
    f_codes = nn.functional.normalize(f_codes, dim=1)
    g_codes = nn.functional.normalize(g_codes, dim=1)
    role_subject = nn.functional.normalize(role_subject, dim=0)
    role_object = nn.functional.normalize(role_object, dim=0)

    with torch.no_grad():
        for f in range(vocab.n_f):
            model.token_emb.weight[vocab.f_id(f)].copy_(scale * f_codes[f])
            if not model.model_cfg.tie_embeddings:
                model.lm_head.weight[vocab.f_id(f)].copy_(scale * f_codes[f])
        for g in range(vocab.n_g):
            model.token_emb.weight[vocab.g_id(g)].copy_(scale * g_codes[g])
            if not model.model_cfg.tie_embeddings:
                model.lm_head.weight[vocab.g_id(g)].copy_(scale * g_codes[g])
        for entity_idx, (f, g) in enumerate(vocab.subject_attrs):
            vector = scale * (f_codes[f] + g_codes[g] + 0.5 * role_subject)
            model.token_emb.weight[vocab.subject_id(entity_idx)].copy_(vector)
        for entity_idx, (f, g) in enumerate(vocab.object_attrs):
            vector = scale * (f_codes[f] + g_codes[g] + 0.5 * role_object)
            model.token_emb.weight[vocab.object_id(entity_idx)].copy_(vector)
            if not model.model_cfg.tie_embeddings:
                model.lm_head.weight[vocab.object_id(entity_idx)].copy_(vector)


def make_loader(
    sequences: list[list[int]],
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        SequenceListDataset(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_one_batch(
    model: CausalTransformer,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    grad_clip: float,
    object_token_range: tuple[int, int] | None = None,
    object_loss_weight: float = 1.0,
) -> float:
    for group in optimizer.param_groups:
        group["lr"] = lr
    logits = model(x)
    batch_size, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(batch_size * seq_len, vocab_size)
    flat_y = y.reshape(batch_size * seq_len)
    losses = nn.functional.cross_entropy(
        flat_logits,
        flat_y,
        reduction="none",
    )
    if object_token_range is not None and object_loss_weight != 1.0:
        lo, hi = object_token_range
        weights = torch.ones_like(losses)
        object_targets = (flat_y >= lo) & (flat_y < hi)
        weights = torch.where(object_targets, weights * object_loss_weight, weights)
        loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        loss = losses.mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def triple_prompt_inputs(
    vocab: StructuredVocab,
    indices: Iterable[int],
    relation_idx: int = 0,
) -> torch.Tensor:
    prompts = [[vocab.sep_id, vocab.subject_id(i), vocab.relation_token(relation_idx)] for i in indices]
    return torch.tensor(prompts, dtype=torch.long)


def evaluate_mapped_triples(
    model: CausalTransformer,
    vocab: StructuredVocab,
    subject_indices: list[int],
    target_indices: np.ndarray,
    device: torch.device,
    limit: int,
    relation_idx: int = 0,
) -> dict[str, float]:
    model.eval()
    if not subject_indices:
        return {"accuracy": float("nan"), "loss": float("nan")}
    eval_subjects = subject_indices[:limit]
    correct = 0
    object_correct = 0
    top5 = 0
    top10 = 0
    ranks: list[int] = []
    total_loss = 0.0
    total = 0
    object_ids = torch.tensor(
        [vocab.object_id(i) for i in range(vocab.n_entities)],
        device=device,
        dtype=torch.long,
    )
    with torch.no_grad():
        for start in range(0, len(eval_subjects), 256):
            batch_subjects = eval_subjects[start : start + 256]
            x = triple_prompt_inputs(vocab, batch_subjects, relation_idx=relation_idx).to(device)
            y = torch.tensor(
                [vocab.object_id(int(target_indices[i])) for i in batch_subjects],
                device=device,
            )
            logits = model(x)[:, -1, :]
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            object_logits = logits.index_select(dim=-1, index=object_ids)
            object_pred = object_ids[object_logits.argmax(dim=-1)]
            object_correct += (object_pred == y).sum().item()
            target_logits = logits.gather(1, y[:, None])
            batch_ranks = 1 + (logits > target_logits).sum(dim=1)
            ranks.extend(int(rank.item()) for rank in batch_ranks.cpu())
            topk = logits.topk(k=min(10, logits.shape[-1]), dim=-1).indices
            top5 += (topk[:, : min(5, topk.shape[1])] == y[:, None]).any(dim=1).sum().item()
            top10 += (topk == y[:, None]).any(dim=1).sum().item()
            total += len(batch_subjects)
    model.train()
    return {
        "accuracy": correct / max(1, total),
        "object_accuracy": object_correct / max(1, total),
        "top5_accuracy": top5 / max(1, total),
        "top10_accuracy": top10 / max(1, total),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "loss": total_loss / max(1, total),
    }


def evaluate_triples(
    model: CausalTransformer,
    vocab: StructuredVocab,
    indices: list[int],
    device: torch.device,
    limit: int,
) -> dict[str, float]:
    model.eval()
    if not indices:
        return {"accuracy": float("nan"), "loss": float("nan")}
    eval_indices = indices[:limit]
    correct = 0
    object_correct = 0
    top5 = 0
    top10 = 0
    ranks: list[int] = []
    total_loss = 0.0
    total = 0
    object_ids = torch.tensor(
        [vocab.object_id(i) for i in range(vocab.n_entities)],
        device=device,
        dtype=torch.long,
    )
    with torch.no_grad():
        for start in range(0, len(eval_indices), 256):
            batch_indices = eval_indices[start : start + 256]
            x = triple_prompt_inputs(vocab, batch_indices).to(device)
            y = torch.tensor([vocab.object_id(i) for i in batch_indices], device=device)
            logits = model(x)[:, -1, :]
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            object_logits = logits.index_select(dim=-1, index=object_ids)
            object_pred = object_ids[object_logits.argmax(dim=-1)]
            object_correct += (object_pred == y).sum().item()
            target_logits = logits.gather(1, y[:, None])
            batch_ranks = 1 + (logits > target_logits).sum(dim=1)
            ranks.extend(int(rank.item()) for rank in batch_ranks.cpu())
            topk = logits.topk(k=min(10, logits.shape[-1]), dim=-1).indices
            top5 += (topk[:, : min(5, topk.shape[1])] == y[:, None]).any(dim=1).sum().item()
            top10 += (topk == y[:, None]).any(dim=1).sum().item()
            total += len(batch_indices)
    model.train()
    return {
        "accuracy": correct / max(1, total),
        "object_accuracy": object_correct / max(1, total),
        "top5_accuracy": top5 / max(1, total),
        "top10_accuracy": top10 / max(1, total),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "loss": total_loss / max(1, total),
    }


def fit_affine(train_x: np.ndarray, train_y: np.ndarray, ridge: float | None = None) -> np.ndarray:
    """Fit an affine map with ordinary least-squares linear regression.

    The returned matrix has shape ``(input_dim + 1, output_dim)``. The last row is
    the intercept, so callers can keep using ``apply_affine``.

    ``ridge`` is accepted for backwards compatibility with older call sites, but
    affine metrics in this file intentionally use unregularized linear regression.
    """
    del ridge
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression(fit_intercept=True)
    reg.fit(train_x, train_y)
    weights = np.concatenate(
        [reg.coef_.T.astype(train_x.dtype, copy=False), reg.intercept_[None, :].astype(train_x.dtype, copy=False)],
        axis=0,
    )
    return weights


def apply_affine(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def affine_metrics(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, float]:
    if len(train_x) < 2 or len(test_x) < 1:
        return {"r2": float("nan"), "cosine": float("nan")}
    weights = fit_affine(train_x, train_y)
    pred = apply_affine(test_x, weights)
    ss_res = float(np.square(test_y - pred).sum())
    ss_tot = float(np.square(test_y - test_y.mean(axis=0, keepdims=True)).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(test_y, axis=1)
    cosine = float(np.mean(np.sum(pred * test_y, axis=1) / np.maximum(denom, 1e-12)))
    return {"r2": r2, "cosine": cosine}


def affine_nearest_metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    target_indices: list[int],
    candidates_y: np.ndarray,
) -> dict[str, float]:
    if len(train_x) < 2 or len(test_x) < 1:
        return {"accuracy": float("nan"), "mean_rank": float("nan")}
    weights = fit_affine(train_x, train_y)
    pred = apply_affine(test_x, weights)
    pred_norm = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    cand_norm = candidates_y / np.maximum(np.linalg.norm(candidates_y, axis=1, keepdims=True), 1e-12)
    similarities = pred_norm @ cand_norm.T
    predicted = similarities.argmax(axis=1)
    targets = np.asarray(target_indices, dtype=np.int64)
    ranks = 1 + (similarities > similarities[np.arange(len(targets)), targets][:, None]).sum(axis=1)
    return {
        "accuracy": float(np.mean(predicted == targets)),
        "mean_rank": float(np.mean(ranks)),
    }


def attribute_probe_r2(entity_vectors: np.ndarray, attrs: tuple[tuple[int, int], ...]) -> float:
    if len(attrs) < 8:
        return float("nan")
    y = np.asarray(attrs, dtype=np.float64)
    y = (y - y.mean(axis=0, keepdims=True)) / np.maximum(y.std(axis=0, keepdims=True), 1e-6)
    split = max(4, int(0.75 * len(attrs)))
    weights = fit_affine(entity_vectors[:split], y[:split])
    pred = apply_affine(entity_vectors[split:], weights)
    ss_res = float(np.square(y[split:] - pred).sum())
    ss_tot = float(np.square(y[split:] - y[split:].mean(axis=0, keepdims=True)).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def representation_snapshot(
    model: CausalTransformer,
    vocab: StructuredVocab,
    seen_indices: list[int],
    heldout_indices: list[int],
    device: torch.device,
    eval_limit: int,
    affine_train_limit: int,
) -> dict[str, float]:
    token_weights = model.token_emb.weight.detach().cpu().numpy().astype(np.float64)
    unembed_weights = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    train_indices = seen_indices[-affine_train_limit:]
    test_indices = heldout_indices[:eval_limit]

    subj_train = token_weights[[vocab.subject_id(i) for i in train_indices]]
    subj_test = token_weights[[vocab.subject_id(i) for i in test_indices]]
    obj_embed_train = token_weights[[vocab.object_id(i) for i in train_indices]]
    obj_embed_test = token_weights[[vocab.object_id(i) for i in test_indices]]
    obj_unembed_train = unembed_weights[[vocab.object_id(i) for i in train_indices]]
    obj_unembed_test = unembed_weights[[vocab.object_id(i) for i in test_indices]]
    object_vectors = token_weights[[vocab.object_id(i) for i in range(vocab.n_entities)]]

    embed_affine = affine_metrics(subj_train, obj_embed_train, subj_test, obj_embed_test)
    unembed_affine = affine_metrics(subj_train, obj_unembed_train, subj_test, obj_unembed_test)
    embed_nn = affine_nearest_metrics(subj_train, obj_embed_train, subj_test, test_indices, object_vectors)
    unembed_nn = affine_nearest_metrics(
        subj_train,
        obj_unembed_train,
        subj_test,
        test_indices,
        unembed_weights[[vocab.object_id(i) for i in range(vocab.n_entities)]],
    )
    seen_eval = evaluate_triples(model, vocab, seen_indices[-eval_limit:], device, eval_limit)
    heldout_eval = evaluate_triples(model, vocab, heldout_indices, device, eval_limit)
    subject_vectors = token_weights[[vocab.subject_id(i) for i in range(vocab.n_entities)]]

    return {
        "seen_accuracy": seen_eval["accuracy"],
        "seen_object_accuracy": seen_eval["object_accuracy"],
        "seen_top5_accuracy": seen_eval["top5_accuracy"],
        "seen_top10_accuracy": seen_eval["top10_accuracy"],
        "seen_mean_rank": seen_eval["mean_rank"],
        "seen_loss": seen_eval["loss"],
        "heldout_accuracy": heldout_eval["accuracy"],
        "heldout_object_accuracy": heldout_eval["object_accuracy"],
        "heldout_top5_accuracy": heldout_eval["top5_accuracy"],
        "heldout_top10_accuracy": heldout_eval["top10_accuracy"],
        "heldout_mean_rank": heldout_eval["mean_rank"],
        "heldout_loss": heldout_eval["loss"],
        "embedding_affine_r2": embed_affine["r2"],
        "embedding_affine_cosine": embed_affine["cosine"],
        "embedding_affine_nn_accuracy": embed_nn["accuracy"],
        "embedding_affine_nn_mean_rank": embed_nn["mean_rank"],
        "unembedding_affine_r2": unembed_affine["r2"],
        "unembedding_affine_cosine": unembed_affine["cosine"],
        "unembedding_affine_nn_accuracy": unembed_nn["accuracy"],
        "unembedding_affine_nn_mean_rank": unembed_nn["mean_rank"],
        "subject_attr_probe_r2": attribute_probe_r2(subject_vectors, vocab.subject_attrs),
        "object_attr_probe_r2": attribute_probe_r2(object_vectors, vocab.object_attrs),
    }


def plot_history(history: list[dict[str, float]], output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    output_path = output_dir / "phase2_metrics.png"
    x = [row["phase2_triples_seen"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].plot(x, [row["seen_accuracy"] for row in history], label="seen")
    axes[0].plot(x, [row["heldout_accuracy"] for row in history], label="heldout")
    axes[0].set_xlabel("phase-2 triples seen once")
    axes[0].set_ylabel("object accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend()

    axes[1].plot(x, [row["embedding_affine_cosine"] for row in history], label="embedding affine")
    axes[1].plot(x, [row["unembedding_affine_cosine"] for row in history], label="unembedding affine")
    axes[1].set_xlabel("phase-2 triples seen once")
    axes[1].set_ylabel("heldout affine cosine")
    axes[1].set_ylim(-0.1, 1.02)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def experiment1_summary(history: list[dict[str, float]]) -> dict[str, float | bool | str]:
    """Summarize whether Experiment 1 shows an affine phase shift."""
    if not history:
        return {
            "status": "missing_history",
            "affine_phase_shift": False,
            "recall_phase_shift": False,
        }
    first = history[0]
    last = history[-1]
    emb_start = float(first.get("embedding_affine_cosine", float("nan")))
    emb_final = float(last.get("embedding_affine_cosine", float("nan")))
    unemb_start = float(first.get("unembedding_affine_cosine", float("nan")))
    unemb_final = float(last.get("unembedding_affine_cosine", float("nan")))
    seen_start = float(first.get("seen_object_accuracy", first.get("seen_accuracy", float("nan"))))
    seen_final = float(last.get("seen_object_accuracy", last.get("seen_accuracy", float("nan"))))
    heldout_start = float(first.get("heldout_object_accuracy", first.get("heldout_accuracy", float("nan"))))
    heldout_final = float(last.get("heldout_object_accuracy", last.get("heldout_accuracy", float("nan"))))
    emb_nn_final = float(last.get("embedding_affine_nn_accuracy", float("nan")))
    unemb_nn_final = float(last.get("unembedding_affine_nn_accuracy", float("nan")))

    affine_phase_shift = (
        emb_final >= 0.8
        and unemb_final >= 0.8
        and emb_final - emb_start >= 0.25
        and unemb_final - unemb_start >= 0.25
    )
    recall_phase_shift = seen_final >= 0.5 or heldout_final >= 0.1
    return {
        "status": "ok",
        "affine_phase_shift": bool(affine_phase_shift),
        "recall_phase_shift": bool(recall_phase_shift),
        "embedding_affine_cosine_start": emb_start,
        "embedding_affine_cosine_final": emb_final,
        "embedding_affine_cosine_delta": emb_final - emb_start,
        "unembedding_affine_cosine_start": unemb_start,
        "unembedding_affine_cosine_final": unemb_final,
        "unembedding_affine_cosine_delta": unemb_final - unemb_start,
        "embedding_affine_nn_accuracy_final": emb_nn_final,
        "unembedding_affine_nn_accuracy_final": unemb_nn_final,
        "seen_object_accuracy_start": seen_start,
        "seen_object_accuracy_final": seen_final,
        "heldout_object_accuracy_start": heldout_start,
        "heldout_object_accuracy_final": heldout_final,
        "phase2_triples_seen_final": float(last.get("phase2_triples_seen", 0.0)),
    }


def write_experiment1_report(result: dict, output_dir: Path) -> Path:
    """Write a compact Markdown report for the latest Experiment 1 run."""
    summary = result["summary"]
    final = result["history"][-1]
    report_path = output_dir / "summary.md"
    lines = [
        "# Synthetic Experiment 1",
        "",
        f"- Phase-1 sequences: {result['phase1_sequences']}",
        f"- Phase-2 sequences: {result['phase2_sequences']}",
        f"- Unique phase-2 triples: {result['phase2_unique_triples']}",
        f"- Never-trained heldout triples: {result['never_trained_heldout_triples']}",
        f"- Parameters: {result['num_parameters']}",
        "",
        "## Phase-Shift Summary",
        "",
        f"- Affine phase shift: {summary['affine_phase_shift']}",
        f"- Recall phase shift: {summary['recall_phase_shift']}",
        f"- Embedding affine cosine: {summary['embedding_affine_cosine_start']:.4f} -> {summary['embedding_affine_cosine_final']:.4f}",
        f"- Unembedding affine cosine: {summary['unembedding_affine_cosine_start']:.4f} -> {summary['unembedding_affine_cosine_final']:.4f}",
        f"- Final embedding affine NN accuracy: {summary['embedding_affine_nn_accuracy_final']:.4f}",
        f"- Final unembedding affine NN accuracy: {summary['unembedding_affine_nn_accuracy_final']:.4f}",
        f"- Final seen object accuracy: {summary['seen_object_accuracy_final']:.4f}",
        f"- Final heldout object accuracy: {summary['heldout_object_accuracy_final']:.4f}",
        "",
        "## Final Snapshot",
        "",
        f"- Phase: {final['phase']}",
        f"- Phase-2 triples seen: {final['phase2_triples_seen']}",
        f"- Global step: {final['global_step']}",
        f"- Last train loss: {final['last_train_loss']:.4f}",
        f"- Elapsed seconds: {final['elapsed_sec']:.2f}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_experiment1(
    cfg: SyntheticExperimentConfig,
    device: torch.device,
    quiet: bool = False,
) -> dict:
    torch.manual_seed(cfg.curriculum.seed)
    np.random.seed(cfg.curriculum.seed)
    random.seed(cfg.curriculum.seed)
    cfg.curriculum.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg.model, cfg.vocab)
    model.to(device)
    if cfg.curriculum.structured_embedding_init:
        init_entity_embeddings_from_attrs(
            model=model,
            vocab=cfg.vocab,
            scale=cfg.curriculum.init_scale,
            seed=cfg.curriculum.seed,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.curriculum.learning_rate,
        weight_decay=cfg.curriculum.weight_decay,
    )
    phase1_sequences, phase2_sequences, phase1_indices, phase2_indices = build_curriculum_sequences(
        cfg.vocab, cfg.curriculum
    )
    phase1_loader = make_loader(
        phase1_sequences,
        batch_size=cfg.curriculum.batch_size,
        seed=cfg.curriculum.seed,
        shuffle=True,
    )
    phase2_loader = make_loader(
        phase2_sequences,
        batch_size=cfg.curriculum.batch_size,
        seed=cfg.curriculum.seed + 1,
        shuffle=True,
    )
    phase1_max_steps = len(phase1_loader)
    phase2_max_steps = len(phase2_loader)
    phase2_learning_rate = (
        cfg.curriculum.phase2_learning_rate
        if cfg.curriculum.phase2_learning_rate is not None
        else cfg.curriculum.learning_rate
    )
    global_step = 0
    t0 = time.time()

    for x, y in phase1_loader:
        x, y = x.to(device), y.to(device)
        lr = get_lr(global_step, cfg.curriculum.warmup_steps, phase1_max_steps, cfg.curriculum.learning_rate)
        train_one_batch(
            model,
            optimizer,
            x,
            y,
            lr,
            cfg.curriculum.grad_clip,
            object_token_range=(cfg.vocab.object_offset, cfg.vocab.f_offset),
            object_loss_weight=cfg.curriculum.object_loss_weight,
        )
        global_step += 1

    seen_indices = list(phase1_indices)
    never_trained_indices = list(range(cfg.curriculum.phase1_k + len(phase2_indices), cfg.vocab.n_entities))
    heldout_indices = list(phase2_indices) + never_trained_indices
    history: list[dict[str, float]] = []

    def add_snapshot(phase2_seen: int, phase: str, loss: float | None = None) -> None:
        snapshot = representation_snapshot(
            model=model,
            vocab=cfg.vocab,
            seen_indices=seen_indices,
            heldout_indices=heldout_indices,
            device=device,
            eval_limit=cfg.curriculum.eval_limit,
            affine_train_limit=cfg.curriculum.affine_train_limit,
        )
        snapshot.update(
            {
                "phase": phase,
                "phase2_triples_seen": phase2_seen,
                "last_train_loss": loss if loss is not None else float("nan"),
                "global_step": global_step,
                "elapsed_sec": time.time() - t0,
            }
        )
        history.append(snapshot)
        if not quiet:
            print(
                f"{phase:>7} | phase2_seen={phase2_seen:4d} | "
                f"seen_acc={snapshot['seen_accuracy']:.3f} | "
                f"heldout_acc={snapshot['heldout_accuracy']:.3f} | "
                f"emb_aff_cos={snapshot['embedding_affine_cosine']:.3f} | "
                f"unemb_aff_cos={snapshot['unembedding_affine_cosine']:.3f}"
            )

    add_snapshot(0, "phase1")
    phase2_seen = 0
    phase2_exposures = 0
    phase2_unique_seen: set[int] = set()
    last_loss: float | None = None
    next_snapshot = cfg.curriculum.snapshot_every_triples
    phase2_step = 0
    for x, y in phase2_loader:
        batch_indices = [int(token.item() - cfg.vocab.subject_offset) for token in x[:, 1].cpu()]
        x, y = x.to(device), y.to(device)
        if cfg.curriculum.phase2_decay:
            lr = get_lr(
                phase2_step,
                cfg.curriculum.phase2_warmup_steps,
                phase2_max_steps,
                phase2_learning_rate,
            )
        elif cfg.curriculum.phase2_warmup_steps > 0 and phase2_step < cfg.curriculum.phase2_warmup_steps:
            lr = phase2_learning_rate * phase2_step / cfg.curriculum.phase2_warmup_steps
        else:
            lr = phase2_learning_rate
        last_loss = train_one_batch(
            model,
            optimizer,
            x,
            y,
            lr,
            cfg.curriculum.grad_clip,
            object_token_range=(cfg.vocab.object_offset, cfg.vocab.f_offset),
            object_loss_weight=cfg.curriculum.object_loss_weight,
        )
        global_step += 1
        phase2_step += 1
        seen_indices.extend(batch_indices)
        seen_set = set(batch_indices)
        phase2_unique_seen.update(seen_set)
        heldout_indices = [idx for idx in heldout_indices if idx not in seen_set]
        phase2_seen = len(phase2_unique_seen)
        phase2_exposures += len(batch_indices)
        if phase2_seen >= next_snapshot or phase2_exposures == len(phase2_sequences):
            add_snapshot(phase2_seen, "phase2", last_loss)
            while next_snapshot <= phase2_seen:
                next_snapshot += cfg.curriculum.snapshot_every_triples

    plot_path = plot_history(history, cfg.curriculum.output_dir)
    summary = experiment1_summary(history)
    result = {
        "config": {
            "vocab": asdict(cfg.vocab),
            "model": asdict(cfg.model),
            "curriculum": {
                **asdict(cfg.curriculum),
                "output_dir": str(cfg.curriculum.output_dir),
            },
        },
        "num_parameters": model.n_parameters(),
        "phase1_sequences": len(phase1_sequences),
        "phase2_sequences": len(phase2_sequences),
        "phase2_unique_triples": len(phase2_indices),
        "never_trained_heldout_triples": len(never_trained_indices),
        "summary": summary,
        "history": history,
        "plot_path": str(plot_path) if plot_path else None,
    }
    report_path = write_experiment1_report(result, cfg.curriculum.output_dir)
    result["report_path"] = str(report_path)
    result_path = cfg.curriculum.output_dir / "metrics.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    torch.save(model.state_dict(), cfg.curriculum.output_dir / "model.pt")
    result["result_path"] = str(result_path)
    return result


def attrs_to_feature_matrix(
    attrs: tuple[tuple[int, int], ...],
    n_f: int,
    n_g: int,
) -> np.ndarray:
    features = np.zeros((len(attrs), n_f + n_g), dtype=np.float64)
    for row, (f, g) in enumerate(attrs):
        features[row, f] = 1.0
        features[row, n_f + g] = 1.0
    return features


def relation_geometry_metrics(
    subject_features: np.ndarray,
    object_features: np.ndarray,
    target_indices: np.ndarray,
    train_indices: list[int],
    eval_indices: list[int],
) -> dict[str, float]:
    train_x = subject_features[train_indices]
    train_y = object_features[target_indices[train_indices]]
    eval_x = subject_features[eval_indices]
    eval_targets = target_indices[eval_indices].tolist()
    eval_y = object_features[eval_targets]
    metrics = affine_metrics(train_x, train_y, eval_x, eval_y)
    nearest = affine_nearest_metrics(train_x, train_y, eval_x, eval_targets, object_features)
    return {
        "affine_r2": metrics["r2"],
        "affine_cosine": metrics["cosine"],
        "affine_nn_accuracy": nearest["accuracy"],
        "affine_nn_mean_rank": nearest["mean_rank"],
    }


def run_experiment2(cfg: SyntheticExperimentConfig, quiet: bool = False) -> dict:
    """Compare affine-structured and random target relations at matched frequency."""
    rng = np.random.default_rng(cfg.curriculum.seed)
    output_dir = cfg.curriculum.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.arange(cfg.vocab.n_entities)
    rng.shuffle(indices)
    train_n = min(max(cfg.curriculum.phase1_k, 8), cfg.vocab.n_entities // 2)
    eval_n = min(cfg.curriculum.eval_limit, cfg.vocab.n_entities - train_n)
    train_indices = sorted(int(i) for i in indices[:train_n])
    eval_indices = sorted(int(i) for i in indices[train_n : train_n + eval_n])

    subject_features = attrs_to_feature_matrix(cfg.vocab.subject_attrs, cfg.vocab.n_f, cfg.vocab.n_g)
    object_features = attrs_to_feature_matrix(cfg.vocab.object_attrs, cfg.vocab.n_f, cfg.vocab.n_g)
    structured_targets = np.arange(cfg.vocab.n_entities, dtype=np.int64)
    random_targets = rng.permutation(cfg.vocab.n_entities).astype(np.int64)

    result = {
        "config": {
            "vocab": asdict(cfg.vocab),
            "model": asdict(cfg.model),
            "curriculum": {
                **asdict(cfg.curriculum),
                "output_dir": str(cfg.curriculum.output_dir),
            },
        },
        "matched_train_pairs": train_n,
        "matched_eval_pairs": eval_n,
        "relations": {
            "R0_structured": relation_geometry_metrics(
                subject_features,
                object_features,
                structured_targets,
                train_indices,
                eval_indices,
            ),
            "R1_random": relation_geometry_metrics(
                subject_features,
                object_features,
                random_targets,
                train_indices,
                eval_indices,
            ),
        },
    }
    result_path = output_dir / "experiment2_metrics.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    result["result_path"] = str(result_path)
    if not quiet:
        print(json.dumps(result["relations"], indent=2))
    return result


def mapped_relation_snapshot(
    model: CausalTransformer,
    vocab: StructuredVocab,
    target_indices: np.ndarray,
    affine_train_indices: list[int],
    affine_eval_indices: list[int],
    device: torch.device,
    eval_limit: int,
    relation_idx: int = 0,
) -> dict[str, float]:
    token_weights = model.token_emb.weight.detach().cpu().numpy().astype(np.float64)
    unembed_weights = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    object_vectors = token_weights[[vocab.object_id(i) for i in range(vocab.n_entities)]]
    object_unembed = unembed_weights[[vocab.object_id(i) for i in range(vocab.n_entities)]]

    subj_train = token_weights[[vocab.subject_id(i) for i in affine_train_indices]]
    subj_eval = token_weights[[vocab.subject_id(i) for i in affine_eval_indices]]
    train_targets = [int(target_indices[i]) for i in affine_train_indices]
    eval_targets = [int(target_indices[i]) for i in affine_eval_indices]
    obj_embed_train = object_vectors[train_targets]
    obj_embed_eval = object_vectors[eval_targets]
    obj_unembed_train = object_unembed[train_targets]
    obj_unembed_eval = object_unembed[eval_targets]

    embed_affine = affine_metrics(subj_train, obj_embed_train, subj_eval, obj_embed_eval)
    unembed_affine = affine_metrics(subj_train, obj_unembed_train, subj_eval, obj_unembed_eval)
    embed_nn = affine_nearest_metrics(
        subj_train,
        obj_embed_train,
        subj_eval,
        eval_targets,
        object_vectors,
    )
    unembed_nn = affine_nearest_metrics(
        subj_train,
        obj_unembed_train,
        subj_eval,
        eval_targets,
        object_unembed,
    )
    train_eval = evaluate_mapped_triples(
        model,
        vocab,
        affine_train_indices,
        target_indices,
        device,
        eval_limit,
        relation_idx=relation_idx,
    )
    heldout_eval = evaluate_mapped_triples(
        model,
        vocab,
        affine_eval_indices,
        target_indices,
        device,
        eval_limit,
        relation_idx=relation_idx,
    )
    return {
        "train_accuracy": train_eval["accuracy"],
        "train_top10_accuracy": train_eval["top10_accuracy"],
        "train_mean_rank": train_eval["mean_rank"],
        "eval_accuracy": heldout_eval["accuracy"],
        "eval_top10_accuracy": heldout_eval["top10_accuracy"],
        "eval_mean_rank": heldout_eval["mean_rank"],
        "embedding_affine_r2": embed_affine["r2"],
        "embedding_affine_cosine": embed_affine["cosine"],
        "embedding_affine_nn_accuracy": embed_nn["accuracy"],
        "embedding_affine_nn_mean_rank": embed_nn["mean_rank"],
        "unembedding_affine_r2": unembed_affine["r2"],
        "unembedding_affine_cosine": unembed_affine["cosine"],
        "unembedding_affine_nn_accuracy": unembed_nn["accuracy"],
        "unembedding_affine_nn_mean_rank": unembed_nn["mean_rank"],
    }


def train_mapped_relation(
    cfg: SyntheticExperimentConfig,
    target_indices: np.ndarray,
    train_subjects: list[int],
    repeats: int,
    device: torch.device,
    seed_offset: int,
    relation_idx: int = 0,
) -> tuple[CausalTransformer, int, float]:
    torch.manual_seed(cfg.curriculum.seed + seed_offset)
    np.random.seed(cfg.curriculum.seed + seed_offset)
    random.seed(cfg.curriculum.seed + seed_offset)

    model = build_model(cfg.model, cfg.vocab)
    model.to(device)
    if cfg.curriculum.structured_embedding_init:
        init_entity_embeddings_from_attrs(
            model=model,
            vocab=cfg.vocab,
            scale=cfg.curriculum.init_scale,
            seed=cfg.curriculum.seed + seed_offset,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.curriculum.learning_rate,
        weight_decay=cfg.curriculum.weight_decay,
    )
    attr_sequences = [
        seq
        for entity_idx in range(cfg.vocab.n_entities)
        for seq in (subject_attr_sequence(cfg.vocab, entity_idx), object_attr_sequence(cfg.vocab, entity_idx))
    ] * cfg.curriculum.phase1_attr_repeats
    if cfg.curriculum.reverse_attr_repeats > 0:
        attr_sequences.extend(
            [reverse_object_attr_sequence(cfg.vocab, entity_idx) for entity_idx in range(cfg.vocab.n_entities)]
            * cfg.curriculum.reverse_attr_repeats
        )
    triple_sequences = [
        mapped_triple_sequence(
            cfg.vocab,
            subject_idx,
            int(target_indices[subject_idx]),
            relation_idx=relation_idx,
        )
        for _ in range(repeats)
        for subject_idx in train_subjects
    ]
    sequences = attr_sequences + triple_sequences
    loader = make_loader(
        sequences,
        batch_size=cfg.curriculum.batch_size,
        seed=cfg.curriculum.seed + seed_offset,
        shuffle=True,
    )
    last_loss = float("nan")
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        lr = get_lr(step, cfg.curriculum.warmup_steps, len(loader), cfg.curriculum.learning_rate)
        last_loss = train_one_batch(
            model,
            optimizer,
            x,
            y,
            lr,
            cfg.curriculum.grad_clip,
            object_token_range=(cfg.vocab.object_offset, cfg.vocab.f_offset),
            object_loss_weight=cfg.curriculum.object_loss_weight,
        )
    return model, len(loader), last_loss


def run_frequency_contrast(
    cfg: SyntheticExperimentConfig,
    device: torch.device,
    quiet: bool = False,
) -> dict:
    """Train low-frequency structured and high-frequency random relations."""
    rng = np.random.default_rng(cfg.curriculum.seed)
    cfg.curriculum.output_dir.mkdir(parents=True, exist_ok=True)

    shuffled = np.arange(cfg.vocab.n_entities)
    rng.shuffle(shuffled)
    train_n = min(cfg.curriculum.phase2_triples, cfg.vocab.n_entities)
    eval_n = min(cfg.curriculum.eval_limit, cfg.vocab.n_entities - train_n)
    train_subjects = sorted(int(i) for i in shuffled[:train_n])
    never_seen_subjects = sorted(int(i) for i in shuffled[train_n : train_n + eval_n])
    affine_train_subjects = train_subjects
    affine_eval_subjects = never_seen_subjects

    structured_targets = np.arange(cfg.vocab.n_entities, dtype=np.int64)
    random_targets = rng.permutation(cfg.vocab.n_entities).astype(np.int64)
    low_repeats = max(1, cfg.curriculum.phase2_repeats)
    high_repeats = max(low_repeats + 1, cfg.curriculum.phase1_triple_repeats)

    t0 = time.time()
    structured_model, structured_steps, structured_loss = train_mapped_relation(
        cfg,
        structured_targets,
        train_subjects,
        repeats=low_repeats,
        device=device,
        seed_offset=0,
    )
    structured_metrics = mapped_relation_snapshot(
        structured_model,
        cfg.vocab,
        structured_targets,
        affine_train_subjects,
        affine_eval_subjects,
        device,
        cfg.curriculum.eval_limit,
    )

    random_model, random_steps, random_loss = train_mapped_relation(
        cfg,
        random_targets,
        train_subjects,
        repeats=high_repeats,
        device=device,
        seed_offset=1000,
    )
    random_metrics = mapped_relation_snapshot(
        random_model,
        cfg.vocab,
        random_targets,
        affine_train_subjects,
        affine_eval_subjects,
        device,
        cfg.curriculum.eval_limit,
    )

    result = {
        "config": {
            "vocab": asdict(cfg.vocab),
            "model": asdict(cfg.model),
            "curriculum": {
                **asdict(cfg.curriculum),
                "output_dir": str(cfg.curriculum.output_dir),
            },
        },
        "train_subjects": len(train_subjects),
        "never_seen_subjects": len(never_seen_subjects),
        "affine_train_subjects": len(affine_train_subjects),
        "affine_eval_subjects": len(affine_eval_subjects),
        "low_frequency_repeats": low_repeats,
        "high_frequency_repeats": high_repeats,
        "relations": {
            "low_frequency_structured": {
                **structured_metrics,
                "train_steps": structured_steps,
                "last_train_loss": structured_loss,
            },
            "high_frequency_random": {
                **random_metrics,
                "train_steps": random_steps,
                "last_train_loss": random_loss,
            },
        },
        "elapsed_sec": time.time() - t0,
    }
    result_path = cfg.curriculum.output_dir / "frequency_contrast_metrics.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    result["result_path"] = str(result_path)
    if not quiet:
        print(json.dumps(result["relations"], indent=2))
    return result


def prompt_residual_vectors(
    model: CausalTransformer,
    vocab: StructuredVocab,
    subject_indices: list[int],
    relation_idx: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(subject_indices), 256):
            batch_subjects = subject_indices[start : start + 256]
            x = triple_prompt_inputs(vocab, batch_subjects, relation_idx=relation_idx).to(device)
            with model.capture_residuals() as residuals:
                model(x)
            hidden = model.ln_f(residuals[-1])[:, -1, :]
            rows.append(hidden.detach().cpu().numpy().astype(np.float64))
    model.train()
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, model.model_cfg.hidden_dim))


def prompt_relation_metrics(
    model: CausalTransformer,
    vocab: StructuredVocab,
    target_indices: np.ndarray,
    affine_train_indices: list[int],
    affine_eval_indices: list[int],
    relation_idx: int,
    device: torch.device,
    eval_limit: int,
) -> dict[str, float]:
    unembed_weights = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    object_unembed = unembed_weights[[vocab.object_id(i) for i in range(vocab.n_entities)]]
    train_h = prompt_residual_vectors(model, vocab, affine_train_indices, relation_idx, device)
    eval_h = prompt_residual_vectors(model, vocab, affine_eval_indices, relation_idx, device)
    train_targets = [int(target_indices[i]) for i in affine_train_indices]
    eval_targets = [int(target_indices[i]) for i in affine_eval_indices]
    train_y = object_unembed[train_targets]
    eval_y = object_unembed[eval_targets]
    affine = affine_metrics(train_h, train_y, eval_h, eval_y)
    nearest = affine_nearest_metrics(train_h, train_y, eval_h, eval_targets, object_unembed)
    train_eval = evaluate_mapped_triples(
        model,
        vocab,
        affine_train_indices,
        target_indices,
        device,
        eval_limit,
        relation_idx=relation_idx,
    )
    heldout_eval = evaluate_mapped_triples(
        model,
        vocab,
        affine_eval_indices,
        target_indices,
        device,
        eval_limit,
        relation_idx=relation_idx,
    )
    return {
        "train_accuracy": train_eval["accuracy"],
        "train_top10_accuracy": train_eval["top10_accuracy"],
        "train_mean_rank": train_eval["mean_rank"],
        "eval_accuracy": heldout_eval["accuracy"],
        "eval_top10_accuracy": heldout_eval["top10_accuracy"],
        "eval_mean_rank": heldout_eval["mean_rank"],
        "prompt_affine_r2": affine["r2"],
        "prompt_affine_cosine": affine["cosine"],
        "prompt_affine_nn_accuracy": nearest["accuracy"],
        "prompt_affine_nn_mean_rank": nearest["mean_rank"],
    }


def embedding_relation_metrics(
    model: CausalTransformer,
    vocab: StructuredVocab,
    target_indices: np.ndarray,
    affine_train_indices: list[int],
    affine_eval_indices: list[int],
) -> dict[str, float]:
    """Linear-regression affine metrics from subject embeddings to object vectors."""
    token_weights = model.token_emb.weight.detach().cpu().numpy().astype(np.float64)
    unembed_weights = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    subj_train = token_weights[[vocab.subject_id(i) for i in affine_train_indices]]
    subj_eval = token_weights[[vocab.subject_id(i) for i in affine_eval_indices]]
    train_targets = [int(target_indices[i]) for i in affine_train_indices]
    eval_targets = [int(target_indices[i]) for i in affine_eval_indices]
    object_embed = token_weights[[vocab.object_id(i) for i in range(vocab.n_entities)]]
    object_unembed = unembed_weights[[vocab.object_id(i) for i in range(vocab.n_entities)]]

    embed_affine = affine_metrics(
        subj_train,
        object_embed[train_targets],
        subj_eval,
        object_embed[eval_targets],
    )
    embed_nearest = affine_nearest_metrics(
        subj_train,
        object_embed[train_targets],
        subj_eval,
        eval_targets,
        object_embed,
    )
    unembed_affine = affine_metrics(
        subj_train,
        object_unembed[train_targets],
        subj_eval,
        object_unembed[eval_targets],
    )
    unembed_nearest = affine_nearest_metrics(
        subj_train,
        object_unembed[train_targets],
        subj_eval,
        eval_targets,
        object_unembed,
    )
    return {
        "embedding_lr_affine_r2": embed_affine["r2"],
        "embedding_lr_affine_cosine": embed_affine["cosine"],
        "embedding_lr_affine_nn_accuracy": embed_nearest["accuracy"],
        "embedding_lr_affine_nn_mean_rank": embed_nearest["mean_rank"],
        "unembedding_lr_affine_r2": unembed_affine["r2"],
        "unembedding_lr_affine_cosine": unembed_affine["cosine"],
        "unembedding_lr_affine_nn_accuracy": unembed_nearest["accuracy"],
        "unembedding_lr_affine_nn_mean_rank": unembed_nearest["mean_rank"],
    }


def paired_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(np.sum(a * b, axis=1) / np.maximum(denom, 1e-12)))


def run_mixed_relations(
    cfg: SyntheticExperimentConfig,
    device: torch.device,
    quiet: bool = False,
) -> dict:
    """Train one model with structured and random relations.

    By default this preserves the original frequency-contrast setup. Supplying
    structured_relation_repeats and random_relation_repeats allows matched
    frequency tests that isolate relation structure from exposure count.
    """
    if cfg.vocab.n_relations < 2:
        raise ValueError("Mixed relation experiment requires vocab.n_relations >= 2")

    torch.manual_seed(cfg.curriculum.seed)
    np.random.seed(cfg.curriculum.seed)
    random.seed(cfg.curriculum.seed)
    rng = np.random.default_rng(cfg.curriculum.seed)
    cfg.curriculum.output_dir.mkdir(parents=True, exist_ok=True)

    shuffled = np.arange(cfg.vocab.n_entities)
    rng.shuffle(shuffled)
    train_n = min(cfg.curriculum.phase2_triples, cfg.vocab.n_entities)
    eval_n = min(cfg.curriculum.eval_limit, cfg.vocab.n_entities - train_n)
    train_subjects = sorted(int(i) for i in shuffled[:train_n])
    never_seen_subjects = sorted(int(i) for i in shuffled[train_n : train_n + eval_n])
    if cfg.curriculum.structured_target_mode == "identity":
        structured_targets = np.arange(cfg.vocab.n_entities, dtype=np.int64)
    elif cfg.curriculum.structured_target_mode == "f_bucket":
        buckets = cfg.curriculum.structured_target_buckets or cfg.vocab.n_f
        if buckets > cfg.vocab.n_entities:
            raise ValueError("structured_target_buckets must be <= n_entities")
        structured_targets = np.asarray(
            [cfg.vocab.subject_attrs[i][0] % buckets for i in range(cfg.vocab.n_entities)],
            dtype=np.int64,
        )
    else:
        raise ValueError(f"Unknown structured_target_mode: {cfg.curriculum.structured_target_mode}")
    random_targets = rng.permutation(cfg.vocab.n_entities).astype(np.int64)
    structured_repeats = (
        cfg.curriculum.structured_relation_repeats
        if cfg.curriculum.structured_relation_repeats is not None
        else max(1, cfg.curriculum.phase2_repeats)
    )
    random_repeats = (
        cfg.curriculum.random_relation_repeats
        if cfg.curriculum.random_relation_repeats is not None
        else max(structured_repeats + 1, cfg.curriculum.phase1_triple_repeats)
    )
    if structured_repeats < 1 or random_repeats < 1:
        raise ValueError("Relation repeat counts must be positive")
    matched_frequency = structured_repeats == random_repeats

    model = build_model(cfg.model, cfg.vocab)
    model.to(device)
    if cfg.curriculum.structured_embedding_init:
        init_entity_embeddings_from_attrs(
            model=model,
            vocab=cfg.vocab,
            scale=cfg.curriculum.init_scale,
            seed=cfg.curriculum.seed,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.curriculum.learning_rate,
        weight_decay=cfg.curriculum.weight_decay,
    )
    attr_sequences = [
        seq
        for entity_idx in range(cfg.vocab.n_entities)
        for seq in (subject_attr_sequence(cfg.vocab, entity_idx), object_attr_sequence(cfg.vocab, entity_idx))
    ] * cfg.curriculum.phase1_attr_repeats
    if cfg.curriculum.reverse_attr_repeats > 0:
        attr_sequences.extend(
            [reverse_object_attr_sequence(cfg.vocab, entity_idx) for entity_idx in range(cfg.vocab.n_entities)]
            * cfg.curriculum.reverse_attr_repeats
        )
    structured_sequences = [
        mapped_triple_sequence(cfg.vocab, subject_idx, int(structured_targets[subject_idx]), relation_idx=0)
        for _ in range(structured_repeats)
        for subject_idx in train_subjects
    ]
    random_sequences = [
        mapped_triple_sequence(cfg.vocab, subject_idx, int(random_targets[subject_idx]), relation_idx=1)
        for _ in range(random_repeats)
        for subject_idx in train_subjects
    ]
    sequences = attr_sequences + structured_sequences + random_sequences
    loader = make_loader(
        sequences,
        batch_size=cfg.curriculum.batch_size,
        seed=cfg.curriculum.seed,
        shuffle=True,
    )
    last_loss = float("nan")
    t0 = time.time()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        lr = get_lr(step, cfg.curriculum.warmup_steps, len(loader), cfg.curriculum.learning_rate)
        last_loss = train_one_batch(
            model,
            optimizer,
            x,
            y,
            lr,
            cfg.curriculum.grad_clip,
            object_token_range=(cfg.vocab.object_offset, cfg.vocab.f_offset),
            object_loss_weight=cfg.curriculum.object_loss_weight,
        )

    structured_metrics = prompt_relation_metrics(
        model,
        cfg.vocab,
        structured_targets,
        train_subjects,
        never_seen_subjects,
        relation_idx=0,
        device=device,
        eval_limit=cfg.curriculum.eval_limit,
    )
    structured_metrics.update(
        embedding_relation_metrics(
            model,
            cfg.vocab,
            structured_targets,
            train_subjects,
            never_seen_subjects,
        )
    )
    random_metrics = prompt_relation_metrics(
        model,
        cfg.vocab,
        random_targets,
        train_subjects,
        never_seen_subjects,
        relation_idx=1,
        device=device,
        eval_limit=cfg.curriculum.eval_limit,
    )
    random_metrics.update(
        embedding_relation_metrics(
            model,
            cfg.vocab,
            random_targets,
            train_subjects,
            never_seen_subjects,
        )
    )
    rel0_train_h = prompt_residual_vectors(model, cfg.vocab, train_subjects, 0, device)
    rel1_train_h = prompt_residual_vectors(model, cfg.vocab, train_subjects, 1, device)
    rel0_eval_h = prompt_residual_vectors(model, cfg.vocab, never_seen_subjects, 0, device)
    rel1_eval_h = prompt_residual_vectors(model, cfg.vocab, never_seen_subjects, 1, device)
    structured_label = "structured" if matched_frequency else "low_frequency_structured"
    random_label = "random" if matched_frequency else "high_frequency_random"
    result = {
        "config": {
            "vocab": asdict(cfg.vocab),
            "model": asdict(cfg.model),
            "curriculum": {
                **asdict(cfg.curriculum),
                "output_dir": str(cfg.curriculum.output_dir),
            },
        },
        "train_subjects": len(train_subjects),
        "never_seen_subjects": len(never_seen_subjects),
        "structured_repeats": structured_repeats,
        "random_repeats": random_repeats,
        "matched_frequency": matched_frequency,
        "train_steps": len(loader),
        "last_train_loss": last_loss,
        "relations": {
            structured_label: structured_metrics,
            random_label: random_metrics,
        },
        "mechanistic": {
            "same_subject_relation_hidden_cosine_train": paired_cosine(rel0_train_h, rel1_train_h),
            "same_subject_relation_hidden_cosine_never_seen": paired_cosine(rel0_eval_h, rel1_eval_h),
            "relation_hidden_delta_norm_train": float(np.linalg.norm(rel0_train_h - rel1_train_h, axis=1).mean()),
            "relation_hidden_delta_norm_never_seen": float(np.linalg.norm(rel0_eval_h - rel1_eval_h, axis=1).mean()),
        },
        "elapsed_sec": time.time() - t0,
    }
    result_path = cfg.curriculum.output_dir / "mixed_relations_metrics.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    torch.save(model.state_dict(), cfg.curriculum.output_dir / "mixed_model.pt")
    result["result_path"] = str(result_path)
    if not quiet:
        print(json.dumps({"relations": result["relations"], "mechanistic": result["mechanistic"]}, indent=2))
    return result


def synthetic_config_from_toml(raw: dict) -> SyntheticExperimentConfig:
    vocab_raw = dict(raw.get("vocab", {}))
    model_raw = dict(raw.get("model", {}))
    curriculum_raw = dict(raw.get("curriculum", {}))
    if "output_dir" in curriculum_raw:
        output_dir = Path(curriculum_raw["output_dir"])
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        curriculum_raw["output_dir"] = output_dir
    seed = int(raw.get("seed", curriculum_raw.get("seed", 7)))
    curriculum_raw.setdefault("seed", seed)
    vocab_raw.setdefault("seed", seed)

    curriculum = CurriculumConfig(**curriculum_raw)
    vocab_defaults = {
        "n_entities": curriculum.n_entities,
        "n_f": curriculum.n_f,
        "n_g": curriculum.n_g,
        "delta_f": curriculum.delta_f,
        "delta_g": curriculum.delta_g,
        "seed": curriculum.seed,
    }
    vocab_defaults.update(vocab_raw)
    model = TransformerConfig(**model_raw)
    return SyntheticExperimentConfig(
        vocab=StructuredVocab(**vocab_defaults),
        model=model,
        curriculum=curriculum,
    )


def load_synthetic_config(path: Path) -> SyntheticExperimentConfig:
    import tomllib

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return synthetic_config_from_toml(raw)

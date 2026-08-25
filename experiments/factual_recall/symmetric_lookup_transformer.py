"""Symmetric lookup experiment with an ordinary GPT-style transformer.

This module uses the same synthetic world as :mod:`symmetric_lookup`, but it
removes the bespoke bilinear and relation-conditioned heads.  Every fact is
presented as an autoregressive token sequence and every answer is predicted by
the transformer's shared vocabulary readout::

    <bos> subject attribute -> 0/1
    <bos> relation subject object -> n/y

The model is the repository's standard decoder-only transformer: learned token
and position embeddings, causal multi-head self-attention, pre-LayerNorm
residual blocks, 4x GELU MLPs, a final LayerNorm, and a vocabulary projection.
All parameters, including entity embeddings, remain trainable in both stages.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import PROJECT_ROOT, TransformerConfig
from .model import CausalTransformer, build_model
from .symmetric_lookup import (
    SymmetricLookupConfig,
    SymmetricWorld,
    _pair_pools,
    _sample_relation_batch,
    build_world,
    ols_relation_metrics,
)


@dataclass
class TransformerLookupConfig:
    seed: int = 19
    n_entities: int = 64
    n_attributes: int = 10
    hidden_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    tie_embeddings: bool = True
    positive_fraction: float = 0.1
    positive_label_loss_weight: float = 9.0
    arbitrary_relation_fraction: float = 0.5
    batch_size: int = 1024
    attribute_batch_size: int = 640
    phase1_steps: int = 5000
    arbitrary_steps: int = 8000
    joint_steps: int = 8000
    joint_refinement_steps: int = 12000
    joint_refinement_learning_rate: float = 3e-4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    attribute_loss_weight: float = 1.0
    embedding_attribute_loss_weight: float = 0.0
    embedding_attribute_target_scale: float = 0.5
    embedding_attribute_target_mse: float = 0.0025
    ols_representation: str = "learned_attribute_space"
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_every: int = 100
    target_accuracy: float = 0.995
    ols_folds: int = 4
    graph_swaps_per_edge: int = 40
    output_dir: Path = (
        PROJECT_ROOT / "factual_recall" / "results" / "symmetric_lookup_transformer"
    )


@dataclass
class TransformerSweepConfig:
    seeds: list[int] = field(default_factory=lambda: [19, 31, 47])
    hidden_dims: list[int] = field(default_factory=lambda: [16, 24, 32])
    n_layers: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 12])
    n_heads: list[int] = field(default_factory=lambda: [4])
    min_structured_ols_r2: float = 0.95
    max_arbitrary_ols_r2: float = 0.50
    require_all_seeds: bool = True


@dataclass(frozen=True)
class LookupTokenVocabulary:
    """Token layout for the two sentence templates."""

    n_entities: int
    n_attributes: int

    @property
    def bos_id(self) -> int:
        return 0

    @property
    def entity_offset(self) -> int:
        return 1

    @property
    def attribute_offset(self) -> int:
        return self.entity_offset + self.n_entities

    @property
    def relation_offset(self) -> int:
        return self.attribute_offset + self.n_attributes

    @property
    def zero_id(self) -> int:
        return self.relation_offset + self.n_attributes + 1

    @property
    def one_id(self) -> int:
        return self.zero_id + 1

    @property
    def no_id(self) -> int:
        return self.one_id + 1

    @property
    def yes_id(self) -> int:
        return self.no_id + 1

    @property
    def vocab_size(self) -> int:
        return self.yes_id + 1

    def entity_id(self, index: int) -> int:
        return self.entity_offset + index

    def attribute_id(self, index: int) -> int:
        return self.attribute_offset + index

    def relation_id(self, index: int) -> int:
        return self.relation_offset + index

    def attribute_value_id(self, value: int) -> int:
        return self.one_id if value else self.zero_id

    def relation_value_id(self, value: int) -> int:
        return self.yes_id if value else self.no_id

    def token_to_str(self, token_id: int) -> str:
        if token_id == self.bos_id:
            return "<bos>"
        if self.entity_offset <= token_id < self.attribute_offset:
            return f"s{token_id - self.entity_offset}"
        if self.attribute_offset <= token_id < self.relation_offset:
            return f"a{token_id - self.attribute_offset}"
        if self.relation_offset <= token_id < self.zero_id:
            return f"r{token_id - self.relation_offset}"
        return {
            self.zero_id: "0",
            self.one_id: "1",
            self.no_id: "n",
            self.yes_id: "y",
        }.get(token_id, f"?{token_id}")


def _world_config(cfg: TransformerLookupConfig) -> SymmetricLookupConfig:
    return SymmetricLookupConfig(
        seed=cfg.seed,
        n_entities=cfg.n_entities,
        n_attributes=cfg.n_attributes,
        graph_swaps_per_edge=cfg.graph_swaps_per_edge,
    )


def make_model(
    cfg: TransformerLookupConfig, vocab: LookupTokenVocabulary
) -> CausalTransformer:
    model_cfg = TransformerConfig(
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        context_len=4,
        tie_embeddings=cfg.tie_embeddings,
    )
    return build_model(model_cfg, vocab)  # type: ignore[arg-type]


def attribute_prompts(
    vocab: LookupTokenVocabulary,
    subjects: np.ndarray,
    attributes: np.ndarray,
    values: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = np.stack(
        (
            np.full(len(subjects), vocab.bos_id),
            subjects + vocab.entity_offset,
            attributes + vocab.attribute_offset,
        ),
        axis=1,
    )
    targets = np.where(values.astype(bool), vocab.one_id, vocab.zero_id)
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )


def relation_prompts(
    vocab: LookupTokenVocabulary,
    relations: torch.Tensor,
    subjects: torch.Tensor,
    objects: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bos = torch.full_like(relations, vocab.bos_id)
    rows = torch.stack(
        (
            bos,
            relations + vocab.relation_offset,
            subjects + vocab.entity_offset,
            objects + vocab.entity_offset,
        ),
        dim=1,
    )
    targets = torch.where(
        values.bool(),
        torch.full_like(relations, vocab.yes_id),
        torch.full_like(relations, vocab.no_id),
    )
    return rows, targets


def _sample_attribute_batch(
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    subjects = rng.integers(0, world.n_entities, size=batch_size)
    attributes = rng.integers(0, world.n_attributes, size=batch_size)
    values = world.attributes[subjects, attributes]
    return attribute_prompts(vocab, subjects, attributes, values, device)


@torch.no_grad()
def attribute_accuracy(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    device: torch.device,
) -> float:
    subjects, attributes = np.indices((world.n_entities, world.n_attributes))
    values = world.attributes.reshape(-1)
    prompts, targets = attribute_prompts(
        vocab, subjects.reshape(-1), attributes.reshape(-1), values, device
    )
    was_training = model.training
    model.eval()
    predictions = model(prompts)[:, -1].argmax(dim=-1)
    if was_training:
        model.train()
    return float((predictions == targets).float().mean().item())


@torch.no_grad()
def relation_accuracies(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    device: torch.device,
    relation_ids: Sequence[int] | None = None,
    reverse: bool = False,
) -> list[float]:
    ids = list(range(world.n_relations) if relation_ids is None else relation_ids)
    subjects, objects = np.indices((world.n_entities, world.n_entities))
    subjects_t = torch.as_tensor(subjects.reshape(-1), dtype=torch.long, device=device)
    objects_t = torch.as_tensor(objects.reshape(-1), dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    output: list[float] = []
    for relation in ids:
        relations_t = torch.full_like(subjects_t, relation)
        values_t = torch.as_tensor(
            world.labels[relation].reshape(-1), dtype=torch.float32, device=device
        )
        left, right = (objects_t, subjects_t) if reverse else (subjects_t, objects_t)
        prompts, targets = relation_prompts(
            vocab, relations_t, left, right, values_t
        )
        predictions = model(prompts)[:, -1].argmax(dim=-1)
        output.append(float((predictions == targets).float().mean().item()))
    if was_training:
        model.train()
    return output


@torch.no_grad()
def relation_symmetry_agreements(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    device: torch.device,
) -> list[float]:
    """Fraction of pairs whose prediction is unchanged when entities swap."""
    subjects, objects = np.indices((world.n_entities, world.n_entities))
    subjects_t = torch.as_tensor(subjects.reshape(-1), dtype=torch.long, device=device)
    objects_t = torch.as_tensor(objects.reshape(-1), dtype=torch.long, device=device)
    dummy_values = torch.zeros_like(subjects_t, dtype=torch.float32)
    was_training = model.training
    model.eval()
    output: list[float] = []
    for relation in range(world.n_relations):
        relations_t = torch.full_like(subjects_t, relation)
        forward_x, _ = relation_prompts(
            vocab, relations_t, subjects_t, objects_t, dummy_values
        )
        reverse_x, _ = relation_prompts(
            vocab, relations_t, objects_t, subjects_t, dummy_values
        )
        forward = model(forward_x)[:, -1].argmax(dim=-1)
        reverse = model(reverse_x)[:, -1].argmax(dim=-1)
        output.append(float((forward == reverse).float().mean().item()))
    if was_training:
        model.train()
    return output


def all_ols_metrics(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    cfg: TransformerLookupConfig,
) -> list[dict[str, float]]:
    entity_ids = torch.arange(
        vocab.entity_offset,
        vocab.entity_offset + world.n_entities,
        device=model.token_emb.weight.device,
    )
    embeddings = model.token_emb(entity_ids).detach().cpu().numpy()
    return [
        ols_relation_metrics(
            embeddings, world.labels[relation], cfg.ols_folds, cfg.seed
        )
        for relation in range(world.n_relations)
    ]


@torch.no_grad()
def learned_attribute_representations(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    device: torch.device,
) -> np.ndarray:
    """Represent each entity by its ten learned attribute log-odds."""
    subjects, attributes = np.indices((world.n_entities, world.n_attributes))
    dummy_values = np.zeros(subjects.size, dtype=np.int64)
    prompts, _ = attribute_prompts(
        vocab,
        subjects.reshape(-1),
        attributes.reshape(-1),
        dummy_values,
        device,
    )
    was_training = model.training
    model.eval()
    logits = model(prompts)[:, -1]
    log_odds = logits[:, vocab.one_id] - logits[:, vocab.zero_id]
    if was_training:
        model.train()
    return log_odds.reshape(world.n_entities, world.n_attributes).cpu().numpy()


def embedding_attribute_loss(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    cfg: TransformerLookupConfig,
) -> torch.Tensor:
    """Decode attributes with dot products inside the standard embedding table."""
    entity_rows = model.token_emb.weight[
        vocab.entity_offset : vocab.entity_offset + world.n_entities
    ]
    attribute_rows = model.token_emb.weight[
        vocab.attribute_offset : vocab.attribute_offset + world.n_attributes
    ]
    predictions = entity_rows @ attribute_rows.T
    targets = torch.as_tensor(
        world.attributes * 2.0 - 1.0,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    return F.mse_loss(predictions, cfg.embedding_attribute_target_scale * targets)


def attribute_space_ols_metrics(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    cfg: TransformerLookupConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    representations = learned_attribute_representations(
        model, world, vocab, device
    )
    return [
        ols_relation_metrics(
            representations,
            world.labels[relation],
            cfg.ols_folds,
            cfg.seed,
        )
        for relation in range(world.n_relations)
    ]


@torch.no_grad()
def entity_representations_by_layer(
    model: CausalTransformer,
    vocab: LookupTokenVocabulary,
    device: torch.device,
) -> list[np.ndarray]:
    """Return each entity's residual-stream representation after every block."""
    entities = torch.arange(vocab.n_entities, device=device) + vocab.entity_offset
    prompts = torch.stack(
        (torch.full_like(entities, vocab.bos_id), entities), dim=1
    )
    was_training = model.training
    model.eval()
    with model.capture_residuals() as residuals:
        model(prompts)
    outputs = [residual[:, -1].cpu().numpy() for residual in residuals]
    outputs.append(model.ln_f(residuals[-1])[:, -1].cpu().numpy())
    if was_training:
        model.train()
    return outputs


def layerwise_ols_summary(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    cfg: TransformerLookupConfig,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    names = ["embedding_plus_position"] + [
        f"block_{index + 1}" for index in range(model.n_layers)
    ] + ["final_layer_norm"]
    output: list[dict[str, float | int | str]] = []
    for index, representations in enumerate(
        entity_representations_by_layer(model, vocab, device)
    ):
        metrics = [
            ols_relation_metrics(
                representations,
                world.labels[relation],
                cfg.ols_folds,
                cfg.seed,
            )
            for relation in range(world.n_relations)
        ]
        output.append(
            {
                "index": index,
                "name": names[index],
                "min_structured_full_r2": min(
                    row["full_r2"] for row in metrics[: world.n_attributes]
                ),
                "mean_structured_full_r2": float(
                    np.mean(
                        [row["full_r2"] for row in metrics[: world.n_attributes]]
                    )
                ),
                "arbitrary_full_r2": metrics[world.arbitrary_relation]["full_r2"],
            }
        )
    return output


@torch.no_grad()
def relation_conditioned_entity_representations(
    model: CausalTransformer,
    vocab: LookupTokenVocabulary,
    relation: int,
    device: torch.device,
) -> np.ndarray:
    """Represent entities after the causal prefix ``<bos> relation entity``."""
    entities = torch.arange(vocab.n_entities, device=device) + vocab.entity_offset
    prompts = torch.stack(
        (
            torch.full_like(entities, vocab.bos_id),
            torch.full_like(entities, vocab.relation_id(relation)),
            entities,
        ),
        dim=1,
    )
    was_training = model.training
    model.eval()
    with model.capture_residuals() as residuals:
        model(prompts)
    output = model.ln_f(residuals[-1])[:, -1].cpu().numpy()
    if was_training:
        model.train()
    return output


@torch.no_grad()
def relation_conditioned_representations_by_layer(
    model: CausalTransformer,
    vocab: LookupTokenVocabulary,
    relation: int,
    device: torch.device,
) -> list[np.ndarray]:
    entities = torch.arange(vocab.n_entities, device=device) + vocab.entity_offset
    prompts = torch.stack(
        (
            torch.full_like(entities, vocab.bos_id),
            torch.full_like(entities, vocab.relation_id(relation)),
            entities,
        ),
        dim=1,
    )
    was_training = model.training
    model.eval()
    with model.capture_residuals() as residuals:
        model(prompts)
    output = [residual[:, -1].cpu().numpy() for residual in residuals]
    output.append(model.ln_f(residuals[-1])[:, -1].cpu().numpy())
    if was_training:
        model.train()
    return output


def relation_conditioned_ols_metrics(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    cfg: TransformerLookupConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """OLS on the relation-conditioned entity state before the second entity."""
    return [
        ols_relation_metrics(
            relation_conditioned_entity_representations(
                model, vocab, relation, device
            ),
            world.labels[relation],
            cfg.ols_folds,
            cfg.seed,
        )
        for relation in range(world.n_relations)
    ]


def relation_conditioned_layerwise_ols_summary(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    cfg: TransformerLookupConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    names = ["embedding_plus_position"] + [
        f"block_{index + 1}" for index in range(model.n_layers)
    ] + ["final_layer_norm"]
    representations = [
        relation_conditioned_representations_by_layer(
            model, vocab, relation, device
        )
        for relation in range(world.n_relations)
    ]
    output: list[dict[str, object]] = []
    for layer_index, name in enumerate(names):
        metrics = [
            ols_relation_metrics(
                representations[relation][layer_index],
                world.labels[relation],
                cfg.ols_folds,
                cfg.seed,
            )
            for relation in range(world.n_relations)
        ]
        structured_r2 = [
            row["full_r2"] for row in metrics[: world.n_attributes]
        ]
        output.append(
            {
                "index": layer_index,
                "name": name,
                "structured_full_r2": structured_r2,
                "min_structured_full_r2": min(structured_r2),
                "mean_structured_full_r2": float(np.mean(structured_r2)),
                "arbitrary_full_r2": metrics[world.arbitrary_relation]["full_r2"],
            }
        )
    return output


def _train_stage(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    cfg: TransformerLookupConfig,
    relation_ids: Sequence[int],
    sampling_relation_ids: Sequence[int] | None,
    adaptive_sampling: bool,
    max_steps: int,
    seed_offset: int,
    device: torch.device,
    quiet: bool,
) -> dict[str, object]:
    """Train all transformer parameters with relation and attribute rehearsal."""
    rng = np.random.default_rng(cfg.seed + seed_offset)
    pools = _pair_pools(world.labels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    relation_class_weights = torch.ones(vocab.vocab_size, device=device)
    relation_class_weights[vocab.yes_id] = cfg.positive_label_loss_weight
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = (-1.0, -1.0, -1.0, -float("inf"))
    current_sampling_ids = list(
        relation_ids if sampling_relation_ids is None else sampling_relation_ids
    )
    model.train()
    for step in range(1, max_steps + 1):
        relations, subjects, objects, values = _sample_relation_batch(
            world,
            current_sampling_ids,
            cfg.batch_size,
            cfg.positive_fraction,
            rng,
            pools,
            device,
        )
        relation_x, relation_y = relation_prompts(
            vocab, relations, subjects, objects, values
        )
        attribute_x, attribute_y = _sample_attribute_batch(
            world, vocab, cfg.attribute_batch_size, rng, device
        )
        relation_loss = F.cross_entropy(
            model(relation_x)[:, -1], relation_y, weight=relation_class_weights
        )
        attribute_loss = F.cross_entropy(model(attribute_x)[:, -1], attribute_y)
        embedding_loss = embedding_attribute_loss(model, world, vocab, cfg)
        loss = (
            relation_loss
            + cfg.attribute_loss_weight * attribute_loss
            + cfg.embedding_attribute_loss_weight * embedding_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        warmup = min(1.0, step / max(1, cfg.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * warmup
        optimizer.step()

        if step % cfg.eval_every == 0 or step == max_steps:
            accuracies = relation_accuracies(
                model, world, vocab, device, relation_ids
            )
            attributes = attribute_accuracy(model, world, vocab, device)
            row = {
                "step": step,
                "loss": float(loss.item()),
                "attribute_accuracy": attributes,
                "embedding_attribute_mse": float(embedding_loss.item()),
                "min_relation_accuracy": min(accuracies),
                "mean_relation_accuracy": float(np.mean(accuracies)),
            }
            history.append(row)
            embedding_ready = (
                cfg.embedding_attribute_loss_weight == 0.0
                or embedding_loss.item() <= cfg.embedding_attribute_target_mse
            )
            score = (
                float(embedding_ready),
                min(accuracies),
                attributes,
                -float(embedding_loss.item()),
            )
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
            if adaptive_sampling:
                errors = np.maximum(1.0 - np.asarray(accuracies), 1e-3)
                relative_errors = errors / errors.max()
                repeats = 1 + np.rint(9.0 * relative_errors).astype(int)
                current_sampling_ids = [
                    relation
                    for relation, repeat in zip(relation_ids, repeats, strict=True)
                    for _ in range(int(repeat))
                ]
            if not quiet:
                print(
                    f"step={step:5d} attr={attributes:.4f} "
                    f"relation_min={min(accuracies):.4f} "
                    f"relation_mean={np.mean(accuracies):.4f}",
                    flush=True,
                )
            if (
                attributes >= cfg.target_accuracy
                and min(accuracies) >= cfg.target_accuracy
                and embedding_ready
            ):
                break
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "steps": step,
        "history": history,
        "best_min_relation_accuracy": best_score[1],
        "best_attribute_accuracy": best_score[2],
        "best_embedding_attribute_mse": -best_score[3],
    }


def claim_checks(result: dict[str, object]) -> dict[str, bool]:
    cfg = result["config"]
    phase1 = result["phase1_metrics"]
    final = result["final_metrics"]
    ols_key = (
        "input_embedding_ols"
        if cfg["ols_representation"] == "input_embedding"
        else "learned_attribute_space_ols"
    )
    checks = {
        "phase1_attributes_learned": phase1["attribute_accuracy"] >= cfg["target_accuracy"],
        "phase1_structured_relations_learned": min(phase1["relation_accuracy"])
        >= cfg["target_accuracy"],
        "all_parameters_remained_trainable": result["frozen_parameter_count"] == 0,
        "final_attributes_retained": final["attribute_accuracy"] >= cfg["target_accuracy"],
        "final_all_relations_learned": min(final["relation_accuracy"])
        >= cfg["target_accuracy"],
        "ordered_and_reversed_pairs_learned": min(final["reverse_relation_accuracy"])
        >= cfg["target_accuracy"],
        "symmetric_predictions_learned": min(final["symmetry_agreement"])
        >= cfg["target_accuracy"],
        "structured_relations_ols_linear": final[ols_key][
            "min_structured_full_r2"
        ]
        >= 0.95,
        "arbitrary_relation_ols_nonlinear": final[ols_key][
            "arbitrary"
        ]["full_r2"]
        < 0.50,
    }
    checks["all"] = all(checks.values())
    return checks


def run_single(
    cfg: TransformerLookupConfig,
    device: torch.device,
    quiet: bool = False,
) -> dict[str, object]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    world = build_world(_world_config(cfg))
    vocab = LookupTokenVocabulary(cfg.n_entities, cfg.n_attributes)
    model = make_model(cfg, vocab).to(device)
    started = time.time()

    phase1_training = _train_stage(
        model,
        world,
        vocab,
        cfg,
        list(range(world.n_attributes)),
        None,
        True,
        cfg.phase1_steps,
        101,
        device,
        quiet,
    )
    phase1_metrics = {
        "attribute_accuracy": attribute_accuracy(model, world, vocab, device),
        "relation_accuracy": relation_accuracies(
            model, world, vocab, device, list(range(world.n_attributes))
        ),
    }
    phase1_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    arbitrary_training = _train_stage(
        model,
        world,
        vocab,
        cfg,
        [world.arbitrary_relation],
        None,
        False,
        cfg.arbitrary_steps,
        202,
        device,
        quiet,
    )
    arbitrary_stage_metrics = {
        "attribute_accuracy": attribute_accuracy(model, world, vocab, device),
        "arbitrary_accuracy": relation_accuracies(
            model, world, vocab, device, [world.arbitrary_relation]
        )[0],
    }
    # The arbitrary-only stage is a capacity test.  The final joint stage starts
    # from the shared structured checkpoint, as in the original protocol.
    model.load_state_dict(phase1_state)
    embedding_before_joint = model.token_emb.weight.detach().clone()

    if not 0.0 < cfg.arbitrary_relation_fraction < 1.0:
        raise ValueError("arbitrary_relation_fraction must be strictly between 0 and 1")
    arbitrary_repeats = max(
        1,
        round(
            cfg.arbitrary_relation_fraction
            * world.n_attributes
            / (1.0 - cfg.arbitrary_relation_fraction)
        ),
    )
    joint_sampling_ids = list(range(world.n_attributes)) + [
        world.arbitrary_relation
    ] * arbitrary_repeats
    joint_training = _train_stage(
        model,
        world,
        vocab,
        cfg,
        list(range(world.n_relations)),
        joint_sampling_ids,
        False,
        cfg.joint_steps,
        303,
        device,
        quiet,
    )
    joint_refinement: dict[str, object] | None = None
    if cfg.joint_refinement_steps > 0:
        initial_learning_rate = cfg.learning_rate
        cfg.learning_rate = cfg.joint_refinement_learning_rate
        joint_refinement = _train_stage(
            model,
            world,
            vocab,
            cfg,
            list(range(world.n_relations)),
            joint_sampling_ids,
            False,
            cfg.joint_refinement_steps,
            50_303,
            device,
            quiet,
        )
        cfg.learning_rate = initial_learning_rate
    final_relations = relation_accuracies(model, world, vocab, device)
    reverse_relations = relation_accuracies(
        model, world, vocab, device, reverse=True
    )
    symmetry_agreements = relation_symmetry_agreements(
        model, world, vocab, device
    )
    ols = all_ols_metrics(model, world, vocab, cfg)
    layerwise_ols = layerwise_ols_summary(model, world, vocab, cfg, device)
    contextual_ols = relation_conditioned_ols_metrics(
        model, world, vocab, cfg, device
    )
    attribute_ols = attribute_space_ols_metrics(
        model, world, vocab, cfg, device
    )
    contextual_layerwise_ols = relation_conditioned_layerwise_ols_summary(
        model, world, vocab, cfg, device
    )
    result: dict[str, object] = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "architecture": {
            "type": "decoder-only causal transformer",
            "token_embedding": "learned",
            "position_embedding": "learned absolute",
            "normalization": "pre-LayerNorm with final LayerNorm",
            "attention": "causal multi-head self-attention",
            "mlp": "4x expansion with GELU",
            "residual_connections": True,
            "readout": "shared vocabulary linear projection",
            "tie_embeddings": cfg.tie_embeddings,
            "special_relation_heads": False,
        },
        "vocabulary_size": vocab.vocab_size,
        "num_parameters": model.n_parameters(),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "world": {
            "attribute_sentences": len(world.attribute_sentences()),
            "relation_sentences": len(world.relation_sentences()),
            "positive_fraction_in_training_batches": cfg.positive_fraction,
            "positive_label_loss_weight": cfg.positive_label_loss_weight,
            "arbitrary_relation_fraction_in_joint_batches": float(
                arbitrary_repeats / len(joint_sampling_ids)
            ),
            "symmetric": bool(np.all(world.labels == world.labels.transpose(0, 2, 1))),
        },
        "phase1_training": phase1_training,
        "phase1_metrics": phase1_metrics,
        "arbitrary_training": arbitrary_training,
        "arbitrary_stage_metrics": arbitrary_stage_metrics,
        "joint_training": joint_training,
        "joint_refinement": joint_refinement,
        "final_metrics": {
            "attribute_accuracy": attribute_accuracy(model, world, vocab, device),
            "relation_accuracy": final_relations,
            "reverse_relation_accuracy": reverse_relations,
            "symmetry_agreement": symmetry_agreements,
            "min_structured_accuracy": min(final_relations[: world.n_attributes]),
            "arbitrary_accuracy": final_relations[world.arbitrary_relation],
            "embedding_max_drift_during_joint": float(
                (model.token_emb.weight.detach() - embedding_before_joint).abs().max().item()
            ),
            "input_embedding_ols": {
                "structured": ols[: world.n_attributes],
                "arbitrary": ols[world.arbitrary_relation],
                "min_structured_full_r2": min(
                    row["full_r2"] for row in ols[: world.n_attributes]
                ),
                "mean_structured_full_r2": float(
                    np.mean([row["full_r2"] for row in ols[: world.n_attributes]])
                ),
            },
            "learned_attribute_space_ols": {
                "structured": attribute_ols[: world.n_attributes],
                "arbitrary": attribute_ols[world.arbitrary_relation],
                "min_structured_full_r2": min(
                    row["full_r2"] for row in attribute_ols[: world.n_attributes]
                ),
                "mean_structured_full_r2": float(
                    np.mean(
                        [row["full_r2"] for row in attribute_ols[: world.n_attributes]]
                    )
                ),
            },
            "layerwise_ols": layerwise_ols,
            # This is the primary affine-map test for a transformer.  The
            # representation is read immediately after ``<bos> relation entity``,
            # before the second entity is available to the model.
            "relation_conditioned_ols": {
                "structured": contextual_ols[: world.n_attributes],
                "arbitrary": contextual_ols[world.arbitrary_relation],
                "min_structured_full_r2": min(
                    row["full_r2"]
                    for row in contextual_ols[: world.n_attributes]
                ),
                "mean_structured_full_r2": float(
                    np.mean(
                        [
                            row["full_r2"]
                            for row in contextual_ols[: world.n_attributes]
                        ]
                    )
                ),
            },
            "relation_conditioned_layerwise_ols": contextual_layerwise_ols,
        },
        "elapsed_sec": time.time() - started,
    }
    result["claim_checks"] = claim_checks(result)
    result["_state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    return result


def _json_ready(result: dict[str, object]) -> dict[str, object]:
    output = copy.copy(result)
    output.pop("_state_dict", None)
    return output


def refine_joint_checkpoint(
    cfg: TransformerLookupConfig,
    checkpoint: Path,
    device: torch.device,
    quiet: bool = False,
) -> dict[str, object]:
    """Continue a joint checkpoint at a lower learning rate."""
    torch.manual_seed(cfg.seed + 50_000)
    np.random.seed(cfg.seed + 50_000)
    random.seed(cfg.seed + 50_000)
    world = build_world(_world_config(cfg))
    vocab = LookupTokenVocabulary(cfg.n_entities, cfg.n_attributes)
    model = make_model(cfg, vocab).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    arbitrary_repeats = max(
        1,
        round(
            cfg.arbitrary_relation_fraction
            * world.n_attributes
            / (1.0 - cfg.arbitrary_relation_fraction)
        ),
    )
    joint_sampling_ids = list(range(world.n_attributes)) + [
        world.arbitrary_relation
    ] * arbitrary_repeats
    started = time.time()
    training = _train_stage(
        model,
        world,
        vocab,
        cfg,
        list(range(world.n_relations)),
        joint_sampling_ids,
        False,
        cfg.joint_steps,
        50_303,
        device,
        quiet,
    )
    relations = relation_accuracies(model, world, vocab, device)
    reverse_relations = relation_accuracies(model, world, vocab, device, reverse=True)
    symmetry_agreements = relation_symmetry_agreements(
        model, world, vocab, device
    )
    attribute_ols = attribute_space_ols_metrics(model, world, vocab, cfg, device)
    input_embedding_ols = all_ols_metrics(model, world, vocab, cfg)
    contextual_ols = relation_conditioned_ols_metrics(
        model, world, vocab, cfg, device
    )
    result: dict[str, object] = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "source_checkpoint": str(checkpoint),
        "num_parameters": model.n_parameters(),
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "training": training,
        "final_metrics": {
            "attribute_accuracy": attribute_accuracy(model, world, vocab, device),
            "relation_accuracy": relations,
            "reverse_relation_accuracy": reverse_relations,
            "symmetry_agreement": symmetry_agreements,
            "min_structured_accuracy": min(relations[: world.n_attributes]),
            "arbitrary_accuracy": relations[world.arbitrary_relation],
            "input_embedding_ols": {
                "structured": input_embedding_ols[: world.n_attributes],
                "arbitrary": input_embedding_ols[world.arbitrary_relation],
                "min_structured_full_r2": min(
                    row["full_r2"]
                    for row in input_embedding_ols[: world.n_attributes]
                ),
                "mean_structured_full_r2": float(
                    np.mean(
                        [
                            row["full_r2"]
                            for row in input_embedding_ols[: world.n_attributes]
                        ]
                    )
                ),
            },
            "learned_attribute_space_ols": {
                "structured": attribute_ols[: world.n_attributes],
                "arbitrary": attribute_ols[world.arbitrary_relation],
                "min_structured_full_r2": min(
                    row["full_r2"] for row in attribute_ols[: world.n_attributes]
                ),
                "mean_structured_full_r2": float(
                    np.mean(
                        [row["full_r2"] for row in attribute_ols[: world.n_attributes]]
                    )
                ),
            },
            "relation_conditioned_ols": {
                "structured": contextual_ols[: world.n_attributes],
                "arbitrary": contextual_ols[world.arbitrary_relation],
                "min_structured_full_r2": min(
                    row["full_r2"] for row in contextual_ols[: world.n_attributes]
                ),
            },
        },
        "elapsed_sec": time.time() - started,
    }
    final = result["final_metrics"]
    ols_key = (
        "input_embedding_ols"
        if cfg.ols_representation == "input_embedding"
        else "learned_attribute_space_ols"
    )
    result["checks"] = {
        "attributes_retained": final["attribute_accuracy"] >= cfg.target_accuracy,
        "all_relations_learned": min(final["relation_accuracy"])
        >= cfg.target_accuracy,
        "ordered_and_reversed_pairs_learned": min(final["reverse_relation_accuracy"])
        >= cfg.target_accuracy,
        "symmetric_predictions_learned": min(final["symmetry_agreement"])
        >= cfg.target_accuracy,
        "structured_relations_ols_linear": final[ols_key][
            "min_structured_full_r2"
        ]
        >= 0.95,
        "arbitrary_relation_ols_nonlinear": final[ols_key][
            "arbitrary"
        ]["full_r2"]
        < 0.50,
        "all_parameters_trainable": result["frozen_parameter_count"] == 0,
    }
    result["checks"]["all"] = all(result["checks"].values())
    result["_state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    return result


def _candidate_architectures(
    sweep: TransformerSweepConfig,
) -> list[tuple[int, int, int]]:
    candidates = [
        (hidden_dim, n_layers, n_heads)
        for hidden_dim in sweep.hidden_dims
        for n_layers in sweep.n_layers
        for n_heads in sweep.n_heads
        if hidden_dim % n_heads == 0
    ]
    return sorted(candidates, key=lambda row: (row[0] * row[0] * row[1], row))


def run_sweep(
    base_cfg: TransformerLookupConfig,
    sweep: TransformerSweepConfig,
    device: torch.device,
    quiet: bool = False,
) -> dict[str, object]:
    base_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for hidden_dim, n_layers, n_heads in _candidate_architectures(sweep):
        candidate_runs: list[dict[str, object]] = []
        if not quiet:
            print(
                f"\n=== transformer d={hidden_dim} layers={n_layers} heads={n_heads} ===",
                flush=True,
            )
        for seed in sweep.seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg.seed = seed
            cfg.hidden_dim = hidden_dim
            cfg.n_layers = n_layers
            cfg.n_heads = n_heads
            result = run_single(cfg, device, quiet=quiet)
            state = result.pop("_state_dict")
            candidate_runs.append(result)
            runs.append(result)
            if not quiet:
                metrics = result["final_metrics"]
                ols_key = (
                    "input_embedding_ols"
                    if cfg.ols_representation == "input_embedding"
                    else "learned_attribute_space_ols"
                )
                print(
                    f"seed={seed} pass={result['claim_checks']['all']} "
                    f"min={min(metrics['relation_accuracy']):.4f} "
                    f"structured_ols={metrics[ols_key]['min_structured_full_r2']:.3f} "
                    f"random_ols={metrics[ols_key]['arbitrary']['full_r2']:.3f}",
                    flush=True,
                )
            if result["claim_checks"]["all"]:
                torch.save(
                    state,
                    base_cfg.output_dir
                    / f"transformer_d{hidden_dim}_l{n_layers}_h{n_heads}_seed{seed}.pt",
                )

        ols_key = (
            "input_embedding_ols"
            if cfg.ols_representation == "input_embedding"
            else "learned_attribute_space_ols"
        )
        pass_count = sum(
            row["claim_checks"]["all"]
            and row["final_metrics"][ols_key]["min_structured_full_r2"]
            >= sweep.min_structured_ols_r2
            and row["final_metrics"][ols_key]["arbitrary"]["full_r2"]
            <= sweep.max_arbitrary_ols_r2
            for row in candidate_runs
        )
        passes = (
            pass_count == len(candidate_runs)
            if sweep.require_all_seeds
            else pass_count > 0
        )
        if passes:
            candidate = {
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "num_parameters": candidate_runs[0]["num_parameters"],
                "seeds_passed": pass_count,
                "seeds_total": len(candidate_runs),
            }
            if best is None or candidate["num_parameters"] < best["num_parameters"]:
                best = candidate
            if sweep.require_all_seeds:
                break

    output = {
        "base_config": {**asdict(base_cfg), "output_dir": str(base_cfg.output_dir)},
        "sweep": asdict(sweep),
        "runs": runs,
        "best": best,
    }
    path = base_cfg.output_dir / "symmetric_lookup_transformer_sweep.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["result_path"] = str(path)
    return output


def load_config(
    path: Path,
) -> tuple[TransformerLookupConfig, TransformerSweepConfig]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    experiment = dict(raw.get("experiment", {}))
    if "output_dir" in experiment:
        output_dir = Path(experiment["output_dir"])
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        experiment["output_dir"] = output_dir
    return (
        TransformerLookupConfig(**experiment),
        TransformerSweepConfig(**raw.get("sweep", {})),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/runs/symmetric_lookup_transformer.toml"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-joint", type=Path)
    args = parser.parse_args()

    cfg, sweep = load_config(Path(args.config))
    explicit_seed = args.seed is not None
    if explicit_seed:
        cfg.seed = args.seed
    if args.output_dir:
        output_dir = Path(args.output_dir)
        cfg.output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.resume_joint is not None:
        result = refine_joint_checkpoint(
            cfg, args.resume_joint, device, quiet=args.quiet
        )
        state = result.pop("_state_dict")
        result_path = cfg.output_dir / f"symmetric_lookup_transformer_refined_seed{cfg.seed}.json"
        checkpoint_path = cfg.output_dir / f"symmetric_lookup_transformer_refined_seed{cfg.seed}.pt"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        torch.save(state, checkpoint_path)
        print(
            json.dumps(
                {
                    "result_path": str(result_path),
                    "checkpoint_path": str(checkpoint_path),
                    "checks": result["checks"],
                },
                indent=2,
            )
        )
    elif args.single:
        result = run_single(cfg, device, quiet=args.quiet)
        state = result.pop("_state_dict")
        suffix = f"_seed{cfg.seed}" if explicit_seed else "_single"
        result_path = cfg.output_dir / f"symmetric_lookup_transformer{suffix}.json"
        checkpoint_path = cfg.output_dir / f"symmetric_lookup_transformer{suffix}.pt"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        torch.save(state, checkpoint_path)
        print(
            json.dumps(
                {
                    "result_path": str(result_path),
                    "checkpoint_path": str(checkpoint_path),
                    "checks": result["claim_checks"],
                },
                indent=2,
            )
        )
    else:
        result = run_sweep(cfg, sweep, device, quiet=args.quiet)
        print(json.dumps({"result_path": result["result_path"], "best": result["best"]}, indent=2))


if __name__ == "__main__":
    main()

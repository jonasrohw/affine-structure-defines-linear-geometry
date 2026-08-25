"""Linear-versus-arbitrary symmetric relation experiment.

The experiment realizes the sentence schemas requested for the revised Goal 2::

    subject attribute 0/1
    relation subject object y/n

There are ten balanced binary attributes and ten structured relations.  Structured
relation ``r_i`` is true exactly when its two entities have the same value for
attribute ``i``.  The eleventh relation is a symmetric, degree-matched random
lookup table with no attribute rule.

Because an equality relation has multiple positive objects, its OLS target for a
subject is defined as the centroid of all positive object embeddings.  OLS is fit
from subject embeddings to these positive-object centroids.  This is the
many-target analogue of the subject-to-object OLS metric used by ``linear_shift``.

The no-attention model has three components:

* a learned entity embedding and linear attribute decoder;
* a symmetric bilinear (affine-relation) score for every relation; and
* an optional relation-conditioned MLP residual over symmetric pair features.

Training is staged.  First the attributes and ten structured relations are
learned.  The arbitrary relation is then learned with progressively wider/deeper
nonlinear residuals.  Finally, a fresh set of all eleven relation heads is trained
jointly.  Entity embeddings can either be frozen after phase 1 or left trainable
with attribute rehearsal; the final reported ablation uses the latter policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .config import PROJECT_ROOT


@dataclass
class SymmetricLookupConfig:
    seed: int = 19
    n_entities: int = 64
    n_attributes: int = 10
    embedding_dim: int = 10
    nonlinear_width: int = 256
    nonlinear_depth: int = 2
    relation_embedding_dim: int = 8
    positive_fraction: float = 0.1
    batch_size: int = 1024
    phase1_steps: int = 2500
    arbitrary_steps: int = 5000
    joint_steps: int = 6000
    learning_rate: float = 3e-3
    attr_loss_weight: float = 1.0
    freeze_embeddings: bool = True
    relation_attr_rehearsal_weight: float = 1.0
    run_joint_on_stage_failure: bool = False
    nonlinear_base_penalty: float = 0.1
    eval_every: int = 100
    target_accuracy: float = 0.995
    ols_folds: int = 4
    graph_swaps_per_edge: int = 40
    output_dir: Path = (
        PROJECT_ROOT / "factual_recall" / "results" / "symmetric_lookup"
    )


@dataclass
class SweepConfig:
    seeds: list[int] = field(default_factory=lambda: [19, 31, 47])
    embedding_dims: list[int] = field(default_factory=lambda: [6, 8, 10, 12])
    nonlinear_widths: list[int] = field(
        default_factory=lambda: [0, 32, 40, 48, 52, 56, 64, 128, 256, 512]
    )
    nonlinear_depths: list[int] = field(default_factory=lambda: [1, 2])
    min_structured_ols_r2: float = 0.95
    max_arbitrary_ols_r2: float = 0.50
    max_arbitrary_affine_accuracy: float = 0.90
    require_all_seeds: bool = True


@dataclass
class SymmetricWorld:
    attributes: np.ndarray
    labels: np.ndarray
    arbitrary_relation: int

    @property
    def n_entities(self) -> int:
        return int(self.attributes.shape[0])

    @property
    def n_attributes(self) -> int:
        return int(self.attributes.shape[1])

    @property
    def n_relations(self) -> int:
        return int(self.labels.shape[0])

    def attribute_sentences(self) -> list[tuple[int, int, int]]:
        """Return all ``(subject, attribute, value)`` sentences."""
        return [
            (subject, attribute, int(self.attributes[subject, attribute]))
            for subject in range(self.n_entities)
            for attribute in range(self.n_attributes)
        ]

    def relation_sentences(self) -> list[tuple[int, int, int, int]]:
        """Return all ``(relation, subject, object, y/n)`` sentences."""
        return [
            (relation, subject, obj, int(self.labels[relation, subject, obj]))
            for relation in range(self.n_relations)
            for subject in range(self.n_entities)
            for obj in range(self.n_entities)
        ]


def _balanced_binary_attributes(
    n_entities: int, n_attributes: int, rng: np.random.Generator
) -> np.ndarray:
    if n_entities % 2:
        raise ValueError("n_entities must be even so every attribute is balanced")
    if n_entities > 2**n_attributes:
        raise ValueError("Cannot assign unique binary rows with these dimensions")
    for _ in range(10_000):
        columns = []
        for _attribute in range(n_attributes):
            column = np.array(
                [0] * (n_entities // 2) + [1] * (n_entities // 2), dtype=np.int64
            )
            rng.shuffle(column)
            columns.append(column)
        attributes = np.stack(columns, axis=1)
        centered_rank = np.linalg.matrix_rank(attributes.astype(float) - 0.5)
        if len(np.unique(attributes, axis=0)) == n_entities and centered_rank == n_attributes:
            return attributes
    raise RuntimeError("Could not sample unique, balanced, full-rank attributes")


def _circulant_regular_edges(n_vertices: int, degree: int) -> set[tuple[int, int]]:
    if not (0 <= degree < n_vertices):
        raise ValueError("degree must be in [0, n_vertices)")
    if n_vertices * degree % 2:
        raise ValueError("n_vertices * degree must be even")
    edges: set[tuple[int, int]] = set()
    half = degree // 2
    for vertex in range(n_vertices):
        for offset in range(1, half + 1):
            other = (vertex + offset) % n_vertices
            edges.add(tuple(sorted((vertex, other))))
    if degree % 2:
        if n_vertices % 2:
            raise ValueError("Odd regular degree requires an even vertex count")
        for vertex in range(n_vertices // 2):
            edges.add((vertex, vertex + n_vertices // 2))
    expected = n_vertices * degree // 2
    if len(edges) != expected:
        raise AssertionError((len(edges), expected))
    return edges


def _random_regular_adjacency(
    n_vertices: int,
    degree: int,
    rng: np.random.Generator,
    swaps_per_edge: int,
) -> np.ndarray:
    """Randomize a simple regular graph with degree-preserving edge swaps."""
    edges = _circulant_regular_edges(n_vertices, degree)
    attempts = max(1, swaps_per_edge * len(edges))
    for _ in range(attempts):
        edge_list = tuple(edges)
        first_index, second_index = rng.integers(0, len(edge_list), size=2)
        if first_index == second_index:
            continue
        a, b = edge_list[int(first_index)]
        c, d = edge_list[int(second_index)]
        if len({a, b, c, d}) != 4:
            continue
        if rng.random() < 0.5:
            c, d = d, c
        new_first = tuple(sorted((a, d)))
        new_second = tuple(sorted((c, b)))
        if new_first in edges or new_second in edges or new_first == new_second:
            continue
        edges.remove(tuple(sorted((a, b))))
        edges.remove(tuple(sorted((c, d))))
        edges.add(new_first)
        edges.add(new_second)

    adjacency = np.zeros((n_vertices, n_vertices), dtype=bool)
    for left, right in edges:
        adjacency[left, right] = True
        adjacency[right, left] = True
    if not np.all(adjacency.sum(axis=1) == degree):
        raise AssertionError("Degree-preserving swaps changed the degree sequence")
    return adjacency


def build_world(cfg: SymmetricLookupConfig) -> SymmetricWorld:
    rng = np.random.default_rng(cfg.seed)
    attributes = _balanced_binary_attributes(cfg.n_entities, cfg.n_attributes, rng)
    structured = np.stack(
        [
            attributes[:, attribute, None] == attributes[None, :, attribute]
            for attribute in range(cfg.n_attributes)
        ],
        axis=0,
    )

    # A balanced equality relation has n/2 positives in every row, including
    # the diagonal.  Match this exactly with a random (n/2 - 1)-regular graph
    # plus positive self-pairs.
    positives_per_row = cfg.n_entities // 2
    arbitrary = _random_regular_adjacency(
        cfg.n_entities,
        positives_per_row - 1,
        rng,
        cfg.graph_swaps_per_edge,
    )
    np.fill_diagonal(arbitrary, True)
    if any(np.array_equal(arbitrary, relation) for relation in structured):
        raise RuntimeError("Random lookup accidentally equals a structured relation")
    labels = np.concatenate([structured, arbitrary[None, :, :]], axis=0)
    return SymmetricWorld(
        attributes=attributes,
        labels=labels,
        arbitrary_relation=cfg.n_attributes,
    )


class SymmetricBilinearHead(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        self.linear = nn.Parameter(torch.zeros(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(()))
        self.nonlinear_gate = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        weight = 0.5 * (self.weight + self.weight.T)
        bilinear = torch.einsum("bd,de,be->b", left, weight, right)
        linear = (left + right) @ self.linear
        return bilinear + linear + self.bias


class SymmetricLookupModel(nn.Module):
    """No-attention pair model with an affine path and nonlinear residual."""

    def __init__(self, cfg: SymmetricLookupConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.entity_embeddings = nn.Embedding(cfg.n_entities, cfg.embedding_dim)
        self.attribute_decoder = nn.Linear(cfg.embedding_dim, cfg.n_attributes)
        self.relation_heads = nn.ModuleList(
            [SymmetricBilinearHead(cfg.embedding_dim) for _ in range(cfg.n_attributes + 1)]
        )
        self.relation_embeddings = nn.Embedding(
            cfg.n_attributes + 1, cfg.relation_embedding_dim
        )
        pair_dim = 3 * cfg.embedding_dim + cfg.relation_embedding_dim
        if cfg.nonlinear_width > 0:
            layers: list[nn.Module] = []
            input_dim = pair_dim
            for _ in range(cfg.nonlinear_depth):
                layers.extend(
                    [nn.Linear(input_dim, cfg.nonlinear_width), nn.GELU()]
                )
                input_dim = cfg.nonlinear_width
            layers.append(nn.Linear(input_dim, 1))
            self.nonlinear = nn.Sequential(*layers)
        else:
            self.nonlinear = None
        nn.init.normal_(self.entity_embeddings.weight, std=0.2)
        nn.init.normal_(self.relation_embeddings.weight, std=0.02)

    def attribute_values(self) -> torch.Tensor:
        return self.attribute_decoder(self.entity_embeddings.weight)

    @staticmethod
    def symmetric_pair_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.cat((left + right, (left - right).abs(), left * right), dim=-1)

    def relation_logits(
        self,
        relations: torch.Tensor,
        subjects: torch.Tensor,
        objects: torch.Tensor,
        include_nonlinear: bool = True,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        left = self.entity_embeddings(subjects)
        right = self.entity_embeddings(objects)
        logits = torch.empty(len(relations), device=left.device, dtype=left.dtype)
        gates = torch.empty_like(logits)
        for relation in relations.unique().tolist():
            mask = relations == relation
            head = self.relation_heads[int(relation)]
            logits[mask] = head(left[mask], right[mask])
            gates[mask] = head.nonlinear_gate

        residual = torch.zeros_like(logits)
        if include_nonlinear and self.nonlinear is not None:
            pair = self.symmetric_pair_features(left, right)
            relation_features = self.relation_embeddings(relations)
            residual = self.nonlinear(torch.cat((pair, relation_features), dim=-1)).squeeze(-1)
            logits = logits + gates * residual
        if return_residual:
            return logits, residual
        return logits

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _pair_pools(labels: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    pools: dict[tuple[int, int], np.ndarray] = {}
    for relation in range(labels.shape[0]):
        flat = labels[relation].reshape(-1)
        pools[(relation, 0)] = np.flatnonzero(~flat)
        pools[(relation, 1)] = np.flatnonzero(flat)
    return pools


def _sample_relation_batch(
    world: SymmetricWorld,
    relation_ids: Sequence[int],
    batch_size: int,
    positive_fraction: float,
    rng: np.random.Generator,
    pools: dict[tuple[int, int], np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_positive = int(round(batch_size * positive_fraction))
    values = np.zeros(batch_size, dtype=np.int64)
    values[:n_positive] = 1
    rng.shuffle(values)
    relations = rng.choice(np.asarray(relation_ids), size=batch_size, replace=True)
    flat_pairs = np.empty(batch_size, dtype=np.int64)
    for relation in np.unique(relations):
        for value in (0, 1):
            mask = (relations == relation) & (values == value)
            count = int(mask.sum())
            if count:
                flat_pairs[mask] = rng.choice(
                    pools[(int(relation), value)], size=count, replace=True
                )
    subjects = flat_pairs // world.n_entities
    objects = flat_pairs % world.n_entities
    return (
        torch.as_tensor(relations, device=device, dtype=torch.long),
        torch.as_tensor(subjects, device=device, dtype=torch.long),
        torch.as_tensor(objects, device=device, dtype=torch.long),
        torch.as_tensor(values, device=device, dtype=torch.float32),
    )


@torch.no_grad()
def attribute_metrics(model: SymmetricLookupModel, world: SymmetricWorld) -> dict[str, float]:
    predictions = model.attribute_values().detach().cpu().numpy()
    targets = world.attributes.astype(np.float32) * 2.0 - 1.0
    return {
        "accuracy": float(np.mean((predictions > 0) == (targets > 0))),
        "mse": float(np.mean(np.square(predictions - targets))),
    }


@torch.no_grad()
def relation_accuracies(
    model: SymmetricLookupModel,
    world: SymmetricWorld,
    device: torch.device,
    include_nonlinear: bool,
    relation_ids: Iterable[int] | None = None,
) -> list[float]:
    relations_to_evaluate = list(
        range(world.n_relations) if relation_ids is None else relation_ids
    )
    subjects, objects = np.indices((world.n_entities, world.n_entities))
    subjects_t = torch.as_tensor(subjects.reshape(-1), device=device, dtype=torch.long)
    objects_t = torch.as_tensor(objects.reshape(-1), device=device, dtype=torch.long)
    output = []
    model.eval()
    for relation in relations_to_evaluate:
        relation_t = torch.full_like(subjects_t, relation)
        logits = model.relation_logits(
            relation_t, subjects_t, objects_t, include_nonlinear=include_nonlinear
        )
        predictions = logits.detach().cpu().numpy().reshape(
            world.n_entities, world.n_entities
        ) > 0
        output.append(float(np.mean(predictions == world.labels[relation])))
    model.train()
    return output


def _positive_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    degrees = labels.sum(axis=1, keepdims=True).astype(np.float64)
    if np.any(degrees == 0):
        raise ValueError("Every subject must have at least one positive object")
    return labels.astype(np.float64) @ embeddings / degrees


def ols_relation_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
    seed: int,
) -> dict[str, float]:
    """Fit subject -> positive-object-centroid OLS, with full and CV metrics."""
    x = embeddings.astype(np.float64)
    y = _positive_centroids(x, labels)
    full = LinearRegression(fit_intercept=True).fit(x, y)
    full_prediction = full.predict(x)
    centered = y - y.mean(axis=0, keepdims=True)
    ss_tot = float(np.square(centered).sum())
    full_ss_res = float(np.square(y - full_prediction).sum())

    cv_prediction = np.empty_like(y)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_indices, eval_indices in splitter.split(x):
        regression = LinearRegression(fit_intercept=True).fit(x[fit_indices], y[fit_indices])
        cv_prediction[eval_indices] = regression.predict(x[eval_indices])
    cv_ss_res = float(np.square(y - cv_prediction).sum())
    denom = np.maximum(
        np.linalg.norm(y, axis=1) * np.linalg.norm(cv_prediction, axis=1), 1e-12
    )

    # Use the predicted centroid to rank objects; all relations have the same
    # degree, so precision@degree is directly comparable.
    centered_objects = x - x.mean(axis=0, keepdims=True)
    centered_predictions = cv_prediction - x.mean(axis=0, keepdims=True)
    scores = centered_predictions @ centered_objects.T
    degree = int(labels.sum(axis=1)[0])
    top_indices = np.argpartition(scores, -degree, axis=1)[:, -degree:]
    precision = np.mean(
        [labels[row, top_indices[row]].mean() for row in range(len(labels))]
    )
    return {
        "full_r2": 1.0 - full_ss_res / max(ss_tot, 1e-12),
        "full_relative_error": math.sqrt(full_ss_res / max(ss_tot, 1e-12)),
        "cv_r2": 1.0 - cv_ss_res / max(ss_tot, 1e-12),
        "cv_relative_error": math.sqrt(cv_ss_res / max(ss_tot, 1e-12)),
        "cv_cosine": float(np.mean(np.sum(y * cv_prediction, axis=1) / denom)),
        "cv_precision_at_degree": float(precision),
    }


def all_ols_metrics(
    model: SymmetricLookupModel, world: SymmetricWorld, cfg: SymmetricLookupConfig
) -> list[dict[str, float]]:
    embeddings = model.entity_embeddings.weight.detach().cpu().numpy()
    return [
        ols_relation_metrics(embeddings, world.labels[relation], cfg.ols_folds, cfg.seed)
        for relation in range(world.n_relations)
    ]


def _train_phase1(
    model: SymmetricLookupModel,
    world: SymmetricWorld,
    cfg: SymmetricLookupConfig,
    device: torch.device,
    quiet: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed + 101)
    pools = _pair_pools(world.labels)
    parameters: list[nn.Parameter] = list(model.entity_embeddings.parameters())
    parameters += list(model.attribute_decoder.parameters())
    for head in model.relation_heads[: world.n_attributes]:
        parameters += list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=0.0)
    attr_targets = torch.as_tensor(
        world.attributes.astype(np.float32) * 2.0 - 1.0, device=device
    )
    structured_ids = list(range(world.n_attributes))
    history: list[dict[str, float]] = []
    for step in range(1, cfg.phase1_steps + 1):
        relations, subjects, objects, targets = _sample_relation_batch(
            world,
            structured_ids,
            cfg.batch_size,
            cfg.positive_fraction,
            rng,
            pools,
            device,
        )
        logits = model.relation_logits(
            relations, subjects, objects, include_nonlinear=False
        )
        relation_loss = F.binary_cross_entropy_with_logits(logits, targets)
        attr_loss = F.mse_loss(model.attribute_values(), attr_targets)
        loss = relation_loss + cfg.attr_loss_weight * attr_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()

        if step % cfg.eval_every == 0 or step == cfg.phase1_steps:
            accuracies = relation_accuracies(
                model, world, device, False, structured_ids
            )
            attrs = attribute_metrics(model, world)
            row = {
                "step": float(step),
                "loss": float(loss.item()),
                "attribute_accuracy": attrs["accuracy"],
                "min_structured_accuracy": min(accuracies),
                "mean_structured_accuracy": float(np.mean(accuracies)),
            }
            history.append(row)
            if not quiet:
                print(
                    f"phase1 step={step:5d} attr={attrs['accuracy']:.3f} "
                    f"structured_min={min(accuracies):.3f}",
                    flush=True,
                )
            if (
                attrs["accuracy"] >= cfg.target_accuracy
                and min(accuracies) >= cfg.target_accuracy
                and attrs["mse"] < 1e-3
            ):
                break
    return {"steps": step, "history": history}


def _freeze_representation(model: SymmetricLookupModel) -> None:
    model.entity_embeddings.weight.requires_grad_(False)
    for parameter in model.attribute_decoder.parameters():
        parameter.requires_grad_(False)


def _disable_structured_nonlinear_gates(
    model: SymmetricLookupModel, n_structured: int, freeze: bool
) -> None:
    """Keep a newly trained arbitrary residual from perturbing old relations."""
    for head in model.relation_heads[:n_structured]:
        with torch.no_grad():
            head.nonlinear_gate.zero_()
        head.nonlinear_gate.requires_grad_(not freeze)


def _train_relations(
    model: SymmetricLookupModel,
    world: SymmetricWorld,
    cfg: SymmetricLookupConfig,
    device: torch.device,
    relation_ids: Sequence[int],
    max_steps: int,
    seed_offset: int,
    quiet: bool,
    penalize_structured_residual: bool,
    train_representation: bool = False,
) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    pools = _pair_pools(world.labels)
    parameters: list[nn.Parameter] = []
    for relation in relation_ids:
        parameters += list(model.relation_heads[relation].parameters())
    if model.nonlinear is not None:
        parameters += list(model.nonlinear.parameters())
        parameters += list(model.relation_embeddings.parameters())
    if train_representation:
        parameters += list(model.entity_embeddings.parameters())
        if cfg.relation_attr_rehearsal_weight > 0:
            parameters += list(model.attribute_decoder.parameters())
    # Deduplicate shared parameters while preserving order.
    parameters = list({id(parameter): parameter for parameter in parameters}.values())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=0.0)
    attr_targets = torch.as_tensor(
        world.attributes.astype(np.float32) * 2.0 - 1.0, device=device
    )
    history: list[dict[str, float]] = []
    for step in range(1, max_steps + 1):
        relations, subjects, objects, targets = _sample_relation_batch(
            world,
            relation_ids,
            cfg.batch_size,
            cfg.positive_fraction,
            rng,
            pools,
            device,
        )
        logits, residual = model.relation_logits(
            relations,
            subjects,
            objects,
            include_nonlinear=True,
            return_residual=True,
        )
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        if penalize_structured_residual and model.nonlinear is not None:
            structured_mask = relations < world.arbitrary_relation
            if structured_mask.any():
                loss = loss + cfg.nonlinear_base_penalty * residual[
                    structured_mask
                ].square().mean()
        if train_representation and cfg.relation_attr_rehearsal_weight > 0:
            loss = loss + cfg.relation_attr_rehearsal_weight * F.mse_loss(
                model.attribute_values(), attr_targets
            )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()

        if step % cfg.eval_every == 0 or step == max_steps:
            full = relation_accuracies(model, world, device, True, relation_ids)
            affine = relation_accuracies(model, world, device, False, relation_ids)
            row = {
                "step": float(step),
                "loss": float(loss.item()),
                "min_full_accuracy": min(full),
                "mean_full_accuracy": float(np.mean(full)),
                "min_affine_accuracy": min(affine),
                "mean_affine_accuracy": float(np.mean(affine)),
            }
            history.append(row)
            if not quiet:
                print(
                    f"relations step={step:5d} full_min={min(full):.3f} "
                    f"affine_min={min(affine):.3f}",
                    flush=True,
                )
            if min(full) >= cfg.target_accuracy:
                break
    return {"steps": step, "history": history}


def run_single(
    cfg: SymmetricLookupConfig,
    device: torch.device,
    quiet: bool = False,
    run_joint: bool = True,
) -> dict[str, object]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    world = build_world(cfg)
    model = SymmetricLookupModel(cfg).to(device)
    started = time.time()
    phase1 = _train_phase1(model, world, cfg, device, quiet)
    phase1_attrs = attribute_metrics(model, world)
    phase1_accuracy = relation_accuracies(
        model, world, device, False, range(world.n_attributes)
    )
    ols = all_ols_metrics(model, world, cfg)
    frozen_before = model.entity_embeddings.weight.detach().clone()
    phase1_attribute_decoder = copy.deepcopy(model.attribute_decoder.state_dict())
    if cfg.freeze_embeddings:
        _freeze_representation(model)
    _disable_structured_nonlinear_gates(model, world.n_attributes, freeze=True)

    representation_ready = (
        phase1_attrs["accuracy"] >= cfg.target_accuracy
        and min(phase1_accuracy) >= cfg.target_accuracy
        and min(row["full_r2"] for row in ols[: world.n_attributes]) >= 0.95
        # Use the full-data residual for the categorical non-affinity claim.
        # CV is retained as a robustness metric but is not the definition:
        # unlike the old Goal 2 probe, this check asks whether one affine map
        # can reproduce even the complete set of known random facts.
        and ols[world.arbitrary_relation]["full_r2"] < 0.50
    )
    if representation_ready:
        arbitrary_phase = _train_relations(
            model,
            world,
            cfg,
            device,
            [world.arbitrary_relation],
            cfg.arbitrary_steps,
            202,
            quiet,
            False,
            train_representation=not cfg.freeze_embeddings,
        )
    else:
        arbitrary_phase = {
            "steps": 0,
            "history": [],
            "skipped_because_representation_failed": True,
        }
    embedding_drift = float(
        (model.entity_embeddings.weight.detach() - frozen_before).abs().max().item()
    )
    arbitrary_full = relation_accuracies(
        model, world, device, True, [world.arbitrary_relation]
    )[0]
    arbitrary_affine = relation_accuracies(
        model, world, device, False, [world.arbitrary_relation]
    )[0]
    preserved_structured = relation_accuracies(
        model, world, device, True, range(world.n_attributes)
    )
    attributes_after_arbitrary = attribute_metrics(model, world)
    ols_after_arbitrary = all_ols_metrics(model, world, cfg)

    joint_result: dict[str, object] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stage_ready = (
        phase1_attrs["accuracy"] >= cfg.target_accuracy
        and min(phase1_accuracy) >= cfg.target_accuracy
        and min(
            row["full_r2"]
            for row in ols_after_arbitrary[: world.n_attributes]
        )
        >= 0.95
        and ols_after_arbitrary[world.arbitrary_relation]["full_r2"] < 0.50
        and arbitrary_full >= cfg.target_accuracy
        and arbitrary_affine < 0.90
        and min(preserved_structured) >= cfg.target_accuracy
        and (not cfg.freeze_embeddings or embedding_drift == 0.0)
    )
    if run_joint and (stage_ready or cfg.run_joint_on_stage_failure):
        torch.manual_seed(cfg.seed + 10_000)
        joint = SymmetricLookupModel(cfg).to(device)
        # The fresh joint run starts from the original phase-1 representation,
        # not the representation modified by arbitrary-only continuation.
        with torch.no_grad():
            joint.entity_embeddings.weight.copy_(frozen_before)
            joint.attribute_decoder.load_state_dict(phase1_attribute_decoder)
        if cfg.freeze_embeddings:
            _freeze_representation(joint)
        _disable_structured_nonlinear_gates(joint, world.n_attributes, freeze=False)
        joint_training = _train_relations(
            joint,
            world,
            cfg,
            device,
            list(range(world.n_relations)),
            cfg.joint_steps,
            303,
            quiet,
            True,
            train_representation=not cfg.freeze_embeddings,
        )
        joint_full = relation_accuracies(joint, world, device, True)
        joint_affine = relation_accuracies(joint, world, device, False)
        joint_attributes = attribute_metrics(joint, world)
        joint_ols = all_ols_metrics(joint, world, cfg)
        joint_embedding_drift = float(
            (joint.entity_embeddings.weight.detach() - frozen_before).abs().max().item()
        )
        joint_result = {
            "training": joint_training,
            "full_accuracy": joint_full,
            "affine_only_accuracy": joint_affine,
            "min_structured_full_accuracy": min(joint_full[: world.n_attributes]),
            "min_structured_affine_accuracy": min(
                joint_affine[: world.n_attributes]
            ),
            "arbitrary_full_accuracy": joint_full[world.arbitrary_relation],
            "arbitrary_affine_accuracy": joint_affine[world.arbitrary_relation],
            "attribute_metrics": joint_attributes,
            "ols": {
                "structured": joint_ols[: world.n_attributes],
                "arbitrary": joint_ols[world.arbitrary_relation],
                "min_structured_full_r2": min(
                    row["full_r2"]
                    for row in joint_ols[: world.n_attributes]
                ),
            },
            "embedding_max_drift": joint_embedding_drift,
        }
        best_state = {key: value.detach().cpu() for key, value in joint.state_dict().items()}

    result: dict[str, object] = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "num_parameters": model.n_parameters(),
        "world": {
            "n_attribute_sentences": len(world.attribute_sentences()),
            "n_relation_sentences": len(world.relation_sentences()),
            "positives_per_relation": world.labels.sum(axis=(1, 2)).tolist(),
            "positives_per_subject": world.labels.sum(axis=2)[:, 0].tolist(),
            "symmetric": bool(
                np.all(world.labels == world.labels.transpose(0, 2, 1))
            ),
        },
        "phase1": phase1,
        "attribute_metrics": phase1_attrs,
        "structured_affine_accuracy": phase1_accuracy,
        "ols": {
            "structured": ols[: world.n_attributes],
            "arbitrary": ols[world.arbitrary_relation],
            "min_structured_full_r2": min(
                row["full_r2"] for row in ols[: world.n_attributes]
            ),
            "min_structured_cv_r2": min(
                row["cv_r2"] for row in ols[: world.n_attributes]
            ),
        },
        "ols_after_arbitrary": {
            "structured": ols_after_arbitrary[: world.n_attributes],
            "arbitrary": ols_after_arbitrary[world.arbitrary_relation],
            "min_structured_full_r2": min(
                row["full_r2"]
                for row in ols_after_arbitrary[: world.n_attributes]
            ),
            "min_structured_cv_r2": min(
                row["cv_r2"]
                for row in ols_after_arbitrary[: world.n_attributes]
            ),
        },
        "arbitrary_phase": arbitrary_phase,
        "arbitrary_full_accuracy": arbitrary_full,
        "arbitrary_affine_accuracy": arbitrary_affine,
        "preserved_structured_accuracy": preserved_structured,
        "attributes_after_arbitrary": attributes_after_arbitrary,
        "embedding_max_drift": embedding_drift,
        "joint_requested": run_joint,
        "joint_skipped_because_stage_failed": bool(
            run_joint and joint_result is None and not stage_ready
        ),
        "joint": joint_result,
        "elapsed_sec": time.time() - started,
    }
    result["claim_checks"] = claim_checks(result)
    if best_state is not None:
        result["_state_dict"] = best_state
    return result


def claim_checks(result: dict[str, object]) -> dict[str, bool]:
    cfg = result["config"]
    ols = result["ols_after_arbitrary"]
    joint = result.get("joint")
    checks = {
        "attributes_learned": result["attribute_metrics"]["accuracy"]
        >= cfg["target_accuracy"],
        "attributes_retained_after_arbitrary": result[
            "attributes_after_arbitrary"
        ]["accuracy"]
        >= cfg["target_accuracy"],
        "structured_relations_learned": min(result["structured_affine_accuracy"])
        >= cfg["target_accuracy"],
        "structured_relations_ols_linear": ols["min_structured_full_r2"] >= 0.95,
        "arbitrary_relation_ols_nonlinear": ols["arbitrary"]["full_r2"] < 0.50,
        "arbitrary_relation_learned": result["arbitrary_full_accuracy"]
        >= cfg["target_accuracy"],
        "arbitrary_needs_nonlinear_path": result["arbitrary_affine_accuracy"] < 0.90,
        "embedding_policy_satisfied": (
            result["embedding_max_drift"] == 0.0
            if cfg["freeze_embeddings"]
            else True
        ),
        "structured_relations_preserved": min(result["preserved_structured_accuracy"])
        >= cfg["target_accuracy"],
    }
    if joint is not None:
        checks.update(
            {
                "joint_all_relations_learned": min(joint["full_accuracy"])
                >= cfg["target_accuracy"],
                "joint_structured_affine_path_sufficient": joint[
                    "min_structured_affine_accuracy"
                ]
                >= cfg["target_accuracy"],
                "joint_arbitrary_needs_nonlinear_path": joint[
                    "arbitrary_affine_accuracy"
                ]
                < 0.90,
                "joint_attributes_retained": joint["attribute_metrics"]["accuracy"]
                >= cfg["target_accuracy"],
                "joint_structured_relations_ols_linear": joint["ols"][
                    "min_structured_full_r2"
                ]
                >= 0.95,
                "joint_arbitrary_relation_ols_nonlinear": joint["ols"][
                    "arbitrary"
                ]["full_r2"]
                < 0.50,
            }
        )
    elif result.get("joint_requested"):
        checks["joint_all_relations_learned"] = False
        checks["joint_structured_affine_path_sufficient"] = False
        checks["joint_arbitrary_needs_nonlinear_path"] = False
        checks["joint_attributes_retained"] = False
        checks["joint_structured_relations_ols_linear"] = False
        checks["joint_arbitrary_relation_ols_nonlinear"] = False
    checks["all"] = all(checks.values())
    return checks


def _json_ready(result: dict[str, object]) -> dict[str, object]:
    result = copy.copy(result)
    result.pop("_state_dict", None)
    return result


def _candidate_architectures(sweep: SweepConfig) -> list[tuple[int, int]]:
    candidates = {(0, 0)}
    candidates.update(
        (width, depth)
        for width in sweep.nonlinear_widths
        if width > 0
        for depth in sweep.nonlinear_depths
    )
    return sorted(candidates, key=lambda item: (item[0] * max(1, item[1]), item))


def run_sweep(
    base_cfg: SymmetricLookupConfig,
    sweep: SweepConfig,
    device: torch.device,
    quiet: bool = False,
) -> dict[str, object]:
    """Grid search and select the smallest architecture passing every check."""
    output_dir = base_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    best: tuple[int, dict[str, object], list[dict[str, object]]] | None = None
    for embedding_dim in sweep.embedding_dims:
        for width, depth in _candidate_architectures(sweep):
            seed_results: list[dict[str, object]] = []
            if not quiet:
                print(
                    f"\n=== grid d={embedding_dim} width={width} depth={depth} ===",
                    flush=True,
                )
            for seed in sweep.seeds:
                cfg = copy.deepcopy(base_cfg)
                cfg.seed = seed
                cfg.embedding_dim = embedding_dim
                cfg.nonlinear_width = width
                cfg.nonlinear_depth = depth
                result = run_single(cfg, device, quiet=quiet, run_joint=True)
                state = result.pop("_state_dict", None)
                seed_results.append(result)
                rows.append(result)
                if not quiet:
                    checks = result["claim_checks"]
                    print(
                        f"seed={seed} all={checks['all']} "
                        f"baseOLS={result['ols']['min_structured_full_r2']:.3f} "
                        f"rndOLS={result['ols']['arbitrary']['cv_r2']:.3f} "
                        f"rnd={result['arbitrary_full_accuracy']:.3f}/"
                        f"{result['arbitrary_affine_accuracy']:.3f}",
                        flush=True,
                    )
                if state is not None and result["claim_checks"]["all"]:
                    checkpoint = output_dir / (
                        f"candidate_d{embedding_dim}_w{width}_l{depth}_seed{seed}.pt"
                    )
                    torch.save(state, checkpoint)

            pass_count = sum(
                result["claim_checks"]["all"]
                and result["ols_after_arbitrary"]["min_structured_full_r2"]
                >= sweep.min_structured_ols_r2
                and result["ols_after_arbitrary"]["arbitrary"]["full_r2"]
                <= sweep.max_arbitrary_ols_r2
                and result["arbitrary_affine_accuracy"]
                <= sweep.max_arbitrary_affine_accuracy
                for result in seed_results
            )
            passes = pass_count == len(seed_results) if sweep.require_all_seeds else pass_count > 0
            if passes:
                parameter_count = int(seed_results[0]["num_parameters"])
                summary = {
                    "embedding_dim": embedding_dim,
                    "nonlinear_width": width,
                    "nonlinear_depth": depth,
                    "num_parameters": parameter_count,
                    "seeds_passed": pass_count,
                    "seeds_total": len(seed_results),
                }
                if best is None or parameter_count < best[0]:
                    best = (parameter_count, summary, seed_results)

    serializable_rows = [_json_ready(result) for result in rows]
    output: dict[str, object] = {
        "base_config": {**asdict(base_cfg), "output_dir": str(base_cfg.output_dir)},
        "sweep": asdict(sweep),
        "runs": serializable_rows,
        "best": None if best is None else best[1],
    }
    metrics_path = output_dir / "symmetric_lookup_sweep.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["result_path"] = str(metrics_path)
    return output


def load_config(path: Path) -> tuple[SymmetricLookupConfig, SweepConfig]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    experiment = dict(raw.get("experiment", {}))
    if "output_dir" in experiment:
        output_dir = Path(experiment["output_dir"])
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        experiment["output_dir"] = output_dir
    cfg = SymmetricLookupConfig(**experiment)
    sweep = SweepConfig(**raw.get("sweep", {}))
    return cfg, sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/runs/symmetric_lookup_suggested.toml"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    cfg, sweep = load_config(Path(args.config))
    explicit_seed = args.seed is not None
    if explicit_seed:
        cfg.seed = args.seed
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        cfg.output_dir = output_dir
    device = torch.device(args.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if args.single:
        result = run_single(cfg, device, quiet=args.quiet, run_joint=True)
        state = result.pop("_state_dict", None)
        suffix = f"_seed{cfg.seed}" if explicit_seed else "_single"
        result_path = cfg.output_dir / f"symmetric_lookup{suffix}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if state is not None:
            torch.save(state, cfg.output_dir / f"symmetric_lookup{suffix}.pt")
        print(json.dumps({"result_path": str(result_path), "checks": result["claim_checks"]}, indent=2))
    else:
        result = run_sweep(cfg, sweep, device, quiet=args.quiet)
        print(json.dumps({"result_path": result["result_path"], "best": result["best"]}, indent=2))


if __name__ == "__main__":
    main()

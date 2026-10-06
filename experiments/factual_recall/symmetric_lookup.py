"""Balanced symmetric datasets and overdetermined affine probes."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

@dataclass
class SymmetricLookupConfig:
    seed: int = 19
    n_entities: int = 64
    n_attributes: int = 10
    graph_swaps_per_edge: int = 40

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

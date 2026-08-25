"""Mechanistic analysis and paper figures for the revised Goal 2 experiment.

The analysis operates on final joint checkpoints from ``symmetric_lookup``.  It
combines exact path decomposition with causal interventions:

* structured heads are compared with the product of their decoded attribute;
* decoded attribute coordinates are flipped while other coordinates are held fixed;
* clean arbitrary-relation activations are patched into corrupted pairs;
* individual hidden units are patched and mean-ablated; and
* linear probes track where the arbitrary label becomes linearly readable.

Outputs are a JSON report and two multi-panel figures (PDF/PNG/SVG).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROJECT_ROOT
from .symmetric_lookup import (
    SymmetricLookupConfig,
    SymmetricLookupModel,
    SymmetricWorld,
    build_world,
)


DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "factual_recall" / "results" / "symmetric_lookup_unfrozen"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "factual_recall"
    / "results"
    / "symmetric_lookup_mechanistic"
)
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"


def _config_from_result(raw: dict[str, Any]) -> SymmetricLookupConfig:
    valid = {field.name for field in fields(SymmetricLookupConfig)}
    values = {key: value for key, value in raw["config"].items() if key in valid}
    values["output_dir"] = Path(values["output_dir"])
    return SymmetricLookupConfig(**values)


def _paths_for_seed(result_dir: Path, seed: int) -> tuple[Path, Path]:
    suffix = "single" if seed == 19 else f"seed{seed}"
    return (
        result_dir / f"symmetric_lookup_{suffix}.json",
        result_dir / f"symmetric_lookup_{suffix}.pt",
    )


def load_joint_model(
    result_dir: Path, seed: int, device: torch.device
) -> tuple[SymmetricLookupConfig, SymmetricWorld, SymmetricLookupModel, dict[str, Any]]:
    result_path, checkpoint_path = _paths_for_seed(result_dir, seed)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = _config_from_result(result)
    world = build_world(cfg)
    model = SymmetricLookupModel(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return cfg, world, model, result


def all_ordered_pairs(world: SymmetricWorld, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    subjects, objects = np.indices((world.n_entities, world.n_entities))
    return (
        torch.as_tensor(subjects.reshape(-1), device=device, dtype=torch.long),
        torch.as_tensor(objects.reshape(-1), device=device, dtype=torch.long),
    )


def relation_components(
    model: SymmetricLookupModel,
    relation: int,
    subjects: torch.Tensor,
    objects: torch.Tensor,
) -> dict[str, torch.Tensor]:
    relations = torch.full_like(subjects, relation)
    full, raw_residual = model.relation_logits(
        relations,
        subjects,
        objects,
        include_nonlinear=True,
        return_residual=True,
    )
    affine = model.relation_logits(
        relations, subjects, objects, include_nonlinear=False
    )
    gated_residual = full - affine
    return {
        "full": full,
        "affine": affine,
        "raw_residual": raw_residual,
        "gated_residual": gated_residual,
    }


def binary_accuracy(logits: torch.Tensor, labels: np.ndarray | torch.Tensor) -> float:
    if isinstance(labels, np.ndarray):
        labels = torch.as_tensor(labels.reshape(-1), device=logits.device)
    return float(((logits > 0) == labels.bool()).float().mean().item())


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.square(target - prediction).sum())
    total = float(np.square(target - target.mean()).sum())
    return 1.0 - residual / max(total, 1e-12)


def _frob_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(-1)
    right = right.reshape(-1)
    denominator = left.norm() * right.norm()
    return float((left @ right / denominator.clamp_min(1e-12)).item())


@torch.no_grad()
def analyze_paths_and_structured_algorithm(
    model: SymmetricLookupModel,
    world: SymmetricWorld,
    device: torch.device,
) -> dict[str, Any]:
    subjects, objects = all_ordered_pairs(world, device)
    decoded = model.attribute_values()
    path_rows: list[dict[str, float]] = []
    structured_rows: list[dict[str, float]] = []
    labels_pm = torch.as_tensor(
        world.labels.astype(np.float32) * 2.0 - 1.0, device=device
    )

    for relation in range(world.n_relations):
        components = relation_components(model, relation, subjects, objects)
        labels = world.labels[relation]
        label_sign = labels_pm[relation].reshape(-1)
        head = model.relation_heads[relation]
        path_rows.append(
            {
                "relation": relation,
                "full_accuracy": binary_accuracy(components["full"], labels),
                "affine_accuracy": binary_accuracy(components["affine"], labels),
                "nonlinear_only_accuracy": binary_accuracy(
                    components["gated_residual"], labels
                ),
                "gate": float(head.nonlinear_gate.item()),
                "mean_abs_affine": float(components["affine"].abs().mean().item()),
                "mean_abs_nonlinear": float(
                    components["gated_residual"].abs().mean().item()
                ),
                "affine_signed_margin": float(
                    (label_sign * components["affine"]).mean().item()
                ),
                "nonlinear_signed_margin": float(
                    (label_sign * components["gated_residual"]).mean().item()
                ),
            }
        )

        if relation < world.n_attributes:
            attribute_product = (
                decoded[:, relation, None] * decoded[None, :, relation]
            ).reshape(-1)
            affine = components["affine"]
            design = torch.stack(
                (attribute_product, torch.ones_like(attribute_product)), dim=1
            )
            solution = torch.linalg.lstsq(design, affine[:, None]).solution[:, 0]
            predicted = design @ solution
            symmetric_weight = 0.5 * (head.weight + head.weight.T)
            decoder_direction = model.attribute_decoder.weight[relation]
            ideal_weight = torch.outer(decoder_direction, decoder_direction)
            structured_rows.append(
                {
                    "relation": relation,
                    "attribute_product_r2": _r2(
                        affine.cpu().numpy(), predicted.cpu().numpy()
                    ),
                    "attribute_product_correlation": float(
                        torch.corrcoef(torch.stack((attribute_product, affine)))[0, 1].item()
                    ),
                    "weight_outer_product_cosine": _frob_cosine(
                        symmetric_weight, ideal_weight
                    ),
                }
            )

    # Causal coordinate intervention.  The attribute decoder is square at the
    # selected minimum architecture.  Use its pseudoinverse to flip one decoded
    # coordinate while leaving the other nine fixed in decoder space.
    embeddings = model.entity_embeddings.weight
    decoder_values = model.attribute_values()
    decoder_map = model.attribute_decoder.weight.T
    decoder_inverse = torch.linalg.pinv(decoder_map)
    flip_rows: list[dict[str, float]] = []
    for relation in range(world.n_attributes):
        target_delta = torch.zeros_like(decoder_values)
        target_delta[:, relation] = -2.0 * decoder_values[:, relation]
        target_embeddings = embeddings + target_delta @ decoder_inverse

        control_relation = (relation + 1) % world.n_attributes
        control_delta = torch.zeros_like(decoder_values)
        control_delta[:, control_relation] = -2.0 * decoder_values[:, control_relation]
        control_embeddings = embeddings + control_delta @ decoder_inverse

        left = embeddings[subjects]
        target_right = target_embeddings[objects]
        control_right = control_embeddings[objects]
        head = model.relation_heads[relation]
        target_logits = head(left, target_right)
        control_logits = head(left, control_right)
        original_labels = world.labels[relation].reshape(-1)
        flip_rows.append(
            {
                "relation": relation,
                "targeted_flip_toggles_decision": binary_accuracy(
                    target_logits, ~original_labels
                ),
                "off_target_flip_preserves_decision": binary_accuracy(
                    control_logits, original_labels
                ),
            }
        )

    return {
        "paths": path_rows,
        "structured_algorithm": structured_rows,
        "attribute_flip": flip_rows,
    }


def mlp_activations(
    model: SymmetricLookupModel,
    relations: torch.Tensor,
    subjects: torch.Tensor,
    objects: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if model.nonlinear is None or len(model.nonlinear) != 5:
        raise ValueError("Mechanistic analysis expects the selected two-layer MLP")
    left = model.entity_embeddings(subjects)
    right = model.entity_embeddings(objects)
    pair = model.symmetric_pair_features(left, right)
    inputs = torch.cat((pair, model.relation_embeddings(relations)), dim=-1)
    pre1 = model.nonlinear[0](inputs)
    hidden1 = F.gelu(pre1)
    pre2 = model.nonlinear[2](hidden1)
    hidden2 = F.gelu(pre2)
    residual = model.nonlinear[4](hidden2).squeeze(-1)
    affine = torch.empty_like(residual)
    gate = torch.empty_like(residual)
    for relation in relations.unique().tolist():
        mask = relations == relation
        head = model.relation_heads[int(relation)]
        affine[mask] = head(left[mask], right[mask])
        gate[mask] = head.nonlinear_gate
    return {
        "left": left,
        "right": right,
        "pair": pair,
        "input": inputs,
        "pre1": pre1,
        "hidden1": hidden1,
        "pre2": pre2,
        "hidden2": hidden2,
        "residual": residual,
        "affine": affine,
        "gate": gate,
        "full": affine + gate * residual,
    }


def _mlp_from_input(model: SymmetricLookupModel, inputs: torch.Tensor) -> torch.Tensor:
    hidden1 = F.gelu(model.nonlinear[0](inputs))
    hidden2 = F.gelu(model.nonlinear[2](hidden1))
    return model.nonlinear[4](hidden2).squeeze(-1)


def _mlp_from_hidden1(model: SymmetricLookupModel, hidden1: torch.Tensor) -> torch.Tensor:
    hidden2 = F.gelu(model.nonlinear[2](hidden1))
    return model.nonlinear[4](hidden2).squeeze(-1)


def _mlp_from_hidden2(model: SymmetricLookupModel, hidden2: torch.Tensor) -> torch.Tensor:
    return model.nonlinear[4](hidden2).squeeze(-1)


def _normalized_recovery(
    patched: torch.Tensor, clean: torch.Tensor, corrupted: torch.Tensor
) -> float:
    denominator = (clean - corrupted).mean()
    return float(((patched - corrupted).mean() / denominator.clamp_min(1e-8)).item())


def select_clean_corrupted_pairs(
    model: SymmetricLookupModel,
    world: SymmetricWorld,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One positive and one negative object per subject, with subject fixed."""
    rng = np.random.default_rng(seed + 909)
    relation = world.arbitrary_relation
    subjects: list[int] = []
    clean_objects: list[int] = []
    corrupted_objects: list[int] = []
    for subject in range(world.n_entities):
        positives = np.flatnonzero(world.labels[relation, subject]).tolist()
        positives = [obj for obj in positives if obj != subject] or positives
        negatives = np.flatnonzero(~world.labels[relation, subject]).tolist()
        subjects.append(subject)
        clean_objects.append(int(rng.choice(positives)))
        corrupted_objects.append(int(rng.choice(negatives)))
    subject_t = torch.tensor(subjects, device=device)
    clean_t = torch.tensor(clean_objects, device=device)
    corrupted_t = torch.tensor(corrupted_objects, device=device)
    relation_t = torch.full_like(subject_t, relation)
    clean = mlp_activations(model, relation_t, subject_t, clean_t)
    corrupted = mlp_activations(model, relation_t, subject_t, corrupted_t)
    correct = (clean["full"] > 0) & (corrupted["full"] < 0)
    if int(correct.sum()) < world.n_entities // 2:
        raise RuntimeError("Too few correctly classified clean/corrupted pairs")
    return subject_t[correct], clean_t[correct], corrupted_t[correct]


def _probe_accuracy(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs"),
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(classifier, features, labels, cv=splitter).mean())


@torch.no_grad()
def analyze_arbitrary_mlp(
    model: SymmetricLookupModel,
    world: SymmetricWorld,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    relation = world.arbitrary_relation
    subjects, objects = all_ordered_pairs(world, device)
    relations = torch.full_like(subjects, relation)
    activations = mlp_activations(model, relations, subjects, objects)
    labels = world.labels[relation].reshape(-1)

    # Causal nonlinearity ablation: retain all trained affine parameters but
    # replace one or both GELUs by the identity function.
    identity_hidden1 = activations["pre1"]
    identity_pre2 = model.nonlinear[2](identity_hidden1)
    both_identity_residual = model.nonlinear[4](identity_pre2).squeeze(-1)
    first_identity_residual = model.nonlinear[4](F.gelu(identity_pre2)).squeeze(-1)
    second_identity_residual = model.nonlinear[4](activations["pre2"]).squeeze(-1)
    nonlinearity_ablation = {
        "both_gelus_identity_accuracy": binary_accuracy(
            activations["affine"] + activations["gate"] * both_identity_residual,
            labels,
        ),
        "first_gelu_identity_accuracy": binary_accuracy(
            activations["affine"] + activations["gate"] * first_identity_residual,
            labels,
        ),
        "second_gelu_identity_accuracy": binary_accuracy(
            activations["affine"] + activations["gate"] * second_identity_residual,
            labels,
        ),
    }

    # Remove duplicate directions for representation probes; the architecture is
    # exactly symmetric, so (s,o) and (o,s) have identical pair codes.
    upper = np.triu_indices(world.n_entities)
    upper_numpy = upper[0] * world.n_entities + upper[1]
    upper_flat = torch.as_tensor(upper_numpy, device=device, dtype=torch.long)
    probe_rows = {
        "raw_pair_features": activations["pair"][upper_flat],
        "pre_gelu_1": activations["pre1"][upper_flat],
        "post_gelu_1": activations["hidden1"][upper_flat],
        "pre_gelu_2": activations["pre2"][upper_flat],
        "post_gelu_2": activations["hidden2"][upper_flat],
    }
    probe_labels = labels[upper_numpy]
    probe_accuracy = {
        name: _probe_accuracy(rows.cpu().numpy(), probe_labels, seed)
        for name, rows in probe_rows.items()
    }

    subject_t, clean_objects, corrupted_objects = select_clean_corrupted_pairs(
        model, world, device, seed
    )
    relation_t = torch.full_like(subject_t, relation)
    clean = mlp_activations(model, relation_t, subject_t, clean_objects)
    corrupted = mlp_activations(model, relation_t, subject_t, corrupted_objects)
    clean_full, corrupted_full = clean["full"], corrupted["full"]
    gate = corrupted["gate"]

    patching: dict[str, float] = {}
    patching["affine_branch"] = _normalized_recovery(
        clean["affine"] + gate * corrupted["residual"],
        clean_full,
        corrupted_full,
    )
    pair_dim = model.cfg.embedding_dim
    group_slices = {
        "pair_sum": slice(0, pair_dim),
        "pair_abs_difference": slice(pair_dim, 2 * pair_dim),
        "pair_product": slice(2 * pair_dim, 3 * pair_dim),
        "relation_embedding": slice(3 * pair_dim, clean["input"].shape[1]),
    }
    for name, group_slice in group_slices.items():
        patched_input = corrupted["input"].clone()
        patched_input[:, group_slice] = clean["input"][:, group_slice]
        patched = corrupted["affine"] + gate * _mlp_from_input(model, patched_input)
        patching[name] = _normalized_recovery(
            patched, clean_full, corrupted_full
        )
    patched_hidden1 = corrupted["affine"] + gate * _mlp_from_hidden1(
        model, clean["hidden1"]
    )
    patched_hidden2 = corrupted["affine"] + gate * _mlp_from_hidden2(
        model, clean["hidden2"]
    )
    patched_residual = corrupted["affine"] + gate * clean["residual"]
    patching["hidden_layer_1"] = _normalized_recovery(
        patched_hidden1, clean_full, corrupted_full
    )
    patching["hidden_layer_2"] = _normalized_recovery(
        patched_hidden2, clean_full, corrupted_full
    )
    patching["nonlinear_output"] = _normalized_recovery(
        patched_residual, clean_full, corrupted_full
    )

    width = model.cfg.nonlinear_width
    unit_effects_1 = []
    unit_effects_2 = []
    for unit in range(width):
        patched_h1 = corrupted["hidden1"].clone()
        patched_h1[:, unit] = clean["hidden1"][:, unit]
        logits_h1 = corrupted["affine"] + gate * _mlp_from_hidden1(model, patched_h1)
        unit_effects_1.append(
            _normalized_recovery(logits_h1, clean_full, corrupted_full)
        )

        patched_h2 = corrupted["hidden2"].clone()
        patched_h2[:, unit] = clean["hidden2"][:, unit]
        logits_h2 = corrupted["affine"] + gate * _mlp_from_hidden2(model, patched_h2)
        unit_effects_2.append(
            _normalized_recovery(logits_h2, clean_full, corrupted_full)
        )

    importance = np.abs(np.asarray(unit_effects_2))
    order = np.argsort(-importance)
    mean_hidden2 = activations["hidden2"].mean(dim=0)
    k_values = [0, 1, 2, 4, 8, 16, 32, width]
    ablation_rows: list[dict[str, float]] = []
    for k in k_values:
        selected = order[:k]
        mean_ablate = activations["hidden2"].clone()
        if k:
            mean_ablate[:, selected] = mean_hidden2[selected]
        mean_ablate_logits = activations["affine"] + activations[
            "gate"
        ] * _mlp_from_hidden2(model, mean_ablate)

        keep_only = mean_hidden2[None, :].expand_as(activations["hidden2"]).clone()
        if k:
            keep_only[:, selected] = activations["hidden2"][:, selected]
        keep_only_logits = activations["affine"] + activations[
            "gate"
        ] * _mlp_from_hidden2(model, keep_only)
        ablation_rows.append(
            {
                "k": k,
                "mean_ablate_top_k_accuracy": binary_accuracy(
                    mean_ablate_logits, labels
                ),
                "keep_only_top_k_accuracy": binary_accuracy(keep_only_logits, labels),
            }
        )

    return {
        "probe_accuracy": probe_accuracy,
        "nonlinearity_ablation": nonlinearity_ablation,
        "activation_patching": patching,
        "patched_pair_count": int(len(subject_t)),
        "unit_effects_layer1": unit_effects_1,
        "unit_effects_layer2": unit_effects_2,
        "unit_ablation": ablation_rows,
        "full_accuracy": binary_accuracy(activations["full"], labels),
        "affine_accuracy": binary_accuracy(activations["affine"], labels),
    }


def analyze_seed(
    result_dir: Path, seed: int, device: torch.device
) -> dict[str, Any]:
    cfg, world, model, training_result = load_joint_model(result_dir, seed, device)
    structured = analyze_paths_and_structured_algorithm(model, world, device)
    arbitrary = analyze_arbitrary_mlp(model, world, device, seed)
    return {
        "seed": seed,
        "config": {
            "n_entities": cfg.n_entities,
            "n_attributes": cfg.n_attributes,
            "embedding_dim": cfg.embedding_dim,
            "nonlinear_width": cfg.nonlinear_width,
            "nonlinear_depth": cfg.nonlinear_depth,
            "num_parameters": training_result["num_parameters"],
        },
        "joint_training": {
            "attribute_accuracy": training_result["joint"]["attribute_metrics"][
                "accuracy"
            ],
            "full_accuracy": training_result["joint"]["full_accuracy"],
            "affine_accuracy": training_result["joint"]["affine_only_accuracy"],
            "embedding_max_drift": training_result["joint"]["embedding_max_drift"],
            "ols": training_result["joint"]["ols"],
        },
        **structured,
        "arbitrary_mlp": arbitrary,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n_attributes = results[0]["config"]["n_attributes"]
    relation_rows = []
    for relation in range(n_attributes + 1):
        rows = [result["paths"][relation] for result in results]
        relation_rows.append(
            {
                "relation": relation,
                **{
                    key: dict(zip(("mean", "std"), _mean_std([row[key] for row in rows])))
                    for key in (
                        "full_accuracy",
                        "affine_accuracy",
                        "nonlinear_only_accuracy",
                        "gate",
                        "mean_abs_affine",
                        "mean_abs_nonlinear",
                        "affine_signed_margin",
                        "nonlinear_signed_margin",
                    )
                },
            }
        )

    patch_names = list(results[0]["arbitrary_mlp"]["activation_patching"])
    probe_names = list(results[0]["arbitrary_mlp"]["probe_accuracy"])
    aggregate_result = {
        "relations": relation_rows,
        "structured_algorithm": {},
        "attribute_flip": {},
        "activation_patching": {},
        "nonlinearity_ablation": {},
        "probe_accuracy": {},
        "unit_ablation": [],
    }
    for metric in (
        "attribute_product_r2",
        "attribute_product_correlation",
        "weight_outer_product_cosine",
    ):
        values = [
            row[metric]
            for result in results
            for row in result["structured_algorithm"]
        ]
        aggregate_result["structured_algorithm"][metric] = dict(
            zip(("mean", "std"), _mean_std(values))
        )
    for metric in (
        "targeted_flip_toggles_decision",
        "off_target_flip_preserves_decision",
    ):
        values = [
            row[metric]
            for result in results
            for row in result["attribute_flip"]
        ]
        aggregate_result["attribute_flip"][metric] = dict(
            zip(("mean", "std"), _mean_std(values))
        )
    for name in patch_names:
        values = [result["arbitrary_mlp"]["activation_patching"][name] for result in results]
        aggregate_result["activation_patching"][name] = dict(
            zip(("mean", "std"), _mean_std(values))
        )
    for name in results[0]["arbitrary_mlp"]["nonlinearity_ablation"]:
        values = [
            result["arbitrary_mlp"]["nonlinearity_ablation"][name]
            for result in results
        ]
        aggregate_result["nonlinearity_ablation"][name] = dict(
            zip(("mean", "std"), _mean_std(values))
        )
    for name in probe_names:
        values = [result["arbitrary_mlp"]["probe_accuracy"][name] for result in results]
        aggregate_result["probe_accuracy"][name] = dict(
            zip(("mean", "std"), _mean_std(values))
        )
    k_values = [row["k"] for row in results[0]["arbitrary_mlp"]["unit_ablation"]]
    for index, k in enumerate(k_values):
        row: dict[str, Any] = {"k": k}
        for metric in ("mean_ablate_top_k_accuracy", "keep_only_top_k_accuracy"):
            values = [result["arbitrary_mlp"]["unit_ablation"][index][metric] for result in results]
            row[metric] = dict(zip(("mean", "std"), _mean_std(values)))
        aggregate_result["unit_ablation"].append(row)
    return aggregate_result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(directory / f"{stem}.{extension}", dpi=240)
    plt.close(figure)


def plot_results(results: list[dict[str, Any]], aggregate_result: dict[str, Any], figure_dir: Path) -> None:
    _style()
    colors = {"structured": "#2B6CB0", "arbitrary": "#C53030", "nonlinear": "#D69E2E"}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))

    # A: OLS full-data reconstruction after final joint training.
    axis = axes[0, 0]
    for result in results:
        ols = result["joint_training"]["ols"]
        values = [row["full_r2"] for row in ols["structured"]] + [ols["arbitrary"]["full_r2"]]
        axis.plot(range(11), values, marker="o", linewidth=1, alpha=0.55)
    axis.axvspan(9.65, 10.35, color=colors["arbitrary"], alpha=0.10)
    axis.axhline(0.95, color="0.45", linestyle="--", linewidth=1)
    axis.set_xticks(range(11), [f"r{i}" for i in range(10)] + ["random"], rotation=35)
    axis.set_ylim(0, 1.04)
    axis.set_ylabel("full-data OLS $R^2$")
    axis.set_title("A  Affine reconstruction after joint training")

    # B: full versus affine-only behavior.
    axis = axes[0, 1]
    structured_full = [
        np.mean(result["joint_training"]["full_accuracy"][:10]) for result in results
    ]
    structured_affine = [
        np.mean(result["joint_training"]["affine_accuracy"][:10]) for result in results
    ]
    random_full = [result["joint_training"]["full_accuracy"][10] for result in results]
    random_affine = [result["joint_training"]["affine_accuracy"][10] for result in results]
    random_identity = [
        result["arbitrary_mlp"]["nonlinearity_ablation"][
            "both_gelus_identity_accuracy"
        ]
        for result in results
    ]
    groups = [
        structured_full,
        structured_affine,
        random_full,
        random_affine,
        random_identity,
    ]
    means = [np.mean(group) for group in groups]
    stds = [np.std(group) for group in groups]
    bar_colors = [
        colors["structured"],
        "#90CDF4",
        colors["arbitrary"],
        "#FEB2B2",
        "#F6AD55",
    ]
    axis.bar(range(5), means, yerr=stds, capsize=3, color=bar_colors)
    axis.set_xticks(
        range(5),
        [
            "structured\nfull",
            "structured\nbase",
            "random\nfull",
            "random\nbase",
            "random\nGELU$\\to$I",
        ],
    )
    axis.set_ylim(0.45, 1.02)
    axis.set_ylabel("truth-table accuracy")
    axis.set_title("B  The random relation causally needs both GELUs")

    # C: path magnitudes and gates.
    axis = axes[1, 0]
    relations = aggregate_result["relations"]
    affine = [row["mean_abs_affine"]["mean"] for row in relations]
    nonlinear = [row["mean_abs_nonlinear"]["mean"] for row in relations]
    x = np.arange(11)
    axis.bar(x - 0.18, affine, width=0.36, label="bilinear base", color=colors["structured"])
    axis.bar(x + 0.18, nonlinear, width=0.36, label="GELU MLP", color=colors["nonlinear"])
    axis.set_xticks(x, [f"r{i}" for i in range(10)] + ["random"], rotation=35)
    axis.set_ylabel("mean absolute logit contribution")
    axis.set_title("C  Learned path decomposition")
    axis.legend(frameon=False)

    # D: causal attribute-coordinate flips.
    axis = axes[1, 1]
    target = aggregate_result["attribute_flip"]["targeted_flip_toggles_decision"]
    control = aggregate_result["attribute_flip"]["off_target_flip_preserves_decision"]
    axis.bar(
        [0, 1],
        [target["mean"], control["mean"]],
        yerr=[target["std"], control["std"]],
        capsize=3,
        color=[colors["structured"], "#718096"],
    )
    axis.set_xticks([0, 1], ["flip queried\nattribute", "flip control\nattribute"])
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("expected decision achieved")
    axis.set_title("D  Causal test of the equality circuit")

    figure.tight_layout()
    _save_figure(figure, figure_dir, "goal2_symmetric_lookup_results")


def plot_mechanism(results: list[dict[str, Any]], aggregate_result: dict[str, Any], figure_dir: Path) -> None:
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
    blue, gold, red = "#2B6CB0", "#D69E2E", "#C53030"

    # A: activation patching.
    axis = axes[0, 0]
    names = [
        "affine_branch",
        "pair_sum",
        "pair_abs_difference",
        "pair_product",
        "hidden_layer_1",
        "hidden_layer_2",
        "nonlinear_output",
    ]
    labels = ["base head", "sum", "|diff|", "product", "hidden 1", "hidden 2", "MLP out"]
    means = [aggregate_result["activation_patching"][name]["mean"] for name in names]
    stds = [aggregate_result["activation_patching"][name]["std"] for name in names]
    axis.bar(range(len(names)), means, yerr=stds, capsize=3, color=[blue] + [gold] * 6)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.axhline(1, color="0.5", linewidth=0.8, linestyle="--")
    axis.set_xticks(range(len(names)), labels, rotation=35, ha="right")
    axis.set_ylabel("normalized clean-logit recovery")
    axis.set_title("A  Clean-to-corrupted activation patching")

    # B: linear readability by MLP stage.
    axis = axes[0, 1]
    stage_names = ["raw_pair_features", "pre_gelu_1", "post_gelu_1", "pre_gelu_2", "post_gelu_2"]
    stage_labels = ["pair code", "pre 1", "post 1", "pre 2", "post 2"]
    means = [aggregate_result["probe_accuracy"][name]["mean"] for name in stage_names]
    stds = [aggregate_result["probe_accuracy"][name]["std"] for name in stage_names]
    axis.errorbar(range(5), means, yerr=stds, marker="o", color=red, capsize=3)
    axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    axis.set_xticks(range(5), stage_labels, rotation=25)
    axis.set_ylim(0.45, 1.02)
    axis.set_ylabel("5-fold linear-probe accuracy")
    axis.set_title("B  The random edge label becomes linearly readable")

    # C: unit ablation / sufficiency curve.
    axis = axes[1, 0]
    rows = aggregate_result["unit_ablation"]
    k = [row["k"] for row in rows]
    ablate = [row["mean_ablate_top_k_accuracy"]["mean"] for row in rows]
    keep = [row["keep_only_top_k_accuracy"]["mean"] for row in rows]
    axis.plot(k, ablate, marker="o", label="mean-ablate top $k$", color=red)
    axis.plot(k, keep, marker="s", label="keep only top $k$", color=blue)
    axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    axis.set_xlabel("second-layer units, ranked by patching effect")
    axis.set_ylabel("arbitrary-relation accuracy")
    axis.set_title("C  Distributed nonlinear memory")
    axis.legend(frameon=False)

    # D: sorted single-unit causal patching effects.
    axis = axes[1, 1]
    for layer, color, label in (("unit_effects_layer1", blue, "hidden 1"), ("unit_effects_layer2", gold, "hidden 2")):
        curves = []
        for result in results:
            effects = np.abs(np.asarray(result["arbitrary_mlp"][layer]))
            curves.append(np.sort(effects)[::-1])
        curve = np.mean(curves, axis=0)
        axis.plot(np.arange(1, len(curve) + 1), curve, color=color, label=label)
    axis.set_xlabel("unit rank")
    axis.set_ylabel("absolute normalized patch effect")
    axis.set_title("D  No single lookup neuron")
    axis.legend(frameon=False)

    figure.tight_layout()
    _save_figure(figure, figure_dir, "goal2_symmetric_lookup_mechanism")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--seeds", default="19,31,47")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    device = torch.device(args.device)
    results = [analyze_seed(args.result_dir, seed, device) for seed in seeds]
    aggregate_result = aggregate(results)
    report = {"seeds": seeds, "per_seed": results, "aggregate": aggregate_result}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "mechanistic_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_results(results, aggregate_result, args.figure_dir)
    plot_mechanism(results, aggregate_result, args.figure_dir)
    print(f"wrote {report_path}")
    print(f"wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()

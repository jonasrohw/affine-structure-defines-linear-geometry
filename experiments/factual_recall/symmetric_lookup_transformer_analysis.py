"""Analysis and publication figures for the standard-transformer Goal 2 run.

The behavioral claim and the mechanistic diagnostics are kept separate.  The
primary OLS test uses the model's ten learned attribute log-odds as the entity
representation.  Ablations, logit-lens evaluation, attention summaries, and
residual-stream patching then describe how the transformer computes relation
answers without treating the OLS result as a causal circuit identification.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F

from .config import PROJECT_ROOT
from .model import CausalTransformer, TransformerBlock
from .symmetric_lookup import _positive_centroids, build_world
from .symmetric_lookup_transformer import (
    LookupTokenVocabulary,
    _world_config,
    attribute_accuracy,
    learned_attribute_representations,
    load_config,
    make_model,
    relation_accuracies,
    relation_conditioned_layerwise_ols_summary,
    relation_prompts,
    relation_symmetry_agreements,
)


SEEDS = (19, 31, 47)


def _json_float(value: float | np.floating) -> float:
    return float(value)


def _mean_sd(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _attention(
    block: TransformerBlock,
    x: torch.Tensor,
    uniform: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one attention module and also return its attention probabilities."""
    batch_size, seq_len, hidden_dim = x.shape
    qkv = block.attn.qkv(x)
    q, k, v = qkv.split(hidden_dim, dim=2)

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(
            batch_size, seq_len, block.attn.n_heads, block.attn.head_dim
        ).transpose(1, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
    )
    if uniform:
        weights = causal_mask.to(x.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights[None, None].expand(batch_size, block.attn.n_heads, -1, -1)
    else:
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
    output = weights @ v
    output = output.transpose(1, 2).contiguous().view(
        batch_size, seq_len, hidden_dim
    )
    return block.attn.out_proj(output), weights


def _block_forward(
    block: TransformerBlock,
    x: torch.Tensor,
    *,
    gelu_identity: bool = False,
    no_attention: bool = False,
    uniform_attention: bool = False,
    no_mlp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = block.ln1(x)
    attention_output, weights = _attention(
        block, normalized, uniform=uniform_attention
    )
    if no_attention:
        attention_output = torch.zeros_like(attention_output)
    x = x + attention_output
    hidden = block.mlp.fc1(block.ln2(x))
    if not gelu_identity:
        hidden = F.gelu(hidden)
    mlp_output = block.mlp.fc2(hidden)
    if no_mlp:
        mlp_output = torch.zeros_like(mlp_output)
    return x + mlp_output, weights


@torch.no_grad()
def diagnostic_forward(
    model: CausalTransformer,
    input_ids: torch.Tensor,
    *,
    mode: str = "full",
    ablate_attention_layer: int | None = None,
    ablate_mlp_layer: int | None = None,
    return_cache: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Forward pass with controlled component ablations and residual capture."""
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    x = model.token_emb(input_ids) + model.pos_emb(positions)
    residuals = [x]
    attention_weights: list[torch.Tensor] = []
    for layer, block in enumerate(model.blocks):
        x, weights = _block_forward(
            block,
            x,
            gelu_identity=mode == "gelu_identity",
            no_attention=(mode == "no_attention" or layer == ablate_attention_layer),
            uniform_attention=mode == "uniform_attention",
            no_mlp=(mode == "no_mlp" or layer == ablate_mlp_layer),
        )
        residuals.append(x)
        attention_weights.append(weights)
    logits = model.lm_head(model.ln_f(x))
    if not return_cache:
        residuals = []
        attention_weights = []
    return logits, residuals, attention_weights


@torch.no_grad()
def _finish_from_boundary(
    model: CausalTransformer,
    residual: torch.Tensor,
    boundary: int,
) -> torch.Tensor:
    x = residual
    for block in model.blocks[boundary:]:
        x, _ = _block_forward(block, x)
    return model.lm_head(model.ln_f(x))


def _all_relation_prompts(
    world,
    vocab: LookupTokenVocabulary,
    relation: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    subjects, objects = np.indices((world.n_entities, world.n_entities))
    subjects_t = torch.as_tensor(subjects.reshape(-1), device=device)
    objects_t = torch.as_tensor(objects.reshape(-1), device=device)
    relations_t = torch.full_like(subjects_t, relation)
    values_t = torch.as_tensor(
        world.labels[relation].reshape(-1), dtype=torch.float32, device=device
    )
    return relation_prompts(
        vocab, relations_t, subjects_t, objects_t, values_t
    )


@torch.no_grad()
def binary_relation_accuracies(
    model: CausalTransformer,
    world,
    vocab: LookupTokenVocabulary,
    device: torch.device,
    *,
    mode: str = "full",
    ablate_attention_layer: int | None = None,
    ablate_mlp_layer: int | None = None,
) -> list[float]:
    output: list[float] = []
    for relation in range(world.n_relations):
        prompts, targets = _all_relation_prompts(world, vocab, relation, device)
        logits, _, _ = diagnostic_forward(
            model,
            prompts,
            mode=mode,
            ablate_attention_layer=ablate_attention_layer,
            ablate_mlp_layer=ablate_mlp_layer,
        )
        predictions = (
            logits[:, -1, vocab.yes_id] > logits[:, -1, vocab.no_id]
        )
        expected = targets == vocab.yes_id
        output.append(float((predictions == expected).float().mean().item()))
    return output


@torch.no_grad()
def logit_lens_and_attention(
    model: CausalTransformer,
    world,
    vocab: LookupTokenVocabulary,
    device: torch.device,
) -> tuple[dict[str, list[float]], dict[str, list[list[float]]]]:
    n_boundaries = model.n_layers + 1
    correct = np.zeros((world.n_relations, n_boundaries), dtype=np.float64)
    attention = np.zeros((world.n_relations, model.n_layers, 4), dtype=np.float64)
    for relation in range(world.n_relations):
        prompts, targets = _all_relation_prompts(world, vocab, relation, device)
        _, residuals, weights = diagnostic_forward(
            model, prompts, return_cache=True
        )
        expected = targets == vocab.yes_id
        for boundary, residual in enumerate(residuals):
            logits = model.lm_head(model.ln_f(residual))[:, -1]
            predictions = logits[:, vocab.yes_id] > logits[:, vocab.no_id]
            correct[relation, boundary] = float(
                (predictions == expected).float().mean().item()
            )
        for layer, layer_weights in enumerate(weights):
            # Average over prompts and heads for the final object's query.
            attention[relation, layer] = (
                layer_weights[:, :, -1, :].mean(dim=(0, 1)).cpu().numpy()
            )
    return (
        {
            "structured_mean": correct[: world.n_attributes].mean(axis=0).tolist(),
            "structured_min": correct[: world.n_attributes].min(axis=0).tolist(),
            "arbitrary": correct[world.arbitrary_relation].tolist(),
        },
        {
            "structured_mean": attention[: world.n_attributes].mean(axis=0).tolist(),
            "arbitrary": attention[world.arbitrary_relation].tolist(),
        },
    )


@torch.no_grad()
def causal_trace(
    model: CausalTransformer,
    world,
    vocab: LookupTokenVocabulary,
    device: torch.device,
) -> dict[str, object]:
    """Patch clean subject/object residuals into negative prompts.

    Clean and corrupted prompts share the relation and object.  The clean
    subject is a positive neighbor; the corrupted subject is a negative
    neighbor.  Patching at the subject position tracks when subject identity is
    transferred, while patching at the final object position tracks when the
    pair decision has accumulated there.
    """
    per_relation: list[dict[str, object]] = []
    for relation in range(world.n_relations):
        prompts, targets = _all_relation_prompts(world, vocab, relation, device)
        logits, _, _ = diagnostic_forward(model, prompts)
        prediction = (
            logits[:, -1, vocab.yes_id] > logits[:, -1, vocab.no_id]
        ).reshape(world.n_entities, world.n_entities)
        clean_subjects: list[int] = []
        corrupt_subjects: list[int] = []
        objects: list[int] = []
        for obj in range(world.n_entities):
            positive = np.flatnonzero(
                world.labels[relation, :, obj]
                & prediction[:, obj].detach().cpu().numpy()
            )
            negative = np.flatnonzero(
                ~world.labels[relation, :, obj]
                & ~prediction[:, obj].detach().cpu().numpy()
            )
            if len(positive) and len(negative):
                clean_subjects.append(int(positive[0]))
                corrupt_subjects.append(int(negative[0]))
                objects.append(obj)
        relation_t = torch.full(
            (len(objects),), relation, dtype=torch.long, device=device
        )
        object_t = torch.as_tensor(objects, dtype=torch.long, device=device)
        clean_subject_t = torch.as_tensor(
            clean_subjects, dtype=torch.long, device=device
        )
        corrupt_subject_t = torch.as_tensor(
            corrupt_subjects, dtype=torch.long, device=device
        )
        clean_x, _ = relation_prompts(
            vocab,
            relation_t,
            clean_subject_t,
            object_t,
            torch.ones_like(relation_t, dtype=torch.float32),
        )
        corrupt_x, _ = relation_prompts(
            vocab,
            relation_t,
            corrupt_subject_t,
            object_t,
            torch.zeros_like(relation_t, dtype=torch.float32),
        )
        clean_logits, clean_residuals, _ = diagnostic_forward(
            model, clean_x, return_cache=True
        )
        corrupt_logits, corrupt_residuals, _ = diagnostic_forward(
            model, corrupt_x, return_cache=True
        )
        clean_margin = (
            clean_logits[:, -1, vocab.yes_id]
            - clean_logits[:, -1, vocab.no_id]
        )
        corrupt_margin = (
            corrupt_logits[:, -1, vocab.yes_id]
            - corrupt_logits[:, -1, vocab.no_id]
        )
        denominator = (clean_margin - corrupt_margin).sum().item()
        position_curves: dict[str, list[float]] = {}
        for name, position in (("subject", 2), ("object", 3)):
            curve: list[float] = []
            for boundary in range(model.n_layers + 1):
                patched = corrupt_residuals[boundary].clone()
                patched[:, position] = clean_residuals[boundary][:, position]
                patched_logits = _finish_from_boundary(model, patched, boundary)
                patched_margin = (
                    patched_logits[:, -1, vocab.yes_id]
                    - patched_logits[:, -1, vocab.no_id]
                )
                recovery = (
                    (patched_margin - corrupt_margin).sum().item()
                    / max(denominator, 1e-12)
                )
                curve.append(float(recovery))
            position_curves[name] = curve
        per_relation.append(
            {
                "relation": relation,
                "n_pairs": len(objects),
                **position_curves,
            }
        )

    def group_curve(key: str, rows: list[dict[str, object]]) -> list[float]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return values.mean(axis=0).tolist()

    structured = per_relation[: world.n_attributes]
    arbitrary = [per_relation[world.arbitrary_relation]]
    return {
        "per_relation": per_relation,
        "structured_mean": {
            "subject": group_curve("subject", structured),
            "object": group_curve("object", structured),
        },
        "arbitrary": {
            "subject": group_curve("subject", arbitrary),
            "object": group_curve("object", arbitrary),
        },
    }


def _full_ols_r2(representations: np.ndarray, labels: np.ndarray) -> float:
    x = np.column_stack(
        (representations.astype(np.float64), np.ones(len(representations)))
    )
    y = _positive_centroids(representations.astype(np.float64), labels)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ coefficients
    total = np.square(y - y.mean(axis=0, keepdims=True)).sum()
    residual = np.square(y - prediction).sum()
    return float(1.0 - residual / max(float(total), 1e-12))


def permutation_null(
    representations: np.ndarray,
    labels: np.ndarray,
    seed: int,
    n_permutations: int,
) -> list[float]:
    rng = np.random.default_rng(seed + 80_000)
    output: list[float] = []
    for _ in range(n_permutations):
        permutation = rng.permutation(len(labels))
        permuted = labels[np.ix_(permutation, permutation)]
        output.append(_full_ols_r2(representations, permuted))
    return output


def _attribute_geometry(
    representations: np.ndarray, attributes: np.ndarray
) -> dict[str, object]:
    signed = attributes.astype(np.float64) * 2.0 - 1.0
    correlations = np.empty((signed.shape[1], signed.shape[1]), dtype=np.float64)
    for representation_index in range(signed.shape[1]):
        for attribute_index in range(signed.shape[1]):
            correlations[representation_index, attribute_index] = np.corrcoef(
                representations[:, representation_index], signed[:, attribute_index]
            )[0, 1]
    mask = ~np.eye(signed.shape[1], dtype=bool)
    return {
        "sign_accuracy": float(
            np.mean((representations > 0) == attributes.astype(bool))
        ),
        "correlations": correlations.tolist(),
        "mean_diagonal_correlation": float(np.diag(correlations).mean()),
        "mean_absolute_off_diagonal_correlation": float(
            np.abs(correlations[mask]).mean()
        ),
    }


def _load_recorded_metrics(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def analyze_seed(
    seed: int,
    config_path: Path,
    checkpoint_dir: Path,
    device: torch.device,
    n_permutations: int,
) -> dict[str, object]:
    cfg, _ = load_config(config_path)
    cfg.seed = seed
    world = build_world(_world_config(cfg))
    vocab = LookupTokenVocabulary(cfg.n_entities, cfg.n_attributes)
    model = make_model(cfg, vocab).to(device)
    checkpoint = checkpoint_dir / f"symmetric_lookup_transformer_refined_seed{seed}.pt"
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    result_path = checkpoint.with_suffix(".json")
    recorded = _load_recorded_metrics(result_path)
    recorded_final = recorded["final_metrics"]

    # Validate that the diagnostic implementation exactly matches the model.
    prompts, _ = _all_relation_prompts(world, vocab, 0, device)
    direct = model(prompts[:64])
    manual, _, _ = diagnostic_forward(model, prompts[:64])
    max_forward_error = float((direct - manual).abs().max().item())
    if max_forward_error > 1e-5:
        raise AssertionError(f"manual forward mismatch: {max_forward_error}")

    modes = (
        "full",
        "gelu_identity",
        "uniform_attention",
        "no_mlp",
        "no_attention",
    )
    ablations: dict[str, object] = {}
    for mode in modes:
        accuracies = binary_relation_accuracies(
            model, world, vocab, device, mode=mode
        )
        ablations[mode] = {
            "relation_accuracy": accuracies,
            "structured_mean": float(np.mean(accuracies[: world.n_attributes])),
            "structured_min": float(np.min(accuracies[: world.n_attributes])),
            "arbitrary": accuracies[world.arbitrary_relation],
        }

    layer_ablations: dict[str, list[dict[str, float | int]]] = {
        "attention": [],
        "mlp": [],
    }
    for layer in range(model.n_layers):
        for component in ("attention", "mlp"):
            accuracies = binary_relation_accuracies(
                model,
                world,
                vocab,
                device,
                ablate_attention_layer=layer if component == "attention" else None,
                ablate_mlp_layer=layer if component == "mlp" else None,
            )
            layer_ablations[component].append(
                {
                    "layer": layer + 1,
                    "structured_mean": float(
                        np.mean(accuracies[: world.n_attributes])
                    ),
                    "structured_min": float(
                        np.min(accuracies[: world.n_attributes])
                    ),
                    "arbitrary": accuracies[world.arbitrary_relation],
                }
            )

    logit_lens, attention = logit_lens_and_attention(
        model, world, vocab, device
    )
    trace = causal_trace(model, world, vocab, device)
    conditioned_ols = relation_conditioned_layerwise_ols_summary(
        model, world, vocab, cfg, device
    )
    representations = learned_attribute_representations(
        model, world, vocab, device
    )
    arbitrary_r2 = recorded_final["learned_attribute_space_ols"]["arbitrary"][
        "full_r2"
    ]
    null = permutation_null(
        representations,
        world.labels[world.arbitrary_relation],
        seed,
        n_permutations,
    )
    percentile = float(100.0 * np.mean(np.asarray(null) <= arbitrary_r2))

    return {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "num_parameters": model.n_parameters(),
        "manual_forward_max_abs_error": max_forward_error,
        "recorded_final_metrics": recorded_final,
        "full_vocab_recomputed": {
            "attribute_accuracy": attribute_accuracy(model, world, vocab, device),
            "relation_accuracy": relation_accuracies(
                model, world, vocab, device
            ),
            "symmetry_agreement": relation_symmetry_agreements(
                model, world, vocab, device
            ),
        },
        "attribute_geometry": _attribute_geometry(
            representations, world.attributes
        ),
        "permutation_null": {
            "n": n_permutations,
            "r2": null,
            "observed_r2": arbitrary_r2,
            "observed_percentile": percentile,
        },
        "component_ablations": ablations,
        "layer_ablations": layer_ablations,
        "logit_lens": logit_lens,
        "attention": attention,
        "causal_trace": trace,
        "relation_conditioned_layerwise_ols": conditioned_ols,
    }


def summarize(seed_results: list[dict[str, object]]) -> dict[str, object]:
    final = [row["recorded_final_metrics"] for row in seed_results]
    learned_ols = [row["learned_attribute_space_ols"] for row in final]
    metrics = {
        "attribute_accuracy": _mean_sd(row["attribute_accuracy"] for row in final),
        "min_structured_accuracy": _mean_sd(
            row["min_structured_accuracy"] for row in final
        ),
        "arbitrary_accuracy": _mean_sd(row["arbitrary_accuracy"] for row in final),
        "min_structured_full_r2": _mean_sd(
            row["min_structured_full_r2"] for row in learned_ols
        ),
        "mean_structured_full_r2": _mean_sd(
            row["mean_structured_full_r2"] for row in learned_ols
        ),
        "arbitrary_full_r2": _mean_sd(
            row["arbitrary"]["full_r2"] for row in learned_ols
        ),
        "arbitrary_cv_r2": _mean_sd(
            row["arbitrary"]["cv_r2"] for row in learned_ols
        ),
        "min_symmetry_agreement": _mean_sd(
            min(row["full_vocab_recomputed"]["symmetry_agreement"])
            for row in seed_results
        ),
        "attribute_sign_accuracy": _mean_sd(
            row["attribute_geometry"]["sign_accuracy"] for row in seed_results
        ),
        "arbitrary_null_percentile": _mean_sd(
            row["permutation_null"]["observed_percentile"]
            for row in seed_results
        ),
    }

    ablation_summary: dict[str, object] = {}
    modes = seed_results[0]["component_ablations"].keys()
    for mode in modes:
        ablation_summary[mode] = {
            key: _mean_sd(
                row["component_ablations"][mode][key] for row in seed_results
            )
            for key in ("structured_mean", "structured_min", "arbitrary")
        }

    def curve_summary(path: tuple[str, ...]) -> dict[str, list[float]]:
        arrays = []
        for row in seed_results:
            value = row
            for key in path:
                value = value[key]
            arrays.append(value)
        data = np.asarray(arrays, dtype=np.float64)
        return {
            "mean": data.mean(axis=0).tolist(),
            "sd": data.std(axis=0, ddof=1).tolist(),
        }

    curves = {
        "logit_lens_structured_mean": curve_summary(
            ("logit_lens", "structured_mean")
        ),
        "logit_lens_arbitrary": curve_summary(("logit_lens", "arbitrary")),
        "trace_structured_subject": curve_summary(
            ("causal_trace", "structured_mean", "subject")
        ),
        "trace_structured_object": curve_summary(
            ("causal_trace", "structured_mean", "object")
        ),
        "trace_arbitrary_subject": curve_summary(
            ("causal_trace", "arbitrary", "subject")
        ),
        "trace_arbitrary_object": curve_summary(
            ("causal_trace", "arbitrary", "object")
        ),
    }
    return {
        "n_seeds": len(seed_results),
        "metrics": metrics,
        "component_ablations": ablation_summary,
        "curves": curves,
    }


def _style_axis(axis: plt.Axes, panel: str, title: str) -> None:
    axis.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18, linewidth=0.7)


def plot_results(
    seed_results: list[dict[str, object]], output_stem: Path
) -> None:
    colors = {"structured": "#1f5a7a", "arbitrary": "#b24a3a", "other": "#777777"}
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)

    axis = axes[0, 0]
    categories = ("attributes", "structured\n(min)", "arbitrary", "symmetry\n(min)")
    values = []
    for row in seed_results:
        final = row["recorded_final_metrics"]
        values.append(
            [
                final["attribute_accuracy"],
                final["min_structured_accuracy"],
                final["arbitrary_accuracy"],
                min(row["full_vocab_recomputed"]["symmetry_agreement"]),
            ]
        )
    values_array = np.asarray(values)
    axis.bar(
        np.arange(4),
        values_array.mean(axis=0),
        color=["#4d7c8a", colors["structured"], colors["arbitrary"], "#748c5b"],
        alpha=0.82,
    )
    for seed_index, row in enumerate(values_array):
        axis.scatter(np.arange(4) + (seed_index - 1) * 0.045, row, color="black", s=13, zorder=3)
    axis.axhline(0.995, color="black", linestyle="--", linewidth=0.8)
    axis.set_ylim(0.97, 1.001)
    axis.set_xticks(np.arange(4), categories)
    axis.set_ylabel("full-table accuracy / agreement")
    _style_axis(axis, "A", "All behaviors pass in three seeds")

    axis = axes[0, 1]
    structured_x = np.arange(10)
    for seed_index, row in enumerate(seed_results):
        ols = row["recorded_final_metrics"]["learned_attribute_space_ols"]
        structured = [item["full_r2"] for item in ols["structured"]]
        axis.scatter(
            structured_x + (seed_index - 1) * 0.08,
            structured,
            color=colors["structured"],
            s=18,
            alpha=0.78,
        )
        axis.scatter(
            [10.5 + (seed_index - 1) * 0.08],
            [ols["arbitrary"]["full_r2"]],
            color=colors["arbitrary"],
            s=30,
            zorder=4,
        )
    all_null = np.concatenate(
        [np.asarray(row["permutation_null"]["r2"]) for row in seed_results]
    )
    violin = axis.violinplot(all_null, positions=[12.0], widths=0.65, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(colors["other"])
        body.set_alpha(0.45)
    axis.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axis.set_xticks([0, 2, 4, 6, 8, 10.5, 12.0], ["r0", "r2", "r4", "r6", "r8", "random", "null"])
    axis.set_xlim(-0.6, 12.6)
    axis.set_ylim(-0.02, 1.03)
    axis.set_ylabel(r"attribute-space OLS $R^2$")
    _style_axis(axis, "B", "Affine equality maps, null-scale random map")

    axis = axes[1, 0]
    for row in seed_results:
        seed = row["seed"]
        base_path = PROJECT_ROOT / "factual_recall" / "results" / f"symmetric_lookup_transformer_d24_l6_seed{seed}" / f"symmetric_lookup_transformer_seed{seed}.json"
        base = _load_recorded_metrics(base_path)
        history = base["joint_training"]["history"]
        steps = [item["step"] for item in history]
        axis.plot(steps, [item["min_relation_accuracy"] for item in history], color=colors["arbitrary"], alpha=0.36)
        axis.plot(steps, [item["mean_relation_accuracy"] for item in history], color=colors["structured"], alpha=0.36)
        refined = row["recorded_final_metrics"]
        axis.scatter([8200], [min(refined["relation_accuracy"])], color=colors["arbitrary"], s=16)
        axis.scatter([8200], [np.mean(refined["relation_accuracy"])], color=colors["structured"], s=16)
    axis.plot([], [], color=colors["structured"], label="mean of 11")
    axis.plot([], [], color=colors["arbitrary"], label="minimum of 11")
    axis.axhline(0.995, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("joint-training step")
    axis.set_ylabel("full-table accuracy")
    axis.set_ylim(0.45, 1.01)
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis, "C", "Joint learning retains the structured relations")

    axis = axes[1, 1]
    correlation = np.mean(
        [np.asarray(row["attribute_geometry"]["correlations"]) for row in seed_results],
        axis=0,
    )
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axis.set_xlabel("true attribute")
    axis.set_ylabel("learned log-odds coordinate")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    axis.set_title("D  Explicit attribute prompts define the OLS space", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axis, label="correlation", shrink=0.82)

    for suffix in ("pdf", "png", "svg"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def plot_mechanism(
    seed_results: list[dict[str, object]],
    summary: dict[str, object],
    output_stem: Path,
) -> None:
    structured_color = "#1f5a7a"
    arbitrary_color = "#b24a3a"
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    boundaries = np.arange(7)
    labels = ["input"] + [f"B{i}" for i in range(1, 7)]

    axis = axes[0, 0]
    for key, color, label in (
        ("logit_lens_structured_mean", structured_color, "structured mean"),
        ("logit_lens_arbitrary", arbitrary_color, "random"),
    ):
        curve = summary["curves"][key]
        mean = np.asarray(curve["mean"])
        sd = np.asarray(curve["sd"])
        axis.plot(boundaries, mean, marker="o", color=color, label=label)
        axis.fill_between(boundaries, mean - sd, mean + sd, color=color, alpha=0.13)
    axis.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axis.set_xticks(boundaries, labels)
    axis.set_ylim(0.35, 1.02)
    axis.set_ylabel("binary logit-lens accuracy")
    axis.legend(frameon=False)
    _style_axis(axis, "A", "Relation labels become readable late")

    axis = axes[0, 1]
    modes = ("full", "gelu_identity", "uniform_attention", "no_mlp", "no_attention")
    names = ("full", "GELU to\nidentity", "uniform\nattention", "no MLP", "no attention")
    x = np.arange(len(modes))
    width = 0.36
    structured = [summary["component_ablations"][mode]["structured_mean"]["mean"] for mode in modes]
    arbitrary = [summary["component_ablations"][mode]["arbitrary"]["mean"] for mode in modes]
    axis.bar(x - width / 2, structured, width, color=structured_color, label="structured mean")
    axis.bar(x + width / 2, arbitrary, width, color=arbitrary_color, label="random")
    for seed_index, row in enumerate(seed_results):
        offset = (seed_index - 1) * 0.028
        axis.scatter(x - width / 2 + offset, [row["component_ablations"][mode]["structured_mean"] for mode in modes], color="black", s=8, zorder=3)
        axis.scatter(x + width / 2 + offset, [row["component_ablations"][mode]["arbitrary"] for mode in modes], color="black", s=8, zorder=3)
    axis.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axis.set_xticks(x, names)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("binary-choice accuracy")
    axis.legend(frameon=False, loc="upper right")
    _style_axis(axis, "B", "Both attention and MLP computation matter")

    axis = axes[1, 0]
    for group, color, linestyle in (
        ("structured", structured_color, "-"),
        ("arbitrary", arbitrary_color, "--"),
    ):
        for position, marker in (("subject", "o"), ("object", "s")):
            key = f"trace_{group}_{position}"
            curve = summary["curves"][key]
            mean = np.asarray(curve["mean"])
            sd = np.asarray(curve["sd"])
            axis.plot(boundaries, mean, color=color, linestyle=linestyle, marker=marker, label=f"{group}, {position}")
            axis.fill_between(boundaries, mean - sd, mean + sd, color=color, alpha=0.08)
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.set_xticks(boundaries, labels)
    axis.set_ylabel("normalized logit recovery")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    _style_axis(axis, "C", "Causal tracing follows subject-to-object transfer")

    axis = axes[1, 1]
    structured_attention = np.asarray(
        [row["attention"]["structured_mean"] for row in seed_results]
    ).mean(axis=0)
    arbitrary_attention = np.asarray(
        [row["attention"]["arbitrary"] for row in seed_results]
    ).mean(axis=0)
    positions = ("BOS", "relation", "subject", "object")
    position_colors = ("#b8b8b8", "#8c6bb1", "#4c78a8", "#e28e2c")
    x = np.arange(1, 7)
    width = 0.34
    structured_bottom = np.zeros(6)
    arbitrary_bottom = np.zeros(6)
    for position in range(4):
        axis.bar(
            x - width / 2,
            structured_attention[:, position],
            width,
            bottom=structured_bottom,
            color=position_colors[position],
            alpha=0.55,
            hatch="///",
            edgecolor="white",
            linewidth=0.25,
        )
        axis.bar(
            x + width / 2,
            arbitrary_attention[:, position],
            width,
            bottom=arbitrary_bottom,
            color=position_colors[position],
            edgecolor="white",
            linewidth=0.25,
        )
        structured_bottom += structured_attention[:, position]
        arbitrary_bottom += arbitrary_attention[:, position]
    axis.set_xticks(x, [f"B{i}\nS  R" for i in range(1, 7)])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("attention from final token")
    handles = [Patch(facecolor=color, label=label) for color, label in zip(position_colors, positions, strict=True)]
    handles.extend(
        [
            Patch(facecolor="white", edgecolor="gray", hatch="///", label="S: structured"),
            Patch(facecolor="gray", label="R: random"),
        ]
    )
    axis.legend(handles=handles, frameon=False, ncol=3, fontsize=8, loc="lower center")
    _style_axis(axis, "D", "Attention routes relation and entity information")

    for suffix in ("pdf", "png", "svg"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runs/symmetric_lookup_transformer.toml"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "factual_recall/results/symmetric_lookup_transformer_d24_l6_refined"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "factual_recall/results/symmetric_lookup_transformer_d24_l6_refined/analysis.json"
        ),
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("paper/figures")
    )
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    seed_results = [
        analyze_seed(
            seed,
            args.config,
            args.checkpoint_dir,
            device,
            args.permutations,
        )
        for seed in SEEDS
    ]
    result = {
        "analysis": "standard-transformer Goal 2",
        "config": asdict(load_config(args.config)[0]),
        "seeds": seed_results,
        "summary": summarize(seed_results),
    }
    result["config"]["output_dir"] = str(result["config"]["output_dir"])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plot_results(
        seed_results, args.figure_dir / "goal2_transformer_results"
    )
    plot_mechanism(
        seed_results,
        result["summary"],
        args.figure_dir / "goal2_transformer_mechanism",
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

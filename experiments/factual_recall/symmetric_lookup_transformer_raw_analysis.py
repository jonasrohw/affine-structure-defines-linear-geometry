"""Mechanistic checks for the raw-embedding symmetric lookup transformer.

This analysis deliberately evaluates every relation on the complete table.  It
therefore never asks an affine model to generalize an unobserved random fact.
The transformer and OLS see the same learned subject pairs; the comparison is
between the transformer's behavior and an affine map of its raw entity-token
representations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from .figure_style import COLORS, TEXT_WIDTH, apply_style, save_figure
from .symmetric_lookup import build_world, ols_relation_metrics
from .symmetric_lookup_transformer import (
    LookupTokenVocabulary,
    TransformerLookupConfig,
    _world_config,
    attribute_accuracy,
    make_model,
    relation_accuracies,
    relation_symmetry_agreements,
)
from .symmetric_lookup_transformer_analysis import (
    binary_relation_accuracies,
    causal_trace,
    diagnostic_forward,
    logit_lens_and_attention,
    permutation_null,
    _all_relation_prompts,
)


def candidate_config(seed: int) -> TransformerLookupConfig:
    return TransformerLookupConfig(seed=seed)


def raw_entity_representations(
    model: torch.nn.Module, vocab: LookupTokenVocabulary
) -> np.ndarray:
    start = vocab.entity_offset
    stop = start + vocab.n_entities
    return model.token_emb.weight[start:stop].detach().cpu().numpy()


def full_table_ols(world, representations: np.ndarray, seed: int) -> list[dict[str, float]]:
    return [
        ols_relation_metrics(
            representations,
            world.labels[relation],
            seed=seed,
            n_folds=4,
        )
        for relation in range(world.n_relations)
    ]


def analyze_seed(
    seed: int,
    device: torch.device,
    n_permutations: int,
    checkpoint: Path,
) -> dict[str, object]:
    cfg = candidate_config(seed)
    world = build_world(_world_config(cfg))
    vocab = LookupTokenVocabulary(cfg.n_entities, cfg.n_attributes)
    model = make_model(cfg, vocab).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    prompts, _ = _all_relation_prompts(world, vocab, 0, device)
    direct = model(prompts[:64])
    manual, _, _ = diagnostic_forward(model, prompts[:64])
    max_forward_error = float((direct - manual).abs().max().item())
    if max_forward_error > 1e-5:
        raise AssertionError(f"manual forward mismatch: {max_forward_error}")

    relation_accuracy = relation_accuracies(model, world, vocab, device)
    symmetry = relation_symmetry_agreements(model, world, vocab, device)
    representations = raw_entity_representations(model, vocab)
    ols = full_table_ols(world, representations, seed)
    observed_random_r2 = ols[world.arbitrary_relation]["full_r2"]
    null = permutation_null(
        representations,
        world.labels[world.arbitrary_relation],
        seed,
        n_permutations,
    )

    component_ablations: dict[str, object] = {}
    for mode in (
        "full",
        "gelu_identity",
        "uniform_attention",
        "no_mlp",
        "no_attention",
    ):
        accuracies = binary_relation_accuracies(
            model, world, vocab, device, mode=mode
        )
        component_ablations[mode] = {
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
    return {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "num_parameters": model.n_parameters(),
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "manual_forward_max_abs_error": max_forward_error,
        "behavior": {
            "attribute_accuracy": attribute_accuracy(
                model, world, vocab, device
            ),
            "relation_accuracy": relation_accuracy,
            "min_structured_accuracy": min(
                relation_accuracy[: world.n_attributes]
            ),
            "arbitrary_accuracy": relation_accuracy[world.arbitrary_relation],
            "symmetry_agreement": symmetry,
            "min_symmetry_agreement": min(symmetry),
        },
        "raw_embedding_ols": {
            "structured": ols[: world.n_attributes],
            "arbitrary": ols[world.arbitrary_relation],
            "min_structured_full_r2": min(
                row["full_r2"] for row in ols[: world.n_attributes]
            ),
            "mean_structured_full_r2": float(
                np.mean(
                    [row["full_r2"] for row in ols[: world.n_attributes]]
                )
            ),
        },
        "permutation_null": {
            "n": n_permutations,
            "r2": null,
            "mean": float(np.mean(null)),
            "sd": float(np.std(null, ddof=1)),
            "observed_r2": observed_random_r2,
            "observed_percentile": float(
                100.0 * np.mean(np.asarray(null) <= observed_random_r2)
            ),
        },
        "component_ablations": component_ablations,
        "layer_ablations": layer_ablations,
        "logit_lens": logit_lens,
        "attention": attention,
        "causal_trace": trace,
    }


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    modes = rows[0]["component_ablations"].keys()
    return {
        "n_seeds": len(rows),
        "num_parameters": rows[0]["num_parameters"],
        "attribute_accuracy": mean_sd(
            [row["behavior"]["attribute_accuracy"] for row in rows]
        ),
        "min_structured_accuracy": mean_sd(
            [row["behavior"]["min_structured_accuracy"] for row in rows]
        ),
        "arbitrary_accuracy": mean_sd(
            [row["behavior"]["arbitrary_accuracy"] for row in rows]
        ),
        "min_symmetry_agreement": mean_sd(
            [row["behavior"]["min_symmetry_agreement"] for row in rows]
        ),
        "min_structured_full_r2": mean_sd(
            [row["raw_embedding_ols"]["min_structured_full_r2"] for row in rows]
        ),
        "arbitrary_full_r2": mean_sd(
            [row["raw_embedding_ols"]["arbitrary"]["full_r2"] for row in rows]
        ),
        "arbitrary_cv_r2": mean_sd(
            [row["raw_embedding_ols"]["arbitrary"]["cv_r2"] for row in rows]
        ),
        "arbitrary_null_percentile": mean_sd(
            [row["permutation_null"]["observed_percentile"] for row in rows]
        ),
        "component_ablations": {
            mode: {
                key: mean_sd(
                    [row["component_ablations"][mode][key] for row in rows]
                )
                for key in ("structured_mean", "structured_min", "arbitrary")
            }
            for mode in modes
        },
    }


def _style_axis(axis: plt.Axes, panel: str, title: str) -> None:
    axis.set_title(rf"\textbf{{{panel}  {title}}}", loc="left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y")


def plot_results(rows: list[dict[str, object]], output_stem: Path, histories: dict[int, list]) -> None:
    apply_style()
    structured_color = COLORS["structured"]
    arbitrary_color = COLORS["random"]
    figure, axes = plt.subplots(
        2, 2, figsize=(TEXT_WIDTH, 5.15), constrained_layout=True
    )

    axis = axes[0, 0]
    categories = ("attributes", "structured\n(min)", "arbitrary", "symmetry\n(min)")
    values = np.asarray(
        [
            [
                row["behavior"]["attribute_accuracy"],
                row["behavior"]["min_structured_accuracy"],
                row["behavior"]["arbitrary_accuracy"],
                row["behavior"]["min_symmetry_agreement"],
            ]
            for row in rows
        ]
    )
    axis.bar(
        np.arange(4),
        values.mean(axis=0),
        color=(COLORS["blue"], structured_color, arbitrary_color, COLORS["lavender"]),
        alpha=0.82,
    )
    for seed_index, seed_values in enumerate(values):
        axis.scatter(
            np.arange(4) + (seed_index - 1) * 0.045,
            seed_values,
            color=COLORS["ink"],
            s=13,
            zorder=3,
        )
    axis.axhline(0.995, color=COLORS["ink"], linestyle="--", linewidth=0.8)
    axis.set_ylim(0.99, 1.0005)
    axis.set_xticks(np.arange(4), categories)
    axis.set_ylabel("Accuracy / agreement")
    _style_axis(axis, "A", "Complete-table behavior")

    axis = axes[0, 1]
    for seed_index, row in enumerate(rows):
        ols = row["raw_embedding_ols"]
        structured_full = [item["full_r2"] for item in ols["structured"]]
        structured_cv = [item["cv_r2"] for item in ols["structured"]]
        offset = (seed_index - 1) * 0.07
        axis.scatter(
            np.arange(10) + offset,
            structured_full,
            color=structured_color,
            s=18,
            alpha=0.8,
        )
        axis.scatter(
            np.arange(10) + offset,
            structured_cv,
            facecolors="none",
            edgecolors=structured_color,
            s=24,
            alpha=0.7,
        )
        axis.scatter(
            [10.7 + offset],
            [ols["arbitrary"]["full_r2"]],
            color=arbitrary_color,
            s=32,
            zorder=4,
        )
        axis.scatter(
            [10.7 + offset],
            [ols["arbitrary"]["cv_r2"]],
            facecolors="none",
            edgecolors=arbitrary_color,
            s=38,
            zorder=4,
        )
    axis.axhline(0.95, color=COLORS["ink"], linestyle="--", linewidth=0.8)
    axis.axhline(0.0, color=COLORS["ink"], linewidth=0.6)
    axis.set_xticks([0, 2, 4, 6, 8, 10.7], ["r0", "r2", "r4", "r6", "r8", "random"])
    axis.set_xlim(-0.6, 11.25)
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel(r"raw-embedding OLS $R^2$")
    axis.scatter([], [], color=COLORS["ink"], label="full-data fit")
    axis.scatter([], [], facecolors="none", edgecolors=COLORS["ink"], label="4-fold CV")
    axis.legend(frameon=False, loc="lower left")
    _style_axis(axis, "B", "Affine fit by relation")

    axis = axes[1, 0]
    for row in rows:
        seed = row["seed"]
        history = histories[seed]
        steps = [item["step"] for item in history]
        axis.plot(
            steps,
            [item["arbitrary_accuracy"] for item in history],
            color=arbitrary_color, linestyle="--",
            alpha=0.8,
        )
        axis.plot(
            steps,
            [item["min_structured_accuracy"] for item in history],
            color=structured_color,
            alpha=0.8,
        )
        axis.scatter(
            [history[-1]["step"]],
            [row["behavior"]["arbitrary_accuracy"]],
            color=arbitrary_color,
            s=16,
        )
        axis.scatter(
            [history[-1]["step"]],
            [row["behavior"]["min_structured_accuracy"]],
            color=structured_color,
            s=16,
        )
    axis.plot([], [], color=structured_color, label="structured minimum")
    axis.plot([], [], color=arbitrary_color, linestyle="--", label="random")
    axis.axhline(0.995, color=COLORS["ink"], linestyle="--", linewidth=0.8)
    axis.set_xlabel("joint-training step")
    axis.set_ylabel("complete-table accuracy")
    axis.set_ylim(0.45, 1.01)
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis, "C", "Learning during joint training")

    axis = axes[1, 1]
    full_values = [
        row["raw_embedding_ols"]["arbitrary"]["full_r2"] for row in rows
    ]
    cv_values = [
        row["raw_embedding_ols"]["arbitrary"]["cv_r2"] for row in rows
    ]
    precision_values = [
        row["raw_embedding_ols"]["arbitrary"]["cv_precision_at_degree"]
        for row in rows
    ]
    x = np.arange(len(rows))
    width = 0.24
    axis.bar(x - width, full_values, width, color=arbitrary_color, label=r"full $R^2$")
    axis.bar(x, cv_values, width, color=COLORS["apricot"], hatch="//", label=r"CV $R^2$")
    axis.bar(x + width, precision_values, width, color=COLORS["lavender"], label="CV positive precision")
    axis.axhline(0.5, color=COLORS["ink"], linestyle=":", linewidth=0.8)
    axis.axhline(0.0, color=COLORS["ink"], linewidth=0.6)
    axis.set_xticks(x, [f"seed {row['seed']}" for row in rows])
    axis.set_ylim(-1.05, 0.62)
    axis.set_ylabel("score")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis, "D", "Random-relation fit by seed")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output_stem)
    plt.close(figure)


def _curve_summary(rows: list[dict[str, object]], path: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        arrays.append(value)
    data = np.asarray(arrays, dtype=np.float64)
    return data.mean(axis=0), (data.std(axis=0, ddof=1) if len(data) > 1 else np.zeros_like(data[0]))


def plot_mechanism(rows: list[dict[str, object]], output_stem: Path) -> None:
    apply_style()
    structured_color = COLORS["structured"]
    arbitrary_color = COLORS["random"]
    figure, axes = plt.subplots(
        2, 2, figsize=(TEXT_WIDTH, 5.15), constrained_layout=True
    )
    boundaries = np.arange(9)
    boundary_labels = ["input"] + [f"B{i}" for i in range(1, 9)]

    axis = axes[0, 0]
    for path, color, label in (
        (("logit_lens", "structured_mean"), structured_color, "structured mean"),
        (("logit_lens", "arbitrary"), arbitrary_color, "random"),
    ):
        mean, sd = _curve_summary(rows, path)
        axis.plot(boundaries, mean, marker="s" if label == "random" else "o",
                  linestyle="--" if label == "random" else "-", color=color, label=label)
        axis.fill_between(boundaries, mean - sd, mean + sd, color=color, alpha=0.13)
    axis.axhline(0.5, color=COLORS["ink"], linestyle=":", linewidth=0.8)
    axis.set_xticks(boundaries, boundary_labels)
    axis.set_ylim(0.35, 1.02)
    axis.set_ylabel("binary logit-lens accuracy")
    axis.legend(frameon=False)
    _style_axis(axis, "A", "Pair-label readout")

    axis = axes[0, 1]
    modes = ("full", "gelu_identity", "uniform_attention", "no_mlp", "no_attention")
    names = ("full", "GELU to\nidentity", "uniform\nattention", "no MLP", "no\nattention")
    x = np.arange(len(modes))
    width = 0.36
    structured = [
        np.mean([row["component_ablations"][mode]["structured_mean"] for row in rows])
        for mode in modes
    ]
    arbitrary = [
        np.mean([row["component_ablations"][mode]["arbitrary"] for row in rows])
        for mode in modes
    ]
    axis.bar(x - width / 2, structured, width, color=structured_color, label="structured mean")
    axis.bar(x + width / 2, arbitrary, width, color=arbitrary_color,
             edgecolor=COLORS["muted"], hatch="//", label="random")
    for seed_index, row in enumerate(rows):
        offset = (seed_index - 1) * 0.028
        axis.scatter(
            x - width / 2 + offset,
            [row["component_ablations"][mode]["structured_mean"] for mode in modes],
            color=COLORS["ink"], s=8, zorder=3,
        )
        axis.scatter(
            x + width / 2 + offset,
            [row["component_ablations"][mode]["arbitrary"] for mode in modes],
            color=COLORS["ink"], s=8, zorder=3,
        )
    axis.axhline(0.5, color=COLORS["ink"], linestyle=":", linewidth=0.8)
    axis.set_xticks(x, names)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("binary-choice accuracy")
    axis.legend(frameon=False, loc="upper right")
    _style_axis(axis, "B", "Component interventions")

    axis = axes[1, 0]
    for group, color, linestyle in (
        ("structured_mean", structured_color, "-"),
        ("arbitrary", arbitrary_color, "--"),
    ):
        for position, marker in (("subject", "o"), ("object", "s")):
            mean, sd = _curve_summary(rows, ("causal_trace", group, position))
            label_group = "structured" if group == "structured_mean" else "random"
            axis.plot(
                boundaries, mean, color=color, linestyle=linestyle,
                marker=marker, label=f"{label_group}, {position}",
            )
            axis.fill_between(boundaries, mean - sd, mean + sd, color=color, alpha=0.08)
    axis.axhline(0.0, color=COLORS["ink"], linewidth=0.6)
    axis.set_xticks(boundaries, boundary_labels)
    axis.set_ylabel("normalized yes-logit recovery")
    axis.legend(frameon=False, ncol=2, columnspacing=0.6,
                loc="upper center", bbox_to_anchor=(0.5, -0.15))
    _style_axis(axis, "C", "Subject-to-object integration")

    axis = axes[1, 1]
    layers = np.arange(1, 9)
    for component, color, marker in (
        ("attention", COLORS["lavender"], "o"),
        ("mlp", COLORS["apricot"], "s"),
    ):
        data = np.asarray(
            [
                [item["arbitrary"] for item in row["layer_ablations"][component]]
                for row in rows
            ]
        )
        mean = data.mean(axis=0)
        sd = data.std(axis=0, ddof=1) if len(data) > 1 else np.zeros_like(data[0])
        axis.plot(layers, mean, color=color, marker=marker, label=f"remove one {component}")
        axis.fill_between(layers, mean - sd, mean + sd, color=color, alpha=0.13)
    axis.axhline(0.5, color=COLORS["ink"], linestyle=":", linewidth=0.8)
    axis.set_xticks(layers)
    axis.set_ylim(0.45, 1.02)
    axis.set_xlabel("block")
    axis.set_ylabel("random-relation accuracy")
    axis.legend(frameon=False)
    _style_axis(axis, "D", "Layer interventions")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output_stem)
    plt.close(figure)

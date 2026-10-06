"""Transformation-level affine and Jacobian tests for the final Goal 1/2 models.

The original probe asks whether a map fitted on trained subjects predicts
targets for subjects whose relation facts were never observed.  That is a
useful shared-rule/generalization test, but it does not determine whether the
computation executed by the transformer is affine.

This module instead fixes one relation prompt and tests six explicit continuous
maps implemented by the trained one-layer transformer:

``attention_core``
    normalized subject state -> relation-position attention update
``attention``
    subject residual -> relation-position post-attention residual
``mlp_core``
    normalized post-attention state -> MLP update
``mlp``
    post-attention residual -> post-MLP residual
``readout``
    post-MLP residual -> centered object-logit vector
``full_lookup``
    subject residual -> centered object-logit vector

For each map we report:

* four-fold affine fits on three memorized subjects per target, each evaluated
  on the fourth so every memorized subject is held out once;
* the midpoint/additivity defect between memorized-subject pairs;
* dispersion of exact Jacobians across held-out memorized subjects; and
* accuracy after substituting the cross-validated affine prediction.

The first three quantities test the transformation itself.  The last measures
the weaker property of decision equivalence and is never used as proof that the
original function is affine.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .linear_shift import World
from .linear_shift_mechanistic_analysis import replay_and_verify
from .model import CausalTransformer


TensorMap = Callable[[torch.Tensor], torch.Tensor]
STAGES = (
    "attention_core",
    "attention",
    "mlp_core",
    "mlp",
    "readout",
    "full_lookup",
)


def target_stratified_affine_folds(
    world: World,
    trained_subjects: Sequence[int],
    relation: str,
) -> list[tuple[list[int], list[int]]]:
    """Rotate the held-out member within every four-subject target bucket."""
    by_target: dict[int, list[int]] = {}
    for subject in trained_subjects:
        by_target.setdefault(world.target(relation, subject), []).append(subject)
    bucket_sizes = {len(members) for members in by_target.values()}
    if len(bucket_sizes) != 1:
        raise ValueError(f"Target buckets have unequal sizes: {sorted(bucket_sizes)}")
    bucket_size = next(iter(bucket_sizes))
    if bucket_size < 2:
        raise ValueError("Need at least two memorized subjects per target")

    folds = []
    for heldout_position in range(bucket_size):
        fit: list[int] = []
        evaluation: list[int] = []
        for target in sorted(by_target):
            members = sorted(by_target[target])
            evaluation.append(members[heldout_position])
            fit.extend(
                member
                for position, member in enumerate(members)
                if position != heldout_position
            )
        folds.append((sorted(fit), sorted(evaluation)))
    return folds


def fit_affine_map(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit y = x @ weight + bias by least squares in float64."""
    x64 = x.detach().to(dtype=torch.float64)
    y64 = y.detach().to(dtype=torch.float64)
    design = torch.cat(
        [x64, torch.ones((len(x64), 1), dtype=x64.dtype, device=x64.device)], dim=1
    )
    solution = torch.linalg.lstsq(design, y64).solution
    return solution[:-1], solution[-1]


def apply_affine_map(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return x.to(dtype=weight.dtype) @ weight + bias


def affine_reconstruction_metrics(actual: torch.Tensor, predicted: torch.Tensor) -> dict[str, float]:
    actual64 = actual.detach().to(dtype=torch.float64)
    predicted64 = predicted.detach().to(dtype=torch.float64)
    residual = actual64 - predicted64
    centered = actual64 - actual64.mean(dim=0, keepdim=True)
    ss_res = residual.square().sum()
    ss_tot = centered.square().sum().clamp_min(1e-24)
    row_denom = actual64.norm(dim=1) * predicted64.norm(dim=1)
    cosine = (actual64 * predicted64).sum(dim=1) / row_denom.clamp_min(1e-24)
    return {
        "r2": float((1.0 - ss_res / ss_tot).item()),
        "centered_relative_l2_error": float((residual.norm() / centered.norm().clamp_min(1e-24)).item()),
        "output_relative_l2_error": float((residual.norm() / actual64.norm().clamp_min(1e-24)).item()),
        "mean_cosine": float(cosine.mean().item()),
        "rmse": float(residual.square().mean().sqrt().item()),
        "max_abs_error": float(residual.abs().max().item()),
    }


def evaluate_rows(function: TensorMap, rows: torch.Tensor) -> torch.Tensor:
    return torch.stack([function(row) for row in rows])


def different_target_pairs(
    world: World,
    subjects: Sequence[int],
    relation: str,
) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left in range(len(subjects))
        for right in range(left + 1, len(subjects))
        if world.target(relation, subjects[left]) != world.target(relation, subjects[right])
    ]


def midpoint_metrics(
    function: TensorMap,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> dict[str, float]:
    midpoint_inputs = torch.stack([(inputs[left] + inputs[right]) / 2 for left, right in pairs])
    midpoint_outputs = evaluate_rows(function, midpoint_inputs)
    affine_midpoints = torch.stack([(outputs[left] + outputs[right]) / 2 for left, right in pairs])
    pair_scales = torch.stack([(outputs[left] - outputs[right]) / 2 for left, right in pairs])
    defects = midpoint_outputs - affine_midpoints
    denominator = pair_scales.norm().clamp_min(1e-24)
    per_pair_scale = pair_scales.flatten(1).norm(dim=1).clamp_min(1e-24)
    per_pair_defect = defects.flatten(1).norm(dim=1)
    return {
        "relative_frobenius_defect": float((defects.norm() / denominator).item()),
        "mean_pair_relative_defect": float((per_pair_defect / per_pair_scale).mean().item()),
        "max_pair_relative_defect": float((per_pair_defect / per_pair_scale).max().item()),
        "rmse": float(defects.square().mean().sqrt().item()),
        "max_abs_error": float(defects.abs().max().item()),
    }


def jacobian_metrics(
    function: TensorMap,
    inputs: torch.Tensor,
    affine_weight: torch.Tensor,
) -> dict[str, float]:
    jacobians = torch.stack([torch.func.jacrev(function)(row) for row in inputs])
    jacobians64 = jacobians.detach().to(dtype=torch.float64)
    mean_jacobian = jacobians64.mean(dim=0)
    centered = jacobians64 - mean_jacobian
    affine_jacobian = affine_weight.detach().to(dtype=torch.float64).T
    affine_errors = jacobians64 - affine_jacobian
    flat = jacobians64.flatten(1)
    mean_flat = mean_jacobian.flatten()
    cosine = (flat * mean_flat).sum(dim=1) / (
        flat.norm(dim=1) * mean_flat.norm()
    ).clamp_min(1e-24)
    per_row_relative = centered.flatten(1).norm(dim=1) / flat.norm(dim=1).clamp_min(1e-24)
    centered_inputs = inputs.detach().to(dtype=torch.float64)
    centered_inputs = centered_inputs - centered_inputs.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.linalg.svd(centered_inputs, full_matrices=False)
    rank_threshold = singular_values.max().clamp_min(1e-24) * 1e-6
    tangent_rank = int((singular_values > rank_threshold).sum().item())
    tangent_basis = right_vectors[:tangent_rank].T
    tangent_jacobians = jacobians64 @ tangent_basis
    tangent_mean = tangent_jacobians.mean(dim=0)
    tangent_centered = tangent_jacobians - tangent_mean
    tangent_affine = affine_jacobian @ tangent_basis
    return {
        "mean_relative_dispersion": float(per_row_relative.mean().item()),
        "max_relative_dispersion": float(per_row_relative.max().item()),
        "pooled_relative_dispersion": float((centered.norm() / jacobians64.norm().clamp_min(1e-24)).item()),
        "mean_cosine_to_mean_jacobian": float(cosine.mean().item()),
        "affine_fit_relative_error": float(
            (affine_errors.norm() / jacobians64.norm().clamp_min(1e-24)).item()
        ),
        "empirical_tangent_rank": tangent_rank,
        "tangent_pooled_relative_dispersion": float(
            (tangent_centered.norm() / tangent_jacobians.norm().clamp_min(1e-24)).item()
        ),
        "affine_fit_tangent_relative_error": float(
            (
                (tangent_jacobians - tangent_affine).norm()
                / tangent_jacobians.norm().clamp_min(1e-24)
            ).item()
        ),
        "mean_jacobian_frobenius_norm": float(mean_jacobian.norm().item()),
    }


def centered_object_logits(model: CausalTransformer, residual: torch.Tensor, world: World) -> torch.Tensor:
    logits = model.lm_head(model.ln_f(residual))
    object_logits = logits[world.object_offset : world.f_offset]
    return object_logits - object_logits.mean()


def build_stage_maps(
    model: CausalTransformer,
    world: World,
    relation: str,
    subjects: Sequence[int],
    device: torch.device,
) -> tuple[
    dict[str, TensorMap],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Build the six maps, their on-subject values, and decision bases."""
    relation_token = world.relation_token(relation)
    positions = torch.arange(3, device=device)
    sep_residual = model.token_emb.weight[world.sep] + model.pos_emb(positions[0])
    relation_residual = model.token_emb.weight[relation_token] + model.pos_emb(positions[2])
    subject_tokens = torch.tensor([world.subject(subject) for subject in subjects], device=device)
    subject_residuals = model.token_emb(subject_tokens) + model.pos_emb(positions[1])
    block = model.blocks[0]
    normalized_sep = block.ln1(sep_residual)
    normalized_relation = block.ln1(relation_residual)

    def attention_core(normalized_subject: torch.Tensor) -> torch.Tensor:
        normalized = torch.stack((normalized_sep, normalized_subject, normalized_relation))
        hidden_dim = normalized.shape[-1]
        qkv = block.attn.qkv(normalized)
        q, k, values = qkv.split(hidden_dim, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(3, block.attn.n_heads, block.attn.head_dim).transpose(0, 1)

        q, k, values = split_heads(q), split_heads(k), split_heads(values)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        causal_mask = torch.tril(torch.ones(3, 3, device=scores.device, dtype=torch.bool))
        patterns = F.softmax(scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
        heads = patterns @ values
        joined = heads.transpose(0, 1).contiguous().view(3, hidden_dim)
        return block.attn.out_proj(joined)[-1]

    def attention(subject_residual: torch.Tensor) -> torch.Tensor:
        return relation_residual + attention_core(block.ln1(subject_residual))

    def mlp_core(normalized_post_attention: torch.Tensor) -> torch.Tensor:
        return block.mlp(normalized_post_attention)

    def mlp(post_attention_residual: torch.Tensor) -> torch.Tensor:
        return post_attention_residual + mlp_core(block.ln2(post_attention_residual))

    def readout(post_mlp_residual: torch.Tensor) -> torch.Tensor:
        return centered_object_logits(model, post_mlp_residual, world)

    def full_lookup(subject_residual: torch.Tensor) -> torch.Tensor:
        return readout(mlp(attention(subject_residual)))

    functions: dict[str, TensorMap] = {
        "attention_core": attention_core,
        "attention": attention,
        "mlp_core": mlp_core,
        "mlp": mlp,
        "readout": readout,
        "full_lookup": full_lookup,
    }
    with torch.no_grad():
        normalized_subjects = block.ln1(subject_residuals)
        attention_core_outputs = evaluate_rows(attention_core, normalized_subjects)
        attention_outputs = evaluate_rows(attention, subject_residuals)
        if not torch.allclose(
            attention_outputs,
            relation_residual.unsqueeze(0) + attention_core_outputs,
            atol=2e-5,
            rtol=0.0,
        ):
            raise RuntimeError("Attention core and block decompositions do not match")
        normalized_attention_outputs = block.ln2(attention_outputs)
        mlp_core_outputs = evaluate_rows(mlp_core, normalized_attention_outputs)
        mlp_outputs = evaluate_rows(mlp, attention_outputs)
        readout_outputs = evaluate_rows(readout, mlp_outputs)
        prompt_tokens = torch.tensor(
            [
                [world.sep, world.subject(subject), relation_token]
                for subject in subjects
            ],
            dtype=torch.long,
            device=device,
        )
        model_object_logits = model(prompt_tokens)[:, -1, world.object_offset : world.f_offset]
        model_object_logits = model_object_logits - model_object_logits.mean(dim=1, keepdim=True)
        if not torch.allclose(readout_outputs, model_object_logits, atol=2e-5, rtol=0.0):
            max_error = float((readout_outputs - model_object_logits).abs().max().item())
            raise RuntimeError(f"Stage-map decomposition does not match the model: {max_error}")
    inputs = {
        "attention_core": normalized_subjects,
        "attention": subject_residuals,
        "mlp_core": normalized_attention_outputs,
        "mlp": attention_outputs,
        "readout": mlp_outputs,
        "full_lookup": subject_residuals,
    }
    outputs = {
        "attention_core": attention_core_outputs,
        "attention": attention_outputs,
        "mlp_core": mlp_core_outputs,
        "mlp": mlp_outputs,
        "readout": readout_outputs,
        "full_lookup": readout_outputs,
    }
    decision_bases = {
        "attention_core": relation_residual,
        "mlp_core": attention_outputs,
    }
    return functions, inputs, outputs, decision_bases


def prediction_metrics(
    stage: str,
    predicted: torch.Tensor,
    model: CausalTransformer,
    world: World,
    targets: torch.Tensor,
    decision_base: torch.Tensor | None = None,
) -> dict[str, float]:
    block = model.blocks[0]
    if stage == "attention_core":
        if decision_base is None:
            raise ValueError("attention_core requires the relation residual as its decision base")
        states = decision_base.to(dtype=block.ln2.weight.dtype).unsqueeze(0) + predicted.to(
            dtype=block.ln2.weight.dtype
        )
        states = states + block.mlp(block.ln2(states))
        logits = torch.stack([centered_object_logits(model, state, world) for state in states])
    elif stage == "attention":
        states = predicted.to(dtype=block.ln2.weight.dtype)
        states = states + block.mlp(block.ln2(states))
        logits = torch.stack([centered_object_logits(model, state, world) for state in states])
    elif stage == "mlp_core":
        if decision_base is None:
            raise ValueError("mlp_core requires post-attention residuals as its decision base")
        states = decision_base.to(dtype=model.ln_f.weight.dtype) + predicted.to(
            dtype=model.ln_f.weight.dtype
        )
        logits = torch.stack([centered_object_logits(model, state, world) for state in states])
    elif stage == "mlp":
        states = predicted.to(dtype=model.ln_f.weight.dtype)
        logits = torch.stack([centered_object_logits(model, state, world) for state in states])
    else:
        logits = predicted
    predictions = logits.argmax(dim=1)
    return {
        "target_accuracy": float((predictions.cpu() == targets.cpu()).float().mean().item()),
    }


def analyze_seed(
    seed: int,
    device: torch.device,
    *,
    relation: str = "random",
    occurrences: int = 64,
    jacobian_subject_limit: int | None = None,
) -> dict:
    if relation not in ("random", "semantic"):
        raise ValueError(relation)
    model, world, trained_subjects, replay = replay_and_verify(seed, occurrences, device)
    folds = target_stratified_affine_folds(world, trained_subjects, relation)
    ordered_subjects = sorted(trained_subjects)
    subject_index = {subject: index for index, subject in enumerate(ordered_subjects)}
    fold_indices = [
        (
            [subject_index[subject] for subject in fit_subjects],
            [subject_index[subject] for subject in evaluation_subjects],
        )
        for fit_subjects, evaluation_subjects in folds
    ]
    diagnostic_fit_indices, diagnostic_evaluation_indices = fold_indices[0]
    diagnostic_subjects = folds[0][1]
    functions, inputs, outputs, decision_bases = build_stage_maps(
        model, world, relation, ordered_subjects, device
    )
    pairs = different_target_pairs(world, diagnostic_subjects, relation)
    targets = torch.tensor(
        [world.target(relation, subject) for subject in ordered_subjects], dtype=torch.long
    )

    stage_results = {}
    for stage in STAGES:
        function = functions[stage]
        stage_inputs = inputs[stage]
        stage_outputs = outputs[stage]
        predicted = torch.empty_like(stage_outputs, dtype=torch.float64)
        fold_weights = []
        for fit_indices, evaluation_indices in fold_indices:
            weight, bias = fit_affine_map(stage_inputs[fit_indices], stage_outputs[fit_indices])
            predicted[evaluation_indices] = apply_affine_map(
                stage_inputs[evaluation_indices], weight, bias
            )
            fold_weights.append(weight)
        actual = stage_outputs
        eval_inputs = stage_inputs[diagnostic_evaluation_indices]
        eval_outputs = stage_outputs[diagnostic_evaluation_indices]
        jacobian_inputs = eval_inputs
        if jacobian_subject_limit is not None:
            jacobian_inputs = jacobian_inputs[:jacobian_subject_limit]
        stage_results[stage] = {
            "input_dim": int(stage_inputs.shape[1]),
            "output_dim": int(stage_outputs.shape[1]),
            "affine_folds": len(folds),
            "affine_fit_subjects_per_fold": len(diagnostic_fit_indices),
            "crossvalidated_subjects": len(ordered_subjects),
            "jacobian_subjects": len(jacobian_inputs),
            "affine_reconstruction": affine_reconstruction_metrics(actual, predicted),
            "affine_decision": prediction_metrics(
                stage,
                predicted,
                model,
                world,
                targets,
                (
                    decision_bases[stage]
                    if stage == "mlp_core"
                    else decision_bases.get(stage)
                ),
            ),
            "midpoint": midpoint_metrics(
                function,
                eval_inputs,
                eval_outputs,
                pairs,
            ),
            "jacobian": jacobian_metrics(function, jacobian_inputs, fold_weights[0]),
        }

    return {
        "seed": seed,
        "relation": relation,
        "occurrences_per_triple": occurrences,
        "replay_verified": replay["verified_against_archive"],
        "definition": {
            "attention_core": "normalized subject state -> relation-position attention update",
            "attention": "subject residual -> relation-position post-attention residual",
            "mlp_core": "normalized post-attention state -> MLP update",
            "mlp": "post-attention residual -> post-MLP residual",
            "readout": "post-MLP residual -> centered object-logit contrasts",
            "full_lookup": "subject residual -> centered object-logit contrasts",
            "affine_test": "cross-validated vector reconstruction on disjoint memorized subjects",
            "j_equivalence_test": "constancy of exact Jacobians across disjoint memorized subjects",
            "continuous_domain": "natural continuous extension of the residual stream with separator and the analyzed relation token fixed",
        },
        "split": {
            "folds": len(folds),
            "fit_subjects_per_fold": len(diagnostic_fit_indices),
            "evaluation_subjects_per_fold": len(diagnostic_evaluation_indices),
            "crossvalidated_subjects": len(ordered_subjects),
            "fit_per_target": 3,
            "evaluation_per_target": 1,
        },
        "stages": stage_results,
    }


def mean_sd(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_relation_runs(runs: Sequence[dict]) -> dict:
    fields = {
        "affine_reconstruction": (
            "r2",
            "centered_relative_l2_error",
            "output_relative_l2_error",
            "mean_cosine",
        ),
        "affine_decision": ("target_accuracy",),
        "midpoint": (
            "relative_frobenius_defect",
            "mean_pair_relative_defect",
            "max_pair_relative_defect",
        ),
        "jacobian": (
            "mean_relative_dispersion",
            "max_relative_dispersion",
            "pooled_relative_dispersion",
            "mean_cosine_to_mean_jacobian",
            "affine_fit_relative_error",
            "tangent_pooled_relative_dispersion",
            "affine_fit_tangent_relative_error",
        ),
    }
    result = {}
    for stage in STAGES:
        result[stage] = {}
        for section, names in fields.items():
            result[stage][section] = {
                name: mean_sd([run["stages"][stage][section][name] for run in runs])
                for name in names
            }
    return result


def aggregate_runs(runs: Sequence[dict]) -> dict:
    return {
        relation: aggregate_relation_runs([run for run in runs if run["relation"] == relation])
        for relation in sorted({run["relation"] for run in runs})
    }

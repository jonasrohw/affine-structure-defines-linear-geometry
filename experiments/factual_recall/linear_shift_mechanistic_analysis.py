"""Mechanistic audit of the matched-frequency linear/nonlinear experiment.

The script deterministically retrains the reported F=64 category models, expands
their one transformer block into named intermediate tensors, and measures which
paths and components are causally required for semantic generalization and random
fact memorization. It writes raw JSON plus publication figures.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import PROJECT_ROOT
from .linear_shift import (
    LinearShiftConfig,
    World,
    build_corpus,
    ols_probe,
    train_one_epoch,
)
from .model import CausalTransformer


RELATIONS = ("semantic", "random")
POSITIONS = ("separator", "subject", "relation")
SEMANTIC_COLOR = "#0072B2"
RANDOM_COLOR = "#D55E00"
INK = "#222222"
MUTED = "#666666"
GRID = "#dddddd"


def prompts(world: World, relation: str, subjects: list[int], device: torch.device) -> torch.Tensor:
    relation_id = world.relation_token(relation)
    return torch.tensor(
        [[world.sep, world.subject(i), relation_id] for i in subjects],
        dtype=torch.long,
        device=device,
    )


def forward_cache(model: CausalTransformer, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Expand the single block and retain its exact intermediate tensors."""
    if len(model.blocks) != 1:
        raise ValueError("This audit is defined for the reported one-layer model")
    block = model.blocks[0]
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    resid_pre = model.token_emb(input_ids) + model.pos_emb(positions)

    ln1 = block.ln1(resid_pre)
    hidden_dim = ln1.shape[-1]
    qkv = block.attn.qkv(ln1)
    q, k, v = qkv.split(hidden_dim, dim=-1)

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(
            batch_size,
            seq_len,
            block.attn.n_heads,
            block.attn.head_dim,
        ).transpose(1, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).bool()
    attention = F.softmax(scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
    head_values = attention @ v
    concatenated = head_values.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
    attention_out = block.attn.out_proj(concatenated)
    resid_mid = resid_pre + attention_out

    mlp_input = block.ln2(resid_mid)
    mlp_pre = block.mlp.fc1(mlp_input)
    mlp_act = F.gelu(mlp_pre)
    mlp_out = block.mlp.fc2(mlp_act)
    resid_post = resid_mid + mlp_out
    normalized = model.ln_f(resid_post)
    logits = model.lm_head(normalized)
    return {
        "resid_pre": resid_pre,
        "ln1": ln1,
        "q": q,
        "k": k,
        "v": v,
        "attention": attention,
        "head_values": head_values,
        "attention_out": attention_out,
        "resid_mid": resid_mid,
        "mlp_input": mlp_input,
        "mlp_pre": mlp_pre,
        "mlp_act": mlp_act,
        "mlp_out": mlp_out,
        "resid_post": resid_post,
        "normalized": normalized,
        "logits": logits,
    }


def decode_residual(model: CausalTransformer, residual: torch.Tensor) -> torch.Tensor:
    return model.lm_head(model.ln_f(residual))


def continue_from_attention(
    model: CausalTransformer,
    resid_pre_last: torch.Tensor,
    attention_out_last: torch.Tensor,
    mlp_activations: torch.Tensor | None = None,
    linearize_gelu: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Continue the final-position computation after an attention intervention."""
    block = model.blocks[0]
    resid_mid = resid_pre_last + attention_out_last
    mlp_input = block.ln2(resid_mid)
    mlp_pre = block.mlp.fc1(mlp_input)
    if mlp_activations is not None:
        mlp_act = mlp_activations
    elif linearize_gelu:
        mlp_act = mlp_pre
    else:
        mlp_act = F.gelu(mlp_pre)
    mlp_out = block.mlp.fc2(mlp_act)
    resid_post = resid_mid + mlp_out
    logits = decode_residual(model, resid_post)
    return logits, {
        "resid_mid": resid_mid,
        "mlp_input": mlp_input,
        "mlp_pre": mlp_pre,
        "mlp_act": mlp_act,
        "mlp_out": mlp_out,
        "resid_post": resid_post,
    }


def target_ids(world: World, relation: str, subjects: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [world.object(world.target(relation, i)) for i in subjects],
        dtype=torch.long,
        device=device,
    )


def behavior_metrics(
    logits: torch.Tensor,
    world: World,
    relation: str,
    subjects: list[int],
) -> dict[str, float]:
    """Full-vocabulary recall plus object-restricted rank and margin."""
    device = logits.device
    targets = target_ids(world, relation, subjects, device)
    target_logits = logits.gather(1, targets[:, None]).squeeze(1)
    object_ids = torch.arange(world.object_offset, world.f_offset, device=device)
    object_logits = logits[:, object_ids]
    object_targets = targets - world.object_offset
    masked = object_logits.clone()
    masked.scatter_(1, object_targets[:, None], float("-inf"))
    ranks = 1 + (object_logits > target_logits[:, None]).sum(dim=1)
    return {
        "recall": float((logits.argmax(dim=1) == targets).float().mean().item()),
        "loss": float(F.cross_entropy(logits, targets).item()),
        "object_rank": float(ranks.float().mean().item()),
        "object_margin": float((target_logits - masked.max(dim=1).values).mean().item()),
        "target_logit": float(target_logits.mean().item()),
    }


def head_contributions(model: CausalTransformer, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    """Projected per-head writes; their sum is exactly attention_out."""
    block = model.blocks[0]
    rows = []
    for head in range(block.attn.n_heads):
        start = head * block.attn.head_dim
        end = start + block.attn.head_dim
        weight = block.attn.out_proj.weight[:, start:end]
        rows.append(cache["head_values"][:, head] @ weight.T)
    return torch.stack(rows, dim=1)


def source_contributions(model: CausalTransformer, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    """Projected attention writes split by source position and summed over heads."""
    block = model.blocks[0]
    contributions = []
    query_pos = cache["attention"].shape[-2] - 1
    for source_pos in range(cache["attention"].shape[-1]):
        per_head = []
        for head in range(block.attn.n_heads):
            start = head * block.attn.head_dim
            end = start + block.attn.head_dim
            weight = block.attn.out_proj.weight[:, start:end]
            weighted_value = (
                cache["attention"][:, head, query_pos, source_pos, None]
                * cache["v"][:, head, source_pos]
            )
            per_head.append(weighted_value @ weight.T)
        contributions.append(torch.stack(per_head, dim=0).sum(dim=0))
    return torch.stack(contributions, dim=1)


def attention_last_from_qkv(
    model: CausalTransformer,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the final-position write from explicitly supplied q, k, and v."""
    block = model.blocks[0]
    scores = torch.einsum("bhd,bhpd->bhp", query, keys) / math.sqrt(block.attn.head_dim)
    weights = F.softmax(scores, dim=-1)
    head_values = torch.einsum("bhp,bhpd->bhd", weights, values)
    concatenated = head_values.reshape(head_values.shape[0], -1)
    return block.attn.out_proj(concatenated), weights


def stage_metrics(
    model: CausalTransformer,
    world: World,
    relation: str,
    subjects: list[int],
    cache: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    return {
        "embedding": behavior_metrics(
            decode_residual(model, cache["resid_pre"][:, -1]), world, relation, subjects
        ),
        "after_attention": behavior_metrics(
            decode_residual(model, cache["resid_mid"][:, -1]), world, relation, subjects
        ),
        "after_mlp": behavior_metrics(cache["logits"][:, -1], world, relation, subjects),
    }


def intervention_suite(
    model: CausalTransformer,
    world: World,
    relation: str,
    subjects: list[int],
    cache: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    resid_pre = cache["resid_pre"][:, -1]
    attention_out = cache["attention_out"][:, -1]
    baseline = behavior_metrics(cache["logits"][:, -1], world, relation, subjects)
    no_attention, _ = continue_from_attention(model, resid_pre, torch.zeros_like(attention_out))
    no_mlp = decode_residual(model, cache["resid_mid"][:, -1])
    linearized, _ = continue_from_attention(
        model, resid_pre, attention_out, linearize_gelu=True
    )
    result = {
        "baseline": baseline,
        "zero_attention": behavior_metrics(no_attention, world, relation, subjects),
        "zero_mlp": behavior_metrics(no_mlp, world, relation, subjects),
        "linearize_gelu": behavior_metrics(linearized, world, relation, subjects),
    }

    per_head = head_contributions(model, cache)[:, :, -1]
    for head in range(per_head.shape[1]):
        logits, _ = continue_from_attention(
            model, resid_pre, attention_out - per_head[:, head]
        )
        result[f"ablate_head_{head}"] = behavior_metrics(
            logits, world, relation, subjects
        )

    per_source = source_contributions(model, cache)
    for source_pos, label in enumerate(POSITIONS):
        logits, _ = continue_from_attention(
            model, resid_pre, attention_out - per_source[:, source_pos]
        )
        result[f"ablate_{label}_value_path"] = behavior_metrics(
            logits, world, relation, subjects
        )
    return result


def relation_path_patching(
    model: CausalTransformer,
    world: World,
    subjects: list[int],
    caches: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Swap the relation-dependent direct and attention paths separately."""
    result = {}
    for base_relation, source_relation in (("semantic", "random"), ("random", "semantic")):
        base = caches[base_relation]
        source = caches[source_relation]
        base_resid = base["resid_pre"][:, -1]
        base_attn = base["attention_out"][:, -1]
        source_resid = source["resid_pre"][:, -1]
        source_attn = source["attention_out"][:, -1]
        conditions = {
            "base": (base_resid, base_attn),
            "swap_direct_residual": (source_resid, base_attn),
            "swap_attention_write": (base_resid, source_attn),
            "swap_both": (source_resid, source_attn),
        }
        rows = {}
        for label, (resid, attn) in conditions.items():
            logits, _ = continue_from_attention(model, resid, attn)
            rows[label] = {
                "base_target": behavior_metrics(
                    logits, world, base_relation, subjects
                ),
                "source_target": behavior_metrics(
                    logits, world, source_relation, subjects
                ),
            }
        result[f"{base_relation}_to_{source_relation}"] = rows
    return result


def relation_qkv_patching(
    model: CausalTransformer,
    world: World,
    subjects: list[int],
    caches: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Locate the relation instruction in the query versus its self key/value."""
    result = {}
    for base_relation, source_relation in (("semantic", "random"), ("random", "semantic")):
        base = caches[base_relation]
        source = caches[source_relation]
        base_q = base["q"][:, :, -1]
        base_k = base["k"].clone()
        base_v = base["v"].clone()
        source_q = source["q"][:, :, -1]
        source_k = source["k"][:, :, -1]
        source_v = source["v"][:, :, -1]
        variants: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
            "base": (base_q, base_k, base_v),
            "swap_query": (source_q, base_k, base_v),
        }
        key_swapped = base_k.clone()
        key_swapped[:, :, -1] = source_k
        variants["swap_relation_self_key"] = (base_q, key_swapped, base_v)
        value_swapped = base_v.clone()
        value_swapped[:, :, -1] = source_v
        variants["swap_relation_self_value"] = (base_q, base_k, value_swapped)
        both_k = base_k.clone()
        both_v = base_v.clone()
        both_k[:, :, -1] = source_k
        both_v[:, :, -1] = source_v
        variants["swap_query_and_self_kv"] = (source_q, both_k, both_v)

        rows = {}
        for label, (q, k, v) in variants.items():
            attention_out, _ = attention_last_from_qkv(model, q, k, v)
            logits, _ = continue_from_attention(
                model, base["resid_pre"][:, -1], attention_out
            )
            rows[label] = {
                "base_target": behavior_metrics(logits, world, base_relation, subjects),
                "source_target": behavior_metrics(logits, world, source_relation, subjects),
            }
        result[f"{base_relation}_to_{source_relation}"] = rows
    return result


def subject_kv_patching(
    model: CausalTransformer,
    world: World,
    relation: str,
    subjects: list[int],
    cache: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Move a different subject's key/value into each prompt and track its answer."""
    labels = np.array([world.target(relation, i) for i in subjects])
    source_order = None
    for shift in range(1, len(subjects)):
        candidate = np.roll(np.arange(len(subjects)), shift)
        if np.all(labels[candidate] != labels):
            source_order = candidate
            break
    if source_order is None:
        raise RuntimeError("Could not construct a target-changing subject permutation")
    source_idx = torch.tensor(source_order, dtype=torch.long, device=cache["q"].device)
    query = cache["q"][:, :, -1]
    base_k = cache["k"].clone()
    base_v = cache["v"].clone()
    source_k = base_k[source_idx, :, 1]
    source_v = base_v[source_idx, :, 1]
    variants: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "base": (base_k, base_v),
    }
    key_swapped = base_k.clone()
    key_swapped[:, :, 1] = source_k
    variants["swap_subject_key"] = (key_swapped, base_v)
    value_swapped = base_v.clone()
    value_swapped[:, :, 1] = source_v
    variants["swap_subject_value"] = (base_k, value_swapped)
    both_k = base_k.clone()
    both_v = base_v.clone()
    both_k[:, :, 1] = source_k
    both_v[:, :, 1] = source_v
    variants["swap_subject_key_and_value"] = (both_k, both_v)

    rows = {}
    source_subjects = [subjects[int(i)] for i in source_order]
    for label, (keys, values) in variants.items():
        attention_out, _ = attention_last_from_qkv(model, query, keys, values)
        logits, _ = continue_from_attention(
            model, cache["resid_pre"][:, -1], attention_out
        )
        rows[label] = {
            "base_target": behavior_metrics(logits, world, relation, subjects),
            "source_target": behavior_metrics(logits, world, relation, source_subjects),
        }
    return {"permutation_shift": int(np.where(source_order == 0)[0][0]), "conditions": rows}


def activation_path_patching(
    model: CausalTransformer,
    world: World,
    subjects: list[int],
    caches: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Swap post-GELU coefficients while preserving the base residual paths."""
    result = {}
    for base_relation, source_relation in (("semantic", "random"), ("random", "semantic")):
        base = caches[base_relation]
        source = caches[source_relation]
        logits, _ = continue_from_attention(
            model,
            base["resid_pre"][:, -1],
            base["attention_out"][:, -1],
            mlp_activations=source["mlp_act"][:, -1],
        )
        result[f"{base_relation}_to_{source_relation}"] = {
            "base_target": behavior_metrics(logits, world, base_relation, subjects),
            "source_target": behavior_metrics(logits, world, source_relation, subjects),
        }
    return result


def target_direction_scores(
    model: CausalTransformer,
    world: World,
    relation: str,
    subjects: list[int],
    activations: torch.Tensor,
) -> torch.Tensor:
    """Rank neurons by their mean raw-logit write toward the correct object."""
    object_rows = model.lm_head.weight[world.object_offset : world.f_offset]
    all_mean = object_rows.mean(dim=0)
    targets = torch.tensor(
        [world.target(relation, i) for i in subjects],
        dtype=torch.long,
        device=activations.device,
    )
    target_directions = object_rows[targets] - all_mean
    columns = model.blocks[0].mlp.fc2.weight.T
    alignment = target_directions @ columns.T
    return (activations * alignment).mean(dim=0)


def neuron_ablation_suite(
    model: CausalTransformer,
    world: World,
    subjects: list[int],
    caches: dict[str, dict[str, torch.Tensor]],
) -> dict[str, object]:
    rankings = {}
    for relation in RELATIONS:
        scores = target_direction_scores(
            model,
            world,
            relation,
            subjects,
            caches[relation]["mlp_act"][:, -1],
        )
        rankings[relation] = torch.argsort(scores, descending=True)

    rows = {}
    for ranked_for in RELATIONS:
        rows[ranked_for] = {}
        for k in (8, 16, 32, 64, 128):
            ids = rankings[ranked_for][:k]
            per_eval_relation = {}
            for eval_relation in RELATIONS:
                cache = caches[eval_relation]
                acts = cache["mlp_act"][:, -1].clone()
                acts[:, ids] = 0.0
                logits, _ = continue_from_attention(
                    model,
                    cache["resid_pre"][:, -1],
                    cache["attention_out"][:, -1],
                    mlp_activations=acts,
                )
                per_eval_relation[eval_relation] = behavior_metrics(
                    logits, world, eval_relation, subjects
                )
            rows[ranked_for][str(k)] = per_eval_relation

    top_semantic = set(rankings["semantic"][:32].detach().cpu().tolist())
    top_random = set(rankings["random"][:32].detach().cpu().tolist())
    return {
        "ablations": rows,
        "top32_overlap": len(top_semantic & top_random),
        "top32_semantic": sorted(top_semantic),
        "top32_random": sorted(top_random),
    }


def eta_squared(features: torch.Tensor, labels: np.ndarray) -> np.ndarray:
    """Per-feature fraction of variance explained by a categorical target."""
    values = features.detach().cpu().numpy().astype(np.float64)
    grand = values.mean(axis=0)
    total = np.square(values - grand).sum(axis=0)
    between = np.zeros(values.shape[1], dtype=np.float64)
    for label in np.unique(labels):
        group = values[labels == label]
        between += len(group) * np.square(group.mean(axis=0) - grand)
    return between / np.maximum(total, 1e-12)


def neuron_selectivity(
    world: World,
    subjects: list[int],
    caches: dict[str, dict[str, torch.Tensor]],
) -> dict[str, object]:
    rows = {}
    for relation in RELATIONS:
        labels = np.array([world.target(relation, i) for i in subjects])
        eta = eta_squared(caches[relation]["mlp_act"][:, -1], labels)
        rows[relation] = {
            "mean_eta_squared": float(eta.mean()),
            "median_eta_squared": float(np.median(eta)),
            "neurons_eta_squared_gt_0_5": int((eta > 0.5).sum()),
            "top10_mean_eta_squared": float(np.sort(eta)[-10:].mean()),
        }
    sem_act = caches["semantic"]["mlp_act"][:, -1]
    rnd_act = caches["random"]["mlp_act"][:, -1]
    mean_difference = rnd_act.mean(dim=0) - sem_act.mean(dim=0)
    pooled_std = torch.sqrt(0.5 * sem_act.var(dim=0) + 0.5 * rnd_act.var(dim=0) + 1e-12)
    d = mean_difference / pooled_std
    rows["relation_gating"] = {
        "neurons_abs_cohens_d_gt_1": int((d.abs() > 1).sum().item()),
        "neurons_abs_cohens_d_gt_2": int((d.abs() > 2).sum().item()),
        "median_abs_cohens_d": float(d.abs().median().item()),
    }
    return rows


def representation_geometry(
    model: CausalTransformer,
    world: World,
    trained_subjects: list[int],
    caches: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Within/between cosine evidence for the attribute and random geometries."""
    embeddings = model.token_emb.weight[
        [world.subject(i) for i in trained_subjects]
    ].detach()
    embeddings = F.normalize(embeddings, dim=1)
    similarities = embeddings @ embeddings.T
    eye = torch.eye(len(trained_subjects), dtype=torch.bool, device=embeddings.device)
    sem_labels = torch.tensor(
        [world.target("semantic", i) for i in trained_subjects], device=embeddings.device
    )
    rnd_labels = torch.tensor(
        [world.target("random", i) for i in trained_subjects], device=embeddings.device
    )

    def gap(labels: torch.Tensor) -> tuple[float, float, float]:
        same = (labels[:, None] == labels[None, :]) & ~eye
        different = labels[:, None] != labels[None, :]
        within = similarities[same].mean()
        between = similarities[different].mean()
        return float(within.item()), float(between.item()), float((within - between).item())

    sem_within, sem_between, sem_gap = gap(sem_labels)
    rnd_within, rnd_between, rnd_gap = gap(rnd_labels)
    return {
        "semantic_same_target_cosine": sem_within,
        "semantic_different_target_cosine": sem_between,
        "semantic_cosine_gap": sem_gap,
        "random_same_target_cosine": rnd_within,
        "random_different_target_cosine": rnd_between,
        "random_cosine_gap": rnd_gap,
    }


def trained_affine_cross_validation(
    model: CausalTransformer,
    world: World,
    relation: str,
    subjects: list[int],
) -> dict[str, object]:
    """Four-fold target-stratified OLS within the relation-trained subjects."""
    from sklearn.linear_model import LinearRegression

    embeddings = model.token_emb.weight.detach().cpu().numpy().astype(np.float64)
    unembedding = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    object_rows = unembedding[[world.object(j) for j in range(world.n_objects)]]
    labels = np.array([world.target(relation, i) for i in subjects])
    groups = {label: np.flatnonzero(labels == label).tolist() for label in range(world.n_objects)}
    if {len(indices) for indices in groups.values()} != {4}:
        raise ValueError("The reported balanced training split must have four subjects per target")
    folds = []
    for fold in range(4):
        eval_indices = np.array([groups[label][fold] for label in range(world.n_objects)])
        fit_mask = np.ones(len(subjects), dtype=bool)
        fit_mask[eval_indices] = False
        fit_indices = np.flatnonzero(fit_mask)
        fit_x = embeddings[[world.subject(subjects[i]) for i in fit_indices]]
        fit_y = object_rows[labels[fit_indices]]
        eval_x = embeddings[[world.subject(subjects[i]) for i in eval_indices]]
        prediction = LinearRegression(fit_intercept=True).fit(fit_x, fit_y).predict(eval_x)
        prediction /= np.maximum(np.linalg.norm(prediction, axis=1, keepdims=True), 1e-12)
        candidates = object_rows / np.maximum(np.linalg.norm(object_rows, axis=1, keepdims=True), 1e-12)
        predicted_labels = (prediction @ candidates.T).argmax(axis=1)
        folds.append(float(np.mean(predicted_labels == labels[eval_indices])))
    return {"accuracy": float(np.mean(folds)), "fold_accuracies": folds}


def tensor_norms(cache: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(cache[key][:, -1].norm(dim=1).mean().item())
        for key in ("resid_pre", "attention_out", "resid_mid", "mlp_out", "resid_post")
    }


def example_forward(
    model: CausalTransformer,
    world: World,
    subject: int,
    relation: str,
    cache: dict[str, torch.Tensor],
) -> dict[str, object]:
    logits_by_stage = {
        "embedding": decode_residual(model, cache["resid_pre"][:, -1]),
        "after_attention": decode_residual(model, cache["resid_mid"][:, -1]),
        "after_mlp": cache["logits"][:, -1],
    }
    target = world.object(world.target(relation, subject))
    stages = {}
    for label, logits in logits_by_stage.items():
        object_logits = logits[0, world.object_offset : world.f_offset]
        top = int(object_logits.argmax().item())
        target_rank = int(1 + (object_logits > logits[0, target]).sum().item())
        stages[label] = {
            "top_object": top,
            "target_object": world.target(relation, subject),
            "target_object_rank": target_rank,
            "target_full_vocab_probability": float(logits[0].softmax(dim=0)[target].item()),
        }
    attention = cache["attention"][0, :, -1].detach().cpu().numpy()
    pre = cache["mlp_pre"][0, -1]
    act = cache["mlp_act"][0, -1]
    return {
        "subject": subject,
        "attributes": list(world.subject_attrs[subject]),
        "relation": relation,
        "target_object": world.target(relation, subject),
        "attention_by_head": [
            {POSITIONS[pos]: float(attention[head, pos]) for pos in range(3)}
            for head in range(attention.shape[0])
        ],
        "mlp_positive_pre_activations": int((pre > 0).sum().item()),
        "mlp_activations_gt_0_05": int((act > 0.05).sum().item()),
        "norms": tensor_norms(cache),
        "stages": stages,
    }


def analyze_seed(seed: int, device: torch.device) -> dict[str, object]:
    cfg = LinearShiftConfig(
        seed=seed,
        n_triples=128,
        occurrences=64,
        attr_repeats=128,
    )
    world = World("category", seed)
    sequences, trained_subjects = build_corpus(cfg, world)
    model, steps, final_loss = train_one_epoch(cfg, world, sequences, device)
    model.eval()

    all_subjects = {
        "trained": trained_subjects,
        "heldout": world.heldout_subjects,
    }
    caches_by_split: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    with torch.no_grad():
        for split, subjects in all_subjects.items():
            caches_by_split[split] = {
                relation: forward_cache(model, prompts(world, relation, subjects, device))
                for relation in RELATIONS
            }

    result: dict[str, object] = {
        "seed": seed,
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "train_steps": steps,
        "last_train_loss": final_loss,
        "num_parameters": model.n_parameters(),
        "probe": {
            relation: ols_probe(model, world, relation, "embedding", device)
            for relation in RELATIONS
        },
        "splits": {},
    }
    for split, subjects in all_subjects.items():
        split_result = {}
        for relation in RELATIONS:
            cache = caches_by_split[split][relation]
            split_result[relation] = {
                "stages": stage_metrics(model, world, relation, subjects, cache),
                "interventions": intervention_suite(
                    model, world, relation, subjects, cache
                ),
                "attention": {
                    "mean_final_query_by_head": [
                        {
                            POSITIONS[pos]: float(
                                cache["attention"][:, head, -1, pos].mean().item()
                            )
                            for pos in range(3)
                        }
                        for head in range(model.model_cfg.n_heads)
                    ],
                    "mean_norms": tensor_norms(cache),
                },
            }
        split_result["relation_path_patching"] = relation_path_patching(
            model, world, subjects, caches_by_split[split]
        )
        split_result["relation_qkv_patching"] = relation_qkv_patching(
            model, world, subjects, caches_by_split[split]
        )
        split_result["activation_path_patching"] = activation_path_patching(
            model, world, subjects, caches_by_split[split]
        )
        split_result["subject_kv_patching"] = {
            relation: subject_kv_patching(
                model, world, relation, subjects, caches_by_split[split][relation]
            )
            for relation in RELATIONS
        }
        result["splits"][split] = split_result

    result["neuron_ablations"] = neuron_ablation_suite(
        model, world, trained_subjects, caches_by_split["trained"]
    )
    result["neuron_selectivity"] = neuron_selectivity(
        world, trained_subjects, caches_by_split["trained"]
    )
    result["representation_geometry"] = representation_geometry(
        model, world, trained_subjects, caches_by_split["trained"]
    )
    result["trained_affine_cross_validation"] = {
        relation: trained_affine_cross_validation(
            model, world, relation, trained_subjects
        )
        for relation in RELATIONS
    }

    example_subject = 37
    result["worked_example"] = {}
    with torch.no_grad():
        for relation in RELATIONS:
            example_cache = forward_cache(
                model, prompts(world, relation, [example_subject], device)
            )
            result["worked_example"][relation] = example_forward(
                model, world, example_subject, relation, example_cache
            )

    return result


def aggregate_scalar(rows: list[float]) -> dict[str, float]:
    values = np.asarray(rows, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "values": values.tolist(),
    }


def nested_get(row: dict[str, object], path: tuple[str, ...]) -> float:
    value: object = row
    for key in path:
        value = value[key]  # type: ignore[index]
    return float(value)


def aggregate_results(seed_results: list[dict[str, object]]) -> dict[str, object]:
    paths: dict[str, tuple[str, ...]] = {}
    for split in ("trained", "heldout"):
        for relation in RELATIONS:
            for stage in ("embedding", "after_attention", "after_mlp"):
                for metric in ("recall", "object_margin", "object_rank"):
                    paths[f"{split}.{relation}.stage.{stage}.{metric}"] = (
                        "splits", split, relation, "stages", stage, metric
                    )
            for intervention in (
                "baseline",
                "zero_attention",
                "zero_mlp",
                "linearize_gelu",
                "ablate_separator_value_path",
                "ablate_subject_value_path",
                "ablate_relation_value_path",
                "ablate_head_0",
                "ablate_head_1",
                "ablate_head_2",
                "ablate_head_3",
            ):
                paths[f"{split}.{relation}.intervention.{intervention}.recall"] = (
                    "splits", split, relation, "interventions", intervention, "recall"
                )
            for head in range(4):
                for position in POSITIONS:
                    paths[f"{split}.{relation}.attention.h{head}.{position}"] = (
                        "splits", split, relation, "attention",
                        "mean_final_query_by_head", str(head), position,
                    )
    # Lists need integer indexing, so handle the attention paths separately below.
    aggregate: dict[str, object] = {}
    for label, path in paths.items():
        try:
            values = [nested_get(row, path) for row in seed_results]
        except (KeyError, TypeError):
            if ".attention." not in label:
                raise
            split, relation, _, head_label, position = label.split(".")
            head = int(head_label[1:])
            values = [
                float(row["splits"][split][relation]["attention"]["mean_final_query_by_head"][head][position])
                for row in seed_results
            ]
        aggregate[label] = aggregate_scalar(values)

    for relation in RELATIONS:
        aggregate[f"probe.heldout.{relation}.accuracy"] = aggregate_scalar(
            [float(row["probe"][relation]["probe_accuracy"]) for row in seed_results]
        )
        aggregate[f"probe.trained_cv.{relation}.accuracy"] = aggregate_scalar(
            [
                float(row["trained_affine_cross_validation"][relation]["accuracy"])
                for row in seed_results
            ]
        )

    for direction in ("semantic_to_random", "random_to_semantic"):
        for condition in (
            "base",
            "swap_query",
            "swap_relation_self_key",
            "swap_relation_self_value",
            "swap_query_and_self_kv",
        ):
            for target_kind in ("base_target", "source_target"):
                label = f"qkv_patch.{direction}.{condition}.{target_kind}.recall"
                aggregate[label] = aggregate_scalar(
                    [
                        float(
                            row["splits"]["trained"]["relation_qkv_patching"][direction][condition][target_kind]["recall"]
                        )
                        for row in seed_results
                    ]
                )

    for relation in RELATIONS:
        for condition in (
            "base",
            "swap_subject_key",
            "swap_subject_value",
            "swap_subject_key_and_value",
        ):
            for target_kind in ("base_target", "source_target"):
                label = f"subject_patch.{relation}.{condition}.{target_kind}.recall"
                aggregate[label] = aggregate_scalar(
                    [
                        float(
                            row["splits"]["trained"]["subject_kv_patching"][relation]["conditions"][condition][target_kind]["recall"]
                        )
                        for row in seed_results
                    ]
                )

    for ranked_for in RELATIONS:
        for k in (8, 16, 32, 64, 128):
            for eval_relation in RELATIONS:
                label = f"neuron_ablation.{ranked_for}.top{k}.eval_{eval_relation}.recall"
                values = [
                    float(row["neuron_ablations"]["ablations"][ranked_for][str(k)][eval_relation]["recall"])
                    for row in seed_results
                ]
                aggregate[label] = aggregate_scalar(values)

    for relation in RELATIONS:
        for key in (
            "mean_eta_squared",
            "median_eta_squared",
            "neurons_eta_squared_gt_0_5",
            "top10_mean_eta_squared",
        ):
            aggregate[f"selectivity.{relation}.{key}"] = aggregate_scalar(
                [float(row["neuron_selectivity"][relation][key]) for row in seed_results]
            )
    for key in seed_results[0]["representation_geometry"]:
        aggregate[f"geometry.{key}"] = aggregate_scalar(
            [float(row["representation_geometry"][key]) for row in seed_results]
        )
    return aggregate


def format_mean_sd(aggregate: dict[str, object], key: str, digits: int = 2) -> str:
    row = aggregate[key]
    return f"{row['mean']:.{digits}f} +/- {row['sd']:.{digits}f}"


def draw_forward_figure(
    seed_result: dict[str, object], aggregate: dict[str, object], output_path: Path
) -> None:
    """A detailed, data-annotated diagram of the exact one-block computation."""
    fig = plt.figure(figsize=(13.4, 8.0), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str = "#f7f7f7", lw: float = 1.0, fontsize: float = 9) -> None:
        patch = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=INK, linewidth=lw)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = INK, lw: float = 1.2) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": color, "lw": lw})

    ax.text(0.35, 7.70, "Exact forward pass for prompt  x  a_37  r", fontsize=16, weight="bold", color=INK)
    ax.text(0.35, 7.37, "One layer, d=64, four 16-dimensional heads, 256 GELU units, untied 265-way unembedding", fontsize=9.5, color=MUTED)

    # Shared computational spine.
    token_x = [0.5, 1.75, 3.0]
    for x, label in zip(token_x, ("x\nposition 0", "a_37\nposition 1", "r_0 or r_1\nposition 2")):
        box(x, 6.35, 1.0, 0.62, label, "#ffffff", fontsize=8.5)
        arrow(x + 0.5, 6.35, x + 0.5, 6.03)
        box(x, 5.42, 1.0, 0.60, "token emb.\n+ position emb.", "#eeeeee", fontsize=8)

    box(4.55, 5.36, 1.30, 0.72, "LN1\nper position", "#eeeeee")
    for x in token_x:
        arrow(x + 1.0, 5.72, 4.55, 5.72, lw=0.8)
    arrow(5.85, 5.72, 6.20, 5.72)
    box(6.20, 5.24, 2.42, 0.98, "4-head causal attention\nq,k,v = W_QKV LN1(z)\nalpha = softmax(qk^T / sqrt(16))\nA = sum_h W_O^h sum_p alpha_hp v_hp", "#d9edf7", fontsize=8.2)
    arrow(8.62, 5.72, 9.00, 5.72)
    box(9.00, 5.36, 1.18, 0.72, "residual add\nz_A = z_0 + A", "#eeeeee", fontsize=8.5)
    arrow(10.18, 5.72, 10.52, 5.72)
    box(10.52, 5.36, 0.70, 0.72, "LN2", "#eeeeee")
    arrow(11.22, 5.72, 11.55, 5.72)
    box(11.55, 5.16, 1.35, 1.12, "MLP dictionary\nu = W_1 LN2(z_A)+b_1\na = GELU(u)  [256]\nM = W_2 a+b_2", "#fce8d5", fontsize=7.8)

    # Residual and readout below the MLP.
    arrow(12.22, 5.16, 12.22, 4.68)
    box(11.50, 4.02, 1.45, 0.65, "residual add\nz_1 = z_A + M", "#eeeeee", fontsize=8.5)
    arrow(11.50, 4.34, 10.95, 4.34)
    box(9.72, 4.02, 1.23, 0.65, "final LN", "#eeeeee")
    arrow(9.72, 4.34, 9.18, 4.34)
    box(7.60, 4.02, 1.58, 0.65, "logits = U LN_f(z_1)\nargmax over 265 tokens", "#eeeeee", fontsize=8.2)
    ax.text(6.18, 5.00, "relation route: swapping q switches the answered task (0.98 recall)", fontsize=7.8, color="#245B78")
    ax.text(6.18, 4.77, "entity route: swapping the subject v switches the answer (1.00 / 0.995)", fontsize=7.8, color="#245B78")

    # Relation-specific explanations.
    semantic_example = seed_result["worked_example"]["semantic"]
    random_example = seed_result["worked_example"]["random"]
    ax.plot([0.35, 13.05], [3.62, 3.62], color=GRID, lw=1)
    box(0.45, 1.05, 5.95, 2.25, "", "#f2f8fc", lw=1.2)
    ax.text(0.72, 3.03, "r_0: semantic / linear route", fontsize=12, weight="bold", color=SEMANTIC_COLOR)
    ax.text(0.72, 2.68, "a_37 carries f_5 geometry learned from attribute sentences.", fontsize=8.3, color=INK)
    ax.text(0.72, 2.37, "The r_0 query selects a semantic attention readout; the subject value", fontsize=8.3, color=INK)
    ax.text(0.72, 2.16, "supplies its content. The MLP increases the b_5 margin.", fontsize=8.3, color=INK)
    sem_stage = semantic_example["stages"]
    ax.text(0.72, 1.75, f"Worked pass: object rank {sem_stage['embedding']['target_object_rank']} -> {sem_stage['after_attention']['target_object_rank']} -> {sem_stage['after_mlp']['target_object_rank']}; final P(b_5)={sem_stage['after_mlp']['target_full_vocab_probability']:.3f}", fontsize=8.3, color=SEMANTIC_COLOR)
    ax.text(0.72, 1.40, "Held-out full-pass recall: " + format_mean_sd(aggregate, "heldout.semantic.stage.after_mlp.recall"), fontsize=8.3, color=INK)
    ax.text(0.72, 1.19, "Held-out embedding affine probe: 0.89 +/- 0.06", fontsize=8.3, color=INK)

    box(6.95, 1.05, 5.95, 2.25, "", "#fff5ed", lw=1.2)
    ax.text(7.22, 3.03, "r_1: random / memorized route", fontsize=12, weight="bold", color=RANDOM_COLOR)
    ax.text(7.22, 2.68, "a_37 has no attribute rule for b_27; repeated triples reshape its embedding.", fontsize=8.3, color=INK)
    ax.text(7.22, 2.37, "The r_1 query selects a different attention readout of the subject value;", fontsize=8.3, color=INK)
    ax.text(7.22, 2.16, "the MLP increases the b_27 margin. Unseen assignments remain unknowable.", fontsize=8.3, color=INK)
    rnd_stage = random_example["stages"]
    ax.text(7.22, 1.82, f"Worked pass: object rank {rnd_stage['embedding']['target_object_rank']} -> {rnd_stage['after_attention']['target_object_rank']} -> {rnd_stage['after_mlp']['target_object_rank']}; final P(b_27)={rnd_stage['after_mlp']['target_full_vocab_probability']:.3f}", fontsize=8.3, color=RANDOM_COLOR)
    ax.text(7.22, 1.53, "Train recall: " + format_mean_sd(aggregate, "trained.random.stage.after_mlp.recall"), fontsize=8.3, color=INK)
    ax.text(7.22, 1.32, "Trained-subject affine CV: " + format_mean_sd(aggregate, "probe.trained_cv.random.accuracy"), fontsize=8.3, color=INK)
    ax.text(7.22, 1.11, "Probe on unseen random assignments: 0.02 +/- 0.02", fontsize=8.3, color=INK)

    ax.text(0.45, 0.58, "Causal reading: query/value patches and component ablations recompute every downstream operation. Arrows describe the measured circuit, not a post-hoc classifier.", fontsize=8.3, color=MUTED)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_evidence_figure(aggregate: dict[str, object], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.65), dpi=180)

    # A: stage-wise recall.
    ax = axes[0]
    stages = ("embedding", "after_attention", "after_mlp")
    x = np.arange(len(stages))
    series = (
        ("heldout", "semantic", SEMANTIC_COLOR, "semantic held-out"),
        ("trained", "random", RANDOM_COLOR, "random trained"),
        ("heldout", "random", MUTED, "random held-out"),
    )
    for split, relation, color, label in series:
        means = [aggregate[f"{split}.{relation}.stage.{stage}.recall"]["mean"] for stage in stages]
        sds = [aggregate[f"{split}.{relation}.stage.{stage}.recall"]["sd"] for stage in stages]
        ax.errorbar(x, means, yerr=sds, marker="o", capsize=3, lw=2, color=color, label=label)
    ax.set_xticks(x, ("relation\nembedding", "+ attention", "+ GELU MLP"))
    ax.set_ylabel("full-vocabulary top-1 recall")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("(a) where the answer appears", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    # B: causal ablations.
    ax = axes[1]
    interventions = ("baseline", "zero_attention", "zero_mlp", "linearize_gelu", "ablate_subject_value_path")
    labels = ("baseline", "zero\nattention", "zero\nMLP", "replace GELU\nby identity", "remove subject\nvalue path")
    width = 0.36
    x = np.arange(len(interventions))
    for offset, relation, split, color, label in (
        (-width / 2, "semantic", "heldout", SEMANTIC_COLOR, "semantic held-out"),
        (width / 2, "random", "trained", RANDOM_COLOR, "random trained"),
    ):
        means = [aggregate[f"{split}.{relation}.intervention.{name}.recall"]["mean"] for name in interventions]
        sds = [aggregate[f"{split}.{relation}.intervention.{name}.recall"]["sd"] for name in interventions]
        ax.bar(x + offset, means, width, yerr=sds, capsize=2, color=color, label=label)
    ax.set_xticks(x, labels, fontsize=7.2)
    ax.set_ylim(0, 1.08)
    ax.set_title("(b) causal component tests", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # C: distinguish trained-fact decodability from unseen-target transfer.
    ax = axes[2]
    x = np.arange(2)
    width = 0.34
    for offset, probe_kind, label in (
        (-width / 2, "trained_cv", "trained-subject CV"),
        (width / 2, "heldout", "held-out subjects"),
    ):
        means = [aggregate[f"probe.{probe_kind}.{relation}.accuracy"]["mean"] for relation in RELATIONS]
        sds = [aggregate[f"probe.{probe_kind}.{relation}.accuracy"]["sd"] for relation in RELATIONS]
        colors = [SEMANTIC_COLOR, RANDOM_COLOR]
        bars = ax.bar(x + offset, means, width, yerr=sds, capsize=3, color=colors, alpha=1.0 if probe_kind == "trained_cv" else 0.45, label=label)
        if probe_kind == "heldout":
            for bar in bars:
                bar.set_hatch("//")
    ax.axhline(1 / 32, color=MUTED, ls=":", lw=1)
    ax.set_xticks(x, ("semantic", "random"))
    ax.set_ylabel("affine nearest-object accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("(c) the held-out probe tests transfer", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    for ax in axes:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="19,31,47")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factual_recall/results/final_mechanistic_analysis"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("paper/figures"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    figure_dir = args.figure_dir if args.figure_dir.is_absolute() else PROJECT_ROOT / args.figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    seed_results = []
    for seed in [int(value) for value in args.seeds.split(",")]:
        print(f"training and auditing seed {seed}", flush=True)
        seed_results.append(analyze_seed(seed, device))

    aggregate = aggregate_results(seed_results)
    payload = {
        "method": {
            "seeds": [row["seed"] for row in seed_results],
            "checkpoint_reconstruction": "deterministic retraining with the reported F=64 configuration",
            "intervention_semantics": "remove or replace a named final-position write, then recompute every downstream operation",
        },
        "seeds": seed_results,
        "aggregate": aggregate,
    }
    json_path = output_dir / "mechanistic_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figure_paths = {
        "forward_pdf": figure_dir / "linear_vs_nonlinear_forward_pass.pdf",
        "forward_png": figure_dir / "linear_vs_nonlinear_forward_pass.png",
        "evidence_pdf": figure_dir / "linear_vs_nonlinear_mechanistic_evidence.pdf",
        "evidence_png": figure_dir / "linear_vs_nonlinear_mechanistic_evidence.png",
    }
    draw_forward_figure(seed_results[0], aggregate, figure_paths["forward_pdf"])
    draw_forward_figure(seed_results[0], aggregate, figure_paths["forward_png"])
    draw_evidence_figure(aggregate, figure_paths["evidence_pdf"])
    draw_evidence_figure(aggregate, figure_paths["evidence_png"])
    print(json.dumps({"results": str(json_path), **{key: str(value) for key, value in figure_paths.items()}}, indent=2))


if __name__ == "__main__":
    main()

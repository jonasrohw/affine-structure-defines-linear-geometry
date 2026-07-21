"""Causal circuit analysis for the two final linear-shift claims.

This module deterministically replays the archived Goal 1 (F=16) and Goal 2
(F=64) runs, verifies the replay against the stored behavioral metrics, and
then analyzes the actual one-layer forward pass.  The analysis deliberately
combines descriptive decoding with causal interventions:

* a stage-wise logit lens over input, post-attention, and post-MLP residuals;
* exhaustive interventions over all 2^4 attention-head subsets;
* discovery/evaluation-split MLP-neuron ablation and sufficiency curves;
* clean/corrupt activation patching at subject-value, head, and neuron sites;
* attribution patching, validated against exact activation patching; and
* class-geometry tests on the values transported by attention.

The model is small enough that exact patching is preferable to relying on
probes alone.  All reported component names (H0--H3 and N0--N255) refer to
literal units in ``CausalTransformer.blocks[0]``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import PROJECT_ROOT
from .linear_shift import (
    LinearShiftConfig,
    World,
    behavioral_check,
    build_corpus,
    ols_probe,
    train_one_epoch,
)
from .model import CausalTransformer


GOALS = {
    "goal1": {"occurrences": 16, "relation": "semantic"},
    "goal2": {"occurrences": 64, "relation": "random"},
}
TOKENS = ("separator", "subject", "relation")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) != len(values):
        for group, count in enumerate(counts):
            if count > 1:
                ids = np.flatnonzero(inverse == group)
                ranks[ids] = ranks[ids].mean()
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def prompt_ids(world: World, subjects: Sequence[int], relation: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[world.sep, world.subject(subject), world.relation_token(relation)] for subject in subjects],
        dtype=torch.long,
        device=device,
    )


def target_token_ids(world: World, subjects: Sequence[int], relation: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [world.object(world.target(relation, subject)) for subject in subjects],
        dtype=torch.long,
        device=device,
    )


def object_metrics(logits: torch.Tensor, target_tokens: torch.Tensor, world: World) -> dict[str, float]:
    """Metrics at the final prompt position, both full-vocabulary and object-only."""
    final = logits[:, -1, :]
    target = final.gather(1, target_tokens[:, None]).squeeze(1)
    object_token_ids = torch.arange(world.object_offset, world.f_offset, device=final.device)
    object_logits = final.index_select(1, object_token_ids)
    local_targets = target_tokens - world.object_offset
    masked = object_logits.clone()
    masked.scatter_(1, local_targets[:, None], float("-inf"))
    best_wrong = masked.max(dim=1).values
    ranks = 1 + (object_logits > target[:, None]).sum(dim=1)
    return {
        "full_vocab_accuracy": float((final.argmax(dim=1) == target_tokens).float().mean().item()),
        "object_accuracy": float((object_logits.argmax(dim=1) == local_targets).float().mean().item()),
        "object_mean_rank": float(ranks.float().mean().item()),
        "mean_target_logit": float(target.mean().item()),
        "mean_object_margin": float((target - best_wrong).mean().item()),
        "mean_object_probability": float(
            F.softmax(object_logits, dim=1).gather(1, local_targets[:, None]).mean().item()
        ),
        "cross_entropy": float(F.cross_entropy(final, target_tokens).item()),
    }


def margin_vector(logits: torch.Tensor, positive_tokens: torch.Tensor, negative_tokens: torch.Tensor) -> torch.Tensor:
    final = logits[:, -1, :]
    positive = final.gather(1, positive_tokens[:, None]).squeeze(1)
    negative = final.gather(1, negative_tokens[:, None]).squeeze(1)
    return positive - negative


def manual_forward(
    model: CausalTransformer,
    input_ids: torch.Tensor | None = None,
    *,
    x0: torch.Tensor | None = None,
    retain_intermediate_grads: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact one-layer forward pass with named, intervention-ready compute units."""
    if model.n_layers != 1:
        raise ValueError("The final Goal 1/2 experiments use exactly one layer")
    if (input_ids is None) == (x0 is None):
        raise ValueError("Supply exactly one of input_ids or x0")
    block = model.blocks[0]
    if x0 is None:
        assert input_ids is not None
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x0 = model.token_emb(input_ids) + model.pos_emb(positions)
    else:
        seq_len = x0.shape[1]

    ln1 = block.ln1(x0)
    batch_size, _, hidden_dim = ln1.shape
    qkv = block.attn.qkv(ln1)
    q, k, v_flat = qkv.split(hidden_dim, dim=2)

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)

    q, k, values = split_heads(q), split_heads(k), split_heads(v_flat)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x0.device, dtype=torch.bool))
    attention = F.softmax(scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
    heads = attention @ values
    if retain_intermediate_grads:
        heads.retain_grad()
    joined = heads.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
    attn_out = block.attn.out_proj(joined)
    x_attn = x0 + attn_out
    ln2 = block.ln2(x_attn)
    mlp_pre = block.mlp.fc1(ln2)
    mlp_act = F.gelu(mlp_pre)
    if retain_intermediate_grads:
        mlp_act.retain_grad()
    mlp_out = block.mlp.fc2(mlp_act)
    x_mlp = x_attn + mlp_out
    logits = model.lm_head(model.ln_f(x_mlp))
    cache = {
        "x0": x0,
        "ln1": ln1,
        "q": q,
        "k": k,
        "values": values,
        "attention": attention,
        "heads": heads,
        "attn_out": attn_out,
        "x_attn": x_attn,
        "ln2": ln2,
        "mlp_pre": mlp_pre,
        "mlp_act": mlp_act,
        "mlp_out": mlp_out,
        "x_mlp": x_mlp,
    }
    return logits, cache


def forward_from_heads(
    model: CausalTransformer,
    cache: dict[str, torch.Tensor],
    heads: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    block = model.blocks[0]
    batch_size, _, seq_len, _ = heads.shape
    joined = heads.transpose(1, 2).contiguous().view(batch_size, seq_len, model.model_cfg.hidden_dim)
    attn_out = block.attn.out_proj(joined)
    x_attn = cache["x0"] + attn_out
    ln2 = block.ln2(x_attn)
    mlp_pre = block.mlp.fc1(ln2)
    mlp_act = F.gelu(mlp_pre)
    mlp_out = block.mlp.fc2(mlp_act)
    x_mlp = x_attn + mlp_out
    logits = model.lm_head(model.ln_f(x_mlp))
    return logits, {"x_attn": x_attn, "mlp_act": mlp_act, "x_mlp": x_mlp}


def forward_from_mlp_acts(
    model: CausalTransformer,
    cache: dict[str, torch.Tensor],
    mlp_act: torch.Tensor,
) -> torch.Tensor:
    mlp_out = model.blocks[0].mlp.fc2(mlp_act)
    x_mlp = cache["x_attn"] + mlp_out
    return model.lm_head(model.ln_f(x_mlp))


def logit_lens(model: CausalTransformer, cache: dict[str, torch.Tensor], targets: torch.Tensor, world: World) -> dict:
    result = {}
    for label, residual in [
        ("input_relation_residual", cache["x0"]),
        ("post_attention_residual", cache["x_attn"]),
        ("post_mlp_residual", cache["x_mlp"]),
    ]:
        logits = model.lm_head(model.ln_f(residual))
        result[label] = object_metrics(logits, targets, world)
    return result


def replay_and_verify(
    seed: int,
    occurrences: int,
    device: torch.device,
) -> tuple[CausalTransformer, World, list[int], dict]:
    cfg = LinearShiftConfig(
        world="category",
        probe="embedding",
        seed=seed,
        n_triples=128,
        occurrences=occurrences,
        attr_repeats=128,
        output_dir=Path("unused"),
    )
    world = World(cfg.world, cfg.seed, subjects_per_bucket=cfg.subjects_per_bucket, comp_nf=cfg.comp_nf)
    sequences, trained_subjects = build_corpus(cfg, world)
    model, steps, last_loss = train_one_epoch(cfg, world, sequences, device)
    model.eval()

    replay = {"train_steps": steps, "last_train_loss": last_loss, "relations": {}}
    for relation in ("semantic", "random"):
        replay["relations"][relation] = {
            **ols_probe(model, world, relation, "embedding", device),
            "trained_check": behavioral_check(model, world, relation, trained_subjects, device),
            "heldout_check": behavioral_check(model, world, relation, world.heldout_subjects, device),
        }
    reference_path = PROJECT_ROOT / f"factual_recall/results/final_category_seed{seed}/linear_shift_metrics.json"
    rows = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = next(
        row
        for row in rows
        if row["config"]["n_triples"] == 128 and row["config"]["occurrences"] == occurrences
    )
    comparisons = {
        "last_train_loss_abs_error": abs(last_loss - reference["last_train_loss"]),
        "semantic_probe_abs_error": abs(
            replay["relations"]["semantic"]["probe_accuracy"]
            - reference["relations"]["semantic"]["probe_accuracy"]
        ),
        "semantic_train_recall_abs_error": abs(
            replay["relations"]["semantic"]["trained_check"]["recall"]
            - reference["relations"]["semantic"]["trained_check"]["recall"]
        ),
        "semantic_heldout_recall_abs_error": abs(
            replay["relations"]["semantic"]["heldout_check"]["recall"]
            - reference["relations"]["semantic"]["heldout_check"]["recall"]
        ),
        "random_probe_abs_error": abs(
            replay["relations"]["random"]["probe_accuracy"]
            - reference["relations"]["random"]["probe_accuracy"]
        ),
        "random_train_recall_abs_error": abs(
            replay["relations"]["random"]["trained_check"]["recall"]
            - reference["relations"]["random"]["trained_check"]["recall"]
        ),
        "random_heldout_recall_abs_error": abs(
            replay["relations"]["random"]["heldout_check"]["recall"]
            - reference["relations"]["random"]["heldout_check"]["recall"]
        ),
    }
    verified = all(error < 1e-7 for error in comparisons.values())
    if not verified:
        raise RuntimeError(f"Replay for seed={seed}, F={occurrences} did not match: {comparisons}")
    return model, world, trained_subjects, {
        "verified_exactly": verified,
        "reference_path": str(reference_path.relative_to(PROJECT_ROOT)),
        "errors": comparisons,
        "replay": replay,
    }


def split_discovery_evaluation(
    world: World,
    trained_subjects: Sequence[int],
    relation: str,
    goal: str,
) -> tuple[list[int], list[int]]:
    """Use true held-out subjects for Goal 1 and a target-stratified split for Goal 2."""
    if goal == "goal1":
        return list(trained_subjects), list(world.heldout_subjects)
    by_target: dict[int, list[int]] = {}
    for subject in trained_subjects:
        by_target.setdefault(world.target(relation, subject), []).append(subject)
    discovery: list[int] = []
    evaluation: list[int] = []
    for target in sorted(by_target):
        members = sorted(by_target[target])
        cut = max(1, len(members) // 2)
        discovery.extend(members[:cut])
        evaluation.extend(members[cut:])
    return sorted(discovery), sorted(evaluation)


def centroid_geometry(
    discovery_repr: np.ndarray,
    discovery_labels: Sequence[int],
    evaluation_repr: np.ndarray,
    evaluation_labels: Sequence[int],
) -> dict[str, float]:
    labels = sorted(set(int(x) for x in discovery_labels))
    centroids = np.stack(
        [discovery_repr[np.asarray(discovery_labels) == label].mean(axis=0) for label in labels]
    )
    center = discovery_repr.mean(axis=0, keepdims=True)
    total_ss = float(np.square(discovery_repr - center).sum())
    within_ss = 0.0
    for label, centroid in zip(labels, centroids):
        rows = discovery_repr[np.asarray(discovery_labels) == label]
        within_ss += float(np.square(rows - centroid).sum())
    eta_squared = 1.0 - within_ss / max(total_ss, 1e-12)
    eval_norm = evaluation_repr / np.maximum(np.linalg.norm(evaluation_repr, axis=1, keepdims=True), 1e-12)
    centroid_norm = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    predictions = np.asarray(labels)[(eval_norm @ centroid_norm.T).argmax(axis=1)]
    return {
        "centroid_cosine_accuracy": float(np.mean(predictions == np.asarray(evaluation_labels))),
        "class_eta_squared": eta_squared,
    }


@torch.no_grad()
def representation_geometry(
    model: CausalTransformer,
    world: World,
    relation: str,
    discovery: Sequence[int],
    evaluation: Sequence[int],
    device: torch.device,
) -> dict:
    discovery_ids = prompt_ids(world, discovery, relation, device)
    evaluation_ids = prompt_ids(world, evaluation, relation, device)
    _, d_cache = manual_forward(model, discovery_ids)
    _, e_cache = manual_forward(model, evaluation_ids)
    d_labels = [world.target(relation, subject) for subject in discovery]
    e_labels = [world.target(relation, subject) for subject in evaluation]
    stages = {
        "subject_input_embedding": (
            d_cache["x0"][:, 1, :],
            e_cache["x0"][:, 1, :],
        ),
        "attention_head_outputs": (
            d_cache["heads"][:, :, -1, :].flatten(1),
            e_cache["heads"][:, :, -1, :].flatten(1),
        ),
        "post_attention_residual": (
            d_cache["x_attn"][:, -1, :],
            e_cache["x_attn"][:, -1, :],
        ),
        "mlp_activation": (
            d_cache["mlp_act"][:, -1, :],
            e_cache["mlp_act"][:, -1, :],
        ),
        "post_mlp_residual": (
            d_cache["x_mlp"][:, -1, :],
            e_cache["x_mlp"][:, -1, :],
        ),
    }
    result = {}
    for name, (d_tensor, e_tensor) in stages.items():
        result[name] = centroid_geometry(
            d_tensor.cpu().numpy(), d_labels, e_tensor.cpu().numpy(), e_labels
        )
    return result


@torch.no_grad()
def stage_and_attention_analysis(
    model: CausalTransformer,
    world: World,
    relation: str,
    subjects: Sequence[int],
    device: torch.device,
) -> tuple[dict, dict[str, torch.Tensor]]:
    inputs = prompt_ids(world, subjects, relation, device)
    targets = target_token_ids(world, subjects, relation, device)
    logits, cache = manual_forward(model, inputs)
    exact_error = float((logits - model(inputs)).abs().max().item())
    if exact_error > 2e-5:
        raise RuntimeError(f"Manual forward mismatch: {exact_error}")
    attention = cache["attention"][:, :, -1, :].mean(dim=0)
    per_head = []
    for head in range(model.model_cfg.n_heads):
        per_head.append(
            {
                "head": head,
                "mean_attention": {
                    TOKENS[position]: float(attention[head, position].item())
                    for position in range(len(TOKENS))
                },
                "mean_output_norm": float(cache["heads"][:, head, -1, :].norm(dim=1).mean().item()),
                "mean_subject_value_contribution_norm": float(
                    (
                        cache["attention"][:, head, -1, 1, None]
                        * cache["values"][:, head, 1, :]
                    ).norm(dim=1).mean().item()
                ),
            }
        )
    return {
        "manual_forward_max_abs_error": exact_error,
        "baseline": object_metrics(logits, targets, world),
        "logit_lens": logit_lens(model, cache, targets, world),
        "attention_heads": per_head,
    }, cache


@torch.no_grad()
def exhaustive_head_interventions(
    model: CausalTransformer,
    world: World,
    relation: str,
    discovery: Sequence[int],
    evaluation: Sequence[int],
    device: torch.device,
) -> dict:
    d_inputs = prompt_ids(world, discovery, relation, device)
    _, d_cache = manual_forward(model, d_inputs)
    head_reference = d_cache["heads"].mean(dim=0, keepdim=True)
    e_inputs = prompt_ids(world, evaluation, relation, device)
    targets = target_token_ids(world, evaluation, relation, device)
    baseline_logits, e_cache = manual_forward(model, e_inputs)
    n_heads = model.model_cfg.n_heads
    subset_rows = []
    value_by_mask: dict[int, float] = {}
    for mask in range(1 << n_heads):
        kept = [head for head in range(n_heads) if mask & (1 << head)]
        modified = head_reference.expand_as(e_cache["heads"]).clone()
        if kept:
            modified[:, kept, :, :] = e_cache["heads"][:, kept, :, :]
        logits, _ = forward_from_heads(model, e_cache, modified)
        metrics = object_metrics(logits, targets, world)
        value_by_mask[mask] = metrics["mean_object_margin"]
        subset_rows.append({"kept_heads": kept, **metrics})

    factorial = math.factorial
    shapley = []
    all_heads = set(range(n_heads))
    for head in range(n_heads):
        value = 0.0
        others = sorted(all_heads - {head})
        for size in range(n_heads):
            weight = factorial(size) * factorial(n_heads - size - 1) / factorial(n_heads)
            for subset in itertools.combinations(others, size):
                mask = sum(1 << item for item in subset)
                value += weight * (value_by_mask[mask | (1 << head)] - value_by_mask[mask])
        shapley.append({"head": head, "margin_shapley": value})

    baseline = object_metrics(baseline_logits, targets, world)
    full_row = next(row for row in subset_rows if len(row["kept_heads"]) == n_heads)
    if abs(full_row["mean_object_margin"] - baseline["mean_object_margin"]) > 1e-5:
        raise RuntimeError("Keeping all heads did not recover the baseline")
    return {
        "intervention": "Heads outside kept_heads are replaced by their discovery-set mean activation.",
        "head_reference_subjects": len(discovery),
        "evaluation_subjects": len(evaluation),
        "baseline": baseline,
        "subsets": subset_rows,
        "margin_shapley": shapley,
    }


@torch.no_grad()
def rank_neurons_by_ablation(
    model: CausalTransformer,
    world: World,
    relation: str,
    discovery: Sequence[int],
    device: torch.device,
) -> tuple[list[int], torch.Tensor, list[dict]]:
    inputs = prompt_ids(world, discovery, relation, device)
    targets = target_token_ids(world, discovery, relation, device)
    baseline_logits, cache = manual_forward(model, inputs)
    baseline_margin = object_metrics(baseline_logits, targets, world)["mean_object_margin"]
    reference = cache["mlp_act"].mean(dim=0, keepdim=True)
    rows = []
    for neuron in range(cache["mlp_act"].shape[-1]):
        modified = cache["mlp_act"].clone()
        modified[:, -1, neuron] = reference[0, -1, neuron]
        logits = forward_from_mlp_acts(model, cache, modified)
        metrics = object_metrics(logits, targets, world)
        rows.append(
            {
                "neuron": neuron,
                "causal_margin_drop": baseline_margin - metrics["mean_object_margin"],
                "ablated_object_accuracy": metrics["object_accuracy"],
            }
        )
    ranking = [row["neuron"] for row in sorted(rows, key=lambda row: row["causal_margin_drop"], reverse=True)]
    return ranking, reference, rows


@torch.no_grad()
def neuron_curves(
    model: CausalTransformer,
    world: World,
    relation: str,
    evaluation: Sequence[int],
    ranking: Sequence[int],
    discovery_reference: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict:
    inputs = prompt_ids(world, evaluation, relation, device)
    targets = target_token_ids(world, evaluation, relation, device)
    baseline_logits, cache = manual_forward(model, inputs)
    reference = discovery_reference.expand_as(cache["mlp_act"])
    baseline = object_metrics(baseline_logits, targets, world)
    widths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    rng = np.random.default_rng(seed + 1701)
    rows = []
    all_neurons = np.arange(cache["mlp_act"].shape[-1])
    for width in widths:
        selected = list(ranking[:width])
        necessity_act = cache["mlp_act"].clone()
        necessity_act[:, -1, selected] = reference[:, -1, selected]
        necessity = object_metrics(forward_from_mlp_acts(model, cache, necessity_act), targets, world)

        sufficiency_act = reference.clone()
        sufficiency_act[:, -1, selected] = cache["mlp_act"][:, -1, selected]
        sufficiency = object_metrics(forward_from_mlp_acts(model, cache, sufficiency_act), targets, world)

        random_margins = []
        random_accuracies = []
        for _ in range(20):
            random_selected = rng.choice(all_neurons, size=width, replace=False).tolist()
            random_act = cache["mlp_act"].clone()
            random_act[:, -1, random_selected] = reference[:, -1, random_selected]
            random_metrics = object_metrics(forward_from_mlp_acts(model, cache, random_act), targets, world)
            random_margins.append(random_metrics["mean_object_margin"])
            random_accuracies.append(random_metrics["object_accuracy"])
        rows.append(
            {
                "k": width,
                "top_neurons": selected,
                "necessity_mean_margin": necessity["mean_object_margin"],
                "necessity_object_accuracy": necessity["object_accuracy"],
                "sufficiency_mean_margin": sufficiency["mean_object_margin"],
                "sufficiency_object_accuracy": sufficiency["object_accuracy"],
                "random_ablation_margin": _mean_sd(random_margins),
                "random_ablation_accuracy": _mean_sd(random_accuracies),
            }
        )
    return {
        "intervention": (
            "Neuron activations are mean-ablated using discovery-set means. Rankings are learned on "
            "discovery subjects and all curves are evaluated on disjoint subjects."
        ),
        "baseline": baseline,
        "top_32_ranking": list(ranking[:32]),
        "curves": rows,
    }


@torch.no_grad()
def mlp_nonlinearity_test(
    model: CausalTransformer,
    world: World,
    relation: str,
    discovery: Sequence[int],
    evaluation: Sequence[int],
    device: torch.device,
) -> dict:
    """Test whether GELU curvature is functionally required on the task manifold.

    For every physical MLP neuron, fit ``activation = slope * preactivation +
    intercept`` on discovery subjects only.  Replacing the true GELUs by these
    per-neuron affine surrogates on disjoint evaluation subjects turns the
    entire MLP into an affine map.  This is more informative than naively
    replacing GELU(x) by x, which is a large off-manifold intervention.
    """
    discovery_inputs = prompt_ids(world, discovery, relation, device)
    _, discovery_cache = manual_forward(model, discovery_inputs)
    x = discovery_cache["mlp_pre"][:, -1, :]
    y = discovery_cache["mlp_act"][:, -1, :]
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    slopes = ((x - x_mean) * (y - y_mean)).sum(dim=0) / (
        (x - x_mean).square().sum(dim=0) + 1e-12
    )
    intercepts = y_mean - slopes * x_mean

    evaluation_inputs = prompt_ids(world, evaluation, relation, device)
    targets = target_token_ids(world, evaluation, relation, device)
    baseline_logits, cache = manual_forward(model, evaluation_inputs)
    surrogate_final = cache["mlp_pre"][:, -1, :] * slopes + intercepts
    surrogate_acts = cache["mlp_act"].clone()
    surrogate_acts[:, -1, :] = surrogate_final
    surrogate_logits = forward_from_mlp_acts(model, cache, surrogate_acts)

    mean_acts = cache["mlp_act"].clone()
    mean_acts[:, -1, :] = y_mean
    mean_logits = forward_from_mlp_acts(model, cache, mean_acts)
    no_mlp_logits = model.lm_head(model.ln_f(cache["x_attn"]))

    true_final = cache["mlp_act"][:, -1, :]
    ss_res = float((true_final - surrogate_final).square().sum().item())
    ss_tot = float((true_final - true_final.mean(dim=0, keepdim=True)).square().sum().item())
    output_true = model.blocks[0].mlp.fc2(true_final)
    output_surrogate = model.blocks[0].mlp.fc2(surrogate_final)
    output_relative_error = float(
        (output_true - output_surrogate).norm().item() / max(output_true.norm().item(), 1e-12)
    )
    return {
        "method": (
            "Per-neuron affine GELU surrogates are fit on discovery preactivations and evaluated "
            "on disjoint subjects; the resulting whole MLP is affine on its input."
        ),
        "activation_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "mlp_output_relative_l2_error": output_relative_error,
        "baseline": object_metrics(baseline_logits, targets, world),
        "affine_gelu_surrogate": object_metrics(surrogate_logits, targets, world),
        "discovery_mean_mlp_activation": object_metrics(mean_logits, targets, world),
        "no_mlp_residual_only": object_metrics(no_mlp_logits, targets, world),
    }


def make_counterfactual_pairs(
    world: World,
    subjects: Sequence[int],
    relation: str,
) -> tuple[list[int], list[int]]:
    clean = list(subjects)
    corrupt = []
    for index, subject in enumerate(clean):
        for offset in range(1, len(clean)):
            candidate = clean[(index + offset) % len(clean)]
            if world.target(relation, candidate) != world.target(relation, subject):
                corrupt.append(candidate)
                break
        else:
            raise ValueError("Could not construct a different-target counterfactual")
    return clean, corrupt


def patch_summary(
    clean_score: float,
    corrupt_score: float,
    patched_score: float,
) -> dict[str, float]:
    total = clean_score - corrupt_score
    return {
        "patched_mean_logit_difference": patched_score,
        "absolute_restoration": patched_score - corrupt_score,
        "normalized_restoration": (patched_score - corrupt_score) / total if abs(total) > 1e-12 else float("nan"),
    }


def activation_and_attribution_patching(
    model: CausalTransformer,
    world: World,
    relation: str,
    subjects: Sequence[int],
    device: torch.device,
) -> dict:
    clean_subjects, corrupt_subjects = make_counterfactual_pairs(world, subjects, relation)
    clean_inputs = prompt_ids(world, clean_subjects, relation, device)
    corrupt_inputs = prompt_ids(world, corrupt_subjects, relation, device)
    clean_targets = target_token_ids(world, clean_subjects, relation, device)
    corrupt_targets = target_token_ids(world, corrupt_subjects, relation, device)
    with torch.no_grad():
        clean_logits, clean_cache = manual_forward(model, clean_inputs)
        corrupt_logits, corrupt_cache = manual_forward(model, corrupt_inputs)
        clean_score = float(margin_vector(clean_logits, clean_targets, corrupt_targets).mean().item())
        corrupt_score = float(margin_vector(corrupt_logits, clean_targets, corrupt_targets).mean().item())

        # Subject-embedding patch: the only changed input token, so this must recover the clean run.
        patched_x0 = corrupt_cache["x0"].clone()
        patched_x0[:, 1, :] = clean_cache["x0"][:, 1, :]
        subject_logits, _ = manual_forward(model, x0=patched_x0)

        all_head_logits, _ = forward_from_heads(model, corrupt_cache, clean_cache["heads"])
        all_mlp_logits = forward_from_mlp_acts(model, corrupt_cache, clean_cache["mlp_act"])

        components = {
            "subject_input_embedding": patch_summary(
                clean_score,
                corrupt_score,
                float(margin_vector(subject_logits, clean_targets, corrupt_targets).mean().item()),
            ),
            "all_attention_heads": patch_summary(
                clean_score,
                corrupt_score,
                float(margin_vector(all_head_logits, clean_targets, corrupt_targets).mean().item()),
            ),
            "all_mlp_neurons": patch_summary(
                clean_score,
                corrupt_score,
                float(margin_vector(all_mlp_logits, clean_targets, corrupt_targets).mean().item()),
            ),
        }

        exact_head_delta = []
        exact_subject_value_delta = []
        head_rows = []
        for head in range(model.model_cfg.n_heads):
            modified = corrupt_cache["heads"].clone()
            modified[:, head, -1, :] = clean_cache["heads"][:, head, -1, :]
            logits, _ = forward_from_heads(model, corrupt_cache, modified)
            patched_score = float(margin_vector(logits, clean_targets, corrupt_targets).mean().item())
            exact_head_delta.append(patched_score - corrupt_score)

            clean_subject_contribution = (
                clean_cache["attention"][:, head, -1, 1, None]
                * clean_cache["values"][:, head, 1, :]
            )
            corrupt_subject_contribution = (
                corrupt_cache["attention"][:, head, -1, 1, None]
                * corrupt_cache["values"][:, head, 1, :]
            )
            value_modified = corrupt_cache["heads"].clone()
            value_modified[:, head, -1, :] += clean_subject_contribution - corrupt_subject_contribution
            value_logits, _ = forward_from_heads(model, corrupt_cache, value_modified)
            value_score = float(margin_vector(value_logits, clean_targets, corrupt_targets).mean().item())
            exact_subject_value_delta.append(value_score - corrupt_score)
            head_rows.append(
                {
                    "head": head,
                    "full_head_patch": patch_summary(clean_score, corrupt_score, patched_score),
                    "subject_value_path_patch": patch_summary(clean_score, corrupt_score, value_score),
                }
            )

        # Pattern vs value decomposition across all heads.
        pattern_heads = clean_cache["attention"] @ corrupt_cache["values"]
        value_heads = corrupt_cache["attention"] @ clean_cache["values"]
        pattern_logits, _ = forward_from_heads(model, corrupt_cache, pattern_heads)
        value_logits, _ = forward_from_heads(model, corrupt_cache, value_heads)
        components["attention_patterns_only"] = patch_summary(
            clean_score,
            corrupt_score,
            float(margin_vector(pattern_logits, clean_targets, corrupt_targets).mean().item()),
        )
        components["attention_values_only"] = patch_summary(
            clean_score,
            corrupt_score,
            float(margin_vector(value_logits, clean_targets, corrupt_targets).mean().item()),
        )

        exact_neuron_delta = []
        for neuron in range(clean_cache["mlp_act"].shape[-1]):
            modified = corrupt_cache["mlp_act"].clone()
            modified[:, -1, neuron] = clean_cache["mlp_act"][:, -1, neuron]
            logits = forward_from_mlp_acts(model, corrupt_cache, modified)
            score = float(margin_vector(logits, clean_targets, corrupt_targets).mean().item())
            exact_neuron_delta.append(score - corrupt_score)

    # Attribution patching on the same corrupted run, validated below against exact patches.
    model.zero_grad(set_to_none=True)
    grad_logits, grad_cache = manual_forward(model, corrupt_inputs, retain_intermediate_grads=True)
    objective = margin_vector(grad_logits, clean_targets, corrupt_targets).mean()
    objective.backward()
    head_attr = (
        (clean_cache["heads"][:, :, -1, :] - grad_cache["heads"][:, :, -1, :].detach())
        * grad_cache["heads"].grad[:, :, -1, :]
    ).sum(dim=(0, 2)).detach().cpu().numpy()
    neuron_attr = (
        (clean_cache["mlp_act"][:, -1, :] - grad_cache["mlp_act"][:, -1, :].detach())
        * grad_cache["mlp_act"].grad[:, -1, :]
    ).sum(dim=0).detach().cpu().numpy()
    model.zero_grad(set_to_none=True)

    exact_head = np.asarray(exact_head_delta)
    exact_neuron = np.asarray(exact_neuron_delta)
    top_neurons = np.argsort(-exact_neuron)[:20]
    neuron_rows = [
        {
            "neuron": int(neuron),
            "exact_absolute_restoration": float(exact_neuron[neuron]),
            "exact_normalized_restoration": float(exact_neuron[neuron] / (clean_score - corrupt_score)),
            "attribution_estimate": float(neuron_attr[neuron]),
        }
        for neuron in top_neurons
    ]
    subject_recovery = components["subject_input_embedding"]["normalized_restoration"]
    if abs(subject_recovery - 1.0) > 2e-4:
        raise RuntimeError(f"Subject-input patch should fully recover the clean run, got {subject_recovery}")
    return {
        "pair_count": len(clean_subjects),
        "metric": "mean logit(clean target) - logit(corrupt target)",
        "clean_mean_logit_difference": clean_score,
        "corrupt_mean_logit_difference": corrupt_score,
        "total_effect": clean_score - corrupt_score,
        "components": components,
        "heads": head_rows,
        "top_exact_neurons": neuron_rows,
        "attribution_validation": {
            "heads_pearson_r": _pearson(head_attr, exact_head),
            "heads_spearman_r": _spearman(head_attr, exact_head),
            "neurons_pearson_r": _pearson(neuron_attr, exact_neuron),
            "neurons_spearman_r": _spearman(neuron_attr, exact_neuron),
            "head_attribution_estimates": head_attr,
            "head_exact_restorations": exact_head,
        },
    }


def relation_gate_patching(
    model: CausalTransformer,
    world: World,
    clean_relation: str,
    subjects: Sequence[int],
    device: torch.device,
) -> dict:
    corrupt_relation = "random" if clean_relation == "semantic" else "semantic"
    clean_inputs = prompt_ids(world, subjects, clean_relation, device)
    corrupt_inputs = prompt_ids(world, subjects, corrupt_relation, device)
    clean_targets = target_token_ids(world, subjects, clean_relation, device)
    corrupt_targets = target_token_ids(world, subjects, corrupt_relation, device)
    with torch.no_grad():
        clean_logits, clean_cache = manual_forward(model, clean_inputs)
        corrupt_logits, corrupt_cache = manual_forward(model, corrupt_inputs)
        clean_score = float(margin_vector(clean_logits, clean_targets, corrupt_targets).mean().item())
        corrupt_score = float(margin_vector(corrupt_logits, clean_targets, corrupt_targets).mean().item())
        patched_x0 = corrupt_cache["x0"].clone()
        patched_x0[:, -1, :] = clean_cache["x0"][:, -1, :]
        token_logits, _ = manual_forward(model, x0=patched_x0)
        head_logits, _ = forward_from_heads(model, corrupt_cache, clean_cache["heads"])
        mlp_logits = forward_from_mlp_acts(model, corrupt_cache, clean_cache["mlp_act"])
        per_head = []
        for head in range(model.model_cfg.n_heads):
            modified = corrupt_cache["heads"].clone()
            modified[:, head, -1, :] = clean_cache["heads"][:, head, -1, :]
            logits, _ = forward_from_heads(model, corrupt_cache, modified)
            score = float(margin_vector(logits, clean_targets, corrupt_targets).mean().item())
            per_head.append({"head": head, **patch_summary(clean_score, corrupt_score, score)})
    return {
        "clean_relation": clean_relation,
        "corrupt_relation": corrupt_relation,
        "clean_mean_logit_difference": clean_score,
        "corrupt_mean_logit_difference": corrupt_score,
        "total_effect": clean_score - corrupt_score,
        "relation_token_embedding": patch_summary(
            clean_score,
            corrupt_score,
            float(margin_vector(token_logits, clean_targets, corrupt_targets).mean().item()),
        ),
        "all_attention_heads": patch_summary(
            clean_score,
            corrupt_score,
            float(margin_vector(head_logits, clean_targets, corrupt_targets).mean().item()),
        ),
        "all_mlp_neurons": patch_summary(
            clean_score,
            corrupt_score,
            float(margin_vector(mlp_logits, clean_targets, corrupt_targets).mean().item()),
        ),
        "heads": per_head,
    }


def analyze_goal_seed(
    goal: str,
    seed: int,
    device: torch.device,
) -> dict:
    spec = GOALS[goal]
    occurrences = int(spec["occurrences"])
    relation = str(spec["relation"])
    model, world, trained_subjects, replay = replay_and_verify(seed, occurrences, device)
    discovery, evaluation = split_discovery_evaluation(world, trained_subjects, relation, goal)
    stage, _ = stage_and_attention_analysis(model, world, relation, evaluation, device)
    geometry = representation_geometry(model, world, relation, discovery, evaluation, device)
    head_interventions = exhaustive_head_interventions(
        model, world, relation, discovery, evaluation, device
    )
    neuron_ranking, neuron_reference, neuron_rows = rank_neurons_by_ablation(
        model, world, relation, discovery, device
    )
    neurons = neuron_curves(
        model,
        world,
        relation,
        evaluation,
        neuron_ranking,
        neuron_reference,
        device,
        seed,
    )
    neurons["discovery_top_20_individual_ablations"] = sorted(
        neuron_rows, key=lambda row: row["causal_margin_drop"], reverse=True
    )[:20]
    nonlinearity = mlp_nonlinearity_test(
        model, world, relation, discovery, evaluation, device
    )
    subject_patching = activation_and_attribution_patching(
        model, world, relation, evaluation, device
    )
    relation_patching = relation_gate_patching(
        model, world, relation, evaluation, device
    )

    # Matched controls use the same replayed model and evaluation subjects.
    control_relation = "random" if relation == "semantic" else "semantic"
    control_stage, _ = stage_and_attention_analysis(
        model, world, control_relation, evaluation, device
    )
    heldout_control = None
    if goal == "goal2":
        heldout_control, _ = stage_and_attention_analysis(
            model, world, relation, world.heldout_subjects, device
        )
    return {
        "goal": goal,
        "seed": seed,
        "occurrences_per_triple": occurrences,
        "relation": relation,
        "discovery_subjects": len(discovery),
        "evaluation_subjects": len(evaluation),
        "evaluation_split": (
            "never-seen relation subjects" if goal == "goal1" else "disjoint target-stratified half of trained subjects"
        ),
        "replay_verification": replay,
        "stage_analysis": stage,
        "matched_relation_stage_control": {"relation": control_relation, **control_stage},
        "random_heldout_control": heldout_control,
        "representation_geometry": geometry,
        "head_interventions": head_interventions,
        "neuron_interventions": neurons,
        "mlp_nonlinearity": nonlinearity,
        "subject_swap_patching": subject_patching,
        "relation_gate_patching": relation_patching,
    }


def subset_metric(run: dict, kept: Sequence[int], key: str) -> float:
    row = next(
        row
        for row in run["head_interventions"]["subsets"]
        if row["kept_heads"] == list(kept)
    )
    return float(row[key])


def curve_metric(run: dict, k: int, key: str) -> float:
    row = next(row for row in run["neuron_interventions"]["curves"] if row["k"] == k)
    return float(row[key])


def aggregate_results(runs: Sequence[dict]) -> dict:
    result = {}
    for goal in GOALS:
        goal_runs = [run for run in runs if run["goal"] == goal]
        per_head = []
        for head in range(4):
            per_head.append(
                {
                    "head": head,
                    "subject_attention": _mean_sd(
                        [
                            run["stage_analysis"]["attention_heads"][head]["mean_attention"]["subject"]
                            for run in goal_runs
                        ]
                    ),
                    "margin_shapley": _mean_sd(
                        [
                            run["head_interventions"]["margin_shapley"][head]["margin_shapley"]
                            for run in goal_runs
                        ]
                    ),
                    "subject_value_patch_recovery": _mean_sd(
                        [
                            run["subject_swap_patching"]["heads"][head]["subject_value_path_patch"][
                                "normalized_restoration"
                            ]
                            for run in goal_runs
                        ]
                    ),
                }
            )
        neuron_margin_fraction = {}
        for k in (8, 16, 32, 64, 128, 256):
            necessity_remaining = []
            sufficiency_recovered = []
            random_remaining = []
            for run in goal_runs:
                baseline_margin = run["neuron_interventions"]["baseline"]["mean_object_margin"]
                all_mean_margin = curve_metric(run, 256, "necessity_mean_margin")
                scale = baseline_margin - all_mean_margin
                row = next(
                    item for item in run["neuron_interventions"]["curves"] if item["k"] == k
                )
                necessity_remaining.append(
                    (row["necessity_mean_margin"] - all_mean_margin) / scale
                )
                sufficiency_recovered.append(
                    (row["sufficiency_mean_margin"] - all_mean_margin) / scale
                )
                random_remaining.append(
                    (row["random_ablation_margin"]["mean"] - all_mean_margin) / scale
                )
            neuron_margin_fraction[str(k)] = {
                "after_top_k_ablation": _mean_sd(necessity_remaining),
                "with_only_top_k_variable": _mean_sd(sufficiency_recovered),
                "after_random_k_ablation": _mean_sd(random_remaining),
            }
        result[goal] = {
            "n_seeds": len(goal_runs),
            "seeds": [run["seed"] for run in goal_runs],
            "baseline_full_vocab_accuracy": _mean_sd(
                [run["stage_analysis"]["baseline"]["full_vocab_accuracy"] for run in goal_runs]
            ),
            "baseline_object_margin": _mean_sd(
                [run["stage_analysis"]["baseline"]["mean_object_margin"] for run in goal_runs]
            ),
            "logit_lens_object_accuracy": {
                stage: _mean_sd(
                    [run["stage_analysis"]["logit_lens"][stage]["object_accuracy"] for run in goal_runs]
                )
                for stage in (
                    "input_relation_residual",
                    "post_attention_residual",
                    "post_mlp_residual",
                )
            },
            "all_heads_mean_ablated_accuracy": _mean_sd(
                [subset_metric(run, [], "object_accuracy") for run in goal_runs]
            ),
            "all_heads_present_accuracy": _mean_sd(
                [subset_metric(run, [0, 1, 2, 3], "object_accuracy") for run in goal_runs]
            ),
            "heads": per_head,
            "subject_input_patch_recovery": _mean_sd(
                [
                    run["subject_swap_patching"]["components"]["subject_input_embedding"][
                        "normalized_restoration"
                    ]
                    for run in goal_runs
                ]
            ),
            "all_heads_patch_recovery": _mean_sd(
                [
                    run["subject_swap_patching"]["components"]["all_attention_heads"][
                        "normalized_restoration"
                    ]
                    for run in goal_runs
                ]
            ),
            "attention_values_patch_recovery": _mean_sd(
                [
                    run["subject_swap_patching"]["components"]["attention_values_only"][
                        "normalized_restoration"
                    ]
                    for run in goal_runs
                ]
            ),
            "attention_patterns_patch_recovery": _mean_sd(
                [
                    run["subject_swap_patching"]["components"]["attention_patterns_only"][
                        "normalized_restoration"
                    ]
                    for run in goal_runs
                ]
            ),
            "all_mlp_patch_recovery": _mean_sd(
                [
                    run["subject_swap_patching"]["components"]["all_mlp_neurons"][
                        "normalized_restoration"
                    ]
                    for run in goal_runs
                ]
            ),
            "attribution_validation": {
                key: _mean_sd(
                    [run["subject_swap_patching"]["attribution_validation"][key] for run in goal_runs]
                )
                for key in (
                    "heads_pearson_r",
                    "heads_spearman_r",
                    "neurons_pearson_r",
                    "neurons_spearman_r",
                )
            },
            "geometry_centroid_accuracy": {
                stage: _mean_sd(
                    [run["representation_geometry"][stage]["centroid_cosine_accuracy"] for run in goal_runs]
                )
                for stage in (
                    "subject_input_embedding",
                    "attention_head_outputs",
                    "post_attention_residual",
                    "mlp_activation",
                    "post_mlp_residual",
                )
            },
            "neuron_necessity_accuracy": {
                str(k): _mean_sd(
                    [curve_metric(run, k, "necessity_object_accuracy") for run in goal_runs]
                )
                for k in (8, 16, 32, 64, 128, 256)
            },
            "neuron_sufficiency_accuracy": {
                str(k): _mean_sd(
                    [curve_metric(run, k, "sufficiency_object_accuracy") for run in goal_runs]
                )
                for k in (8, 16, 32, 64, 128, 256)
            },
            "neuron_subject_specific_margin_fraction": neuron_margin_fraction,
            "mlp_nonlinearity": {
                "activation_r2": _mean_sd(
                    [run["mlp_nonlinearity"]["activation_r2"] for run in goal_runs]
                ),
                "affine_surrogate_accuracy": _mean_sd(
                    [
                        run["mlp_nonlinearity"]["affine_gelu_surrogate"]["object_accuracy"]
                        for run in goal_runs
                    ]
                ),
                "affine_surrogate_margin": _mean_sd(
                    [
                        run["mlp_nonlinearity"]["affine_gelu_surrogate"]["mean_object_margin"]
                        for run in goal_runs
                    ]
                ),
                "no_mlp_accuracy": _mean_sd(
                    [run["mlp_nonlinearity"]["no_mlp_residual_only"]["object_accuracy"] for run in goal_runs]
                ),
                "no_mlp_margin": _mean_sd(
                    [
                        run["mlp_nonlinearity"]["no_mlp_residual_only"]["mean_object_margin"]
                        for run in goal_runs
                    ]
                ),
            },
            "relation_gate": {
                "all_heads_patch_recovery": _mean_sd(
                    [
                        run["relation_gate_patching"]["all_attention_heads"]["normalized_restoration"]
                        for run in goal_runs
                    ]
                ),
                "all_mlp_patch_recovery": _mean_sd(
                    [
                        run["relation_gate_patching"]["all_mlp_neurons"]["normalized_restoration"]
                        for run in goal_runs
                    ]
                ),
            },
            "representative_seed": goal_runs[0]["seed"],
            "representative_top_neurons": goal_runs[0]["neuron_interventions"]["top_32_ranking"],
        }
    return result


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_report(result: dict, path: Path) -> None:
    aggregate = result["aggregate"]
    g1 = aggregate["goal1"]
    g2 = aggregate["goal2"]

    def ms(cell: dict) -> str:
        return f"{cell['mean']:.3f} ± {cell['sd']:.3f}"

    lines = [
        "# Goal 1 and Goal 2: mechanistic forward-pass analysis",
        "",
        "![Forward pass and identified compute units](goal1_goal2_forward_pass.svg)",
        "",
        "## Scope and evidential standard",
        "",
        "This report analyzes the exact one-layer, four-head, 64-dimensional transformer used in the final Goal 1/2 runs. "
        "The archived runs stored metrics but not checkpoints, so every checkpoint was reconstructed by deterministic replay. "
        "For every seed and frequency, final loss, probe accuracy, trained recall, and held-out recall match the archived JSON "
        "to <1e-7. All component claims below therefore refer to the reported models, not a newly tuned surrogate.",
        "",
        "A probe is treated only as evidence that information is available. A compute unit is called part of the mechanism only "
        "when an intervention changes the output. We use exact mean ablation, clean→corrupt activation patching, exhaustive "
        "head-subset tests, and neuron interventions ranked on one subject split and evaluated on a disjoint split. Attribution "
        "patching is used only after its approximation is checked against exact patching.",
        "",
        "## The literal forward pass",
        "",
        "For the final prompt position in `[x, a_i, r]`, the model computes",
        "",
        "```text",
        "x0 = E[r] + P[2]",
        "h_h = Σ_t softmax(q_h(x0)·k_h(x_t)/√16)_t · v_h(x_t),   h = 0…3",
        "xA = x0 + W_O concat(h_0,h_1,h_2,h_3)",
        "m_n = GELU(w_in,n · LN(xA) + b_n),                       n = 0…255",
        "xM = xA + Σ_n m_n w_out,n + b_out",
        "logits = U · LN(xM)",
        "```",
        "",
        "Only the attention heads can move information from the subject at position 1 to the prediction position 2. The MLP "
        "is position-wise: it can transform or sharpen the retrieved state, but it cannot retrieve the subject on its own.",
        "",
        "## Goal 1 — infrequent semantic relation (F=16)",
        "",
        f"On the never-seen relation subjects, the replayed models achieve full-vocabulary accuracy {ms(g1['baseline_full_vocab_accuracy'])}. "
        "The logit lens changes object accuracy from "
        f"{ms(g1['logit_lens_object_accuracy']['input_relation_residual'])} at the bare relation residual, to "
        f"{ms(g1['logit_lens_object_accuracy']['post_attention_residual'])} after attention, and "
        f"{ms(g1['logit_lens_object_accuracy']['post_mlp_residual'])} after the MLP.",
        "",
        "The causal path is:",
        "",
        "1. Attribute training organizes subject embeddings by the semantic target class. A target centroid learned only from "
        f"trained subjects classifies held-out subject embeddings at {ms(g1['geometry_centroid_accuracy']['subject_input_embedding'])}.",
        "2. The semantic relation token supplies the final-position queries. Heads attend to the subject and transport its value "
        "vector to the final position. Replacing every head by its discovery-set mean changes object accuracy from "
        f"{ms(g1['all_heads_present_accuracy'])} to {ms(g1['all_heads_mean_ablated_accuracy'])}.",
        "3. In clean/corrupt subject swaps, patching all head outputs restores "
        f"{ms(g1['all_heads_patch_recovery'])} of the clean-target logit effect. Patching values while leaving corrupted attention "
        f"patterns restores {ms(g1['attention_values_patch_recovery'])}; patching patterns while leaving corrupted values restores "
        f"{ms(g1['attention_patterns_patch_recovery'])}. This separates transported content from routing.",
        "4. The MLP is a distributed readout/sharpening stage. Its neuron rankings are learned on trained subjects and tested on "
        "held-out subjects. Mean-ablating the top 32 leaves "
        f"{ms(g1['neuron_subject_specific_margin_fraction']['32']['after_top_k_ablation'])} of the subject-specific MLP margin; "
        f"keeping only those 32 variable neurons recovers {ms(g1['neuron_subject_specific_margin_fraction']['32']['with_only_top_k_variable'])}. "
        "The full concentration curve, rather than a single cherry-picked unit, is stored in the JSON and plotted in the figure.",
        "5. To isolate activation curvature from mere MLP involvement, each GELU is replaced by a per-neuron affine surrogate fit "
        f"on trained subjects. On held-out subjects this fully affine MLP reaches {ms(g1['mlp_nonlinearity']['affine_surrogate_accuracy'])} "
        f"object accuracy (activation R² {ms(g1['mlp_nonlinearity']['activation_r2'])}); deleting the MLP reaches "
        f"{ms(g1['mlp_nonlinearity']['no_mlp_accuracy'])}.",
        "",
        "Thus Goal 1 is not merely ‘an affine probe works.’ The model uses the same attribute-organized subject code causally: "
        "relation-gated attention moves it to the prediction position, and a distributed MLP/readout converts it to the object logit. "
        "Because the code is shared by subjects with the same f-value, the circuit works on subjects whose relation triple never appeared.",
        "",
        "Holding the subject fixed and swapping only r₀↔r₁ shows where relation selection occurs. For Goal 1, patching the "
        f"four head outputs restores {ms(g1['relation_gate']['all_heads_patch_recovery'])} of the relation effect, whereas patching "
        f"the MLP population restores {ms(g1['relation_gate']['all_mlp_patch_recovery'])}. The semantic selector therefore persists "
        "mainly through the relation residual into the MLP; the heads primarily transport subject content.",
        "",
        "## Goal 2 — frequent random relation (F=64)",
        "",
        f"On a target-stratified, disjoint half of the trained facts, full-vocabulary accuracy is {ms(g2['baseline_full_vocab_accuracy'])}. "
        "The same stage lens changes object accuracy from "
        f"{ms(g2['logit_lens_object_accuracy']['input_relation_residual'])}, to "
        f"{ms(g2['logit_lens_object_accuracy']['post_attention_residual'])}, to "
        f"{ms(g2['logit_lens_object_accuracy']['post_mlp_residual'])}.",
        "",
        "The causal path is:",
        "",
        "1. Repetition has trained a subject-specific code for the arbitrary target. Unlike Goal 1, it is not a rule that transfers "
        "to unseen subjects: archived held-out random recall is chance, while trained recall is 1.0 in every seed.",
        "2. The random-relation token changes the queries/routing. The four head outputs jointly retrieve the memorized subject code. "
        "Mean-ablating all heads changes object accuracy from "
        f"{ms(g2['all_heads_present_accuracy'])} to {ms(g2['all_heads_mean_ablated_accuracy'])}; the exhaustive 16-subset table "
        "and exact Shapley margin effects identify how the four physical heads share this work.",
        "3. Clean/corrupt subject patching restores "
        f"{ms(g2['all_heads_patch_recovery'])} through all head outputs and {ms(g2['attention_values_patch_recovery'])} through "
        "the value path alone. This is the causal signature of lookup: swapping the transported subject code moves the output "
        "toward that other subject's arbitrary target.",
        "4. The post-attention code is transformed by a distributed population of literal GELU units. On disjoint trained facts, "
        f"ablating the top 32 leaves {ms(g2['neuron_subject_specific_margin_fraction']['32']['after_top_k_ablation'])} of the "
        f"subject-specific MLP margin, while retaining only those 32 variable units recovers "
        f"{ms(g2['neuron_subject_specific_margin_fraction']['32']['with_only_top_k_variable'])}. Representative seed-specific IDs "
        "are shown in the figure; IDs are not expected to align across random initializations.",
        "5. The nonlinear-representation claim must not be confused with a claim that GELU curvature alone performs the lookup. "
        f"A per-neuron affine surrogate for every GELU reaches {ms(g2['mlp_nonlinearity']['affine_surrogate_accuracy'])} object "
        f"accuracy on disjoint memorized facts (activation R² {ms(g2['mlp_nonlinearity']['activation_r2'])}); with the whole MLP "
        f"deleted, attention alone reaches {ms(g2['mlp_nonlinearity']['no_mlp_accuracy'])}. The experiment therefore localizes "
        "most arbitrary lookup to learned subject embeddings plus attention values, with the MLP providing margin and residual errors.",
        "",
        "This is a non-generalizing memorization circuit: the model retrieves a learned subject key and transforms it through "
        "attention plus a distributed MLP into the memorized object direction. ‘Nonlinear’ in Goal 2 is the operational storage-format "
        "result—no affine map fitted on trained subjects predicts held-out random targets—not a stronger claim that every successful "
        "trained-subject forward pass requires GELU curvature. There is no class-shared rule for held-out subjects, which is why the "
        "same static affine probe and behavioral generalization both remain at chance.",
        "",
        "With the subject held fixed, swapping r₁ for r₀ and patching the four random-relation head outputs restores "
        f"{ms(g2['relation_gate']['all_heads_patch_recovery'])} of the relation effect; patching the MLP population restores "
        f"{ms(g2['relation_gate']['all_mlp_patch_recovery'])}. At high frequency, relation selection is therefore strongly present "
        "in both the attention retrieval and its downstream transformation.",
        "",
        "## Convergent causal evidence",
        "",
        "| Test | Goal 1 | Goal 2 | Interpretation |",
        "|---|---:|---:|---|",
        f"| All-head patch recovery | {ms(g1['all_heads_patch_recovery'])} | {ms(g2['all_heads_patch_recovery'])} | Subject information causally crosses attention |",
        f"| Value-only patch recovery | {ms(g1['attention_values_patch_recovery'])} | {ms(g2['attention_values_patch_recovery'])} | Content is in transported V vectors |",
        f"| Pattern-only patch recovery | {ms(g1['attention_patterns_patch_recovery'])} | {ms(g2['attention_patterns_patch_recovery'])} | Routing contribution separated from content |",
        f"| Attribution↔exact neuron patch Pearson r | {ms(g1['attribution_validation']['neurons_pearson_r'])} | {ms(g2['attribution_validation']['neurons_pearson_r'])} | Gradient screen checked against interventions |",
        "",
        "## What the analysis does not claim",
        "",
        "The neuron mechanism is distributed, so the report does not claim one universal ‘fact neuron.’ Unit numbers are local to a "
        "seed; the stable result is the computational role and causal concentration curve. Mean ablation is used instead of zeroing "
        "where possible to reduce out-of-distribution artifacts. The affine embedding geometry in Goal 1 is upstream structure; it "
        "becomes a mechanism claim only because subject-value and head patches show that the forward pass consumes it.",
        "",
        "## Reproduction and artifacts",
        "",
        "From `experiments/`:",
        "",
        "```bash",
        "uv run python -m factual_recall.linear_shift_mechanistic_analysis --device cuda",
        "```",
        "",
        "- `goal1_goal2_mechanistic_analysis.json`: every per-seed intervention and aggregate.",
        "- `goal1_goal2_forward_pass.svg`: vector figure used above.",
        "- `goal1_goal2_forward_pass.pdf`: publication-ready vector export.",
        "- `goal1_goal2_forward_pass.png`: high-resolution raster preview.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(result: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    aggregate = result["aggregate"]
    g1 = aggregate["goal1"]
    g2 = aggregate["goal2"]
    runs = result["runs"]
    rep1 = next(run for run in runs if run["goal"] == "goal1" and run["seed"] == g1["representative_seed"])
    rep2 = next(run for run in runs if run["goal"] == "goal2" and run["seed"] == g2["representative_seed"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(15, 12.5), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.68, 0.72, 0.72, 1.08],
        hspace=0.34,
        wspace=0.22,
    )
    generic = fig.add_subplot(grid[0, :])
    goal1_ax = fig.add_subplot(grid[1, :])
    goal2_ax = fig.add_subplot(grid[2, :])
    heads_ax = fig.add_subplot(grid[3, 0])
    neurons_ax = fig.add_subplot(grid[3, 1])

    navy = "#16324F"
    blue = "#2B6CB0"
    cyan = "#2A9D8F"
    orange = "#E07A5F"
    gold = "#E9C46A"
    light = "#F4F7FA"
    gray = "#5E6C76"

    def box(ax, x, y, w, h, text, face, edge=navy, size=9, weight="normal"):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
            linewidth=1.2, edgecolor=edge, facecolor=face
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight)
        return patch

    def arrow(ax, x0, y0, x1, y1, color=gray, width=1.3):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=width, color=color, shrinkA=2, shrinkB=2))

    # Generic architecture row.
    generic.set_xlim(0, 1)
    generic.set_ylim(0, 1)
    generic.axis("off")
    generic.text(0.01, 0.95, "A  Literal forward pass", fontsize=15, weight="bold", color=navy, va="top")
    xs = [0.02, 0.18, 0.35, 0.52, 0.69, 0.85]
    labels = [
        "tokens\n[x, aᵢ, r]",
        "E[token] + P[pos]\nresidual x₀",
        "LN → Q,K,V\n4 heads × 16d",
        "Wₒ + residual\npost-attention xᴬ",
        "LN → 256 GELUs\n+ residual xᴹ",
        "final LN → U\nobject logits",
    ]
    widths = [0.11, 0.13, 0.13, 0.13, 0.13, 0.13]
    faces = [light, "#E8F1FB", "#DFF3F0", "#E8F1FB", "#FBECE7", "#FFF5D6"]
    for x, label, width, face in zip(xs, labels, widths, faces):
        box(generic, x, 0.27, width, 0.42, label, face, size=11, weight="bold" if x in (0.35, 0.69) else "normal")
    for left, width, right in zip(xs[:-1], widths[:-1], xs[1:]):
        arrow(generic, left + width, 0.48, right, 0.48)
    generic.text(0.40, 0.13, "Only attention crosses token positions", ha="center", color=cyan, weight="bold", fontsize=10)
    generic.text(0.76, 0.13, "MLP transforms the retrieved state position-wise", ha="center", color=orange, weight="bold", fontsize=10)

    def circuit_panel(ax, title, subtitle, agg, rep, semantic):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.0, 1.02, title, fontsize=14, weight="bold", color=navy, va="bottom")
        ax.text(0.0, 0.96, subtitle, fontsize=10, color=gray, va="top")
        source_text = "attribute-shared f code\nin E[aᵢ]" if semantic else "subject-specific key\nin E[aᵢ]"
        target_text = "shared rule → $b_{f(i)}$" if semantic else "memorized lookup → $b_{\\rho(i)}$"
        head_text = (
            "H3 + H1 primary\nH0/H2 support\n\ntransport subject V"
            if semantic
            else "H1 primary\nH0/H2 support; H3 ≈ 0\n\nretrieve subject V"
        )
        box(ax, 0.02, 0.55, 0.18, 0.23, source_text, "#E8F1FB", size=11, weight="bold")
        box(ax, 0.29, 0.48, 0.24, 0.37, head_text, "#DFF3F0", size=11, weight="bold")
        box(ax, 0.62, 0.48, 0.18, 0.37, "distributed GELUs\nN0…N255\n\ntransform / sharpen", "#FBECE7", size=11, weight="bold")
        box(ax, 0.84, 0.55, 0.15, 0.23, target_text, "#FFF5D6", size=10, weight="bold")
        relation_label = "r₀" if semantic else "r₁"
        box(ax, 0.545, 0.79, 0.06, 0.06, relation_label, "#F0E8F7", edge="#79589F", size=9, weight="bold")
        arrow(ax, 0.545, 0.81, 0.51, 0.76, color="#79589F", width=1.2)
        arrow(ax, 0.605, 0.81, 0.64, 0.76, color="#79589F", width=1.2)
        arrow(ax, 0.20, 0.665, 0.29, 0.665, color=blue, width=2)
        arrow(ax, 0.53, 0.665, 0.62, 0.665, color=cyan, width=2)
        arrow(ax, 0.80, 0.665, 0.84, 0.665, color=orange, width=2)
        acc = agg["baseline_full_vocab_accuracy"]["mean"]
        head_patch = agg["all_heads_patch_recovery"]["mean"]
        value_patch = agg["attention_values_patch_recovery"]["mean"]
        no_heads = agg["all_heads_mean_ablated_accuracy"]["mean"]
        top_ids = rep["neuron_interventions"]["top_32_ranking"][:8]
        ax.text(0.11, 0.39, f"held-out/validation acc  {acc:.2f}", ha="center", color=blue, weight="bold")
        ax.text(0.41, 0.35, f"all-head patch recovery  {head_patch:.2f}\nvalue-only recovery  {value_patch:.2f}\nall heads mean-ablated acc  {no_heads:.2f}",
                ha="center", va="top", fontsize=10, color=navy)
        ax.text(0.71, 0.31, "representative top units\n" + ", ".join(f"N{i}" for i in top_ids),
                ha="center", va="top", fontsize=9, color=navy)

    circuit_panel(
        goal1_ax,
        "B  Goal 1 compute graph — semantic, F=16",
        "Discovery: 128 trained subjects  →  evaluation: 64 subjects never seen in relation triples",
        g1,
        rep1,
        True,
    )
    circuit_panel(
        goal2_ax,
        "C  Goal 2 compute graph — random, F=64",
        "Discovery/evaluation: target-stratified disjoint halves of the 128 memorized facts",
        g2,
        rep2,
        False,
    )

    # Head causal results.
    heads = np.arange(4)
    width = 0.36
    g1_shap = [row["margin_shapley"]["mean"] for row in g1["heads"]]
    g2_shap = [row["margin_shapley"]["mean"] for row in g2["heads"]]
    g1_sd = [row["margin_shapley"]["sd"] for row in g1["heads"]]
    g2_sd = [row["margin_shapley"]["sd"] for row in g2["heads"]]
    heads_ax.axhline(0, color="#BBC4CA", linewidth=0.8)
    heads_ax.bar(heads - width / 2, g1_shap, width, yerr=g1_sd, label="Goal 1", color=blue, capsize=3)
    heads_ax.bar(heads + width / 2, g2_shap, width, yerr=g2_sd, label="Goal 2", color=orange, capsize=3)
    heads_ax.set_xticks(heads, [f"H{i}" for i in heads])
    heads_ax.set_ylabel("Exact margin Shapley effect")
    heads_ax.set_title("D  Which physical attention heads matter?", loc="left", weight="bold", color=navy)
    heads_ax.legend(frameon=False, ncol=2)
    heads_ax.spines[["top", "right"]].set_visible(False)
    heads_ax.grid(axis="y", alpha=0.18)

    # Neuron causal concentration curves.
    ks = np.array([8, 16, 32, 64, 128, 256])
    for goal, agg, color, marker in [
        ("Goal 1", g1, blue, "o"),
        ("Goal 2", g2, orange, "s"),
    ]:
        necessity = [
            agg["neuron_subject_specific_margin_fraction"][str(k)]["after_top_k_ablation"]["mean"]
            for k in ks
        ]
        sufficiency = [
            agg["neuron_subject_specific_margin_fraction"][str(k)]["with_only_top_k_variable"]["mean"]
            for k in ks
        ]
        neurons_ax.plot(ks, necessity, color=color, marker=marker, linewidth=2, label=f"{goal}: ablate top-k")
        neurons_ax.plot(ks, sufficiency, color=color, marker=marker, linewidth=1.6, linestyle="--", alpha=0.8,
                        label=f"{goal}: keep only top-k")
    neurons_ax.set_xscale("log", base=2)
    neurons_ax.set_xticks(ks, [str(k) for k in ks])
    neurons_ax.set_ylim(-0.04, 1.08)
    neurons_ax.set_xlabel("Number of ranked GELU units (out of 256)")
    neurons_ax.set_ylabel("Fraction of subject-specific MLP margin")
    neurons_ax.set_title("E  The MLP mechanism is distributed", loc="left", weight="bold", color=navy)
    neurons_ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left")
    neurons_ax.spines[["top", "right"]].set_visible(False)
    neurons_ax.grid(alpha=0.18)

    fig.suptitle(
        "From generic transformer blocks to the compute units used for factual recall",
        fontsize=19,
        weight="bold",
        color=navy,
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "Bars/curves: mean over seeds 19, 31, 47; error bars: across-seed SD. "
        "All causal rankings are evaluated on subjects disjoint from discovery.",
        ha="center",
        fontsize=9,
        color=gray,
    )
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(output_dir / f"goal1_goal2_forward_pass.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def run_analysis(seeds: Sequence[int], device_name: str, output_dir: Path) -> dict:
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for goal in GOALS:
        for seed in seeds:
            print(f"analyzing {goal} seed={seed} F={GOALS[goal]['occurrences']}", flush=True)
            runs.append(analyze_goal_seed(goal, seed, device))
    result = {
        "analysis": "Goal 1 / Goal 2 causal forward-pass mechanisms",
        "model": {
            "layers": 1,
            "hidden_dim": 64,
            "attention_heads": 4,
            "head_dim": 16,
            "mlp_neurons": 256,
        },
        "seeds": list(seeds),
        "goals": GOALS,
        "runs": runs,
    }
    result["aggregate"] = aggregate_results(runs)
    json_path = output_dir / "goal1_goal2_mechanistic_analysis.json"
    json_path.write_text(json.dumps(result, indent=2, default=_jsonable), encoding="utf-8")
    write_report(result, output_dir / "goal1_goal2_mechanistic_analysis.md")
    make_figure(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="19,31,47")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factual_recall/results/final_mechanistic_analysis"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    result = run_analysis([int(seed) for seed in args.seeds.split(",")], args.device, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "runs": len(result["runs"]),
                "all_replays_verified": all(
                    run["replay_verification"]["verified_exactly"] for run in result["runs"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

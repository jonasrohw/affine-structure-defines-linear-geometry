"""Causal circuit analysis for the two final linear-shift claims.

This module deterministically replays the archived Goal 1 (F=16) and Goal 2
(F=64) runs, verifies the replay against the stored behavioral metrics, and
then analyzes the learned representations and the computation that uses them.
The analysis deliberately combines descriptive decoding with causal interventions:

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

import itertools
import json
import math
from pathlib import Path
from typing import Sequence

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


GOALS = {"goal1": {"occurrences": 16, "relation": "semantic"}}
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
    world = World(cfg.world, cfg.seed, subjects_per_bucket=cfg.subjects_per_bucket)
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
    # The archived aggregate contains the original F=16 replay measurements.
    # Record drift explicitly; never substitute archived values for fresh measurements.
    reference_path = PROJECT_ROOT / "paper/reference/experiment1_mechanism.json"
    archive = json.loads(reference_path.read_text())
    reference = next((run["replay_verification"]["replay"] for run in archive["runs"]
                      if run["seed"] == seed and run["occurrences_per_triple"] == occurrences), None)
    comparisons = {}
    if reference is not None:
        comparisons["last_train_loss_abs_error"] = abs(last_loss - reference["last_train_loss"])
        for relation in ("semantic", "random"):
            for key, path in (("probe", ("probe_accuracy",)),
                              ("trained", ("trained_check", "recall")),
                              ("heldout", ("heldout_check", "recall"))):
                actual = replay["relations"][relation]
                expected = reference["relations"][relation]
                for part in path:
                    actual, expected = actual[part], expected[part]
                comparisons[f"{relation}_{key}_abs_error"] = abs(actual - expected)
    return model, world, trained_subjects, {
        "verified_against_archive": bool(comparisons) and all(e < 5e-7 for e in comparisons.values()),
        "verification_tolerance": 5e-7,
        "reference_path": "paper/reference/experiment1_mechanism.json",
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
    return {
        "goal": goal,
        "seed": seed,
        "occurrences_per_triple": occurrences,
        "relation": relation,
        "discovery_subjects": len(discovery),
        "evaluation_subjects": len(evaluation),
        "evaluation_split": (
            "never-seen relation subjects"
        ),
        "replay_verification": replay,
        "stage_analysis": stage,
        "matched_relation_stage_control": {"relation": control_relation, **control_stage},
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

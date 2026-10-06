"""The structured and joint next-token training stages of Experiment 2."""
from __future__ import annotations
import json
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .symmetric_lookup import _pair_pools, _sample_relation_batch, build_world
from .symmetric_lookup_transformer import (
    LookupTokenVocabulary, TransformerLookupConfig, _sample_attribute_batch,
    _world_config, all_ols_metrics, attribute_accuracy, make_model,
    relation_accuracies, relation_symmetry_agreements, relation_prompts,
)

def reverse_attribute_prompts(
    vocab: LookupTokenVocabulary,
    attribute_x: torch.Tensor,
    attribute_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn ``subject attribute -> value`` facts into next-token inverse facts."""
    subjects = attribute_x[:, 1]
    prompts = torch.stack(
        (
            torch.full_like(subjects, vocab.bos_id),
            attribute_x[:, 2],
            attribute_y,
        ),
        dim=1,
    )
    return prompts, subjects



def train_structured_geometry(
    cfg: TransformerLookupConfig,
    device: torch.device,
    reverse_attribute_weight: float = 0.0,
) -> dict[str, object]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    world = build_world(_world_config(cfg))
    vocab = LookupTokenVocabulary(cfg.n_entities, cfg.n_attributes)
    model = make_model(cfg, vocab).to(device)
    rng = np.random.default_rng(cfg.seed + 101)
    pools = _pair_pools(world.labels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    class_weights = torch.ones(vocab.vocab_size, device=device)
    class_weights[vocab.yes_id] = cfg.positive_label_loss_weight
    relation_ids = list(range(world.n_attributes))
    sampling_ids = relation_ids.copy()
    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = (-1.0, -1.0, -1.0)

    for step in range(1, cfg.phase1_steps + 1):
        relations, subjects, objects, values = _sample_relation_batch(
            world,
            sampling_ids,
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
        reverse_x, reverse_subjects = reverse_attribute_prompts(
            vocab, attribute_x, attribute_y
        )
        relation_loss = F.cross_entropy(
            model(relation_x)[:, -1], relation_y, weight=class_weights
        )
        attribute_loss = F.cross_entropy(
            model(attribute_x)[:, -1], attribute_y
        )
        reverse_attribute_loss = F.cross_entropy(
            model(reverse_x)[:, -1], reverse_subjects
        )
        loss = (
            relation_loss
            + cfg.attribute_loss_weight * attribute_loss
            + reverse_attribute_weight * reverse_attribute_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        warmup = min(1.0, step / max(1, cfg.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * warmup
        optimizer.step()

        if step % cfg.eval_every != 0 and step != cfg.phase1_steps:
            continue
        attributes = attribute_accuracy(model, world, vocab, device)
        relations_accuracy = relation_accuracies(
            model, world, vocab, device, relation_ids
        )
        ols = all_ols_metrics(model, world, vocab, cfg)
        structured_r2 = [row["full_r2"] for row in ols[: world.n_attributes]]
        row = {
            "step": step,
            "loss": float(loss.item()),
            "reverse_attribute_loss": float(reverse_attribute_loss.item()),
            "attribute_accuracy": attributes,
            "min_structured_accuracy": min(relations_accuracy),
            "mean_structured_accuracy": float(np.mean(relations_accuracy)),
            "min_raw_structured_full_r2": min(structured_r2),
            "mean_raw_structured_full_r2": float(np.mean(structured_r2)),
        }
        history.append(row)
        behavior_pass = float(
            attributes >= cfg.target_accuracy
            and min(relations_accuracy) >= cfg.target_accuracy
        )
        score = (
            behavior_pass,
            min(structured_r2),
            float(np.mean(structured_r2)),
        )
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        errors = np.maximum(1.0 - np.asarray(relations_accuracy), 1e-3)
        repeats = 1 + np.rint(9.0 * errors / errors.max()).astype(int)
        sampling_ids = [
            relation
            for relation, repeat in zip(relation_ids, repeats, strict=True)
            for _ in range(int(repeat))
        ]
        print(json.dumps(row), flush=True)
        model.train()

    if best_state is None:
        raise RuntimeError("No structured checkpoint was evaluated")
    model.load_state_dict(best_state)
    final_ols = all_ols_metrics(model, world, vocab, cfg)
    final_relations = relation_accuracies(
        model, world, vocab, device, relation_ids
    )
    result = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "reverse_attribute_weight": reverse_attribute_weight,
        "num_parameters": model.n_parameters(),
        "history": history,
        "best": {
            "attribute_accuracy": attribute_accuracy(model, world, vocab, device),
            "relation_accuracy": final_relations,
            "min_structured_accuracy": min(final_relations),
            "raw_embedding_ols": final_ols[: world.n_attributes],
            "min_raw_structured_full_r2": min(
                row["full_r2"] for row in final_ols[: world.n_attributes]
            ),
            "mean_raw_structured_full_r2": float(
                np.mean(
                    [row["full_r2"] for row in final_ols[: world.n_attributes]]
                )
            ),
        },
    }
    result["passes_structured_gate"] = bool(
        result["best"]["attribute_accuracy"] >= cfg.target_accuracy
        and result["best"]["min_structured_accuracy"] >= cfg.target_accuracy
        and result["best"]["min_raw_structured_full_r2"] >= 0.95
    )
    result["_state_dict"] = best_state
    return result



def train_joint_geometry(
    cfg: TransformerLookupConfig,
    checkpoint: Path,
    device: torch.device,
    reverse_attribute_weight: float = 4.0,
) -> dict[str, object]:
    """Learn all eleven relations while preserving raw attribute geometry."""
    torch.manual_seed(cfg.seed + 50_000)
    np.random.seed(cfg.seed + 50_000)
    random.seed(cfg.seed + 50_000)
    world = build_world(_world_config(cfg))
    vocab = LookupTokenVocabulary(cfg.n_entities, cfg.n_attributes)
    model = make_model(cfg, vocab).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Every parameter must remain trainable")
    rng = np.random.default_rng(cfg.seed + 50_303)
    pools = _pair_pools(world.labels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    class_weights = torch.ones(vocab.vocab_size, device=device)
    class_weights[vocab.yes_id] = cfg.positive_label_loss_weight
    arbitrary_repeats = round(
        cfg.arbitrary_relation_fraction
        * world.n_attributes
        / (1.0 - cfg.arbitrary_relation_fraction)
    )
    sampling_ids = list(range(world.n_attributes)) + [
        world.arbitrary_relation
    ] * arbitrary_repeats
    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = (-1.0, -1.0, -1.0, -1.0)

    for step in range(1, cfg.joint_steps + 1):
        relations, subjects, objects, values = _sample_relation_batch(
            world,
            sampling_ids,
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
        reverse_x, reverse_subjects = reverse_attribute_prompts(
            vocab, attribute_x, attribute_y
        )
        relation_loss = F.cross_entropy(
            model(relation_x)[:, -1], relation_y, weight=class_weights
        )
        attribute_loss = F.cross_entropy(
            model(attribute_x)[:, -1], attribute_y
        )
        reverse_attribute_loss = F.cross_entropy(
            model(reverse_x)[:, -1], reverse_subjects
        )
        loss = (
            relation_loss
            + cfg.attribute_loss_weight * attribute_loss
            + reverse_attribute_weight * reverse_attribute_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        warmup = min(1.0, step / max(1, cfg.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * warmup
        optimizer.step()

        if step % cfg.eval_every != 0 and step != cfg.joint_steps:
            continue
        attributes = attribute_accuracy(model, world, vocab, device)
        relations_accuracy = relation_accuracies(model, world, vocab, device)
        ols = all_ols_metrics(model, world, vocab, cfg)
        structured_r2 = [row["full_r2"] for row in ols[: world.n_attributes]]
        arbitrary_r2 = ols[world.arbitrary_relation]["full_r2"]
        geometry_pass = float(
            min(structured_r2) >= 0.95 and arbitrary_r2 < 0.50
        )
        behavior_pass = float(
            attributes >= cfg.target_accuracy
            and min(relations_accuracy) >= cfg.target_accuracy
        )
        row = {
            "step": step,
            "loss": float(loss.item()),
            "reverse_attribute_loss": float(reverse_attribute_loss.item()),
            "attribute_accuracy": attributes,
            "min_structured_accuracy": min(
                relations_accuracy[: world.n_attributes]
            ),
            "arbitrary_accuracy": relations_accuracy[world.arbitrary_relation],
            "min_all_relation_accuracy": min(relations_accuracy),
            "min_raw_structured_full_r2": min(structured_r2),
            "mean_raw_structured_full_r2": float(np.mean(structured_r2)),
            "arbitrary_raw_full_r2": arbitrary_r2,
            "arbitrary_raw_cv_r2": ols[world.arbitrary_relation]["cv_r2"],
        }
        history.append(row)
        score = (
            behavior_pass,
            geometry_pass,
            min(relations_accuracy),
            min(structured_r2) - max(arbitrary_r2 - 0.50, 0.0),
        )
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(json.dumps(row), flush=True)
        if behavior_pass and geometry_pass:
            break
        model.train()

    if best_state is None:
        raise RuntimeError("No joint checkpoint was evaluated")
    model.load_state_dict(best_state)
    final_relations = relation_accuracies(model, world, vocab, device)
    final_symmetry = relation_symmetry_agreements(model, world, vocab, device)
    final_ols = all_ols_metrics(model, world, vocab, cfg)
    result = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "source_checkpoint": str(checkpoint),
        "reverse_attribute_weight": reverse_attribute_weight,
        "num_parameters": model.n_parameters(),
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "history": history,
        "final": {
            "attribute_accuracy": attribute_accuracy(model, world, vocab, device),
            "relation_accuracy": final_relations,
            "min_structured_accuracy": min(
                final_relations[: world.n_attributes]
            ),
            "arbitrary_accuracy": final_relations[world.arbitrary_relation],
            "symmetry_agreement": final_symmetry,
            "min_symmetry_agreement": min(final_symmetry),
            "raw_embedding_ols": {
                "structured": final_ols[: world.n_attributes],
                "arbitrary": final_ols[world.arbitrary_relation],
                "min_structured_full_r2": min(
                    row["full_r2"]
                    for row in final_ols[: world.n_attributes]
                ),
                "mean_structured_full_r2": float(
                    np.mean(
                        [
                            row["full_r2"]
                            for row in final_ols[: world.n_attributes]
                        ]
                    )
                ),
            },
        },
    }
    result["passes_joint_gate"] = bool(
        result["final"]["attribute_accuracy"] >= cfg.target_accuracy
        and min(result["final"]["relation_accuracy"]) >= cfg.target_accuracy
        and result["final"]["min_symmetry_agreement"] >= cfg.target_accuracy
        and result["final"]["raw_embedding_ols"][
            "min_structured_full_r2"
        ]
        >= 0.95
        and result["final"]["raw_embedding_ols"]["arbitrary"]["full_r2"]
        < 0.50
        and result["frozen_parameter_count"] == 0
    )
    result["_state_dict"] = best_state
    return result

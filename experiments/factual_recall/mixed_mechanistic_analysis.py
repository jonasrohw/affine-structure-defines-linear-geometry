"""Mechanistic analysis for the mixed structured/random relation experiment."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .model import CausalTransformer, build_model
from .synthetic_curriculum import (
    SyntheticExperimentConfig,
    evaluate_mapped_triples,
    load_synthetic_config,
    prompt_relation_metrics,
    triple_prompt_inputs,
)


def reconstruct_mixed_split(cfg: SyntheticExperimentConfig) -> dict:
    rng = np.random.default_rng(cfg.curriculum.seed)
    shuffled = np.arange(cfg.vocab.n_entities)
    rng.shuffle(shuffled)
    train_n = min(cfg.curriculum.phase2_triples, cfg.vocab.n_entities)
    eval_n = min(cfg.curriculum.eval_limit, cfg.vocab.n_entities - train_n)
    train_subjects = sorted(int(i) for i in shuffled[:train_n])
    never_seen_subjects = sorted(int(i) for i in shuffled[train_n : train_n + eval_n])
    return {
        "train_subjects": train_subjects,
        "never_seen_subjects": never_seen_subjects,
        "structured_targets": np.arange(cfg.vocab.n_entities, dtype=np.int64),
        "random_targets": rng.permutation(cfg.vocab.n_entities).astype(np.int64),
    }


def load_mixed_model(cfg: SyntheticExperimentConfig, checkpoint: Path, device: torch.device) -> CausalTransformer:
    model = build_model(cfg.model, cfg.vocab)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def metric_subset(row: dict[str, float]) -> dict[str, float]:
    keys = [
        "train_accuracy",
        "train_top10_accuracy",
        "train_mean_rank",
        "eval_accuracy",
        "eval_top10_accuracy",
        "eval_mean_rank",
        "prompt_affine_cosine",
        "prompt_affine_nn_accuracy",
        "prompt_affine_nn_mean_rank",
    ]
    return {key: float(row[key]) for key in keys}


def relation_metrics(model: CausalTransformer, cfg: SyntheticExperimentConfig, split: dict, device: torch.device) -> dict:
    structured = prompt_relation_metrics(
        model,
        cfg.vocab,
        split["structured_targets"],
        split["train_subjects"],
        split["never_seen_subjects"],
        relation_idx=0,
        device=device,
        eval_limit=cfg.curriculum.eval_limit,
    )
    random = prompt_relation_metrics(
        model,
        cfg.vocab,
        split["random_targets"],
        split["train_subjects"],
        split["never_seen_subjects"],
        relation_idx=1,
        device=device,
        eval_limit=cfg.curriculum.eval_limit,
    )
    return {
        "structured": metric_subset(structured),
        "random": metric_subset(random),
    }


@contextmanager
def patched_model(model: CausalTransformer, patches: list[tuple[object, str, object]]) -> Iterator[None]:
    originals = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    try:
        for obj, name, value in patches:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in originals:
            setattr(obj, name, value)


def zero_attn_forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def zero_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def linear_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc2(self.fc1(x))


def neuron_ablation_forward(neuron_ids: list[int]):
    ids = torch.tensor(neuron_ids, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        local_ids = ids.to(hidden.device)
        hidden = hidden.clone()
        hidden.index_fill_(-1, local_ids, 0.0)
        return self.fc2(hidden)

    return forward


def component_ablation_suite(
    model: CausalTransformer,
    cfg: SyntheticExperimentConfig,
    split: dict,
    device: torch.device,
) -> dict:
    baseline = relation_metrics(model, cfg, split, device)
    rows = {"baseline": baseline}
    for layer_idx, block in enumerate(model.blocks):
        with patched_model(
            model,
            [(block.attn, "forward", MethodType(zero_attn_forward, block.attn))],
        ):
            rows[f"zero_attn_layer_{layer_idx}"] = relation_metrics(model, cfg, split, device)
        with patched_model(
            model,
            [(block.mlp, "forward", MethodType(zero_mlp_forward, block.mlp))],
        ):
            rows[f"zero_mlp_layer_{layer_idx}"] = relation_metrics(model, cfg, split, device)

    attn_patches = [
        (block.attn, "forward", MethodType(zero_attn_forward, block.attn))
        for block in model.blocks
    ]
    mlp_patches = [
        (block.mlp, "forward", MethodType(zero_mlp_forward, block.mlp))
        for block in model.blocks
    ]
    linear_patches = [
        (block.mlp, "forward", MethodType(linear_mlp_forward, block.mlp))
        for block in model.blocks
    ]
    with patched_model(model, attn_patches):
        rows["zero_all_attention"] = relation_metrics(model, cfg, split, device)
    with patched_model(model, mlp_patches):
        rows["zero_all_mlp"] = relation_metrics(model, cfg, split, device)
    with patched_model(model, linear_patches):
        rows["linearize_all_gelu"] = relation_metrics(model, cfg, split, device)
    return rows


def manual_forward_cache(
    model: CausalTransformer,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    _, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    x = model.token_emb(input_ids) + model.pos_emb(positions)
    caches: list[dict[str, torch.Tensor]] = []
    for block in model.blocks:
        ln1 = block.ln1(x)
        batch_size, _, hidden_dim = ln1.shape
        qkv = block.attn.qkv(ln1)
        q, k, v = qkv.split(hidden_dim, dim=2)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).bool()
        weights = F.softmax(scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
        attn_out = weights @ v
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        attn_out = block.attn.out_proj(attn_out)
        x = x + attn_out
        ln2 = block.ln2(x)
        pre = block.mlp.fc1(ln2)
        act = F.gelu(pre)
        mlp_out = block.mlp.fc2(act)
        x = x + mlp_out
        caches.append(
            {
                "attention_weights": weights.detach().cpu(),
                "mlp_pre": pre.detach().cpu(),
                "mlp_act": act.detach().cpu(),
                "attn_out": attn_out.detach().cpu(),
                "mlp_out": mlp_out.detach().cpu(),
                "resid": x.detach().cpu(),
            }
        )
    logits = model.lm_head(model.ln_f(x))
    return logits, caches


def attention_analysis(
    model: CausalTransformer,
    cfg: SyntheticExperimentConfig,
    split: dict,
    device: torch.device,
) -> dict:
    result: dict[str, list[dict[str, object]]] = {}
    subjects = split["train_subjects"][: min(64, len(split["train_subjects"]))]
    labels = ["sep", "subject", "relation"]
    with torch.no_grad():
        for relation_name, relation_idx in [("structured", 0), ("random", 1)]:
            x = triple_prompt_inputs(cfg.vocab, subjects, relation_idx=relation_idx).to(device)
            _, caches = manual_forward_cache(model, x)
            layer_rows = []
            for layer_idx, cache in enumerate(caches):
                weights = cache["attention_weights"][:, :, -1, :].numpy()
                per_head = weights.mean(axis=0)
                layer_rows.append(
                    {
                        "layer": layer_idx,
                        "mean_final_query_attention": {
                            labels[pos]: float(per_head[:, pos].mean())
                            for pos in range(len(labels))
                        },
                        "per_head_final_query_attention": [
                            {labels[pos]: float(per_head[head, pos]) for pos in range(len(labels))}
                            for head in range(per_head.shape[0])
                        ],
                    }
                )
            result[relation_name] = layer_rows
    return result


def collect_mlp_activations(
    model: CausalTransformer,
    cfg: SyntheticExperimentConfig,
    subjects: list[int],
    relation_idx: int,
    device: torch.device,
) -> list[np.ndarray]:
    rows: list[list[np.ndarray]] | None = None
    with torch.no_grad():
        for start in range(0, len(subjects), 128):
            batch = subjects[start : start + 128]
            x = triple_prompt_inputs(cfg.vocab, batch, relation_idx=relation_idx).to(device)
            _, caches = manual_forward_cache(model, x)
            if rows is None:
                rows = [[] for _ in caches]
            for layer_idx, cache in enumerate(caches):
                rows[layer_idx].append(cache["mlp_act"][:, -1, :].numpy().astype(np.float64))
    return [np.concatenate(layer_rows, axis=0) for layer_rows in (rows or [])]


def neuron_ranking(
    model: CausalTransformer,
    cfg: SyntheticExperimentConfig,
    split: dict,
    device: torch.device,
    top_k: int = 16,
) -> dict:
    subjects = split["train_subjects"]
    rel0 = collect_mlp_activations(model, cfg, subjects, 0, device)
    rel1 = collect_mlp_activations(model, cfg, subjects, 1, device)
    random_targets = split["random_targets"]
    structured_targets = split["structured_targets"]
    unembed = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    rows = []
    top_random_by_layer: dict[int, list[int]] = {}
    top_structured_by_layer: dict[int, list[int]] = {}
    for layer_idx, (act0, act1) in enumerate(zip(rel0, rel1)):
        block = model.blocks[layer_idx]
        fc2 = block.mlp.fc2.weight.detach().cpu().numpy().astype(np.float64)
        mean0 = act0.mean(axis=0)
        mean1 = act1.mean(axis=0)
        pooled_std = np.sqrt(0.5 * act0.var(axis=0) + 0.5 * act1.var(axis=0)) + 1e-6
        d_score = (mean1 - mean0) / pooled_std
        random_target_vecs = unembed[[cfg.vocab.object_id(int(random_targets[i])) for i in subjects]]
        structured_target_vecs = unembed[[cfg.vocab.object_id(int(structured_targets[i])) for i in subjects]]
        random_direct = act1 * (random_target_vecs @ fc2)
        structured_direct = act0 * (structured_target_vecs @ fc2)
        random_direct_mean = random_direct.mean(axis=0)
        structured_direct_mean = structured_direct.mean(axis=0)
        top_random = np.argsort(-random_direct_mean)[:top_k].tolist()
        top_structured = np.argsort(-structured_direct_mean)[:top_k].tolist()
        top_random_by_layer[layer_idx] = [int(i) for i in top_random]
        top_structured_by_layer[layer_idx] = [int(i) for i in top_structured]
        selected = sorted(
            set(np.argsort(-np.abs(d_score))[:top_k].tolist() + top_random + top_structured)
        )
        for neuron_idx in selected:
            rows.append(
                {
                    "layer": layer_idx,
                    "neuron": int(neuron_idx),
                    "relation_d_score_random_minus_structured": float(d_score[neuron_idx]),
                    "mean_activation_structured": float(mean0[neuron_idx]),
                    "mean_activation_random": float(mean1[neuron_idx]),
                    "approx_direct_structured_target_logit": float(structured_direct_mean[neuron_idx]),
                    "approx_direct_random_target_logit": float(random_direct_mean[neuron_idx]),
                }
            )
    rows.sort(
        key=lambda row: max(
            abs(row["relation_d_score_random_minus_structured"]),
            abs(row["approx_direct_random_target_logit"]),
            abs(row["approx_direct_structured_target_logit"]),
        ),
        reverse=True,
    )
    return {
        "top_neurons": rows[: max(25, top_k)],
        "top_random_direct_by_layer": top_random_by_layer,
        "top_structured_direct_by_layer": top_structured_by_layer,
    }


def targeted_neuron_ablations(
    model: CausalTransformer,
    cfg: SyntheticExperimentConfig,
    split: dict,
    device: torch.device,
    ranking: dict,
) -> dict:
    rows = {}
    for label, key in [
        ("ablate_random_direct", "top_random_direct_by_layer"),
        ("ablate_structured_direct", "top_structured_direct_by_layer"),
    ]:
        patches = []
        for layer_idx, neuron_ids in ranking[key].items():
            block = model.blocks[int(layer_idx)]
            patches.append(
                (
                    block.mlp,
                    "forward",
                    MethodType(neuron_ablation_forward(neuron_ids), block.mlp),
                )
            )
        with patched_model(model, patches):
            rows[label] = relation_metrics(model, cfg, split, device)
    return rows


def write_markdown_summary(result: dict, path: Path) -> None:
    baseline = result["component_ablations"]["baseline"]
    linearized = result["component_ablations"]["linearize_all_gelu"]
    random_ablate = result["targeted_neuron_ablations"]["ablate_random_direct"]
    lines = [
        "# Mixed Mechanistic Analysis",
        "",
        "## Baseline",
        f"- Structured prompt affine cosine: {baseline['structured']['prompt_affine_cosine']:.3f}",
        f"- Structured prompt affine NN accuracy: {baseline['structured']['prompt_affine_nn_accuracy']:.3f}",
        f"- Random train top-1: {baseline['random']['train_accuracy']:.3f}",
        f"- Random prompt affine cosine: {baseline['random']['prompt_affine_cosine']:.3f}",
        "",
        "## Nonlinearity",
        f"- Linearizing all GELUs changes random train top-1 from {baseline['random']['train_accuracy']:.3f} to {linearized['random']['train_accuracy']:.3f}.",
        f"- Linearizing all GELUs changes structured affine NN accuracy from {baseline['structured']['prompt_affine_nn_accuracy']:.3f} to {linearized['structured']['prompt_affine_nn_accuracy']:.3f}.",
        "",
        "## Targeted Neuron Ablation",
        f"- Ablating top random-direct MLP neurons changes random train top-1 from {baseline['random']['train_accuracy']:.3f} to {random_ablate['random']['train_accuracy']:.3f}.",
        f"- It changes structured affine NN accuracy from {baseline['structured']['prompt_affine_nn_accuracy']:.3f} to {random_ablate['structured']['prompt_affine_nn_accuracy']:.3f}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(config: Path, checkpoint: Path, output_dir: Path, device_name: str) -> dict:
    cfg = load_synthetic_config(config)
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_mixed_model(cfg, checkpoint, device)
    split = reconstruct_mixed_split(cfg)
    result = {
        "config": {
            "config_path": str(config),
            "checkpoint_path": str(checkpoint),
            "vocab": asdict(cfg.vocab),
            "model": asdict(cfg.model),
            "curriculum": {
                **asdict(cfg.curriculum),
                "output_dir": str(cfg.curriculum.output_dir),
            },
        },
        "split": {
            "train_subjects": len(split["train_subjects"]),
            "never_seen_subjects": len(split["never_seen_subjects"]),
        },
        "component_ablations": component_ablation_suite(model, cfg, split, device),
        "attention": attention_analysis(model, cfg, split, device),
        "neurons": neuron_ranking(model, cfg, split, device),
    }
    result["targeted_neuron_ablations"] = targeted_neuron_ablations(
        model,
        cfg,
        split,
        device,
        result["neurons"],
    )
    result_path = output_dir / "mixed_mechanistic_analysis.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    write_markdown_summary(result, output_dir / "mixed_mechanistic_analysis.md")
    result["result_path"] = str(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/runs/synthetic_mixed_relations_smoke.toml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("factual_recall/results/synthetic_mixed_relations_smoke/mixed_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factual_recall/results/synthetic_mixed_relations_smoke"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_analysis(args.config, args.checkpoint, args.output_dir, args.device)
    print(json.dumps({"result_path": result["result_path"]}, indent=2))


if __name__ == "__main__":
    main()

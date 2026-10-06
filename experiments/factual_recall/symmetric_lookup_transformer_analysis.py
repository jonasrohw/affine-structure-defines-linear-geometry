"""Interventions and residual diagnostics for Experiment 2."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from .model import CausalTransformer, TransformerBlock
from .symmetric_lookup import _positive_centroids
from .symmetric_lookup_transformer import LookupTokenVocabulary, relation_prompts

SEEDS = (19, 31, 47)

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

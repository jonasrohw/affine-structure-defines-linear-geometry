"""GPT model, tokenization and evaluation for Experiment 2."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from .config import PROJECT_ROOT, TransformerConfig
from .model import CausalTransformer, build_model
from .symmetric_lookup import SymmetricLookupConfig, SymmetricWorld, ols_relation_metrics

@dataclass
class TransformerLookupConfig:
    seed: int = 19
    n_entities: int = 64
    n_attributes: int = 10
    hidden_dim: int = 20
    n_layers: int = 8
    n_heads: int = 4
    tie_embeddings: bool = True
    positive_fraction: float = 0.1
    positive_label_loss_weight: float = 9.0
    arbitrary_relation_fraction: float = 0.5
    batch_size: int = 1024
    attribute_batch_size: int = 640
    phase1_steps: int = 6000
    joint_steps: int = 12000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    attribute_loss_weight: float = 1.0
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_every: int = 500
    target_accuracy: float = 0.995
    require_symmetry_for_stopping: bool = True
    ols_folds: int = 4
    graph_swaps_per_edge: int = 40
    output_dir: Path = PROJECT_ROOT / "results" / "experiment2"

@dataclass(frozen=True)
class LookupTokenVocabulary:
    """Token layout for the two sentence templates."""

    n_entities: int
    n_attributes: int

    @property
    def bos_id(self) -> int:
        return 0

    @property
    def entity_offset(self) -> int:
        return 1

    @property
    def attribute_offset(self) -> int:
        return self.entity_offset + self.n_entities

    @property
    def relation_offset(self) -> int:
        return self.attribute_offset + self.n_attributes

    @property
    def zero_id(self) -> int:
        return self.relation_offset + self.n_attributes + 1

    @property
    def one_id(self) -> int:
        return self.zero_id + 1

    @property
    def no_id(self) -> int:
        return self.one_id + 1

    @property
    def yes_id(self) -> int:
        return self.no_id + 1

    @property
    def vocab_size(self) -> int:
        return self.yes_id + 1

    def entity_id(self, index: int) -> int:
        return self.entity_offset + index

    def attribute_id(self, index: int) -> int:
        return self.attribute_offset + index

    def relation_id(self, index: int) -> int:
        return self.relation_offset + index


def _world_config(cfg: TransformerLookupConfig) -> SymmetricLookupConfig:
    return SymmetricLookupConfig(
        seed=cfg.seed,
        n_entities=cfg.n_entities,
        n_attributes=cfg.n_attributes,
        graph_swaps_per_edge=cfg.graph_swaps_per_edge,
    )



def make_model(
    cfg: TransformerLookupConfig, vocab: LookupTokenVocabulary
) -> CausalTransformer:
    model_cfg = TransformerConfig(
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        context_len=4,
        tie_embeddings=cfg.tie_embeddings,
    )
    return build_model(model_cfg, vocab)  # type: ignore[arg-type]



def attribute_prompts(
    vocab: LookupTokenVocabulary,
    subjects: np.ndarray,
    attributes: np.ndarray,
    values: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = np.stack(
        (
            np.full(len(subjects), vocab.bos_id),
            subjects + vocab.entity_offset,
            attributes + vocab.attribute_offset,
        ),
        axis=1,
    )
    targets = np.where(values.astype(bool), vocab.one_id, vocab.zero_id)
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )



def relation_prompts(
    vocab: LookupTokenVocabulary,
    relations: torch.Tensor,
    subjects: torch.Tensor,
    objects: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bos = torch.full_like(relations, vocab.bos_id)
    rows = torch.stack(
        (
            bos,
            relations + vocab.relation_offset,
            subjects + vocab.entity_offset,
            objects + vocab.entity_offset,
        ),
        dim=1,
    )
    targets = torch.where(
        values.bool(),
        torch.full_like(relations, vocab.yes_id),
        torch.full_like(relations, vocab.no_id),
    )
    return rows, targets



def _sample_attribute_batch(
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    subjects = rng.integers(0, world.n_entities, size=batch_size)
    attributes = rng.integers(0, world.n_attributes, size=batch_size)
    values = world.attributes[subjects, attributes]
    return attribute_prompts(vocab, subjects, attributes, values, device)



@torch.no_grad()
def attribute_accuracy(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    device: torch.device,
) -> float:
    subjects, attributes = np.indices((world.n_entities, world.n_attributes))
    values = world.attributes.reshape(-1)
    prompts, targets = attribute_prompts(
        vocab, subjects.reshape(-1), attributes.reshape(-1), values, device
    )
    was_training = model.training
    model.eval()
    predictions = model(prompts)[:, -1].argmax(dim=-1)
    if was_training:
        model.train()
    return float((predictions == targets).float().mean().item())



@torch.no_grad()
def relation_accuracies(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    device: torch.device,
    relation_ids: Sequence[int] | None = None,
    reverse: bool = False,
) -> list[float]:
    ids = list(range(world.n_relations) if relation_ids is None else relation_ids)
    subjects, objects = np.indices((world.n_entities, world.n_entities))
    subjects_t = torch.as_tensor(subjects.reshape(-1), dtype=torch.long, device=device)
    objects_t = torch.as_tensor(objects.reshape(-1), dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    output: list[float] = []
    for relation in ids:
        relations_t = torch.full_like(subjects_t, relation)
        values_t = torch.as_tensor(
            world.labels[relation].reshape(-1), dtype=torch.float32, device=device
        )
        left, right = (objects_t, subjects_t) if reverse else (subjects_t, objects_t)
        prompts, targets = relation_prompts(
            vocab, relations_t, left, right, values_t
        )
        predictions = model(prompts)[:, -1].argmax(dim=-1)
        output.append(float((predictions == targets).float().mean().item()))
    if was_training:
        model.train()
    return output



@torch.no_grad()
def relation_symmetry_agreements(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    device: torch.device,
) -> list[float]:
    """Fraction of pairs whose prediction is unchanged when entities swap."""
    subjects, objects = np.indices((world.n_entities, world.n_entities))
    subjects_t = torch.as_tensor(subjects.reshape(-1), dtype=torch.long, device=device)
    objects_t = torch.as_tensor(objects.reshape(-1), dtype=torch.long, device=device)
    dummy_values = torch.zeros_like(subjects_t, dtype=torch.float32)
    was_training = model.training
    model.eval()
    output: list[float] = []
    for relation in range(world.n_relations):
        relations_t = torch.full_like(subjects_t, relation)
        forward_x, _ = relation_prompts(
            vocab, relations_t, subjects_t, objects_t, dummy_values
        )
        reverse_x, _ = relation_prompts(
            vocab, relations_t, objects_t, subjects_t, dummy_values
        )
        forward = model(forward_x)[:, -1].argmax(dim=-1)
        reverse = model(reverse_x)[:, -1].argmax(dim=-1)
        output.append(float((forward == reverse).float().mean().item()))
    if was_training:
        model.train()
    return output



def all_ols_metrics(
    model: CausalTransformer,
    world: SymmetricWorld,
    vocab: LookupTokenVocabulary,
    cfg: TransformerLookupConfig,
) -> list[dict[str, float]]:
    entity_ids = torch.arange(
        vocab.entity_offset,
        vocab.entity_offset + world.n_entities,
        device=model.token_emb.weight.device,
    )
    embeddings = model.token_emb(entity_ids).detach().cpu().numpy()
    return [
        ols_relation_metrics(
            embeddings, world.labels[relation], cfg.ols_folds, cfg.seed
        )
        for relation in range(world.n_relations)
    ]

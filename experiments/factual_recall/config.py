"""Configuration objects for synthetic factual recall experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

R2_PERM_100: list[int] = [
    83, 53, 70, 45, 44, 39, 22, 80, 10, 0, 18, 30, 73, 33, 90, 4,
    76, 77, 12, 31, 55, 88, 26, 42, 69, 15, 40, 96, 9, 72, 11, 47,
    85, 28, 93, 5, 66, 65, 35, 16, 49, 34, 7, 95, 27, 19, 81, 25,
    62, 13, 24, 3, 17, 38, 8, 78, 6, 64, 36, 89, 56, 99, 54, 43,
    50, 67, 46, 68, 61, 97, 79, 41, 58, 48, 98, 57, 75, 32, 94, 59,
    63, 84, 37, 29, 1, 52, 21, 2, 23, 87, 91, 74, 86, 82, 20, 60,
    71, 14, 92, 51,
]


def make_permutation(size: int, seed: int = 42) -> list[int]:
    """Return a deterministic object permutation for permutation relations."""
    if size == 100 and seed == 42:
        return list(R2_PERM_100)
    return np.random.RandomState(seed).permutation(size).tolist()


@dataclass(frozen=True)
class RelationSpec:
    """Declarative subject-to-object relation."""

    name: str
    kind: Literal["identity", "cyclic_shift", "permutation"]
    shift: int = 0
    seed: int = 42
    permutation: tuple[int, ...] | None = None

    def map_subject(self, subject_idx: int, num_objects: int) -> int:
        if self.kind == "identity":
            return subject_idx
        if self.kind == "cyclic_shift":
            return (subject_idx + self.shift) % num_objects
        if self.kind == "permutation":
            perm = self.permutation or tuple(make_permutation(num_objects, self.seed))
            return perm[subject_idx]
        raise ValueError(f"Unknown relation kind: {self.kind}")


def default_vienna_relations(num_objects: int) -> tuple[RelationSpec, ...]:
    """Relations used by the original Vienna filler-token experiment."""
    return (
        RelationSpec(name="identity", kind="identity"),
        RelationSpec(name="half_shift", kind="cyclic_shift", shift=num_objects // 2),
        RelationSpec(name="seeded_permutation", kind="permutation", seed=42),
    )


@dataclass
class TokenVocabulary:
    """Token layout and relation semantics for subject-relation-object sequences."""

    n_subjects: int = 100
    n_objects: int = 100
    num_fillers: int = 3
    n_train_pairs: int | None = None
    n_heldout_pairs: int = 0
    sep_id: int = 0
    relation_specs: tuple[RelationSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.n_train_pairs is None:
            self.n_train_pairs = self.n_subjects - self.n_heldout_pairs
        if not self.relation_specs:
            self.relation_specs = default_vienna_relations(self.n_objects)

    @property
    def num_relations(self) -> int:
        return len(self.relation_specs)

    @property
    def subject_offset(self) -> int:
        return 1

    @property
    def object_offset(self) -> int:
        return self.subject_offset + self.n_subjects

    @property
    def relation_offset(self) -> int:
        return self.object_offset + self.n_objects

    @property
    def filler_offset(self) -> int:
        return self.relation_offset + self.num_relations

    @property
    def vocab_size(self) -> int:
        return self.filler_offset + self.num_fillers

    @property
    def train_indices(self) -> list[int]:
        return list(range(self.n_train_pairs or 0))

    @property
    def heldout_indices(self) -> list[int]:
        return list(range(self.n_train_pairs or 0, self.n_subjects))

    def subject_id(self, index: int) -> int:
        return self.subject_offset + index

    def object_id(self, index: int) -> int:
        return self.object_offset + index

    def relation_id(self, index: int) -> int:
        return self.relation_offset + index

    def filler_id(self, index: int) -> int:
        return self.filler_offset + index

    @property
    def filler_ids(self) -> list[int]:
        return [self.filler_id(k) for k in range(self.num_fillers)]

    def relation_names(self) -> list[str]:
        return [relation.name for relation in self.relation_specs]

    def apply_relation(self, relation_idx: int, subject_idx: int) -> int:
        return self.relation_specs[relation_idx].map_subject(subject_idx, self.n_objects)

    def token_to_str(self, token_id: int) -> str:
        if token_id == self.sep_id:
            return "x"
        if self.subject_offset <= token_id < self.object_offset:
            return f"a{token_id - self.subject_offset}"
        if self.object_offset <= token_id < self.relation_offset:
            return f"b{token_id - self.object_offset}"
        if self.relation_offset <= token_id < self.filler_offset:
            return f"r{token_id - self.relation_offset}"
        if self.filler_offset <= token_id < self.vocab_size:
            return f"f{token_id - self.filler_offset}"
        return f"?{token_id}"


@dataclass
class TransformerConfig:
    n_layers: int = 1
    hidden_dim: int = 128
    n_heads: int = 4
    context_len: int = 128
    tie_embeddings: bool = False


@dataclass
class TrainingConfig:
    num_sequences: int = 5000
    seq_length: int = 70
    batch_size: int = 64
    max_epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    grad_clip: float = 1.0
    eval_every: int = 10
    checkpoint_dir: Path = PROJECT_ROOT / "vienna" / "checkpoints"
    results_dir: Path = PROJECT_ROOT / "vienna" / "results"


@dataclass
class ExperimentConfig:
    vocab: TokenVocabulary = field(default_factory=TokenVocabulary)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42


def relation_specs_from_dicts(raw_specs: Iterable[dict]) -> tuple[RelationSpec, ...]:
    return tuple(RelationSpec(**spec) for spec in raw_specs)


# Backwards-compatible names used by legacy Vienna scripts.
VocabConfig = TokenVocabulary
ModelConfig = TransformerConfig
TrainConfig = TrainingConfig
Config = ExperimentConfig
R2_PERM = R2_PERM_100
cfg = ExperimentConfig()

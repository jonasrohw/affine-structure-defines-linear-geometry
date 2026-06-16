"""
Configuration for the Salzburg experiment (no-filler variant).

Sequence format: [x, r_k, a_i, b_j] — 4 tokens per triple.
Relation comes before subject so the subject representation is relation-conditioned.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).parent.parent

# Hard-coded random permutation for R2 (a_i -> b_{R2_PERM[i]}).
# Same as vienna — generated once with: np.random.RandomState(42).permutation(100).tolist()
R2_PERM_100: List[int] = [
    83, 53, 70, 45, 44, 39, 22, 80, 10,  0, 18, 30, 73, 33, 90,  4,
    76, 77, 12, 31, 55, 88, 26, 42, 69, 15, 40, 96,  9, 72, 11, 47,
    85, 28, 93,  5, 66, 65, 35, 16, 49, 34,  7, 95, 27, 19, 81, 25,
    62, 13, 24,  3, 17, 38,  8, 78,  6, 64, 36, 89, 56, 99, 54, 43,
    50, 67, 46, 68, 61, 97, 79, 41, 58, 48, 98, 57, 75, 32, 94, 59,
    63, 84, 37, 29,  1, 52, 21,  2, 23, 87, 91, 74, 86, 82, 20, 60,
    71, 14, 92, 51,
]

R2_PERM = R2_PERM_100


def make_r2_perm(n: int, seed: int = 42) -> List[int]:
    """Generate a deterministic random permutation of size n."""
    return np.random.RandomState(seed).permutation(n).tolist()


@dataclass
class VocabConfig:
    n_subjects: int = 100        # a0 .. a{n-1}
    n_objects: int = 100         # b0 .. b{n-1}
    num_relations: int = 3       # r0, r1, r2
    r2_perm: Optional[List[int]] = None

    sep_id: int = 0
    n_train_pairs: Optional[int] = None
    n_heldout_pairs: int = 0

    def __post_init__(self):
        if self.n_train_pairs is None:
            self.n_train_pairs = self.n_subjects
        if self.r2_perm is None:
            if self.n_subjects == 100:
                self.r2_perm = R2_PERM_100
            else:
                self.r2_perm = make_r2_perm(self.n_subjects)

    @property
    def subject_offset(self) -> int:
        return 1

    @property
    def object_offset(self) -> int:
        return 1 + self.n_subjects

    @property
    def relation_offset(self) -> int:
        return 1 + self.n_subjects + self.n_objects

    @property
    def vocab_size(self) -> int:
        return 1 + self.n_subjects + self.n_objects + self.num_relations

    @property
    def train_indices(self) -> List[int]:
        return list(range(self.n_train_pairs))

    @property
    def heldout_indices(self) -> List[int]:
        return list(range(self.n_train_pairs, self.n_subjects))

    def subject_id(self, i: int) -> int:
        return self.subject_offset + i

    def object_id(self, i: int) -> int:
        return self.object_offset + i

    def relation_id(self, k: int) -> int:
        return self.relation_offset + k

    def apply_relation(self, relation_idx: int, subject_idx: int) -> int:
        """Return the object index for subject_idx under relation relation_idx."""
        if relation_idx == 0:
            return subject_idx
        elif relation_idx == 1:
            return (subject_idx + self.n_objects // 2) % self.n_objects
        elif relation_idx == 2:
            return self.r2_perm[subject_idx]
        else:
            raise ValueError(f"Unknown relation index: {relation_idx}")

    def token_to_str(self, token_id: int) -> str:
        if token_id == self.sep_id:
            return "x"
        elif self.subject_offset <= token_id < self.object_offset:
            return f"a{token_id - self.subject_offset}"
        elif self.object_offset <= token_id < self.relation_offset:
            return f"b{token_id - self.object_offset}"
        elif self.relation_offset <= token_id < self.vocab_size:
            return f"r{token_id - self.relation_offset}"
        else:
            return f"?{token_id}"


@dataclass
class ModelConfig:
    n_layers: int = 1
    hidden_dim: int = 128
    n_heads: int = 4
    context_len: int = 128
    tie_embeddings: bool = False


@dataclass
class TrainConfig:
    num_sequences: int = 5000
    # Each triple is 4 tokens: x r_k a_i b_j
    # 40 = 10 triples × 4 tokens
    seq_length: int = 40
    batch_size: int = 64
    max_epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    grad_clip: float = 1.0
    eval_every: int = 10
    checkpoint_dir: Path = ROOT / "checkpoints"
    results_dir: Path = ROOT / "results"


@dataclass
class Config:
    vocab: VocabConfig = field(default_factory=VocabConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42


cfg = Config()

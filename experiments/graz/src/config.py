"""
Configuration for the Graz experiment.

Entities have two attribute groups:
  - f-attributes: a 3-element subset of {0,...,9}  (120 possible f-types)
  - g-attributes: a 3-element subset of {0,...,9}  (120 possible g-types)

Phase 1: attribute sentences only [sep, entity, attribute] — 3 tokens.
Phase 2: adds relation triples [sep, relation, subject, object] — 4 tokens.

Relations:
  R0: shared f-type.  Subject i → objects with same f-type.
  R1: shared g-type.  Subject i → objects with same g-type.

With n_entities > N_TYPES (120), multiple entities share each type.
"""
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).parent.parent

COMBOS: List[Tuple[int, ...]] = [tuple(sorted(c)) for c in combinations(range(10), 3)]
N_TYPES = len(COMBOS)
assert N_TYPES == 120


def make_combos(n_attrs: int, k: int) -> List[Tuple[int, ...]]:
    return [tuple(sorted(c)) for c in combinations(range(n_attrs), k)]


@dataclass
class VocabConfig:
    n_f_attrs: int = 10
    n_g_attrs: int = 10
    k_per_group: int = 3
    n_entities: int = 120
    r0_shift: int = 0
    r1_shift: int = 60
    num_relations: int = 0
    n_train_subjects: int = 120
    held_out_triples: List[Tuple[int, int]] = field(default_factory=list)
    _has_dummy: bool = False

    sep_id: int = 0

    @property
    def combos(self) -> List[Tuple[int, ...]]:
        return make_combos(self.n_f_attrs, self.k_per_group)

    @property
    def n_types(self) -> int:
        return len(self.combos)

    @property
    def subject_offset(self) -> int:
        return 1

    @property
    def object_offset(self) -> int:
        return 1 + self.n_entities

    @property
    def f_attr_offset(self) -> int:
        return 1 + 2 * self.n_entities

    @property
    def g_attr_offset(self) -> int:
        return self.f_attr_offset + self.n_f_attrs

    @property
    def relation_offset(self) -> int:
        return self.g_attr_offset + self.n_g_attrs

    @property
    def s_marker_id(self) -> int:
        return self.relation_offset + self.num_relations

    @property
    def o_marker_id(self) -> int:
        return self.relation_offset + self.num_relations + 1

    @property
    def y_id(self) -> int:
        return self.relation_offset + self.num_relations + 2

    @property
    def n_id(self) -> int:
        return self.relation_offset + self.num_relations + 3

    def query_id(self, k: int) -> int:
        return self.relation_offset + self.num_relations + 4 + k

    @property
    def dummy_id(self) -> int:
        return self.relation_offset + self.num_relations + 4 + self.num_relations

    @property
    def has_dummy(self) -> bool:
        return self._has_dummy

    @property
    def vocab_size(self) -> int:
        base = 1 + 2 * self.n_entities + self.n_f_attrs + self.n_g_attrs + self.num_relations + 4 + self.num_relations
        return base + (1 if self._has_dummy else 0)

    @property
    def n_test_subjects(self) -> int:
        return self.n_entities - self.n_train_subjects

    @property
    def train_subject_indices(self) -> List[int]:
        return list(range(self.n_train_subjects))

    @property
    def test_subject_indices(self) -> List[int]:
        return list(range(self.n_train_subjects, self.n_entities))

    @property
    def all_triples(self) -> List[Tuple[int, int]]:
        return [(i, k) for i in range(self.n_entities) for k in range(self.num_relations)]

    @property
    def train_triples(self) -> List[Tuple[int, int]]:
        held = set(self.held_out_triples)
        return [t for t in self.all_triples if t not in held]

    def subject_id(self, i: int) -> int:
        return self.subject_offset + i

    def object_id(self, j: int) -> int:
        return self.object_offset + j

    def f_attr_id(self, a: int) -> int:
        return self.f_attr_offset + a

    def g_attr_id(self, a: int) -> int:
        return self.g_attr_offset + a

    def relation_id(self, k: int) -> int:
        return self.relation_offset + k

    def is_relation(self, token_id: int) -> bool:
        return self.relation_offset <= token_id < self.relation_offset + self.num_relations

    def relation_objects(self, subject_idx: int, rel_k: int) -> List[int]:
        """All valid objects for this (subject, relation) pair."""
        if rel_k == 0:
            target = self.subject_f_type(subject_idx)
            return [j for j in range(self.n_entities) if self.object_f_type(j) == target]
        elif rel_k == 1:
            target = self.subject_g_type(subject_idx)
            return [j for j in range(self.n_entities) if self.object_g_type(j) == target]
        raise ValueError(f"Unknown relation {rel_k}")

    def relation_object(self, subject_idx: int, rel_k: int) -> int:
        """Single object for this (subject, relation) — first valid target."""
        return self.relation_objects(subject_idx, rel_k)[0]

    def subject_f_type(self, i: int) -> Tuple[int, ...]:
        return self.combos[i % self.n_types]

    def subject_g_type(self, i: int) -> Tuple[int, ...]:
        base = i % self.n_types
        copy = i // self.n_types
        return self.combos[(base + copy * self.r1_shift) % self.n_types]

    def object_f_type(self, j: int) -> Tuple[int, ...]:
        f_offset = self.n_types // 3
        base = j % self.n_types
        copy = j // self.n_types
        return self.combos[(base + f_offset + copy * self.r0_shift) % self.n_types]

    def object_g_type(self, j: int) -> Tuple[int, ...]:
        base = j % self.n_types
        copy = j // self.n_types
        return self.combos[(base + self.r1_shift + copy * self.r1_shift) % self.n_types]

    def entity_f_type(self, token_id: int) -> Tuple[int, ...]:
        if self.subject_offset <= token_id < self.object_offset:
            return self.subject_f_type(token_id - self.subject_offset)
        elif self.object_offset <= token_id < self.f_attr_offset:
            return self.object_f_type(token_id - self.object_offset)
        raise ValueError(f"Token {token_id} is not an entity")

    def entity_g_type(self, token_id: int) -> Tuple[int, ...]:
        if self.subject_offset <= token_id < self.object_offset:
            return self.subject_g_type(token_id - self.subject_offset)
        elif self.object_offset <= token_id < self.f_attr_offset:
            return self.object_g_type(token_id - self.object_offset)
        raise ValueError(f"Token {token_id} is not an entity")

    def entity_attrs(self, token_id: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return self.entity_f_type(token_id), self.entity_g_type(token_id)

    def valid_attr_ids(self, token_id: int) -> List[int]:
        f_attrs, g_attrs = self.entity_attrs(token_id)
        return ([self.f_attr_id(a) for a in f_attrs] +
                [self.g_attr_id(a) for a in g_attrs])

    def r0_partner(self, i: int) -> int:
        return self.relation_objects(i, 0)[0]

    def r1_partner(self, i: int) -> int:
        return self.relation_objects(i, 1)[0]

    def is_entity(self, token_id: int) -> bool:
        return self.subject_offset <= token_id < self.f_attr_offset

    def is_subject(self, token_id: int) -> bool:
        return self.subject_offset <= token_id < self.object_offset

    def is_object(self, token_id: int) -> bool:
        return self.object_offset <= token_id < self.f_attr_offset

    def token_to_str(self, token_id: int) -> str:
        if token_id == self.sep_id:
            return "x"
        elif self.is_subject(token_id):
            i = token_id - self.subject_offset
            f = "".join(str(d) for d in self.subject_f_type(i))
            g = "".join(str(d) for d in self.subject_g_type(i))
            return f"a_{f}{g}"
        elif self.is_object(token_id):
            j = token_id - self.object_offset
            f = "".join(str(d) for d in self.object_f_type(j))
            g = "".join(str(d) for d in self.object_g_type(j))
            return f"b_{f}{g}"
        elif self.f_attr_offset <= token_id < self.g_attr_offset:
            return f"f_{token_id - self.f_attr_offset}"
        elif self.g_attr_offset <= token_id < self.relation_offset:
            return f"g_{token_id - self.g_attr_offset}"
        elif self.is_relation(token_id):
            return f"r_{token_id - self.relation_offset}"
        elif token_id == self.s_marker_id:
            return "S"
        elif token_id == self.o_marker_id:
            return "O"
        elif token_id == self.y_id:
            return "y"
        elif token_id == self.n_id:
            return "n"
        else:
            for k in range(self.num_relations):
                if token_id == self.query_id(k):
                    return f"Q{k}"
        return f"?{token_id}"


@dataclass
class ModelConfig:
    n_layers: int = 1
    hidden_dim: int = 32
    n_heads: int = 4
    context_len: int = 128
    tie_embeddings: bool = False
    no_mlp: bool = False


@dataclass
class TrainConfig:
    num_sequences: int = 5000
    seq_length: int = 39  # 13 sentences × 3 tokens
    batch_size: int = 64
    max_epochs: int = 800
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

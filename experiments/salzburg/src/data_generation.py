"""
Synthetic data generation for the Salzburg experiment (no-filler variant).

Sequences have the structure:
    x  r_k  a_i  b_{R_k(i)}  x  r_m  a_j  b_{R_m(j)}  ...

Each triple is 4 tokens: separator | relation | subject | object.
Relation comes before subject so the subject position is relation-conditioned.
"""
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config, VocabConfig

TOKENS_PER_TRIPLE = 4  # x, r_k, a_i, b_j


def generate_sequence(
    length: int,
    pair_indices: List[int],
    vocab: VocabConfig,
    rng: random.Random,
) -> List[int]:
    """Generate a single token sequence of triples.

    Each triple: sep | relation | subject | object.
    """
    n_triples = length // TOKENS_PER_TRIPLE
    tokens = []
    for _ in range(n_triples):
        i = rng.choice(pair_indices)
        k = rng.randint(0, vocab.num_relations - 1)
        j = vocab.apply_relation(k, i)
        tokens.append(vocab.sep_id)
        tokens.append(vocab.relation_id(k))
        tokens.append(vocab.subject_id(i))
        tokens.append(vocab.object_id(j))
    return tokens


class SequenceDataset(Dataset):
    """Dataset of fixed-length token sequences."""

    def __init__(
        self,
        num_sequences: int,
        seq_length: int,
        pair_indices: List[int],
        vocab: VocabConfig,
        seed: int = 0,
    ):
        self.pair_indices = pair_indices
        self.vocab = vocab
        rng = random.Random(seed)
        self.sequences = [
            generate_sequence(seq_length, pair_indices, vocab, rng)
            for _ in range(num_sequences)
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def make_dataloader(
    cfg: Config,
    split: str = "train",
    num_sequences: Optional[int] = None,
    seed_offset: int = 0,
) -> DataLoader:
    pair_indices = (
        cfg.vocab.train_indices if split == "train" else cfg.vocab.heldout_indices
    )
    n = num_sequences or cfg.train.num_sequences
    ds = SequenceDataset(
        num_sequences=n,
        seq_length=cfg.train.seq_length,
        pair_indices=pair_indices,
        vocab=cfg.vocab,
        seed=cfg.seed + seed_offset,
    )
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=(split == "train"),
        drop_last=True,
    )


def token_type_mask(tokens: torch.Tensor, vocab: VocabConfig):
    """Return boolean masks for each token type (no fillers).

    Returns:
        (sep_mask, subj_mask, rel_mask, obj_mask)
    """
    sep_mask = tokens == vocab.sep_id
    subj_mask = (tokens >= vocab.subject_offset) & (tokens < vocab.object_offset)
    obj_mask = (tokens >= vocab.object_offset) & (tokens < vocab.relation_offset)
    rel_mask = (tokens >= vocab.relation_offset) & (tokens < vocab.vocab_size)
    return sep_mask, subj_mask, rel_mask, obj_mask

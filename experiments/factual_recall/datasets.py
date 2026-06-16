"""Dataset generation for configurable subject-relation-object sequences."""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig, TokenVocabulary

TOKENS_PER_TRIPLE = 7


def generate_sequence(
    length: int,
    pair_indices: list[int],
    vocab: TokenVocabulary,
    rng: random.Random,
) -> list[int]:
    """Generate one sequence of filler-token relation triples."""
    n_triples = length // TOKENS_PER_TRIPLE
    tokens: list[int] = []
    for _ in range(n_triples):
        subject_idx = rng.choice(pair_indices)
        relation_idx = rng.randint(0, vocab.num_relations - 1)
        object_idx = vocab.apply_relation(relation_idx, subject_idx)
        tokens.append(vocab.sep_id)
        tokens.append(vocab.subject_id(subject_idx))
        tokens.append(vocab.relation_id(relation_idx))
        tokens.extend(vocab.filler_ids)
        tokens.append(vocab.object_id(object_idx))
    return tokens


class TokenSequenceDataset(Dataset):
    """Fixed-length next-token prediction sequences."""

    def __init__(
        self,
        num_sequences: int,
        seq_length: int,
        pair_indices: list[int],
        vocab: TokenVocabulary,
        seed: int = 0,
    ) -> None:
        self.pair_indices = pair_indices
        self.vocab = vocab
        rng = random.Random(seed)
        self.sequences = [
            generate_sequence(seq_length, pair_indices, vocab, rng)
            for _ in range(num_sequences)
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.sequences[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def make_dataloader(
    cfg: ExperimentConfig,
    split: str = "train",
    num_sequences: Optional[int] = None,
    seed_offset: int = 0,
) -> DataLoader:
    pair_indices = cfg.vocab.train_indices if split == "train" else cfg.vocab.heldout_indices
    n_sequences = num_sequences or cfg.train.num_sequences
    dataset = TokenSequenceDataset(
        num_sequences=n_sequences,
        seq_length=cfg.train.seq_length,
        pair_indices=pair_indices,
        vocab=cfg.vocab,
        seed=cfg.seed + seed_offset,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=(split == "train"),
        drop_last=True,
    )


def token_type_mask(tokens: torch.Tensor, vocab: TokenVocabulary):
    """Return boolean masks for separator, subject, relation, filler, and object."""
    sep_mask = tokens == vocab.sep_id
    subj_mask = (tokens >= vocab.subject_offset) & (tokens < vocab.object_offset)
    obj_mask = (tokens >= vocab.object_offset) & (tokens < vocab.relation_offset)
    rel_mask = (tokens >= vocab.relation_offset) & (tokens < vocab.filler_offset)
    filler_mask = (tokens >= vocab.filler_offset) & (tokens < vocab.vocab_size)
    return sep_mask, subj_mask, rel_mask, filler_mask, obj_mask


# Backwards-compatible name used by legacy Vienna scripts.
SequenceDataset = TokenSequenceDataset

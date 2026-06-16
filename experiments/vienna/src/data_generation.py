"""Compatibility wrapper for legacy Vienna imports.

New code should import from `factual_recall.datasets`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factual_recall.datasets import (
    TOKENS_PER_TRIPLE,
    SequenceDataset,
    TokenSequenceDataset,
    generate_sequence,
    make_dataloader,
    token_type_mask,
)

__all__ = [
    "TOKENS_PER_TRIPLE",
    "SequenceDataset",
    "TokenSequenceDataset",
    "generate_sequence",
    "make_dataloader",
    "token_type_mask",
]

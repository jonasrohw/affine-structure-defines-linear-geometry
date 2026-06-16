"""Compatibility wrapper for legacy Vienna imports.

New code should import from `factual_recall.model`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factual_recall.model import (
    CausalSelfAttention,
    CausalTransformer,
    MLP,
    MiniTransformer,
    TransformerBlock,
    build_model,
)

__all__ = [
    "CausalSelfAttention",
    "CausalTransformer",
    "MLP",
    "MiniTransformer",
    "TransformerBlock",
    "build_model",
]

"""Shared model configuration and repository paths."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class TokenVocabulary(Protocol):
    @property
    def vocab_size(self) -> int: ...

@dataclass
class TransformerConfig:
    n_layers: int = 1
    hidden_dim: int = 64
    n_heads: int = 4
    context_len: int = 8
    tie_embeddings: bool = False

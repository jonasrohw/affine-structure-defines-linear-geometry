"""Compatibility wrapper for legacy Vienna imports.

New code should import from `factual_recall.config`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factual_recall.config import (
    Config,
    ExperimentConfig,
    ModelConfig,
    R2_PERM,
    RelationSpec,
    TokenVocabulary,
    TrainConfig,
    TrainingConfig,
    TransformerConfig,
    VocabConfig,
    cfg,
    default_vienna_relations,
    make_permutation,
)

__all__ = [
    "Config",
    "ExperimentConfig",
    "ModelConfig",
    "R2_PERM",
    "RelationSpec",
    "TokenVocabulary",
    "TrainConfig",
    "TrainingConfig",
    "TransformerConfig",
    "VocabConfig",
    "cfg",
    "default_vienna_relations",
    "make_permutation",
]

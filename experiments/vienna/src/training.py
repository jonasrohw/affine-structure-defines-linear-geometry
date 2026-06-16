"""Compatibility wrapper for legacy Vienna imports.

New code should import from `factual_recall.training`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factual_recall.training import evaluate, get_lr, load_checkpoint, train

__all__ = ["evaluate", "get_lr", "load_checkpoint", "train"]

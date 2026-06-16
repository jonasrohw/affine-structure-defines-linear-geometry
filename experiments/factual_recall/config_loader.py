"""Load experiment configs from TOML files."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .config import ExperimentConfig
from .registry import experiment_config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return experiment_config(raw)

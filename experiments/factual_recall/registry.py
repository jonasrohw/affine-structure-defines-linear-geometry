"""Registries for named datasets and model sizes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import (
    ExperimentConfig,
    RelationSpec,
    TokenVocabulary,
    TrainingConfig,
    TransformerConfig,
    relation_specs_from_dicts,
)

MODEL_SIZES: dict[str, TransformerConfig] = {
    "tiny": TransformerConfig(n_layers=1, hidden_dim=32, n_heads=4, context_len=64),
    "small": TransformerConfig(n_layers=1, hidden_dim=128, n_heads=4, context_len=128),
    "medium": TransformerConfig(n_layers=2, hidden_dim=256, n_heads=8, context_len=128),
}

DATASETS: dict[str, TokenVocabulary] = {
    "vienna_filler": TokenVocabulary(),
}


def dataset_config(name: str, overrides: dict | None = None) -> TokenVocabulary:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset config '{name}'. Known: {sorted(DATASETS)}")
    overrides = dict(overrides or {})
    overrides.pop("name", None)
    relation_specs = overrides.pop("relation_specs", None)
    if relation_specs is not None:
        overrides["relation_specs"] = relation_specs_from_dicts(relation_specs)
    return replace(DATASETS[name], **overrides)


def model_config(name: str, overrides: dict | None = None) -> TransformerConfig:
    if name not in MODEL_SIZES:
        raise KeyError(f"Unknown model size '{name}'. Known: {sorted(MODEL_SIZES)}")
    overrides = dict(overrides or {})
    overrides.pop("size", None)
    return replace(MODEL_SIZES[name], **overrides)


def training_config(overrides: dict | None = None) -> TrainingConfig:
    overrides = dict(overrides or {})
    for key in ("checkpoint_dir", "results_dir"):
        if key in overrides:
            overrides[key] = Path(overrides[key])
    return TrainingConfig(**overrides)


def experiment_config(raw: dict) -> ExperimentConfig:
    dataset_raw = dict(raw.get("dataset", {}))
    model_raw = dict(raw.get("model", {}))
    train_raw = dict(raw.get("training", {}))
    seed = int(raw.get("seed", 42))
    dataset_name = dataset_raw.get("name", "vienna_filler")
    model_size = model_raw.get("size", "small")
    return ExperimentConfig(
        vocab=dataset_config(dataset_name, dataset_raw),
        model=model_config(model_size, model_raw),
        train=training_config(train_raw),
        seed=seed,
    )


def known_configs() -> dict[str, list[str]]:
    return {
        "datasets": sorted(DATASETS),
        "model_sizes": sorted(MODEL_SIZES),
    }

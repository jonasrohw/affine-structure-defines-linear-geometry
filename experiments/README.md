# Factual Recall Mechanisms

Synthetic factual recall experiments. The supported entrypoint is UV:

```bash
uv sync --extra dev --extra plots
uv run factual-recall train --config configs/runs/vienna_smoke.toml
```

New datasets and model sizes should be added through config files plus the
Python registries in `factual_recall.registry`.

# Experiment maintenance

Use the root `pyproject.toml` and `uv.lock`. Prefix shell commands with `rtk` as
specified in `/home/jonas/.codex/RTK.md`.

- `uv run --locked pytest -q` checks model, dataset and analysis invariants.
- `uv run --locked reproduce --smoke` exercises the complete computation path.
- `uv run --locked reproduce` trains all paper seeds, regenerates figures/tables,
  compiles the paper and audits measurements against the archived references.
- Keep `paper/reference/` immutable; numerical disagreements are evidence to
  investigate, never a reason to replace reference values with new measurements.
- Preserve dataset RNG ordering, losses, checkpoint selection and probe definitions
  in refactors. Record changes to those choices explicitly.
- Generated artifacts belong in `results/`, not in the Python package.

The canonical manuscript baseline is now `paper/baselines/rtx4090-v1/`, explicitly
adopted after two matching fresh RTX runs. Follow the root README's pinned RTX
Docker workflow for exact paper reproduction; the root uv environment remains a
CPU diagnostic. New baselines require explicit scientific review and repeatability
evidence. Never overwrite `paper/reference/` or silently loosen audit tolerances.

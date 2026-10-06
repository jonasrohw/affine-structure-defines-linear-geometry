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

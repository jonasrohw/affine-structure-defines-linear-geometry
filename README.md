# Anonymous experiment reproduction

This distribution contains experiment code, locked environments, tests, three final
checkpoints, numerical reference results and saved LMEnt observations. It contains
no manuscript or author metadata. The `paper/` directory holds analysis scripts
and scientific evidence. Generated results go under `results/`.

## Coverage

| Component | Included | Reproduction boundary |
| --- | --- | --- |
| Synthetic Experiment 1 | Data generator, frequency and attribute sweeps, training, interventions, Jacobian analyses | Fresh training and analysis for seeds 19, 31, 47 |
| Synthetic Experiment 2 | Data generator, staged training, checkpoint selection, probes, ablations, causal tracing, permutation tests | Fresh training and analysis for seeds 19, 31, 47 |
| Synthetic reference evidence | Final checkpoints, training histories, numerical and tensor baselines | Exact comparison requires the documented RTX environment |
| LMEnt frequency and intervention summaries | 5,388 saved fact observations, 56 intervention relation summaries, hashes and aggregation code | Recomputes aggregates from supplied observations |
| LMEnt original inference and corpus counts | Evaluation settings and source hashes only | Original evaluation code, corpus extraction and counting inputs are absent |
| LMEnt answer-sharing, regressions, probe comparison and case study | Sharing figure script only | Analysis source and per-fact probe inputs are absent |

**This distribution does not yet reproduce every manuscript result.** The missing
LMEnt materials must be supplied before it can support that claim. In particular,
`probe_perfact_all.csv`, `probe_vs_lre.csv` and `cohort.csv` are required by the
sharing figure. These files alone would allow plotting, but the inference, probe
and regression code is also needed to reproduce their measurements.

## Quick check

Use Python 3.13.11 and uv with the supplied lock. Local figure generation needs
Cairo, TeX Live (including `type1cm` and `cm-super`) and `dvipng`. The Docker recipe
pins these system dependencies as well as Python packages.

```sh
uv sync --locked
uv run --locked pytest -q
uv run --locked reproduce --smoke --output results/smoke
```

The smoke run trains one seed with shortened Experiment 2 stages and reduced
Jacobian and permutation samples. It exercises training, checkpoint loading,
analysis, aggregation, figures and table exports. It does not certify scientific
results. The local CPU environment supports diagnostics and full training, but
its output can differ from the RTX reference.

## Full reproduction on the reference hardware

The archived baseline was measured on an AMD Ryzen 9 7950X3D, one NVIDIA RTX 4090,
NVIDIA driver 610.57.04, Linux x86-64, CPython 3.14.2 and PyTorch 2.12.0+cu130.
Experiment 1 runs on CPU and Experiment 2 on GPU. Seeds are fixed at 19, 31 and 47.
There are three independent workers with one numerical CPU thread each. Exact
results on other hardware or runtimes are not guaranteed.

Build the three images in this order. Building downloads public dependencies.
The runner then executes without network access.

```sh
docker build --network host -t factual-recall:reproducible .
docker build --network host -f docker/rtx/Dockerfile -t factual-recall:rtx-runtime docker/rtx
docker build --network host -f Dockerfile.rtx -t factual-recall:rtx .
python scripts/run_rtx.py --python -m pytest -q
python scripts/run_rtx.py --output /app/results/full
```

The runner validates the CPU, GPU and driver before training. A full run writes
`audit.json` and `audit.csv` and exits with an error if any current-reference
measurement, checkpoint tensor or required joint gate differs. The comparison
uses exact numerical equality. Historical comparisons are saved separately.
Wall-clock duration, serialization metadata and filesystem paths are excluded
from scientific comparisons. No reference values are used to train a model.

To resume an interrupted run, append `--resume`. A cached stage is reused only
when its source fingerprint, runtime, configuration and artifact hashes match.
For independent repeatability verification, use two new output directories and
omit `--resume`:

```sh
python scripts/run_rtx.py --output /app/results/run-a
python scripts/run_rtx.py --output /app/results/run-b
python scripts/run_rtx.py --python scripts/verify_rtx.py /app/results/run-a /app/results/run-b --output /app/results/verification
```

The archived evidence records 21,200 identical numerical outputs, 935 identical
checkpoint tensors and three passing joint gates across two fresh runs. This is
historical evidence. See `validation.json` for checks performed on this anonymous
distribution. Full GPU training was not repeated during preparation of this copy.

## Outputs and individual analyses

Each run saves configuration, runtime information, source fingerprints and
artifact hashes in `manifest.json`. Experiment directories contain per-seed
measurements, training histories and checkpoints. `paper/` within the output
contains six generated figure sets, `measurements.json`, `result_values.tex`,
LMEnt aggregates and full-precision CSV tables under `tables/`. No manuscript
source is needed and no manuscript PDF is generated. TeX is used to render plot
labels. `--stage experiment1`, `--stage experiment2` and `--stage report` run
individual stages. `--stage paper` remains an alias for `--stage report`.

To regenerate reports from an existing completed run:

```sh
python scripts/run_rtx.py --stage report --output /app/results/full
```

To recompute only the supplied LMEnt aggregates:

```sh
uv run --locked python paper/analyze_lment_frequency.py --output-dir results/lment/paper
```

The snapshot hashes are checked before aggregation. Original faithfulness
operators were selected using evaluation outcomes. Intervention operators were
selected on validation data. The cohorts and their metrics differ.

The optional sharing plot requires external inputs:

```sh
uv run --locked python paper/make_lment_sharing_figure.py --inputs /path/to/analysis-inputs --output-dir results/sharing/figures
```

## Distribution

Run `sha256sum --check SHA256SUMS` after extracting the release to verify every
distributed file against its recorded checksum.

The anonymous branch has one root commit with a generic author and no inherited
history. The release ZIP contains only tracked files, fixed timestamps and no
Git metadata. Share that ZIP through an anonymous submission channel. The local
worktree's `.git` file and the original repository hosting account identify their
owner and must not be distributed. Public dependency and model identifiers are
retained because they are needed for reproducibility. Code and data may still be
recognizable by comparison with public material, so complete anonymity cannot be
guaranteed.

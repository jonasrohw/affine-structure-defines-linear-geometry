# Factual Recall Mechanisms

Code and evidence for **Frequency Alone Does Not Determine Affine Relation Geometry in Transformers**.
The original repository is preserved on the local `backup` branch.

The paper uses the versioned **rtx4090-v1** baseline in
[`paper/baselines/rtx4090-v1/`](paper/baselines/rtx4090-v1/).
Two independent full runs from fresh initialization matched **21,200 scientific
numeric outputs, 935 checkpoint tensors, and all three final gates exactly**.
All 23 tests pass. This evidence was required before adoption.
The [validation record](validation.json) records the verification and claim review.
Historical A100 measurements in `paper/reference/` remain unchanged.

The source manuscript `paper/main.tex` contains the RTX numerical definitions
and result tables directly. It uses the committed figures in `paper/figures/`
and can be compiled without any generated `results/` directory:

```sh
latexmk -cd -pdf paper/main.tex
```

This writes `paper/main.pdf`. The reproduction pipeline refreshes the marked
numerical block in its generated copy of `main.tex` from each run's measurements.

## Reproduce the paper on the certified RTX setup

Certification is scoped to the Ryzen 9 7950X3D CPU, one RTX 4090, NVIDIA driver
610.57.04, and the pinned Linux x86-64 environment: CPython 3.14.2 (standalone
build 20260114), PyTorch 2.12.0+cu130 and the full lock in `docker/rtx/uv.lock`.
Experiment 1 runs on the CPU; Experiment 2 runs on the GPU. Changing hardware,
drivers or numerical libraries requires new verification. This is not a claim
of universal bitwise reproducibility across systems.

From the repository root:

```sh
docker build --network host -t factual-recall:reproducible .
docker build --network host -f docker/rtx/Dockerfile -t factual-recall:rtx-runtime docker/rtx
docker build --network host -f Dockerfile.rtx -t factual-recall:rtx .
python scripts/run_rtx.py --output /app/results/full
```

The first image supplies the pinned Debian snapshot, Cairo, fonts and LaTeX. The
RTX runtime installs the SHA-256-verified public Python archive and hash-locked
CUDA environment. The reporting-only Cairo additions are also hash pinned. The
launcher checks the certified CPU, GPU and driver, passes through that GPU and
its driver libraries without changing Docker's daemon, and disables networking.
Builds download dependencies; training and report generation are offline.

The full command trains seeds 19, 31 and 47, recomputes both experiments and all
interventions, aggregates the hash-verified LMEnt observations, generates numerical
prose, four tables and six figures, compiles the PDF, and audits the fresh outputs.
The audit uses **zero tolerance** for synthetic values and checkpoint tensor hashes.
The source manuscript is not overwritten by a run. An audit mismatch returns a
nonzero status and preserves the generated evidence.

```sh
python scripts/run_rtx.py --smoke --output /app/results/smoke
python scripts/run_rtx.py --python -m pytest -q
python scripts/run_rtx.py --resume --output /app/results/full
```

Smoke runs exercise the complete pipeline with one seed and shortened Experiment 2;
they do not reproduce paper values. Resume checks source, runtime, configuration
and artifact hashes before reusing completed stages. `--stage experiment1`,
`--stage experiment2` and `--stage paper` are diagnostic entry points. `--no-pdf`
omits LaTeX compilation. Three seed processes run concurrently with one numerical
CPU thread each; the certified command uses the default three workers.

For an independent repeatability check, use two unused output directories and
omit `--resume`:

```sh
python scripts/run_rtx.py --output /app/results/repeat-a
python scripts/run_rtx.py --output /app/results/repeat-b
python scripts/run_rtx.py --python scripts/verify_rtx.py /app/results/repeat-a /app/results/repeat-b --output /app/results/repeat-check
```

The verifier compares all recorded numerical experiment outputs (including losses,
training histories, permutation samples and intervention results), final gates,
and the bytes of every saved tensor. Paths, elapsed time and container/PDF metadata
are excluded. It writes evidence under `results/`; it never automatically replaces
a paper baseline. `--reference historical` explicitly audits the original A100
measurements instead, and is expected to report differences for fresh RTX training.

## Outputs and evidence

- `manifest.json`: source/lock hashes, packages, runtime, seeds, settings and artifact hashes.
- `experiment1/`: sweeps, interventions, affine fits, midpoint and exact Jacobian diagnostics.
- `experiment2/`: configurations, histories, checkpoints, complete-table behavior,
  raw-embedding OLS, component/layer interventions, logit lenses and residual patching.
- `paper/main.pdf`, `paper/result_values.tex`, `paper/measurements.json`: generated paper and numerical prose.
- `audit.json`, `audit.csv`: exact comparison with the current baseline plus LMEnt checks.
- `historical_audit.json`, `historical_audit.csv`: separately retained comparison with the original evidence.

The committed baseline includes exact measurement/tensor-hash snapshots, readable
analyses, final Experiment 2 checkpoints, repeatability evidence, and a claim review.
The paper's checked-in figures and numerical macros come from this baseline.
`paper/data/` contains LMEnt input snapshots and provenance. Complete intermediate
artifacts are generated under ignored `results/`; Docker images are not Git artifacts.

## Changes from the historical experiment

The recovered stage limits and evaluation intervals are retained: 6,000 structured
steps; up to 12,000 joint steps, evaluated every 500 steps for seed 19 and every
1,000 for seeds 31/47; up to 8,000 continuation steps at 3e-4, evaluated every 200;
and up to 3,000 steps at 1e-4 for seeds 19/31, evaluated every 100. Optimizer and RNG
resets, datasets, losses and probes are unchanged.

The RTX baseline requires deterministic operations, fixes the cuBLAS workspace,
disables TF32 and cuDNN benchmarking, and uses one CPU thread per worker. Joint
checkpoint selection and early stopping now include the existing prediction-symmetry
threshold. Previously symmetry was checked only afterwards, allowing an early stop
that failed the final gate. This is an explicit protocol correction, not a claim
that the historical trained weights were recovered.

The methods also clarify two inherited details: attribute batches sample 640
examples **with replacement**, and structured pretraining adapts relation sampling
to errors. Joint training retains its 10:1 random-to-individual-structured ratio.
The exact null percentiles and intervention curves are specific to the newly
reported solutions. The core claims remain supported, but seed 47's random-fit
null percentile changes from 99.2 to 49.2 (seed 19: 38.2 to 94.8; seed 31: 98.4 to
98.0). The former above-98th-percentile statement is removed. The manuscript also
reports the actual minimum structured R-squared, 0.99485, instead of the old
slightly overstated inequality. See the baseline's `claim_review.json`.

## CPU and historical diagnostics

The root `pyproject.toml`, `.python-version` and `uv.lock` retain the Python 3.13.11
CPU workflow. With uv, Cairo, TeX Live and `latexmk` installed:

```sh
uv run --locked pytest -q
uv run --locked reproduce --smoke
uv run --locked reproduce --reference historical
```

CPU training is outside the RTX certification scope. Its full audit can fail;
it must not be presented as successful reproduction of the RTX baseline.
`--experiment2-device cuda --runtime-profile beta` retains the recovered
nondeterministic training/stopping behavior for diagnosis. Runtime settings are
recorded and included in resume signatures.

The five data plots share `experiments/factual_recall/figure_style.py`: Times text
rendered by LaTeX, 8 pt figure labels at the final ACL column/text width, and
10 pt Times captions supplied by the ACL template. The sea-green/coral palette
is shared across experiments, with lavender, apricot, and blue supporting series.
Hatching, markers, and line styles supplement color where applicable. These plots
export as vector PDF and SVG, plus 1200-DPI PNG. The manuscript includes the
vector PDFs, which remain sharp at any magnification; DPI applies only to raster
exports. Plot SVG text is outlined so the appearance does not depend on viewer fonts.
The data-generation schematic retains its original layout, typography, and export
settings, with only its colors updated to match the palette.

To restyle existing measurements without training or changing manuscript tables:

```sh
uv run --locked python paper/make_paper_figures.py --results results/full --figures-only
uv run --locked python paper/make_dataset_figure.py --output-dir results/full/paper/figures
uv run --locked python paper/analyze_lment_frequency.py --output-dir results/full/paper
```

## Reproducibility boundaries

LMEnt aggregation uses 5,388 saved fact-level predictions/counts and a separate
56-relation intervention aggregate, both hash verified. Original pretrained-model
inference, Wikipedia extraction and individual intervention trials remain external
inputs. Neither deterministic training nor adoption of a new synthetic baseline
recreates those observations. Experiment 1 uses population SD; Experiment 2 and
the attribute-transition figure use sample SD.

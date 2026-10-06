# Factual Recall Mechanisms

Code and evidence for **Frequency Alone Does Not Determine Affine Relation Geometry in Transformers**.
The original repository is preserved on the local `backup` branch.

**Exact numerical reproduction is not yet certified.** The full Experiment 1
and LMEnt audit passes 241 of 246 checks; five transformation-table entries
differ from the manuscript. The same differences occur with the pre-refactor
runtime, and the refactored Experiment 1 trainer matches the backup bit-for-bit
for all three reported seeds. References are retained unchanged, and the full
command deliberately fails its audit when values disagree.

The [validation record](validation.json) includes 18 passing tests, local and
offline Docker smoke/PDF runs, and the outstanding numerical disagreements.

Run from the repository root:

```sh
uv run --locked reproduce
```

This trains both synthetic experiments from scratch for seeds 19, 31 and 47,
recomputes the probes, interventions and exact Jacobians, aggregates the saved
LMEnt observations, generates all six manuscript figures and four synthetic
result tables, compiles the paper, and compares fresh measurements with the
reported evidence. Results go to `results/full/`; the source manuscript and
archived references are never overwritten. A numerical disagreement produces
`audit.json` and `audit.csv` and a nonzero exit status. Do not treat completion of
training alone as successful reproduction of the paper.

Local requirements: uv, Cairo (for the SVG dataset figure), and a TeX Live
installation with `latexmk`, the ACL template's packages, and Inconsolata.
Python 3.13.11 is selected by `.python-version`. All Python dependencies,
including transitive dependencies and wheel hashes, are pinned in `uv.lock`.
Linux uses the pinned CPU PyTorch wheel; no GPU or external dataset download is
required. `--no-pdf` omits only LaTeX compilation.

```sh
uv run --locked reproduce --smoke     # one seed, short Experiment 2; exercises PDF generation
uv run --locked reproduce --resume    # verify hashes before reusing completed stages
uv run --locked pytest -q
```

Smoke output is isolated in `results/smoke/` and explicitly marked as a smoke
run. Its values do not reproduce the paper. The smoke run still performs one seed of
the full Experiment 1 sweeps and intervention analysis, with a reduced Jacobian
sample and only two optimizer steps per Experiment 2 stage. Full Experiment 2 is substantially
more expensive: 6,000 structured steps, up to 12,000 joint steps, and the reported
continuations for each seed. The three seed processes run concurrently, each with
one numerical thread. Use `--workers 1` for sequential execution. An interrupted
training stage restarts from its beginning; completed stages can be reused.

The `--stage experiment1`, `--stage experiment2`, and `--stage paper` options are
for diagnosis. The default `all` stage is the complete reproduction command.
`--output PATH` selects a separate artifact directory. Reuse only results made by
the same locked source revision; `--resume` verifies source/configuration and
checkpoint hashes before reuse.

## Docker

```sh
docker build -t factual-recall:reproducible .
docker run --rm -v "$PWD/results:/app/results" factual-recall:reproducible
# Fast integration validation:
docker run --rm -v "$PWD/results:/app/results" factual-recall:reproducible --smoke
```

The Dockerfile pins Python and uv by registry digest and uses the immutable
2026-10-01 Debian package snapshot. `uv sync --locked` installs hash-locked
Python wheels; the container runs with `--frozen`. It includes the fonts and
LaTeX dependencies needed to build the PDF. `/opt/os-packages.txt` records the
resolved OS package versions. Docker build downloads the dependencies; the
completed image can run without network access. Containers run as root by
default, so bind-mounted output ownership follows the Docker host's mapping.

## Outputs and evidence

- `manifest.json`: run status, source/lock hashes, platform, Python/PyTorch build,
  installed packages, deterministic settings, seeds, duration and artifact hashes.
- `experiment1/`: frequency and attribute-repetition sweeps, causal interventions,
  four-fold affine fits, midpoint defects and exact Jacobian diagnostics.
- `experiment2/`: per-seed configurations, training histories, checkpoints,
  complete-table behavior, raw-embedding OLS, component/layer interventions,
  logit lenses and residual patching.
- `paper/main.pdf`: manuscript with newly generated tables and figures.
- `paper/measurements.json`, `audit.json`, `audit.csv`: measured summaries and
  comparisons with the archived paper. Prose remains the reference manuscript;
  inspect the audit before interpreting a generated PDF when values differ.

`paper/reference/` contains original synthetic measurements, used only for the
final comparison. `paper/data/` contains the LMEnt observations and provenance.
`paper/figures/` contains the six originally reported figures for reading the
source paper. The scripts regenerate figures from fresh measurements in the
output directory; they do not copy the archived figures or use hard-coded bars.
The ACL `.sty`/`.bst` files remain because the current manuscript requires them.
Historical experiment projects, configuration searches, duplicate papers,
review iterations, unused artwork, and editor/agent bundles live only on `backup`.

## Reproducibility boundaries

The LMEnt faithfulness analysis is recomputed from 5,388 saved fact-level
predictions/counts. Its separate intervention panel is recomputed from the saved
56-relation aggregate. Both inputs are SHA-256 verified. The original pretrained
model inference, Wikipedia extraction and individual intervention trials are
external inputs; this repository cannot regenerate them from training data.
The provenance manifests retain their original source paths and hashes.

The original Experiment 2 checkpoints and training logs were not committed.
The runner reconstructs the manuscript's width-20, eight-layer configuration and
its 6,000/12,000-step curriculum, followed by 200 steps at 3e-4 (all seeds) and
100 steps at 1e-4 (seeds 19 and 31). It preserves the existing training functions,
including optimizer and RNG resets at each continuation. The structured/joint
evaluation interval is explicitly 500 steps; continuation intervals are 100.
The original analysis JSON contains stale defaults rather than the full training
configuration, so exact recovery of the selected checkpoints is not established
by that JSON. The audit detects discrepancies instead of silently substituting
the archived measurements or retuning seeds to match them.

The implementation samples 640 attribute examples with replacement per batch;
this differs from the manuscript's statement that every batch contains all 640
facts. Structured pretraining also adapts its relation sampling to errors. Joint
training keeps the stated 10:1 random-to-individual-structured sampling ratio.
The training loop ranks evaluated checkpoints and stops joint stages on behavior
and geometry; prediction symmetry is checked afterwards. These inherited details
are retained and exposed for review rather than silently changing the experiment.

Experiment 1's archived tables/analyses use population SD; Experiment 2 uses
sample SD. The attribute-transition figure uses sample SD. Seeded deterministic
CPU operations improve repeatability on the same runtime, but do not guarantee
identical optimization across hardware or library builds. See the official
[PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).

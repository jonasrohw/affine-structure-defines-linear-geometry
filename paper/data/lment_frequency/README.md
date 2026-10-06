# LMEnt frequency observations

`facts.csv` holds 5,388 evaluated facts with predictions and three corpus-count
proxies. `provenance.json` records the original source hashes, evaluation settings,
counting rules and the snapshot SHA-256. The adjacent `lment_causality/` snapshot
contains the separately evaluated intervention relation summaries and provenance.

From the repository root, `uv run --locked reproduce` regenerates all aggregates,
tables and the figure alongside the synthetic experiments. For aggregation only:

```sh
uv run --locked python paper/analyze_lment_frequency.py --output-dir results/lment/paper
```

No pretrained model inference or corpus counting is performed. The historical
faithfulness evaluation selected layers/scales using evaluation outcomes. The
intervention cohort instead selected operators on validation data. These are
different populations and metrics; the saved inputs must not be conflated.

The script verifies snapshot hashes, fact uniqueness, prediction agreement and
valid counts. Quartile tails retain ties, require distinct cutoffs and at least
ten facts per tail, and exclude relations with constant model answer tokens.
Five- and twenty-fact tail thresholds are included as sensitivity analyses.
Every relation/axis, including ineligible cases, is saved in the derived CSVs.
`--source-root PATH` can refresh the historical faithfulness snapshot only when
the original external project and its raw predictions/counts are available.

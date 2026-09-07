# LMEnt frequency aggregation

This directory contains the observations behind Section 8, its figure, and the
pretrained-model appendix. The input is the existing LMEnt-1B-1E `step109672`
analysis in `/mounts/work/jonas/triplets-dataset`, created on 2026-09-04.
The OLMo-7B result directories in that project do not supply these frequency
curves; this aggregation covers one pretrained checkpoint.

Run from the repository root with Python, numpy, pandas, and matplotlib:

```bash
python paper/analyze_lment_frequency.py
```

The installed environment used for this paper is
`/mounts/work/jonas/lment-count-analysis/.venv/bin/python`.
To refresh the snapshot from the original project and validate it against every
raw browser curve:

```bash
python paper/analyze_lment_frequency.py --source-root /mounts/work/jonas/triplets-dataset
```

`facts.csv` contains all 5,388 evaluated facts, their token predictions, and the
three document counts. `provenance.json` records original file hashes, the
snapshot hash, evaluation settings, corpus extraction coverage, relation cues,
and object-label aliases. Refreshing checks unique joins, complete counts, the
evaluation totals, estimation/evaluation separation, and all 174 original curves.

The script writes `summary.json`, `binned_faithfulness.csv`,
`relation_contrasts.csv`, `tail_size_sensitivity.csv`, two LaTeX tables, and the
PDF/PNG figure in `paper/figures`. Rebuilding from the saved snapshot requires
neither a model nor the adjacent project.

Pooled bins weight facts equally and retain all relations. Quartile comparisons
keep ties together, require distinct cutoffs and at least ten facts per tail,
and exclude relations with a constant full-model answer token. Each eligible
relation gets equal weight. Every relation and axis, including ineligible cases,
is retained in `relation_contrasts.csv`; use its `eligible` column for the paper's
comparison. Five- and twenty-fact tail thresholds are sensitivity checks.

The example table deliberately illustrates three decreases and one increase.
The aggregate figure and counts include all eligible relations. These are
descriptive results from a selected decoder at one checkpoint. They do not test
statistical independence, prove the absence of other affine maps, or measure
full-answer correctness. The model's layer and scale were selected using the
reported evaluation set. Counts are corpus proxies and are not exact numbers
of factual statements seen during training.

The new manuscript section and appendix received two writing passes using
`avoid-ai-writing` with a technical voice. The second pass checked direct claims,
defined metrics, repeated framing, and unsupported implications. Scientific
qualifications about selection and proxy counts were retained.

> **SUPERSEDED (2026-07-03).** This setup relied on `structured_embedding_init` (attribute-initialized embeddings) and/or fit-set-size-varying probes; its conclusions do not survive a fixed-size probe with standard initialization. The current experiments, single OLS metric, and results are in `paper/linear_vs_nonlinear_experimental_writeup.tex` (implementation: `factual_recall/linear_shift.py`).

# Frequent Relation: Structured Category vs Random Memorization

Current final setup:

```text
experiments/configs/runs/synthetic_mixed_relations_category_k128_r128.toml
```

This setup uses ordinary end-to-end training. No components are frozen and no targeted
updates are used.

## Data

- 192 subject tokens: `a_0 ... a_191`
- 192 object tokens: `b_0 ... b_191`
- 32 skill/category attributes `f_0 ... f_31`
- 32 auxiliary attributes `g_0 ... g_31`

Each subject has attributes:

```text
a_i -> (f_i, g_i)
```

The structured relation maps each subject to the job-category object determined by
its `f` attribute:

```text
R0(a_i) = b_{f_i}
```

This means many distinct subjects can share the same structured target, like many
applicants with the same skill matching the same job category.

The random relation is:

```text
R1(a_i) = b_{pi(i)}
```

where `pi` is a random permutation.

## Frequency

Both relations have identical frequency:

```text
R0: 128 distinct trained triples, each repeated 128 times
R1: 128 distinct trained triples, each repeated 128 times
```

The remaining 64 subjects are held out from relation training but still appear in
attribute sentences.

## Model

```text
1-layer causal transformer
hidden_dim = 64
n_heads = 4
context_len = 8
object_loss_weight = 4.0
learning_rate = 0.001
```

## Main Result

Seed 19:

| relation | train top-1 | held-out top-1 | held-out top-10 | prompt affine NN | OLS embedding affine NN |
|---|---:|---:|---:|---:|---:|
| R0 structured | 1.000 | 0.875 | 0.953 | 0.812 | 0.969 |
| R1 random | 1.000 | 0.000 | 0.000 | 0.000 | 0.016 |

Replication at 128 repeats:

| seed | R0 train | R0 held-out | R0 OLS affine NN | R1 train | R1 held-out | R1 OLS affine NN |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 1.000 | 0.641 | 0.906 | 1.000 | 0.000 | 0.000 |
| 47 | 1.000 | 0.859 | 0.812 | 1.000 | 0.000 | 0.000 |
| 59 | 1.000 | 0.797 | 0.938 | 1.000 | 0.000 | 0.000 |

## Reproduce

```bash
cd experiments
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
  uv run factual-recall synthetic-mixed-relations \
  --config configs/runs/synthetic_mixed_relations_category_k128_r128.toml \
  --device cpu --quiet
```


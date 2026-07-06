> **SUPERSEDED (2026-07-03).** This setup relied on `structured_embedding_init` (attribute-initialized embeddings) and/or fit-set-size-varying probes; its conclusions do not survive a fixed-size probe with standard initialization. The current experiments, single OLS metric, and results are in `paper/linear_vs_nonlinear_experimental_writeup.tex` (implementation: `factual_recall/linear_shift.py`).

# Exact Ambiguous-Bucket Setup

**Status: superseded for the current goal.** This setup uses
`phase2_update_mode = target_lm_rows`, which freezes most components during Phase 2.
It should not be used as evidence for the current end-to-end frequent-relation claim.
Use `experiments/paper/frequent_relation_linear_vs_random_setup.md` instead.

This file describes only the final setup that satisfied the two requirements:

1. Single-exposure facts are learned as exact facts.
2. At high frequency, exact-token recall is memorized without preserving affine structure, both when semantics support the relation and when they do not.

The implementation is:

```text
experiments/factual_recall/ambiguous_bucket_experiments.py
```

The final runs use seeds `23`, `31`, and `47`.

## Core Idea

Each subject belongs to a semantic bucket. Each semantic bucket has multiple possible object tokens. The subject attributes tell the model which object bucket is correct, but they do not tell the model which exact object inside that bucket is correct.

So the task separates two kinds of knowledge:

- Semantic structure: map a subject bucket to the right object bucket.
- Exact factual identity: choose the correct object token inside that bucket.

A single exposure should only be able to teach the second part: the exact object identity.

## Final Configuration

```text
n_buckets = 32
subjects_per_bucket = 3
objects_per_bucket = 3
n_subjects = 32 * 3 = 96
n_objects = 32 * 3 = 96
n_f = 24
n_g = 24
delta_f = 5
delta_g = 7
attr_repeats = 8
pilot_repeats = 160
single_repeats = 1
high_frequency_repeats = 128
model = 2-layer transformer, hidden_dim 128, 4 heads
batch_size = 8
learning_rate = 0.002
phase2_learning_rate = 0.5
phase2_update_mode = target_lm_rows
object_loss_weight = 64
```

## Vocabulary

The vocabulary contains:

```text
x                  separator token
r_0                semantic relation token
r_1                random relation token
a_i                subject tokens, i = 0 ... 95
b_j                object tokens, j = 0 ... 95
f_k                f-attribute tokens, k = 0 ... 23
g_k                g-attribute tokens, k = 0 ... 23
```

The token layout in code is:

```text
0                  x
1                  r_0
2                  r_1
3 ... 98           a_0 ... a_95
99 ... 194         b_0 ... b_95
195 ... 218        f_0 ... f_23
219 ... 242        g_0 ... g_23
```

## Data Generation

### Step 1: Create Semantic Buckets

There are 32 semantic buckets. For each bucket, sample one pair of hidden subject attributes:

```text
bucket b -> (f_b, g_b)
```

The attributes are sampled from the 24 by 24 grid of possible `(f, g)` pairs.

Example shape:

```text
bucket 0 -> (f_?, g_?)
bucket 1 -> (f_?, g_?)
...
bucket 31 -> (f_?, g_?)
```

The exact sampled values depend on the seed.

### Step 2: Create Target Object Buckets

Each subject bucket maps to an object bucket by a fixed attribute shift:

```text
(f_b, g_b) -> (f_b + 5 mod 24, g_b + 7 mod 24)
```

This is the semantic relation. It is linear/structured at the bucket level.

If a subject is in bucket `b`, the correct object must be in the corresponding shifted object bucket. But there are three objects in that bucket, so the semantic rule alone is not enough to identify the exact object.

### Step 3: Create Subjects

Each bucket has three subjects:

```text
bucket 0: a_0,  a_1,  a_2
bucket 1: a_3,  a_4,  a_5
bucket 2: a_6,  a_7,  a_8
...
bucket 31: a_93, a_94, a_95
```

All three subjects in a bucket share the same hidden attributes:

```text
a_i -> (f_b, g_b)
```

The training sentence for a subject attribute is:

```text
x a_i f_b g_b x
```

### Step 4: Create Objects

Each target object bucket has three object candidates:

```text
bucket 0: b_0,  b_1,  b_2
bucket 1: b_3,  b_4,  b_5
bucket 2: b_6,  b_7,  b_8
...
bucket 31: b_93, b_94, b_95
```

All three objects in a target bucket share the shifted hidden attributes:

```text
b_j -> (f_b + 5 mod 24, g_b + 7 mod 24)
```

The training sentence for an object attribute is:

```text
x b_j f_target g_target x
```

### Step 5: Assign Exact Semantic Targets

For relation `r_0`, each subject is assigned to one exact object inside the correct target bucket.

Within each bucket, the assignment is a random balanced matching:

```text
bucket b subjects: a_{3b}, a_{3b+1}, a_{3b+2}
bucket b objects:  b_{3b}, b_{3b+1}, b_{3b+2}

r_0 target:
a_{3b}   -> one of b_{3b}, b_{3b+1}, b_{3b+2}
a_{3b+1} -> another one
a_{3b+2} -> the remaining one
```

This matching is seed-dependent.

Important: the exact object identity is not recoverable from attributes. Attributes only identify the bucket.

The relation sentence is:

```text
x a_i r_0 b_target x
```

### Step 6: Assign Random Targets

For relation `r_1`, targets are assigned by a global random permutation of all 96 object tokens:

```text
r_1(a_i) = b_{pi(i)}
```

This relation has no support from the bucket attributes.

The relation sentence is:

```text
x a_i r_1 b_random x
```

## Subject Splits

Each bucket has exactly three subjects. They are split by position inside the bucket:

```text
position 0 -> pilot subject
position 1 -> single-exposure subject
position 2 -> held-out subject
```

For bucket `b`:

```text
pilot:          a_{3b}
single-exposure: a_{3b+1}
held-out:       a_{3b+2}
```

Across 32 buckets:

```text
pilot subjects:           32
single-exposure subjects: 32
held-out subjects:        32
```

This makes the single-exposure and held-out groups matched by bucket. For every single-exposure subject, there is a held-out subject with the same semantic bucket structure.

## Experiment 1: Single-Exposure Exact Recall

### Phase 1: Learn the Semantic Scaffold

Phase 1 trains on:

1. Attribute sentences for all subjects and objects, repeated 8 times.
2. Pilot `r_0` relation triples, repeated 160 times.

The pilot triples are:

```text
x a_{3b} r_0 b_target x
```

for every bucket `b`.

This teaches the model:

- how subject attributes work;
- how object attributes work;
- that `r_0` maps a subject bucket to the shifted object bucket.

It does not show relation triples for the single-exposure or held-out subjects.

### Phase 2: One Exact Exposure

Phase 2 shows exactly one `r_0` triple for each single-exposure subject:

```text
x a_{3b+1} r_0 b_target x
```

Each such exact triple appears once.

The held-out subject from the same bucket is never shown:

```text
x a_{3b+2} r_0 b_target x
```

does not appear in training.

### Phase 2 Update Rule

In Phase 2, the model updates only the output row for the target object token.

This is controlled by:

```text
phase2_update_mode = target_lm_rows
```

Concretely:

- The transformer backbone is frozen for the Phase 2 update.
- Non-target output rows are frozen.
- Only the output embeddings for the object tokens that appear as single-exposure targets are updated.

This isolates the question:

Can a mature representation bind one exact object identity after a single exposure?

The update does not let the model relearn the full relation from repeated exposure.

## Experiment 1 Metrics

Evaluation prompt:

```text
x a_i r_0
```

The model must predict the next token.

Metrics:

- Exact top-1: whether the top object prediction is the exact target object.
- Held-out exact top-1: same metric for bucket-matched subjects whose exact triples were never shown.
- Within-bucket top-1: whether the correct object is ranked highest among the three same-bucket candidates.

Final replicated results:

| seed | single exact top-1 | held-out exact top-1 | single within-bucket top-1 |
|---:|---:|---:|---:|
| 23 | 0.781 | 0.000 | 1.000 |
| 31 | 0.812 | 0.000 | 1.000 |
| 47 | 0.844 | 0.000 | 1.000 |
| mean | 0.812 | 0.000 | 1.000 |

This satisfies the single-exposure requirement: the seen-once facts are learned behaviorally, while matched held-out facts are not.

## Experiment 2: High-Frequency Semantic and Random Relations

Experiment 2 uses the same generated subjects, objects, buckets, and exact semantic targets.

The model is trained on two relations:

```text
r_0 = semantic bucket-supported exact mapping
r_1 = random exact mapping
```

For both relations, the same train subjects are used:

```text
pilot subjects + single-exposure subjects = 64 train subjects
held-out subjects = 32 held-out subjects
```

Each train relation fact is repeated 128 times:

```text
x a_i r_0 b_semantic_target x
x a_i r_1 b_random_target x
```

The held-out subjects are never shown in relation triples.

## Experiment 2 Metrics

For each relation:

- Train exact top-1: exact object recall for trained subjects.
- Held-out exact top-1: exact object recall for never-trained subjects.
- Prompt affine nearest-neighbor accuracy: fit an affine map from relation-conditioned prompt hidden states to object unembedding vectors on trained subjects, then test exact object recovery on held-out subjects.

Final replicated high-frequency results at 128 repeats:

| seed | semantic train | semantic held-out | semantic affine NN | random train | random held-out | random affine NN |
|---:|---:|---:|---:|---:|---:|---:|
| 23 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| 31 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| 47 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

This satisfies the high-frequency requirement:

- The semantic relation is memorized exactly at high frequency.
- The random relation is memorized exactly at high frequency.
- Neither relation preserves exact-token affine generalization.

## Reproduction Commands

Run from the repository root:

```bash
cd experiments

env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
  uv run python -m factual_recall.ambiguous_bucket_experiments \
  --device cpu --seed 23 --repeats 128 \
  --n-buckets 32 --subjects-per-bucket 3 --objects-per-bucket 3 \
  --attr-repeats 8 --pilot-repeats 160 --batch-size 8 \
  --learning-rate 0.002 --phase2-learning-rate 0.5 \
  --phase2-update-mode target_lm_rows --object-loss-weight 64 \
  --hidden-dim 128 \
  --output-dir factual_recall/results/ambiguous_bucket_final_seed23_f128

env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
  uv run python -m factual_recall.ambiguous_bucket_experiments \
  --device cpu --seed 31 --repeats 128 \
  --n-buckets 32 --subjects-per-bucket 3 --objects-per-bucket 3 \
  --attr-repeats 8 --pilot-repeats 160 --batch-size 8 \
  --learning-rate 0.002 --phase2-learning-rate 0.5 \
  --phase2-update-mode target_lm_rows --object-loss-weight 64 \
  --hidden-dim 128 \
  --output-dir factual_recall/results/ambiguous_bucket_final_seed31_f128

env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
  uv run python -m factual_recall.ambiguous_bucket_experiments \
  --device cpu --seed 47 --repeats 128 \
  --n-buckets 32 --subjects-per-bucket 3 --objects-per-bucket 3 \
  --attr-repeats 8 --pilot-repeats 160 --batch-size 8 \
  --learning-rate 0.002 --phase2-learning-rate 0.5 \
  --phase2-update-mode target_lm_rows --object-loss-weight 64 \
  --hidden-dim 128 \
  --output-dir factual_recall/results/ambiguous_bucket_final_seed47_f128
```

## Output Files

For each run, the script writes:

```text
ambiguous_bucket_metrics.json
summary.md
```

Example:

```text
experiments/factual_recall/results/ambiguous_bucket_final_seed23_f128/
  ambiguous_bucket_metrics.json
  summary.md
```

# Project Plan v2: Multiple Relations with Filler Tokens

*Written 2026-04-16. Supersedes project_plan.md.*

## 1. Motivation

The original single-relation setup has a fundamental flaw: in a causal LM trained on
`x a_i b_i`, the last-layer hidden state at the `a_i` position has exactly one job —
predict `b_i` next. Finding that the object is recoverable from the subject
representation is then tautological. This plan addresses the flaw with two combined
changes:

1. **Multiple permuted relations** — the same subject token must predict different
   objects depending on relational context, forcing object representations to encode
   entity identity rather than being owned by a single subject.
2. **Filler tokens** — N filler tokens between subject and object decouple the
   subject's last-layer representation from direct next-token prediction of the object.

See `results/experimental_design_critique.md`, `results/multiple_relations_design.md`,
and `results/filler_tokens_design.md` for the reasoning behind these choices.

---

## 2. Sequence Format

Each training example is a sequence of triples separated by `x`:

```
x  a_i  r_k  f0  f1  f2  b_{R_k(i)}  x  a_j  r_m  f0  f1  f2  b_{R_m(j)}  ...
```

**Token order within a triple**: `separator | subject | relation | filler × 3 | object`

The relation token comes *after* the subject. Consequently, at the subject position
`a_i`, the model can only attend to prior context — not to `r_k`. Subject
representations therefore encode entity identity independently of which relation is
active. The relation token then gates the prediction of the correct object through the
filler span.

---

## 3. Vocabulary

| Token set       | Tokens            | Count |
|-----------------|-------------------|-------|
| Subjects        | a0 .. a99         | 100   |
| Objects         | b0 .. b99         | 100   |
| Relations       | r0, r1, r2        | 3     |
| Fillers         | f0, f1, f2        | 3     |
| Separator       | x                 | 1     |
| **Total**       |                   | **207** |

Filler tokens are always written in the fixed order `f0 f1 f2`. Using three distinct
tokens (rather than one repeated token) prevents the model from collapsing the filler
span into a no-op; the tokens are predictable and easily learned, but they occupy
positions that create a processing buffer between subject and object.

---

## 4. Relations

Three bijective mappings over the shared object vocabulary:

| Relation | Token | Mapping                          | Notes                        |
|----------|-------|----------------------------------|------------------------------|
| R0       | r0    | a_i → b_i                        | Identity                     |
| R1       | r1    | a_i → b_{(i+50) mod 100}         | Cyclic shift by 50           |
| R2       | r2    | a_i → b_{π(i)}                   | Fixed random permutation π   |

π is hard-coded (not sampled at runtime) to make analysis and debugging reproducible.
All three relations use the same b0..b99 vocabulary — the objects are shared, only the
pairing differs.

Training sequences interleave all three relations uniformly at random.

---

## 5. What This Design Tests

**Primary question (per-relation linearity):** For each relation R_k, does there exist
an affine map T_k such that

    T_k(h_{a_i}) ≈ h_{b_{R_k(i)}}

with good cross-validated Δcos, where h_{a_i} is the subject representation extracted
*before* the model has seen the relation token?

**Secondary question (linearity of the relation space):** Is the space of affine maps
{T_0, T_1, T_2} itself linearly structured? Concretely, does there exist a base map
T_0 and "relation delta" vectors Δ_1, Δ_2 such that T_k ≈ T_0 + Δ_k? A positive
result would be a second-order linearity finding — the relations themselves form a
linear space, not just each individual relation.

**Layer-by-layer analysis:** Linear structure should be studied at every layer, not
just the last. The last-layer representation at the subject position is no longer a
direct pointer to the object (the next token is f0, not b_j), but intermediate layers
may still reveal where the linear structure emerges in the forward pass.

---

## 6. Representation Extraction

### Subject representations
Run the model on the prefix `x a_i` (no relation token, no fillers, no object).
Extract the hidden state at the `a_i` position at each layer. This is the pure subject
representation, uncontaminated by relational context.

This extraction context matches the training setup: at the `a_i` position during
training, the model has only seen `x a_i`, so the representation is computed
identically.

### Object representations
Run the model on the prefix `x b_j` (symmetric probe, consistent with v1 option B).
Extract the hidden state at the `b_j` position at each layer. Because all three
relations share the same object vocabulary, a single set of object representations
serves all three affine maps.

---

## 7. Affine Fitting and Evaluation

### Per-relation fit
For each relation R_k, collect the matrix pairs:

    S^k = [h_{a_0}, h_{a_1}, ..., h_{a_99}]   (100 × d)
    O^k = [h_{b_{R_k(0)}}, ..., h_{b_{R_k(99)}}]   (100 × d)

Fit T_k = (W_k, b_k) via least squares (same procedure as v1). Use 5-fold CV across
pairs; report mean ± std Δcos per relation per layer.

### Second-order linearity test
After fitting T_0, T_1, T_2, test whether W_1 ≈ W_0 + ΔW_1 and W_2 ≈ W_0 + ΔW_2
for some low-rank ΔW. Quantify via:
- Frobenius distance between each pair of W matrices
- Rank of (W_1 − W_0) and (W_2 − W_0)
- Whether a single shared W with relation-specific bias b_k suffices (pure translation
  hypothesis)

### Cross-relation generalization
Train T_0 on R0 pairs, apply it to predict objects for R1 and R2 pairs. If it
generalizes (Δcos >> 0), the affine maps share geometric structure across relations.

---

## 8. Implementation Changes from v1

The existing modules need the following updates:

| Module                  | Change needed |
|-------------------------|---------------|
| `src/config.py`         | Add `num_relations=3`, `num_fillers=3`, `relation_permutations` |
| `src/data_generation.py`| Generate triples with relation token and fillers; sample relation uniformly; store permutation π |
| `src/model.py`          | No architecture change needed |
| `src/training.py`       | Optionally track per-relation object prediction accuracy |
| `src/representations.py`| Subject extraction: `x a_i` prefix; object extraction unchanged |
| `src/affine.py`         | Loop over relations; add second-order test |
| `src/evaluation.py`     | Report per-relation Δcos; add cross-relation eval |
| `experiments/`          | New script: `analyze_linearity_v2.py` |

---

## 9. Finalized Baseline Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Filler span | 3 (`f0 f1 f2`) | Baseline; ablations (1/2/3/5) after baseline is solid |
| R2 permutation π | Hard-coded | Reproducible; no runtime sampling |
| Model: n_layers | 1 | Same as v1 baseline |
| Model: hidden_dim | 128 | Same as v1 baseline |
| Model: n_heads | 4 | Same as v1 baseline |

---

## 10. Experimental Protocol

1. **Generate dataset** with 3 relations, filler span of 3, all 100 pairs per relation
2. **Train model** (start: 1 layer, hidden_dim=128, 4 heads) until 100% per-relation
   object prediction accuracy
3. **Extract representations** for all subjects and objects at all layers
4. **Fit per-relation affine maps** with 5-fold CV; report Δcos table per layer
5. **Run second-order tests**: rank of (W_1−W_0), (W_2−W_0); pure-translation test
6. **Run cross-relation generalization**: T_0 → R1, R2 Δcos
7. **Ablations** (after baseline is solid):
   - Filler span length: 1, 2, 3, 5 fillers
   - Number of relations: 2, 3, 5
   - Model depth: 1, 2, 3, 4 layers

---

## 11. Success Criteria

| Criterion | Target |
|-----------|--------|
| Training accuracy | 100% object prediction per relation |
| Per-relation Δcos (best layer) | > 0.8 |
| Second-order linearity | rank(W_1 − W_0) ≤ 5 or pure-translation holds |
| Cross-relation generalization | Δcos > 0.5 on held-out relation |

---

## 12. Open Questions

1. At which layer does linear structure appear — and does it differ across relations?
2. Does the subject representation (extracted before the relation token) already encode
   enough to allow all three per-relation affine maps, or does the residual stream
   upstream of the relation token carry no relational information?
3. Does the rank of (W_1 − W_0) reveal anything about the dimensionality of the
   "relation subspace"?
4. How many fillers are needed before the tautology concern fully dissolves?

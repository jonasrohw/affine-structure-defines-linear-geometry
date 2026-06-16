# Plan: LRE Unembedding Evaluation

## Core Insight

The existing Δcos metric is purely geometric — it measures whether ô is *closer* to the correct object in representation space, but all object vectors cluster tightly (mean pairwise cosine ≈ 0.91), so "closer" doesn't discriminate. The LRE paper's actual test is functional: apply the model's own **unembedding layer** (ln_f → lm_head) to ô and check whether the correct object token gets the highest logit.

**Key architectural point**: `residuals[k]` are captured *before* `ln_f`, so the decoding path for ô is exactly `model.ln_f(ô)` → `model.lm_head(...)` → logits over vocab.

---

## Files to Change

### 1. `src/evaluation.py` — Three new functions

- **`decode_ohat_to_logits(O_hat, model)`**: Applies `ln_f` + `lm_head` to a batch of ô vectors → returns `(n, vocab_size)` logits.
- **`unembedding_rank_accuracy(S, correct_object_ids, W, b, model)`**: For each pair, computes ô = W·s + b, decodes via unembedding, finds rank of correct object token. Returns `mean_rank`, `median_rank`, `topk_acc/{1,5,10}`.
- **`subject_direct_unembedding_rank(S, correct_object_ids, model)`** (baseline): Same but decodes *s* directly without the affine map. Distinguishes two cases: (a) affine adds value vs. (b) subject already encodes the object.

### 2. `src/representations.py` — One helper

- **`get_correct_object_ids(cfg, pair_indices)`**: Returns the vocab token id of b_i for each pair, for use as ground truth.

### 3. `experiments/analyze_linearity.py` — Extended evaluation block

After the existing Δcos table, add:

1. **Per-layer unembedding sweep**: For each layer, fit affine on all pairs and report `mean_rank` + top-k accuracy. Produces a second table and a new plot `unembedding_rank_per_layer.pdf`.
2. **Cross-validated unembedding evaluation** (k-fold): Fit affine on 80 pairs, evaluate unembedding rank on the held-out 20. This is the rigorous generalization metric.
3. **Baseline comparison**: Report unembedding rank for *s* directly (no affine) alongside the ô result.

---

## New Evaluation Metrics

| Metric | What it measures |
|---|---|
| `topk_acc/1` for ô | Does ô decode to the correct object at rank 1? |
| `topk_acc/1` for s | Does *s* already decode to the correct object without any affine? |
| `mean_rank` | Average rank of b_i in logit-sorted vocab (1 = perfect, ~100 = chance) |
| CV `topk_acc/1` | Same, cross-validated (W fit on 80 pairs, evaluated on 20) |

---

## Possible Outcomes

1. **Top-1 = 1.0 for ô, low for s** → LRE holds: affine maps s to a functionally correct object representation
2. **Top-1 = 1.0 for both** → The subject representation *already* encodes the correct object; affine is not needed
3. **Top-1 is low for ô** → The tight object cluster means ô is close to the cluster centroid, not b_i specifically; LRE does not hold in the functional sense

---

## Implementation Sequence

1. Add `decode_ohat_to_logits`, `unembedding_rank_accuracy`, `subject_direct_unembedding_rank` to `src/evaluation.py`
2. Add `get_correct_object_ids` to `src/representations.py`
3. Extend `experiments/analyze_linearity.py` with per-layer sweep, CV evaluation, baseline comparison, and new plot
4. Run `python experiments/analyze_linearity.py` — no retraining needed, uses existing checkpoint

---

## Architectural Note on `ln_f`

`ln_f` is `nn.LayerNorm(hidden_dim)`, which normalizes over the last dimension. Calling `model.ln_f(O_hat)` on a tensor of shape `(n, hidden_dim)` correctly normalizes each vector independently — same behavior as in the forward pass where it is applied to `(B, T, hidden_dim)`.

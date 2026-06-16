# MBZ: Simplified Factual Recall with C(x,2) Entity Structure

## Task Overview

Train a 1-layer transformer to memorize relation triples where entities
have a hidden (f, g) pair structure. The model sees only relation
triples — no attribute sentences. Relations are defined by the
attribute structure but the model must discover/memorize this without
being told.

## Entity Structure

Entities are **unordered pairs** from a base set of x elements
{0, 1, ..., x-1}. Each entity is a pair (f, g) with f < g, giving
C(x, 2) = x(x-1)/2 entities.

Example with x=10: entities are (0,1), (0,2), ..., (0,9), (1,2),
(1,3), ..., (8,9) — 45 entities total.

Each entity serves as BOTH subject and object (shared vocabulary,
unlike the graz experiment which has separate subject/object tokens).

## Relation Definitions

Two relations R0 and R1, each a bijection (permutation) on the
C(x,2) entities. Relations are defined via the attribute structure:

**R0 (shared f-attribute):** For each value of f, R0 permutes the
(x-1) entities that share that f-value. Concretely, for entities
with first component f, R0 cycles through their second components:
  - R0((f, g)) = (f, σ_f(g))
  - where σ_f is a fixed cyclic permutation on {0,...,x-1} \ {f}

**R1 (shared g-attribute):** For each value of g, R1 permutes the
(x-1) entities that share that g-value:
  - R1((f, g)) = (τ_g(f), g)
  - where τ_g is a fixed cyclic permutation on {0,...,x-1} \ {g}

Using cyclic permutations (shift by 1 within each attribute group)
is the simplest choice. Both R0 and R1 are bijections on the full
entity set because they permute within disjoint attribute groups.

**Important:** The model never sees the (f, g) decomposition. It sees
each entity as a single opaque token. The attribute structure only
determines WHICH entity maps to which under each relation.

## Training Data

Sequences consist of relation triples only:

    [sep, relation_token, subject_token, object_token]

- sep = special separator token (token 0)
- relation_token: one of {R0, R1}
- subject_token: one of the C(x,2) entity tokens
- object_token: the entity that the subject maps to under that relation

Pack multiple triples into fixed-length sequences (e.g., 48 tokens =
12 triples per sequence).

**Training set:** Use p% of the full set of (entity, relation) pairs.
At p=100%, there are 2 × C(x,2) triples. The model is trained on
next-token prediction — it must predict the object token given
[sep, rel, subj].

**No attribute sentences.** The model has no way to learn the (f, g)
structure except through patterns in the relation triples.

## Vocabulary

- Token 0: separator
- Tokens 1 to C(x,2): entity tokens (shared for subject and object)
- Tokens C(x,2)+1 to C(x,2)+2: relation tokens R0, R1
- Total vocab size: C(x,2) + 3

## Model Architecture

1-layer GPT-style transformer (same architecture as graz/src/model.py):
- Causal attention (1 layer, 4 heads)
- MLP with GELU activation (hidden_dim → 4×hidden_dim → hidden_dim)
- LayerNorm before attention and MLP
- Learned positional embeddings
- Tied input/output embeddings (lm_head = embedding matrix)

## Recommended Parameters

- **x = 10** → C(10,2) = 45 entities, 90 triples at p=100%
- **dim = 64**, 4 attention heads
- **p = 100%** (all triples in training)
- **30 epochs**, batch size 64, lr = 3e-4, AdamW
- Sequence length 48 (12 triples per sequence)
- ~62,500 sequences per epoch (same as graz)

This should converge to 100% accuracy in a few epochs. The task is
intentionally easy — we care about the learned representations, not
whether the model can solve the task.

## Implementation

Use the graz experiment as a reference implementation:
- **graz/src/model.py**: transformer architecture (MiniTransformer)
- **graz/src/config.py**: ModelConfig
- **graz/src/training.py**: learning rate schedule (get_lr)
- **graz/experiments/ravfogel_1layer.py**: closest reference for the
  training loop, vocab class, sequence building, evaluation

Key differences from graz/ravfogel_1layer.py:
1. Shared entity vocabulary (no separate subject/object tokens)
2. Relations defined by attribute structure (not random permutations)
3. No attribute sentences
4. Smaller entity count (45 vs 2025)

### File structure

```
mbz/
  src/
    model.py      — copy or import from graz/src/model.py
    config.py     — copy or import from graz/src/config.py
    training.py   — copy or import from graz/src/training.py
  experiments/
    train.py      — main training + evaluation script
  results/        — output JSON and analysis files
  checkpoints/    — saved model checkpoints
```

You can either copy the graz source files or import them. Copying is
cleaner for an independent project.

### Vocab class sketch

```python
class Cx2Vocab:
    def __init__(self, x=10):
        self.x = x
        self.n_entities = x * (x - 1) // 2
        self.num_relations = 2
        self.sep_id = 0
        self.entity_offset = 1
        self.relation_offset = 1 + self.n_entities

        # Build entity-to-pair and pair-to-entity mappings
        self.entities = []  # list of (f, g) tuples, f < g
        self.pair_to_idx = {}
        idx = 0
        for f in range(x):
            for g in range(f + 1, x):
                self.entities.append((f, g))
                self.pair_to_idx[(f, g)] = idx
                idx += 1

        # Build R0 and R1 as permutations
        self._build_relations()

    def _build_relations(self):
        """R0: cycle within shared-f groups. R1: cycle within shared-g groups."""
        self.perm = [None, None]

        # R0: for each f, cycle the g-values
        r0 = [0] * self.n_entities
        for f in range(self.x):
            partners = sorted(g for g in range(self.x) if g != f)
            # Cyclic shift: g -> next g in sorted order
            for i, g in enumerate(partners):
                next_g = partners[(i + 1) % len(partners)]
                src = self.pair_to_idx[tuple(sorted((f, g)))]
                dst = self.pair_to_idx[tuple(sorted((f, next_g)))]
                r0[src] = dst
        self.perm[0] = r0

        # R1: for each g, cycle the f-values
        r1 = [0] * self.n_entities
        for g in range(self.x):
            partners = sorted(f for f in range(self.x) if f != g)
            for i, f in enumerate(partners):
                next_f = partners[(i + 1) % len(partners)]
                src = self.pair_to_idx[tuple(sorted((f, g)))]
                dst = self.pair_to_idx[tuple(sorted((next_f, g)))]
                r1[src] = dst
        self.perm[1] = r1

    @property
    def vocab_size(self):
        return 1 + self.n_entities + self.num_relations

    def entity_id(self, idx):
        return self.entity_offset + idx

    def relation_id(self, k):
        return self.relation_offset + k

    def relation_object(self, entity_idx, k):
        return self.perm[k][entity_idx]
```

**IMPORTANT:** The `_build_relations` method as sketched above has a
subtlety with unordered pairs — when cycling g-values for a fixed f,
the pair (f, next_g) might need to be stored as (next_g, f) if
next_g < f. The `tuple(sorted(...))` handles this, but verify that
R0 and R1 are actually valid permutations (every entity maps to
exactly one other entity, no fixed points ideally). Print and check
the permutation before training.

## Analysis to Run After Training

1. **Accuracy**: per-relation and overall (should be 100%)
2. **Loss curve**: should converge quickly
3. **Embedding PCA**: do entity embeddings show structure reflecting
   the (f, g) attributes? Check if entities sharing an f-value or
   g-value cluster together in embedding space, even though the model
   was never explicitly told about attributes.
4. **MLP cluster analysis**: same as graz — cluster MLP inputs by
   k-means, fit affine maps per cluster, report R² for k=1,2,5,10,20.
   Does the MLP implement a piecewise-affine transformation?
5. **Neuron analysis**: which neurons are relation-selective? Is there
   a neuron partition between R0 and R1?

## GPU Selection

Always check GPU availability before running. Use:
```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_gpu> python experiments/train.py
```

## Reference

This experiment is related to the graz project in the parent
directory (`../graz/`). The graz project studies how transformers
learn affine transformations implementing factual recall, with
detailed analysis of GELU, ReLU, and GeGLU activation functions
across different dimensionalities. See `graz/results/` for findings.

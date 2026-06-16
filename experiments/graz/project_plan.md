# Graz: Attribute-Based Relational Structure in Embeddings

## Research question

Do transformer embeddings reflect implicit relational structure learned from
attribute associations alone — without ever seeing explicit relation tokens?

In salzburg, the model saw explicit relation-subject-object triples
(`x r_k a_i b_j`). Here, the model only learns which attributes each entity
has. Two entities are implicitly related if they share attributes, but this
relation is never stated directly. The hypothesis is that the learned
embeddings should reflect this shared-attribute structure: entities with the
same f-attributes should have similar embeddings, and entities with the same
g-attributes should have similar embeddings.

## Entities and attributes

Each entity (subject or object) has two groups of attributes:

- **f-attributes**: a 3-element subset of {0, ..., 9} → C(10,3) = 120 possible f-types
- **g-attributes**: a 3-element subset of {0, ..., 9} → C(10,3) = 120 possible g-types

An entity's identity is fully described by its (f-type, g-type) pair. The full
combinatorial space is 120 × 120 = 14,400 possible entities.

**Example**: entity `a_012345` has f-attributes {0, 1, 2} and g-attributes
{3, 4, 5}. Entity `a_567234` has f-attributes {5, 6, 7} and g-attributes
{2, 3, 4}.

## Implicit relations

Two relations are defined but never appear as tokens in the training data:

- **R0 (shared f-attributes)**: subject a and object b are R0-related iff they
  have the same f-type (same 3 f-attributes)
- **R1 (shared g-attributes)**: subject a and object b are R1-related iff they
  have the same g-type (same 3 g-attributes)

**Constraint**: each subject has exactly one R0-partner (object) and exactly
one R1-partner (object).

## Sentence format

Each training sentence is 3 tokens: separator, entity, attribute.

```
x  a_012345  f_0       (subject a_012345 has f-attribute 0)
x  a_012345  g_4       (subject a_012345 has g-attribute 4)
x  b_567234  f_6       (object b_567234 has f-attribute 6)
x  b_567234  g_3       (object b_567234 has g-attribute 3)
```

To generate one sentence: sample an entity (subject or object), sample an
attribute type (f or g with equal probability), sample one of the entity's 3
attributes of that type uniformly. Each entity has 6 possible sentences
(3 f-attributes + 3 g-attributes).

## Dataset construction

### The uniqueness problem

For R0/R1 to have unique partners, each f-type must appear exactly once among
subjects and exactly once among objects, and same for g-types. Otherwise a
subject would have multiple R0- or R1-partners.

### Proposed construction (Latin square, N = 120)

1. Enumerate the C(10,3) = 120 three-element subsets of {0,...,9} in a fixed
   order: c₀ = {0,1,2}, c₁ = {0,1,3}, ..., c₁₁₉ = {7,8,9}.

2. Create **120 subjects** and **120 objects**:

   | Entity | f-type | g-type |
   |--------|--------|--------|
   | Subject i | c_i | c_i |
   | Object j | c_j | c_{(j+60) mod 120} |

3. This gives unique partners for both relations:

   | Relation | Mapping | Permutation |
   |----------|---------|-------------|
   | R0 (shared f) | subject i ↔ object i | identity |
   | R1 (shared g) | subject i ↔ object (i+60) mod 120 | shift by 60 |

4. **Verification**: subject i has g-type c_i. Object (i+60) mod 120 has
   g-type c_{((i+60)+60) mod 120} = c_i. ✓ Each subject has exactly one
   R0-partner and one R1-partner. R0 ≠ R1. ✓

### Properties

- 120 subjects, 120 objects
- Vocabulary: 1 (sep) + 120 (subjects) + 120 (objects) + 10 (f-attrs) + 10
  (g-attrs) = **261 tokens**
- Unique facts: 240 entities × 6 attributes = **1,440 facts**
- Irreducible loss at attribute position: −log(1/6) ≈ 1.79 (since each entity
  has 6 equally likely attributes)

### Why this works

The construction is a Latin square on the (f-type, g-type) grid. Subjects lie
on the diagonal (f-index = g-index). Objects lie on a shifted diagonal
(g-index = f-index + 60 mod 120). The shift ensures R0 and R1 are different
permutations, mirroring salzburg's R0 = identity, R1 = shift by n/2.

### Scaling options

To test within-group similarity more directly (multiple subjects per f-type),
increase to N = 120k with k subjects per f-type. This requires a k-regular
bipartite matching, which is constructible but more involved. Start with
k = 1 (N = 120) and test graded similarity (by attribute overlap: 0/3, 1/3,
2/3, 3/3 shared attributes).

## Training setup

Reuse salzburg's model architecture (1-layer GPT-style transformer) with
appropriate adjustments:

- **Sequence format**: concatenate multiple 3-token sentences:
  `x e₁ a₁ x e₂ a₂ x e₃ a₃ ...`
- **Model**: 1-layer transformer, hidden_dim = 32 (start in salzburg's sweet
  spot), n_heads = 4
- **Loss**: cross-entropy on next-token prediction (all positions)
- **Epochs**: 800 (same as salzburg)

The model needs to learn: given an entity token, predict which attributes it
has. The loss at attribute positions (every 3rd token) is the primary signal.

## Evaluation (Phase 1)

### Embedding similarity by attribute overlap

For all pairs of subjects (i, j), compute:
- Cosine similarity of their learned embeddings
- Number of shared f-attributes (0, 1, 2, or 3 out of 3)
- Number of shared g-attributes (0, 1, 2, or 3 out of 3)

**Hypothesis**: cosine similarity correlates positively with attribute overlap.
Entities with 3/3 shared f-attributes should be most similar; 0/3 should be
least similar. Same for g-attributes.

### Subspace analysis

Test whether the embedding space has a factored structure:
- Is there an f-subspace (linear projection) that captures f-attribute
  similarity?
- Is there a separate g-subspace for g-attribute similarity?
- Can we find these via PCA on attribute-grouped embeddings?

### Cross-entity-type similarity

Do subjects and objects with the same f-type have similar embeddings?
Subject a_i and object b_i share f-type c_i (they are R0-partners). Their
embeddings should reflect this, even though R0 is never explicit in the data.

## Phase 2: Adding explicit relations (future)

Split the 120 subjects into train (e.g., 80) and test (e.g., 40). Add
explicit relation-subject-object triples to the pretraining data for train
subjects only:

```
x r_0 a_i b_{R0(i)}       (explicit R0 triple)
x r_1 a_i b_{R1(i)}       (explicit R1 triple)
```

**Hypothesis**: the model generalizes the relation to test subjects — it can
predict the correct R0/R1 partner for subjects it has only seen in
attribute-association sentences, not in relation triples. The embeddings
learned in Phase 1 should provide the structural foundation for this
generalization.

## Connection to salzburg

| Aspect | Salzburg | Graz |
|--------|----------|------|
| Relations | Explicit (r_k token in sentence) | Implicit (shared attributes) |
| Sentence structure | x r_k a_i b_j (4 tokens) | x a_i f_k (3 tokens) |
| What model learns | Subject → object mapping per relation | Entity → attribute associations |
| Key question | Is the S→O map affine? | Do embeddings reflect attribute structure? |
| Affine structure | Measured via fit_affine + Voronoi | Not primary (similarity-based evaluation instead) |
| Relations | 2–3 permutations | 2 implicit (shared f-type, shared g-type) |

Both projects study how transformers represent relational structure, but from
different angles: salzburg asks whether the *mapping* is structured (affine);
graz asks whether the *representations themselves* are structured
(attribute-factored).

## Changes from original prompt

1. **Typo fix**: "last two indices for g attributes" → "last three indices."
   The examples (e.g., `a_567234`, 6 digits) and the description "indices lmn"
   both confirm 3 g-indices.

2. **Relation naming**: standardized R1/R2 → R0/R1 throughout. The original
   prompt defines "first relation R0" and "second relation R1" but later refers
   to "R1 and R2."

3. **Dataset construction**: proposed the Latin-square approach (N = 120
   subjects and objects) using the diagonal + shifted-diagonal assignment on
   the (f-type, g-type) grid. This gives R0 = identity, R1 = shift by 60 —
   mirroring salzburg's structure.

4. **Evaluation methodology**: added concrete plan for testing the embedding
   structure hypothesis (graded cosine similarity by attribute overlap,
   subspace factoring analysis, cross-entity-type similarity).

5. **Generalization hypothesis clarification**: interpreted "the relation will
   then generalize to subjects/objects that have occurred in relation
   expressions in the training data" as meaning generalization to test
   subjects/objects that have NOT appeared in explicit relation triples but
   HAVE appeared in attribute-association pretraining. (The original phrasing
   says "have occurred," but generalizing to entities already seen in relation
   expressions would not test generalization.)

6. **Scaling note**: the full combinatorial space is 14,400 entities, but
   starting with N = 120 keeps the vocabulary small (261 tokens) and the
   experiment tractable. Noted the option to scale to N = 120k for within-group
   similarity testing.

7. **Irreducible loss**: noted that the minimum achievable loss at attribute
   positions is −log(1/6) ≈ 1.79, since each entity has 6 equally likely
   attributes. This serves as a convergence diagnostic.

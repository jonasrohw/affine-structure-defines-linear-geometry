# Paper Plan: Structure vs Memorization in Transformers

## Working Title

"When Do Transformers Learn Structure? Affine Maps and Compositional Generalization in Synthetic Relational Tasks"

## Thesis

Transformers default to memorization when capacity is abundant. Structured solutions (affine relation maps, compositional type encodings) emerge only under specific pressure: dimensional bottlenecks, auxiliary losses, or contrastive objectives. The two projects (Salzburg, Graz) provide complementary evidence from different angles.

---

## Main Body

### 1. Introduction (~1 page)

- Motivation: LLMs learn relational knowledge, but do they learn *structure* or memorize associations?
- Two complementary studies: (a) whether single relations are implemented as affine maps (Salzburg), (b) whether compositional attribute structure supports generalization to unseen entity pairings (Graz)
- Preview of findings: memorization is the default; structure requires explicit pressure

### 2. Related Work (~0.5 page)

- Hernandez et al. (2024): linear relational embeddings, delta-cosine metric
- Nanda et al.: grokking and mechanistic interpretability
- Compositional generalization literature (SCAN, COGS, etc.)
- Information bottleneck / representation learning theory

### 3. Experimental Setup (~1.5 pages)

#### 3.1 Salzburg: Affine Relation Maps
- Vocabulary: 100 subjects, 100 objects, 2 relations (identity, cyclic shift)
- Data: 4-token sequences [x, r_k, subject, object]
- Model: 1-layer transformer, 4 heads, varying dim (8-128)
- Evaluation: affine map fitting via least-squares, Voronoi classification accuracy

#### 3.2 Graz: Compositional Attribute Relations
- Focus presentation on the 5-feature setup: C(5,2) = 10 types, 100 entities
- Two attribute groups (f-type, g-type); R0 = shared f-type, R1 = shared g-type
- Sentence types: attribute [x, entity, attr], relation [x, r_k, subj, obj], reverse [x, S/O, attr, entity], contrastive [x, Q_k, obj, subj, y/n]
- Model: 1-layer transformer, 4 heads, varying dim (8-64)
- Training: 62.5K sequences, 800 epochs; 20% of triples held out
- Evaluation: held-out triple accuracy, affine probe, contrastive accuracy

### 4. Salzburg Results (~2 pages)

#### 4.1 Affine structure emerges at the right scale

**Table 1: Dimension sweep (2 relations, 100 subjects)**

| dim | Object acc | Affine acc | Interpretation |
|-----|-----------|-----------|----------------|
| 8   | 100%      | 73%       | Bottleneck: no good affine solution exists |
| 16  | 100%      | 97%       | Sweet spot |
| 32  | 100%      | 100%      | Sweet spot |
| 64  | 100%      | 27.5%     | Overparameterized: memorization wins |

- Key finding: non-monotonic relationship between capacity and structure
- Existence test confirms dim=32 can support affine maps — the model *chooses* not to use them at dim=64

#### 4.2 Auxiliary loss nudges toward structure
- At dim=16, tiny auxiliary loss (0.02% of CE) pushes affine accuracy 97% → 100%
- Proves: the structured solution exists and is reachable, but SGD doesn't find it without explicit pressure

#### 4.3 Mechanistic analysis
- Logit lens: attention is 0.987-affine (mechanically linear), MLP provides relation-dependent gating via GELU
- Sublayer decomposition: affine structure lives in attention; MLP implements the multiplicative interaction needed for relation-dependent maps

**Figure 1: Affine accuracy vs dimension (U-shaped curve)**

### 5. Graz Results (~2 pages)

#### 5.1 Capacity controls structure vs memorization

Present the dimension sweep at 5 features (100 entities) as the central result:

**Table 2: Dimension sweep (5 features, C(5,2) = 10 types, 100 entities)**

| dim | Train acc | Held-out acc | × chance | Contrastive +/- | Affine test | Partner sim |
|-----|-----------|-------------|----------|-----------------|-------------|-------------|
| 8   | 100%      | 42.5%       | 4.25×    | 85.2% / 94.0%   | 26.0%       | +0.43/+0.42 → collapsed |
| 16  | 100%      | **90.0%**   | **9.0×** | 88.8% / 97.8%   | 50.0%       | +0.43/+0.42 |
| 32  | 100%      | 85.0%       | 8.5×     | 88.8% / 99.2%   | 55.0%       | +0.18/+0.18 |
| 64  | 100%      | 67.5%       | 6.75×    | 90.0% / 97.6%   | 63.0%       | +0.10/+0.13 |

- Same non-monotonic pattern as Salzburg: dim=16 is the sweet spot, dim=64 memorizes
- Embedding structure gets denser as dim shrinks: partner sim +0.10 → +0.43 from dim=64 to dim=16
- dim=8 collapses: embeddings nearly identical (overlap 0.91), can't represent distinct entities
- All models achieve 100% train accuracy — the difference is purely in generalization

#### 5.2 The compositional task

- 10 entity types defined by combinatorial attributes; relations R0/R1 select on f-type/g-type
- 10 valid targets per relation per subject (chance = 10%)
- 200 total triples, 160 train, 40 held out
- Contrastive sentences [x, Q_k, obj, subj, y/n] provide explicit positive/negative signal about type membership
- Reverse lookups [x, S/O, attr, entity] train attribute→entity mapping (100% across all dims)

#### 5.3 Parallel to Salzburg

- Both show an inverted-U for generalization vs dimension
- Both show memorization as the default at high capacity
- Salzburg measures structure via affine probe; Graz measures it via held-out generalization on novel entity pairings
- The information bottleneck at dim=16 forces compact, compositional representations in both settings

**Figure 2: Dimension sweep at 5 features (inverted U, mirroring Salzburg)**
**Figure 3: Embedding partner similarity vs dim (monotonically decreasing)**

### 6. Discussion (~1.5 pages)

#### 6.1 Unifying theme: the memorization-structure tradeoff
- Both projects show the same pattern: structure only when memorization is expensive
- Salzburg: dimensional bottleneck makes memorization expensive → affine maps emerge
- Graz: dimensional bottleneck + contrastive objective → compositional structure emerges
- In both cases, there's a sweet spot — too little capacity and structure can't be represented, too much and memorization is cheaper

#### 6.2 The affine probe paradox
- In Graz, best generalization (dim=16, 90%) has the *lowest* affine probe accuracy (50%) among functional models
- In Salzburg, dim=64 has 100% object accuracy but 27.5% affine accuracy
- Linear probes measure linear accessibility, not functional utility — structure can be nonlinearly encoded and still support generalization

#### 6.3 Implications for understanding LLMs
- Real LLMs are massively overparameterized → expect memorization as default
- Structured representations may emerge for very frequent patterns (where memorization cost exceeds structure cost)
- Contrastive-like signals in natural data (negation, contrast, questions) may play the role our explicit contrastive sentences play

### 7. Conclusion (~0.5 page)

- Memorization is the path of least resistance for transformers
- Two complementary levers for structure: capacity constraints and objective-level pressure
- The optimal capacity is problem-dependent and non-monotonic
- Implications for training objectives in real systems

---

## Appendices

### A. Graz: Contrastive Learning Ablation (7 features, 441 entities)

Evidence that contrastive learning is a key ingredient, demonstrated at a larger scale:

**A.1 Contrastive vs no-contrastive**
- Phase 3j (no contrastive, dim=32): 50.6% held-out (10.6× chance)
- Phase 3l (with contrastive, dim=64, 62.5K seq): 69.3% held-out (14.6× chance)
- Contrastive forces type-membership encoding; affine probe jumps from 51.8% to 83.0%

**A.2 Contrastive needs sufficient data**
- With 6.25K sequences: contrastive *hurts* generalization (30.7% vs 50.6% baseline)
- Contrastive space is quadratic (441 × 441 × 2 ≈ 390K possible examples)
- With 62.5K sequences (~150K contrastive sentences): near-perfect contrastive accuracy (92%/99%) and 69.3% generalization

**A.3 Exhaustive vs random contrastive sampling**
- Phase 3m: exhaustive enumeration of all positive examples + matched negatives in a pool
- Result: *worse* than random sampling (56.3% vs 69.3% held-out)
- Random sampling provides richer signal: different negatives each epoch, natural frequency variation acts as implicit curriculum

**A.4 More relation data doesn't help**
- Phase 3k: increasing p_relation from 0.33 to 0.50 with 10K sequences
- Result: held-out *decreased* (42.6% vs 50.6% baseline)
- Model memorizes more aggressively with more positive-only relation data

### B. Graz: Phase Progression and Feature Reduction

The path from initial experiments to the 5-feature setup:

**B.1 Phase 2: Pure memorization (10 features, 120 entities)**
- Train/test subject split (80/40): 100% train, 0% test generalization
- Memorization cost: only 175 parameters per fact
- Even when affine probe shows structure in representations, model ignores it

**B.2 Phase 3: Matrix completion improves generalization**
- All subjects appear in relations; 20% of triples held out
- Generalization rises from 0% to 28-35% — partial but still mostly memorization

**B.3 Feature reduction path**

| Features | Types | Entities | dim | Held-out | × chance |
|----------|-------|----------|-----|----------|----------|
| 10, k=3  | 120   | 120      | 32  | 28-35%   | ~35×     |
| 10, k=2  | 45    | 450      | 32  | 31.1%    | 14×      |
| 8, k=2   | 28    | 784      | 32  | 33.2%    | 9.3×     |
| 7, k=2   | 21    | 441      | 32  | 50.6%    | 10.6×    |
| 6, k=2   | 15    | 225      | 64  | 87.8%    | 13.2×    |
| 5, k=2   | 10    | 100      | 16  | 90.0%    | 9.0×     |

- Simpler type space → better generalization, but confounded with other changes (dim, contrastive)
- The trend motivated the 5-feature setup used in the main body

### C. Salzburg: Additional Mechanistic Details

- Full logit lens analysis across layers
- Frozen embedding experiment: embeddings designed for affine maps actually hurt (80% vs 97%)
- Existence test methodology and results

---

## Figures (planned)

### Main body
1. **Schematic**: data generation for both setups (entity structure, sentence types)
2. **Salzburg dimension sweep**: affine accuracy vs dim (U-shaped curve)
3. **Graz dimension sweep**: held-out accuracy vs dim at 5 features (inverted U)
4. **Side-by-side**: both dimension sweeps showing parallel non-monotonic pattern

### Appendix
5. **Graz phase progression**: held-out accuracy across experimental phases
6. **Contrastive data scaling**: generalization vs number of sequences (7 features)

## Tables (planned)

### Main body
1. Salzburg dimension sweep (dim, object acc, affine acc)
2. Graz dimension sweep at 5 features (dim, held-out, contrastive, affine, partner sim)

### Appendix
3. Graz contrastive ablation (with/without, data scaling)
4. Graz feature reduction path (10→5 features)

---

## Key numbers for the main body

### Salzburg
- dim=16 affine: 97%, with aux loss: 100%
- dim=64 affine: 27.5% (memorization)
- Attention affine cosine: 0.987
- Aux loss magnitude: 0.02% of CE loss

### Graz (5 features only)
- dim=16: 90.0% held-out (9× chance) — best
- dim=64: 67.5% held-out (6.75× chance) — memorization
- dim=8: 42.5% held-out — capacity collapse
- All dims: 100% train accuracy, ~89% contrastive positive, ~97% contrastive negative

## Estimated length

- Main body: ~8 pages (single column) or ~5-6 pages (double column)
- Appendices: ~3-4 pages
- Total: ~11-12 pages

## Open questions

1. Which conference format? (ACL, NeurIPS, ICLR, workshop?)
2. How much mechanistic detail in main body vs appendix?
3. Title: emphasize the parallel between the two setups, or lead with the finding?

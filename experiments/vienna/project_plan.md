# Project Plan: Studying the Linear Representation Hypothesis in a Minimal Setup

## 1. Project Overview

**Goal**: Create a minimal toy environment to study the linear representation hypothesis—the idea that language models compute answers by applying an affine transformation to subject representations in the residual stream.

**Hypothesis under investigation**: When a model processes "x a_i", it produces a representation of a_i that, when transformed by an affine function, yields the representation of b_i.

---

## 2. Experimental Setup

### 2.1 Vocabulary Design
- **Total vocabulary size**: 201 tokens
- **Subject tokens**: `a0, a1, ..., a99` (100 tokens)
- **Object tokens**: `b0, b1, ..., b99` (100 tokens)
- **Separator token**: `x` (1 token)

### 2.2 Relation Definition
- Single deterministic relation: `R(a_i) = b_i`
- Each subject a_i maps uniquely to object b_i (bijective mapping)

### 2.3 Corpus Structure
- Sequences of (separator, subject, object) triplets
- Example: `x a5 b5 x a20 b20 x a81 b81 x ...`
- The model learns to predict b_i given the context ending in a_i

---

## 3. Implementation Components

### 3.1 Data Generation Module (`data_generation.py`)

**Responsibilities**:
- Generate vocabulary mappings (token → id, id → token)
- Create training sequences of configurable length
- Create validation/test splits with different random orderings
- Support batch generation for training

**Key functions**:
```python
def create_vocabulary() -> dict
def generate_sequence(length: int, seed: int) -> List[int]
def create_dataloader(num_sequences: int, seq_length: int, batch_size: int) -> DataLoader
```

**Considerations**:
- Ensure uniform sampling of all 100 subject-object pairs
- Consider whether to allow repetitions within a sequence
- Track statistics on pair frequency for debugging

### 3.2 Model Architecture (`model.py`)

**Base model**: Small transformer decoder (GPT-style)

**Configurable hyperparameters**:
- Number of layers: 1-4 (start with 1-2 for interpretability)
- Hidden dimension: 64-256
- Number of attention heads: 2-8
- Context length: 64-256 tokens

**Architecture decisions**:
- Use standard causal attention masking
- Include layer normalization
- No dropout (small dataset, want deterministic behavior for analysis)
- Consider whether to tie input/output embeddings

**Key outputs for analysis**:
- Access to residual stream at each layer
- Access to attention patterns
- Access to MLP activations

### 3.3 Training Module (`training.py`)

**Training objective**: Next-token prediction (causal language modeling)

**Key components**:
- Cross-entropy loss
- AdamW optimizer
- Learning rate scheduler (cosine with warmup)
- Checkpointing for best model

**Training hyperparameters**:
- Learning rate: 1e-3 to 1e-4
- Batch size: 32-128
- Number of epochs: until convergence (likely 50-200)
- Warmup steps: 100-500

**Metrics to track**:
- Overall loss
- Accuracy on predicting b_i given a_i (primary metric)
- Perplexity
- Loss breakdown by token type (separator vs subject vs object)

### 3.4 Representation Extraction (`representations.py`)

**Purpose**: Extract and analyze internal representations

**Key functions**:
```python
def get_residual_stream(model, input_ids, layer: int) -> Tensor
def get_subject_representations(model, layer: int) -> Dict[int, Tensor]
def get_object_representations(model, layer: int) -> Dict[int, Tensor]
```

**Extraction methodology**:
1. For each subject a_i, run the model on context "x a_i"
2. Extract the residual stream at the **a_i position** (last token) — this is the subject representation
3. Repeat for all 100 subjects
4. For object representations, extract the residual stream at the **b_i position** when running "x b_i" — this gives a contextual object representation at the same relative position

**Important clarification — what counts as the "object representation"**:
Three options exist, and the choice changes what the experiment measures:
- **(A) Static embedding**: `embed(b_i)` from the embedding matrix — position-independent, ignores context
- **(B) Contextual residual from "x b_i"**: residual stream at b_i position when b_i is the final token — symmetric with how subject representations are extracted (recommended default)
- **(C) Model state at a_i position when predicting b_i**: the residual at the subject position that directly feeds the next-token prediction — measures whether the model has already "computed" b_i at a_i's position

**Recommendation**: Use option (B) as the primary comparison (symmetric extraction, consistent with Hernandez et al.). Also report option (A) as a baseline to understand what structure exists in the static embeddings before any computation.

### 3.5 Affine Transformation Computation (`affine.py`)

**Goal**: Find the best affine transformation W, b such that:
`W @ repr(a_i) + b ≈ repr(b_i)` for all i

**Method**: Least squares regression
```python
def compute_affine_transformation(
    subject_reprs: Tensor,  # Shape: (100, hidden_dim)
    object_reprs: Tensor    # Shape: (100, hidden_dim)
) -> Tuple[Tensor, Tensor]:  # W: (hidden_dim, hidden_dim), b: (hidden_dim,)
```

**Implementation**:
1. Split the 100 subject-object pairs into a fit set (80 pairs) and eval set (20 pairs) — cross-validate across pairs, not sequences
2. Stack fit-set subject representations into matrix S (80 × d), object representations into O (80 × d)
3. Augment S with ones column for bias: S_aug (80 × (d+1))
4. Solve: [W; b]^T = (S_aug^T @ S_aug)^(-1) @ S_aug^T @ O
5. Evaluate Δcos on the 20 held-out pairs only — this tests generalization, not in-sample fit
6. Use k-fold cross-validation (e.g., k=5) across pairs and report mean ± std Δcos

**Note on W multiplication**: W has shape (hidden_dim, hidden_dim). The reconstruction is computed as `s @ W.T + b` (equivalent to `(Ws + b)^T` in column-vector notation). The `.T` is required because representations are row vectors.

**Variations to explore**:
- Pure linear (W only, no bias)
- Low-rank affine transformations
- Per-layer analysis

### 3.6 Delta Cos Evaluation (`evaluation.py`)

**Delta cosine similarity** is the standard metric for evaluating linearity of relations, as defined by Hernandez et al. (2024).

**Definition**:
```
Δcos = cos(o, ô) - cos(o, s)
```
Where:
- `o` = actual object representation
- `ô` = reconstructed object representation (via affine transformation from subject: ô = Ws + b)
- `s` = subject representation

**Interpretation**:
- Δcos > 0: The affine transformation improves over using the subject directly
- Δcos ≈ 1: Near-perfect linear reconstruction (cos(o, ô) ≈ 1 and cos(o, s) ≈ 0)
- Δcos ≤ 0: The affine transformation does not help

**Implementation**:
```python
def compute_delta_cos(
    subject_reprs: Tensor,  # Shape: (n, hidden_dim)
    object_reprs: Tensor,   # Shape: (n, hidden_dim)
    W: Tensor,              # Shape: (hidden_dim, hidden_dim)
    b: Tensor               # Shape: (hidden_dim,)
) -> float:
    # Reconstruct objects via affine transformation
    object_reconstructed = subject_reprs @ W.T + b  # (n, hidden_dim)

    # Normalize all vectors
    o_norm = F.normalize(object_reprs, dim=1)
    o_hat_norm = F.normalize(object_reconstructed, dim=1)
    s_norm = F.normalize(subject_reprs, dim=1)

    # Compute cosine similarities
    cos_o_ohat = (o_norm * o_hat_norm).sum(dim=1)  # (n,)
    cos_o_s = (o_norm * s_norm).sum(dim=1)          # (n,)

    # Delta cos is the average improvement
    delta_cos = (cos_o_ohat - cos_o_s).mean()
    return delta_cos.item()
```

**Additional metrics**:
- Reconstruction accuracy: How often does W @ repr(a_i) + b closest to repr(b_i)?
- Mean squared error of affine approximation
- Cosine similarity between predicted and actual object representations

---

## 4. Experimental Protocol

### Phase 1: Baseline Training
1. Designate a held-out set of 20 subject-object pairs (e.g., a80–a99 ↔ b80–b99) that never appear in training sequences
2. Generate training data (100k-1M tokens) using only the 80 training pairs
3. Train minimal transformer (1 layer, 64 dim)
4. Verify model achieves >99% accuracy on predicting b_i for the 80 training pairs
5. Save trained model checkpoint

**Rationale for held-out pairs**: Training on a subset of pairs distinguishes genuine linear structure (which would allow the affine map to interpolate/extrapolate) from memorized lookup tables. If the affine transformation learned from training pairs accurately reconstructs held-out object representations, this is strong evidence for systematic linear structure.

### Phase 2: Representation Analysis
1. Extract subject representations from all layers
2. Extract object representations from all layers
3. Visualize representations (PCA, t-SNE)
4. Analyze clustering structure

### Phase 3: Linearity Evaluation
1. Compute delta cos at each layer
2. Identify which layer(s) show highest linearity
3. Plot delta cos vs. layer depth

### Phase 4: Affine Transformation Analysis
1. Compute best-fit affine transformation at each layer
2. Evaluate reconstruction accuracy
3. Analyze properties of W (eigenvalues, rank, condition number)
4. Test: Does W transfer across different initializations?

### Phase 5: Ablation Studies
1. **Model size**: How does linearity change with model capacity?
2. **Training duration**: When does linearity emerge during training?
3. **Data size**: Minimum data needed for linear representations?
4. **Multiple relations**: Add R2(a_i) = b_{(i+50) mod 100} — does linearity hold?

---

## 5. Project Structure

```
vienna/
├── project_plan.md          # This document
├── src/
│   ├── __init__.py
│   ├── config.py            # Hyperparameters and settings
│   ├── data_generation.py   # Synthetic data creation
│   ├── model.py             # Transformer architecture
│   ├── training.py          # Training loop
│   ├── representations.py   # Representation extraction
│   ├── affine.py            # Affine transformation computation
│   └── evaluation.py        # Delta cos and other metrics
├── experiments/
│   ├── train_baseline.py    # Script to train baseline model
│   ├── analyze_linearity.py # Script to run linearity analysis
│   └── ablations.py         # Ablation study scripts
├── notebooks/
│   └── visualization.ipynb  # Interactive analysis and plots
├── checkpoints/             # Saved model weights
├── results/                 # Experiment outputs and figures
└── requirements.txt         # Dependencies
```

---

## 6. References

- Hernandez, E., Sharma, A. S., Haklay, T., Meng, K., Wattenberg, M., Belinkov, Y., & Bau, D. (2024). *Linearity of Relation Decoding in Transformer Language Models*. ICLR 2024.

---

## 7. Dependencies

```
torch>=2.0
numpy
matplotlib
seaborn
scikit-learn  # For PCA, t-SNE
tqdm
wandb         # Optional: experiment tracking
jupyter       # For notebooks
```

---

## 9. Success Criteria

1. **Training success**: Model achieves >99% accuracy on predicting b_i
2. **Linearity detected**: Delta cos > 0.8 at some layer (on held-out pairs; ambitious but feasible in controlled setting)
3. **Affine reconstruction**: >95% accuracy using learned affine transformation
4. **Interpretable results**: Clear visualization of linear structure

---

## 10. Extensions and Future Directions

### 8.1 Multiple Relations
- Add a second relation (e.g., R2 where objects are shifted)
- Study whether different relations have different linear directions
- Investigate relation composition

### 8.2 Non-Bijective Mappings
- Multiple subjects mapping to same object
- Partial mappings (some subjects have no object)

### 8.3 Comparison with Real LLMs
- Extract relation representations from pretrained models
- Compare linearity metrics with this controlled setup

### 8.4 Causal Interventions
- Activation patching: Replace a_i representation with affine-transformed version
- Verify causal role of linear structure

---

## 11. Potential Challenges and Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Model memorizes rather than learns linear structure | Use larger vocab, fewer training examples per pair |
| Representations not separable | Increase model capacity, training time |
| Delta cos metric not discriminative | Implement multiple linearity metrics |
| Affine transformation overfits | Use cross-validation, test on held-out pairs |

---

## 12. Open Questions

1. At which layer does linear structure emerge most clearly?
2. Is the transformation truly affine, or is linear (no bias) sufficient?
3. How does the rank of the learned transformation relate to the task?
4. Does the model learn the same transformation across different random seeds?
5. What is the minimum model capacity required for linear representations?

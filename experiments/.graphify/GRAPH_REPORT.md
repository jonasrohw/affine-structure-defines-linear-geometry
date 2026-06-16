# Graph Report - .  (2026-06-16)

## Corpus Check
- 117 files · ~125,768 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1052 nodes · 2673 edges · 107 communities detected
- Extraction: 69% EXTRACTED · 31% INFERRED · 0% AMBIGUOUS · INFERRED: 834 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output
- Edge kinds: uses: 834 · contains: 524 · calls: 481 · imports_from: 407 · rationale_for: 311 · method: 114 · inherits: 2


## Input Scope
- Requested: all
- Resolved: all (source: cli)
- Included files: 117 · Candidates: recursive
- Excluded: 0 untracked · 0 ignored · 0 sensitive · 0 missing committed

## Graph Freshness
- Built from Git commit: `952594f`
- Compare this hash to `git rev-parse HEAD` before trusting freshness-sensitive graph output.
## God Nodes (most connected - your core abstractions)
1. `VocabConfig` - 239 edges
2. `ModelConfig` - 215 edges
3. `Config` - 196 edges
4. `TrainConfig` - 139 edges
5. `MiniTransformer` - 60 edges
6. `RavfogelVocab` - 56 edges
7. `RavfogelVocabDummy` - 19 edges
8. `SequenceDataset` - 16 edges
9. `ControlSet` - 15 edges
10. `GeGLU_MLP` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Ravfogel-style bijective factual recall with 1-layer standard transformer + dumm` --uses--> `ModelConfig`  [INFERRED]
  graz/experiments/ravfogel_1layer_dummy.py → vienna/src/config.py
- `Evaluate accuracy at dummy position: [sep, r_k, subj_i, dummy] → predict obj.` --uses--> `ModelConfig`  [INFERRED]
  graz/experiments/ravfogel_1layer_dummy.py → vienna/src/config.py
- `Extract residual stream at dummy position (position 3).` --uses--> `ModelConfig`  [INFERRED]
  graz/experiments/ravfogel_1layer_dummy.py → vienna/src/config.py
- `Per-relation affine fit of MLP input → MLP output at dummy position.` --uses--> `ModelConfig`  [INFERRED]
  graz/experiments/ravfogel_1layer_dummy.py → vienna/src/config.py
- `Ravfogel-style bijective factual recall with 1-layer standard transformer.  Setu` --uses--> `ModelConfig`  [INFERRED]
  graz/experiments/ravfogel_1layer.py → vienna/src/config.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.10
Nodes (21): affine_analysis(), build_sequences(), eval_relation_accuracy(), extract_hidden_states(), generate_permutations(), main(), mlp_affine_analysis(), Ravfogel-style bijective factual recall with 1-layer standard transformer + dumm (+13 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (2): make_combos(), VocabConfig

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (20): build_phase1_sequences(), ControlSet, eval_relation_accuracy_ctrl(), eval_relation_accuracy_orig(), ExtendedVocabConfig, flip_attr(), generate_attr_sentence_ctrl(), generate_attr_sentence_orig() (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (21): cosine_similarity_mean(), decode_ohat_to_logits(), delta_cos(), extract_embeddings(), full_report(), mse(), pairwise_similarity_by_overlap(), Evaluation metrics for the Vienna experiment.  Primary metric: Delta cosine simi (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (25): build_pooled_object_matrix(), build_relation_object_matrix(), _extract_sublayer_at_pos(), get_correct_object_ids(), get_object_sublayer_outputs(), get_static_object_embeddings(), get_subject_sublayer_outputs(), Representation extraction from a trained MiniTransformer.  For each subject a_i, (+17 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (24): attention_weight_dependence(), effective_linear_map(), extract_attention_patterns(), head_specialization(), load_model_and_vocab(), main(), ov_circuit_analysis(), Analyze how the 1-layer attention-only (no MLP, no dummy) model implements relat (+16 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (24): affine_analysis(), affine_analysis_all_layers(), affine_analysis_centroid(), _analyze_one_layer(), apply_affine(), build_type_centroids(), extract_subject_hidden(), fit_affine() (+16 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (18): build_model(), MiniTransformer, Small GPT-style decoder-only transformer.      Attributes:         residuals: Wh, Small GPT-style decoder-only transformer.      Attributes:         residuals: Wh, get_post_lnf_object_representations(), get_unembedding_object_representations(), Object representations after applying ln_f to the residual stream.      This is, Object representations from the unembedding (lm_head) matrix.      Returns the s (+10 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (20): apply_lnf_to_filler_reprs(), apply_lnf_to_reprs(), build_pooled_object_matrix(), build_relation_object_matrix(), delta_cos_decomposed(), fmt3(), kfold_decomposed(), main() (+12 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (20): analysis_1_attention_decomposition(), analysis_2_neuron_decomposition(), analysis_3_input_decomposition(), analysis_4_activation_histograms(), analysis_5_weighted_sum(), analysis_6_affine_fit(), extract_all_states(), extract_per_head_attn() (+12 more)

### Community 10 - "Community 10"
Cohesion: 0.18
Nodes (17): affine_analysis_subset(), build_epoch_sequences(), enumerate_facts(), eval_attr_accuracy(), eval_relation_accuracy(), _find_matching(), main(), make_held_out_triples() (+9 more)

### Community 11 - "Community 11"
Cohesion: 0.21
Nodes (16): build_type_centroids(), cosine_similarity(), extract_all_residual_points(), fit_affine(), get_unique_centroids(), main(), precompute_relation_objects(), Residual stream affine analysis for v7 checkpoint.  At each point in the residua (+8 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (2): All valid objects for this (subject, relation) pair., Single object for this (subject, relation) — first valid target.

### Community 13 - "Community 13"
Cohesion: 0.21
Nodes (14): generate_attr_sentence(), generate_contrastive_sentence(), generate_mixed_sequence(), generate_mixed_sequence_with_exhaustive(), generate_rel_sentence(), generate_reverse_sentence(), generate_reverse_short_sentence(), generate_sequence() (+6 more)

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (9): LinearProbe, load_model(), main(), MLPProbe, Embedding probe: measure how much relational structure is in the learned embeddi, run_probes(), train_probe(), GeGLU_MLP (+1 more)

### Community 15 - "Community 15"
Cohesion: 0.16
Nodes (7): RavfogelVocab, compute_epoch_metrics(), extract_mechanism_states(), main(), Trace the time course of mechanism development in Ravfogel bijective training., Extract attention outputs, MLP pre/post-GELU at subject position for all pairs., Compute all mechanism metrics for the current model state.

### Community 16 - "Community 16"
Cohesion: 0.21
Nodes (12): build_phase1_sequences(), flip_attr(), generate_attr_sentence_noisy(), generate_reverse_short_noisy(), main(), make_three_way_split(), precompute_relation_objects(), Noisy attribute experiment: does corrupting the attribute signal degrade learnin (+4 more)

### Community 17 - "Community 17"
Cohesion: 0.19
Nodes (12): build_phase1_sequences(), build_phase2_sequences(), enumerate_facts(), main(), make_three_way_split(), precompute_relation_objects(), One-epoch v7 — Three-way split: train / single-exposure / held-out.  Phase 1: Tr, Sequences with randomly sampled train relation facts + attributes. (+4 more)

### Community 18 - "Community 18"
Cohesion: 0.21
Nodes (14): affine_analysis(), build_frozen_embeddings(), eval_relation_accuracy(), extract_hidden_states(), inject_frozen_embeddings(), main(), make_grad_mask_hook(), mechanism_analysis() (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.21
Nodes (14): cluster_analysis(), extract_all_states(), fit_affine_r2(), load_model(), main(), patch_activation(), Cluster analysis for the dim=64 nonlinear demixing mechanism.  Question: does th, Fit Y = X @ W + b via least squares, return R². (+6 more)

### Community 20 - "Community 20"
Cohesion: 0.13
Nodes (15): _extract_repr_at_pos(), get_all_object_representations(), get_all_subject_representations(), get_filler_representations(), get_object_representations(), get_relation_position_representations(), get_subject_representations(), Extract object representations from artificial context [x, b_j] at position 1. (+7 more)

### Community 21 - "Community 21"
Cohesion: 0.22
Nodes (13): build_epoch_sequences(), enumerate_facts(), eval_attr_accuracy(), eval_relation_accuracy(), main(), make_held_out_triples(), precompute_relation_objects(), One-epoch v3 — Large-scale single-exposure relation learning.  C(10,2) = 45 type (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.30
Nodes (8): affine_analysis(), build_sequences(), eval_relation_accuracy(), extract_hidden_states(), generate_permutations(), main(), Ravfogel-style bijective factual recall with 1-layer standard transformer.  Setu, train_epoch()

### Community 23 - "Community 23"
Cohesion: 0.23
Nodes (12): analysis_decomposition(), analysis_relation_specificity(), analysis_selectivity(), build_type_centroids(), extract_mlp_internals(), main(), precompute_relation_objects(), Neuron-level decomposition of MLP contribution to type-centroid mapping.  Two an (+4 more)

### Community 24 - "Community 24"
Cohesion: 0.23
Nodes (10): affine_analysis_dummy(), build_sequences(), extract_dummy_position_hidden(), main(), make_three_way_split(), precompute_relation_objects(), 2-layer attention-only model with trailing dummy token.  Sequence format: [sep,, Extract hidden states at the dummy position (pos 3) for affine analysis. (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.26
Nodes (12): build_epoch_sequences(), enumerate_facts(), eval_attr_accuracy(), eval_relation_accuracy(), main(), make_held_out_triples(), precompute_relation_objects(), One-epoch v5 — Curriculum training with warmup.  Phase 1 (warmup, epochs 1-2): a (+4 more)

### Community 26 - "Community 26"
Cohesion: 0.23
Nodes (10): build_phase1_sequences(), generate_attr_sentence_filtered(), generate_reverse_short_filtered(), main(), make_three_way_split(), precompute_relation_objects(), Attribute removal experiment: does reducing semantic connectivity degrade learni, Like generate_attr_sentence but never produces the removed attribute. (+2 more)

### Community 27 - "Community 27"
Cohesion: 0.24
Nodes (11): build_pooled_object_matrix(), build_relation_object_matrix(), main(), parse_args(), pca_effective_dim(), Linearity probes at filler positions (Experiment 3 from post_v2_analysis_plan.md, Build (n_subjects * num_relations, d) object matrix matching the pooled f2 order, Build (n_subjects, d) object matrix for a single relation. (+3 more)

### Community 28 - "Community 28"
Cohesion: 0.30
Nodes (11): analysis_1_attention(), analysis_2_neurons(), analysis_3_input(), analysis_4_activations(), analysis_5_weighted_sum(), analysis_6_affine_fit(), extract_all_states(), extract_per_head_attn() (+3 more)

### Community 29 - "Community 29"
Cohesion: 0.29
Nodes (8): affine_analysis_dummy(), build_sequences(), extract_dummy_position_hidden(), main(), make_three_way_split(), precompute_relation_objects(), 1-layer attention-only model with trailing dummy token.  Sequence format: [sep,, train_epoch()

### Community 30 - "Community 30"
Cohesion: 0.31
Nodes (10): affine_accuracy_per_relation(), delta_cos_decomposed(), fmt3(), kfold_decomposed(), logit_lens(), main(), make_cfg(), Run the full experiment pipeline with hidden_dim=32.  With 100 subjects in 32 di (+2 more)

### Community 31 - "Community 31"
Cohesion: 0.22
Nodes (8): evaluate_at_stages(), main(), Logit lens: apply ln_f + lm_head to intermediate residual streams.  Tests whethe, Evaluate object prediction accuracy at multiple residual stream stages.      Sta, main(), Phase 2 dimension sweep: test whether lower dimensionality forces generalization, run_one(), Minimal GPT-style causal transformer for the Vienna experiment.  The model expos

### Community 32 - "Community 32"
Cohesion: 0.33
Nodes (10): build_epoch_sequences(), enumerate_facts(), eval_attr_accuracy(), eval_relation_accuracy(), main(), make_held_out_triples(), precompute_relation_objects(), One-epoch v6 — Graduated curriculum: pilot scaffold + single-exposure main.  Pha (+2 more)

### Community 33 - "Community 33"
Cohesion: 0.20
Nodes (8): Dataset, Add a second relation R2(a_i) = b_{(i+50) mod 100} and check if linearity still, study_relations(), make_dataloader(), Dataset of fixed-length token sequences., Dataset of fixed-length token sequences., Create a DataLoader for the given split.      Args:         cfg: Full experiment, SequenceDataset

### Community 34 - "Community 34"
Cohesion: 0.36
Nodes (9): main(), parse_args(), Ablation studies for the Vienna experiment.  Usage (from vienna/):     python ex, Train a model and return best-layer CV delta_cos on training and heldout pairs., run_experiment(), _save_ablation_plot(), study_data(), study_depth() (+1 more)

### Community 35 - "Community 35"
Cohesion: 0.22
Nodes (7): affine_accuracy_per_relation(), main(), Accuracy of affine-predicted representations when fed to lm_head.  Question: if, K-fold CV: fit affine S→target, apply lm_head, measure object accuracy., evaluate_relations(), Training loop for the Vienna experiment., For each subject and relation, present [x, r_k, a_i] and check object prediction

### Community 36 - "Community 36"
Cohesion: 0.22
Nodes (8): main(), parse_args(), Linearity analysis on a trained model.  Usage (from vienna/):     python experim, cross_relation_eval(), Affine transformation fitting between subject and object representations.  Conve, Test whether the space of per-relation weight matrices is linearly structured., Apply a single affine map (trained on one relation) to other relations.      Arg, second_order_test()

### Community 37 - "Community 37"
Cohesion: 0.36
Nodes (9): affine_accuracy_per_relation(), delta_cos_decomposed(), fmt3(), kfold_decomposed(), logit_lens(), main(), make_cfg(), Run the full experiment pipeline with hidden_dim=32 and 200 subjects.  With 3x20 (+1 more)

### Community 38 - "Community 38"
Cohesion: 0.36
Nodes (9): affine_accuracy_per_relation(), delta_cos_decomposed(), fmt3(), kfold_decomposed(), logit_lens(), main(), make_cfg(), Run the full experiment pipeline with hidden_dim=32 and 2 relations.  Matches th (+1 more)

### Community 39 - "Community 39"
Cohesion: 0.36
Nodes (9): affine_accuracy_per_relation(), delta_cos_decomposed(), fmt3(), kfold_decomposed(), logit_lens(), main(), make_cfg(), Run the full experiment pipeline with hidden_dim=32 and 400 subjects.  With 3x40 (+1 more)

### Community 40 - "Community 40"
Cohesion: 0.36
Nodes (9): extract_representations(), fit_affine_r2(), load_model(), main(), main_v2(), patch_activation(), End-to-end affine R² check: subject embedding → final hidden state.  For each re, Better framing: the model's job is subject_emb → W_u^T object_emb,     where W_u (+1 more)

### Community 41 - "Community 41"
Cohesion: 0.36
Nodes (9): affine_accuracy_per_relation(), kfold_cos(), logit_lens(), main(), make_cfg(), make_unit_vectors(), Frozen-embedding experiment: dim=16, 2 relations, 100 subjects.  Tests option (B, run_existence_test() (+1 more)

### Community 42 - "Community 42"
Cohesion: 0.24
Nodes (7): main(), make_cfg(), Phase 1: Train on attribute associations, evaluate embedding structure.  No expl, main(), make_cfg(), Phase 2: Train on attribute associations + relation triples, test generalization, Centralized configuration for the Vienna linear representation hypothesis experi

### Community 43 - "Community 43"
Cohesion: 0.33
Nodes (6): build_sequences(), main(), make_three_way_split(), precompute_relation_objects(), 1-layer attention-only model (no MLP, no dummy token).  Same data and evaluation, train_epoch()

### Community 44 - "Community 44"
Cohesion: 0.33
Nodes (6): build_phase1_sequences(), main(), make_three_way_split(), precompute_relation_objects(), 2-layer attention-only model (no MLP) — can attention alone implement affine map, train_epoch()

### Community 45 - "Community 45"
Cohesion: 0.36
Nodes (8): build_exhaustive_relation_sequences(), eval_attr_accuracy(), eval_relation_accuracy(), main(), make_held_out_triples(), One-epoch relation learning — Phase 2: relation exposure from phase 1 checkpoint, Pack every train triple exactly once into sequences, interleaved with attribute, run_phase2()

### Community 46 - "Community 46"
Cohesion: 0.36
Nodes (8): eval_attr_accuracy(), eval_relation_accuracy(), main(), make_held_out_triples(), One-epoch v2 — Phase 1: attribute pretraining with k relation triples.  Trains o, Select k triples balanced across relations., run_one_k(), select_k_triples()

### Community 47 - "Community 47"
Cohesion: 0.33
Nodes (6): build_phase1_sequences(), main(), make_three_way_split(), precompute_relation_objects(), Frozen fc2 experiment: can the network learn the type-centroid affine transforma, train_epoch()

### Community 48 - "Community 48"
Cohesion: 0.31
Nodes (8): eval_contrastive_accuracy(), eval_relation_accuracy(), main(), make_held_out_triples(), Phase 3o3_compute: compute scaling — track generalization over training.  Same s, Hold out a random fraction of triples, ensuring every subject keeps ≥1., Evaluate relation accuracy on a set of (subject_idx, rel_k) triples., Evaluate contrastive accuracy (positive and negative).

### Community 49 - "Community 49"
Cohesion: 0.36
Nodes (8): cluster_analysis(), extract_geglu_states(), fit_affine_r2(), main(), patch_mlp_geglu(), GeGLU experiment: does gate/value separation enable cleaner relation gating?  Ge, Extract MLP inputs, gate activations, value activations, outputs., Replace standard MLP with GeGLU MLP in all blocks.

### Community 50 - "Community 50"
Cohesion: 0.22
Nodes (3): CausalSelfAttention, MLP, TransformerBlock

### Community 51 - "Community 51"
Cohesion: 0.36
Nodes (7): analyze_relation_signal(), extract_attention_decomposition(), main(), precompute_relation_objects(), Per-head attention output analysis for v7 checkpoint.  Decomposes attn_out at th, Analyze how attention encodes relation identity., Extract per-head attention outputs at position 2 for all (subject, relation) pai

### Community 52 - "Community 52"
Cohesion: 0.43
Nodes (7): affine_accuracy_per_relation(), kfold_cos(), logit_lens(), main(), make_cfg(), Auxiliary affine loss experiment: dim=16, 2 relations, 100 subjects.  Tests opti, train_with_aux_loss()

### Community 53 - "Community 53"
Cohesion: 0.38
Nodes (6): extract_mlp_io(), fit_affine(), main(), Affine approximation of the mapping from MLP input to post-GELU activations.  Fo, Extract MLP input and post-GELU activations for all (entity, relation) pairs., Fit Y ≈ X @ A + b via least squares. X: (n, d_in), Y: (n, d_out).

### Community 54 - "Community 54"
Cohesion: 0.43
Nodes (6): build_type_centroids(), compute_per_subject_contributions(), main(), precompute_relation_objects(), Per-subject neuron contribution analysis.  Question: does the MLP use the same c, Compute per-neuron alignment contribution for each (subject, relation).      Ret

### Community 55 - "Community 55"
Cohesion: 0.38
Nodes (6): eval_attr_precision_at_k(), eval_entity_precision_at_k(), main(), Top-k coverage analysis for attribute and entity prediction.  Two metrics: 1. At, For each entity, are all k valid f-attrs in the top k? Same for g-attrs., For each attribute, are all 405 valid entities in the top 405 predictions?

### Community 56 - "Community 56"
Cohesion: 0.43
Nodes (6): build_type_centroids(), extract_all(), main(), precompute_relation_objects(), Structure of the MLP weighted sum: how does the network choose activations to pr, Extract activations and residual stream states for all (subject, relation) pairs

### Community 57 - "Community 57"
Cohesion: 0.43
Nodes (6): build_phase2_sequences(), eval_attr_accuracy(), eval_relation_accuracy(), main(), One-epoch v2 — Phase 2: single-epoch relation learning from phase 1 checkpoint., Pack every remaining triple exactly once, with k triples and attrs interleaved.

### Community 58 - "Community 58"
Cohesion: 0.29
Nodes (3): Context manager: fills self._capture with residual tensors during forward., Context manager: fills self._capture with residual tensors during forward., _ResidualCapture

### Community 59 - "Community 59"
Cohesion: 0.29
Nodes (6): create_left_panel(), create_middle_panel(), create_right_panel(), Three parallel equal-length vectors: verb → gerund., Three vectors of different lengths, randomly distributed, different directions., Three parallel equal-length vectors with clouds and x marks.

### Community 60 - "Community 60"
Cohesion: 0.29
Nodes (6): create_left_panel(), create_middle_panel(), create_right_panel(), Three parallel equal-length vectors: verb → gerund., Three vectors of different lengths, randomly distributed, different directions., Three parallel equal-length vectors with clouds and x marks.

### Community 61 - "Community 61"
Cohesion: 0.53
Nodes (5): evaluate(), main(), make_unit_vectors(), optimize(), Decoupled affine existence test.  Fix well-separated object embeddings (random u

### Community 62 - "Community 62"
Cohesion: 0.47
Nodes (5): extract_components(), main(), precompute_relation_objects(), Per-neuron input decomposition: relation component vs subject component.  For ea, Extract h0, attn_out, LN(h0+attn_out), and pre-GELU values for all contexts.

### Community 63 - "Community 63"
Cohesion: 0.47
Nodes (5): cosine_pre_post_ln(), main(), parse_args(), Analyze the effect of LayerNorm on residual stream vectors.  For each f2 and sub, Compute cosine similarity between vectors and their LN'd versions.      Args:

### Community 64 - "Community 64"
Cohesion: 0.47
Nodes (5): build_object_matrix_for_relation(), main(), parse_args(), Linearity analysis for the v2 model (three relations + filler tokens).  Usage (f, Build the (n, d) object matrix for a given relation at a given layer.      For e

### Community 65 - "Community 65"
Cohesion: 0.47
Nodes (5): eval_attr_accuracy(), main(), make_held_out_triples(), One-epoch relation learning — Phase 1: attribute-only pretraining.  Train on for, Evaluate attribute prediction accuracy.

### Community 66 - "Community 66"
Cohesion: 0.53
Nodes (5): eval_relation_accuracy(), main(), make_held_out_triples(), Phase 3o3_compute_seeds: stability test — 5 seeds × 300 epochs.  Same setup as p, run_seed()

### Community 67 - "Community 67"
Cohesion: 0.47
Nodes (5): extract_states(), load_checkpoint(), main(), Follow-up diagnostics: relationship between MLP computation and affine structure, Extract h_pre_mlp, mlp_out, h_final at subject position for all (i, k).

### Community 68 - "Community 68"
Cohesion: 0.33
Nodes (2): Context manager: captures per-block attention and MLP outputs.          Usage:, _SublayerCapture

### Community 69 - "Community 69"
Cohesion: 0.60
Nodes (4): extract_mlp_activations(), main(), precompute_relation_objects(), Histogram of MLP activations at relation-specific neurons for each type.  Identi

### Community 70 - "Community 70"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3b: Matrix completion + reverse lookup sentences.  Same as Phase 3 (all su, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 71 - "Community 71"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3c: Matrix completion + short reverse lookup (S/O markers).  Four auxiliar, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 72 - "Community 72"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3d: Doubled entities (240 subjects, 240 objects) + short reverse.  Key cha, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 73 - "Community 73"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3e: Quadrupled entities (480 subjects, 480 objects) + short reverse.  Key, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 74 - "Community 74"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3: Matrix completion with dense relational exposure.  All 120 subjects app, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 75 - "Community 75"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3f: 8× entities (960 subjects, 960 objects) + short reverse.  Key change f, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 76 - "Community 76"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3g: k_per_group=2 (C(10,2)=45 types) with 450 entities (10 per type).  Key, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 77 - "Community 77"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3h: Symmetric R0/R1 structure (both scattered targets).  Key change from 3, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 78 - "Community 78"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3i: Reduced feature count (8 features, k=2 → C(8,2)=28 types).  Key change, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 79 - "Community 79"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3j: Further reduced feature count (7 features, k=2 → C(7,2)=21 types).  Ke, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 80 - "Community 80"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3k: Same as 3j but with more relational training signal.  Key changes from, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 81 - "Community 81"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3l: Contrastive learning sentences added to phase3j setup.  Key change fro, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 82 - "Community 82"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3m: Exhaustive contrastive learning.  Key change from 3l: instead of rando, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 83 - "Community 83"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3n: Reduce to 6 features for faster iteration.  C(6,2) = 15 types, 15² = 2, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 84 - "Community 84"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3o2: 5 features with dim=32 (vs dim=64 in phase3o).  C(5,2) = 10 types, 10, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 85 - "Community 85"
Cohesion: 0.60
Nodes (4): eval_relation_accuracy(), make_held_out_triples(), Phase 3o3 with 6 features, no contrastive, per-epoch affine probe.  6 features,, run_seed()

### Community 86 - "Community 86"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3o3_k20: data-scaling pilot — 20 subjects (vs 100 in phase3o3).  Same arch, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 87 - "Community 87"
Cohesion: 0.60
Nodes (4): eval_relation_accuracy(), make_held_out_triples(), Phase 3o3 no-contrastive ablation with per-epoch affine probe.  Tracks held-out, run_seed()

### Community 88 - "Community 88"
Cohesion: 0.60
Nodes (4): eval_relation_accuracy(), make_held_out_triples(), Phase 3o3_no_contrastive_seeds: no contrastive ablation, 5 seeds × 300 epochs., run_seed()

### Community 89 - "Community 89"
Cohesion: 0.60
Nodes (4): eval_relation_accuracy(), main(), make_held_out_triples(), Phase 3o3_no_contrastive: ablation — no contrastive sentences, 100 epochs.  Same

### Community 90 - "Community 90"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3o3: 5 features with dim=16.  C(5,2) = 10 types, 10² = 100 entities. Shift, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 91 - "Community 91"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3o4: 5 features with dim=8.  C(5,2) = 10 types, 10² = 100 entities. Shifts, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 92 - "Community 92"
Cohesion: 0.50
Nodes (4): main(), make_held_out_triples(), Phase 3o: Reduce to 5 features.  C(5,2) = 10 types, 10² = 100 entities. Same app, Hold out a random fraction of triples, ensuring every subject keeps ≥1.

### Community 93 - "Community 93"
Cohesion: 0.60
Nodes (4): extract_states(), load_checkpoint(), main(), Test whether the affine structure is explained by LayerNorm being approximately

### Community 94 - "Community 94"
Cohesion: 0.50
Nodes (4): eval_accuracy_on_sequences(), main(), Sanity check: does removing ln_f at inference change object prediction accuracy?, Evaluate object prediction accuracy on generated sequences.      Returns (total_

### Community 95 - "Community 95"
Cohesion: 0.67
Nodes (3): load_checkpoint(), main(), Check the relationship between learned subject embeddings and target object embe

### Community 96 - "Community 96"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Train the baseline model.  Usage (from vienna/):     python experiments/train_ba

### Community 97 - "Community 97"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Train the v2 model (three relations + filler tokens).  Usage (from vienna/):

### Community 98 - "Community 98"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Train the Salzburg model (no fillers, relation before subject).  Sequence format

### Community 99 - "Community 99"
Cohesion: 0.50
Nodes (4): Return boolean masks for each token type.      Args:         tokens: (batch, seq, Return boolean masks for each token type.      Returns: (sep_mask, entity_mask,, Return boolean masks for each token type (no fillers).      Returns:         (se, token_type_mask()

### Community 100 - "Community 100"
Cohesion: 0.67
Nodes (1): Visualize LayerNorm in 2D: 3x3 grid showing input vectors and their normalized v

### Community 101 - "Community 101"
Cohesion: 0.67
Nodes (1): Visualize LayerNorm in 3D: 3x3 grid showing input vectors and their normalized v

### Community 102 - "Community 102"
Cohesion: 0.67
Nodes (2): make_r2_perm(), Generate a deterministic random permutation of size n.

### Community 103 - "Community 103"
Cohesion: 0.67
Nodes (2): Return the object index for subject_idx under relation relation_idx.          R0, Return the object index for subject_idx under relation relation_idx.

### Community 104 - "Community 104"
Cohesion: 0.67
Nodes (2): Args:             input_ids: (B, T) long tensor of token ids.          Returns:, Args:             input_ids: (B, T) long tensor of token ids.          Returns:

### Community 105 - "Community 105"
Cohesion: 1.00
Nodes (1): Generate all figures for the paper.

### Community 106 - "Community 106"
Cohesion: 1.00
Nodes (2): generate_exhaustive_contrastive(), Generate all positive contrastive sentences exhaustively as a flat list.      Fo

## Knowledge Gaps
- **52 isolated node(s):** `Centralized configuration for the Vienna linear representation hypothesis experi`, `All valid objects for this (subject, relation) pair.`, `Single object for this (subject, relation) — first valid target.`, `Generate all figures for the paper.`, `Accuracy of affine-predicted representations when fed to lm_head.  Question: if` (+47 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 1`** (2 nodes): `make_combos()`, `VocabConfig`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (2 nodes): `All valid objects for this (subject, relation) pair.`, `Single object for this (subject, relation) — first valid target.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (2 nodes): `Context manager: captures per-block attention and MLP outputs.          Usage:`, `_SublayerCapture`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 100`** (1 nodes): `Visualize LayerNorm in 2D: 3x3 grid showing input vectors and their normalized v`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 101`** (1 nodes): `Visualize LayerNorm in 3D: 3x3 grid showing input vectors and their normalized v`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 102`** (2 nodes): `make_r2_perm()`, `Generate a deterministic random permutation of size n.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 103`** (2 nodes): `Return the object index for subject_idx under relation relation_idx.          R0`, `Return the object index for subject_idx under relation relation_idx.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 104`** (2 nodes): `Args:             input_ids: (B, T) long tensor of token ids.          Returns:`, `Args:             input_ids: (B, T) long tensor of token ids.          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 105`** (1 nodes): `Generate all figures for the paper.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 106`** (2 nodes): `generate_exhaustive_contrastive()`, `Generate all positive contrastive sentences exhaustively as a flat list.      Fo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModelConfig` connect `Community 9` to `Community 34`, `Community 33`, `Community 53`, `Community 51`, `Community 23`, `Community 69`, `Community 62`, `Community 54`, `Community 11`, `Community 55`, `Community 56`, `Community 5`, `Community 29`, `Community 43`, `Community 24`, `Community 44`, `Community 52`, `Community 37`, `Community 38`, `Community 39`, `Community 30`, `Community 40`, `Community 14`, `Community 41`, `Community 65`, `Community 45`, `Community 46`, `Community 57`, `Community 21`, `Community 10`, `Community 25`, `Community 32`, `Community 47`, `Community 16`, `Community 2`, `Community 17`, `Community 26`, `Community 42`, `Community 31`, `Community 74`, `Community 70`, `Community 71`, `Community 72`, `Community 73`, `Community 75`, `Community 76`, `Community 77`, `Community 78`, `Community 79`, `Community 80`, `Community 81`, `Community 82`, `Community 83`, `Community 84`, `Community 85`, `Community 48`, `Community 66`, `Community 86`, `Community 87`, `Community 89`, `Community 88`, `Community 90`, `Community 91`, `Community 92`, `Community 0`, `Community 95`, `Community 18`, `Community 28`, `Community 93`, `Community 22`, `Community 15`, `Community 67`, `Community 19`, `Community 49`, `Community 96`, `Community 98`, `Community 97`, `Community 50`, `Community 7`, `Community 104`, `Community 58`, `Community 68`?**
  _High betweenness centrality (0.225) - this node is a cross-community bridge._
- **Why does `VocabConfig` connect `Community 1` to `Community 34`, `Community 33`, `Community 53`, `Community 51`, `Community 23`, `Community 69`, `Community 62`, `Community 54`, `Community 11`, `Community 55`, `Community 56`, `Community 5`, `Community 29`, `Community 43`, `Community 24`, `Community 44`, `Community 52`, `Community 37`, `Community 38`, `Community 39`, `Community 30`, `Community 41`, `Community 65`, `Community 45`, `Community 46`, `Community 57`, `Community 21`, `Community 10`, `Community 25`, `Community 32`, `Community 47`, `Community 16`, `Community 2`, `Community 17`, `Community 26`, `Community 42`, `Community 31`, `Community 74`, `Community 70`, `Community 71`, `Community 72`, `Community 73`, `Community 75`, `Community 76`, `Community 77`, `Community 78`, `Community 79`, `Community 80`, `Community 81`, `Community 82`, `Community 83`, `Community 84`, `Community 85`, `Community 48`, `Community 66`, `Community 86`, `Community 87`, `Community 89`, `Community 88`, `Community 90`, `Community 91`, `Community 92`, `Community 96`, `Community 98`, `Community 97`, `Community 36`, `Community 6`, `Community 103`, `Community 12`, `Community 102`, `Community 13`, `Community 99`, `Community 106`, `Community 3`, `Community 50`, `Community 7`, `Community 104`, `Community 58`, `Community 68`?**
  _High betweenness centrality (0.194) - this node is a cross-community bridge._
- **Why does `Config` connect `Community 11` to `Community 34`, `Community 33`, `Community 23`, `Community 55`, `Community 36`, `Community 64`, `Community 5`, `Community 29`, `Community 43`, `Community 24`, `Community 44`, `Community 52`, `Community 37`, `Community 38`, `Community 39`, `Community 30`, `Community 41`, `Community 65`, `Community 45`, `Community 46`, `Community 57`, `Community 21`, `Community 10`, `Community 25`, `Community 32`, `Community 47`, `Community 16`, `Community 2`, `Community 17`, `Community 26`, `Community 42`, `Community 31`, `Community 74`, `Community 70`, `Community 71`, `Community 72`, `Community 73`, `Community 75`, `Community 76`, `Community 77`, `Community 78`, `Community 79`, `Community 80`, `Community 81`, `Community 82`, `Community 83`, `Community 84`, `Community 85`, `Community 48`, `Community 66`, `Community 86`, `Community 87`, `Community 89`, `Community 88`, `Community 90`, `Community 91`, `Community 92`, `Community 96`, `Community 98`, `Community 97`, `Community 6`, `Community 13`, `Community 99`, `Community 106`, `Community 4`, `Community 7`, `Community 20`, `Community 35`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Are the 190 inferred relationships involving `VocabConfig` (e.g. with `Ablation studies for the Vienna experiment.  Usage (from vienna/):     python ex` and `Add a second relation R2(a_i) = b_{(i+50) mod 100} and check if linearity still`) actually correct?**
  _`VocabConfig` has 190 INFERRED edges - model-reasoned connections that need verification._
- **Are the 214 inferred relationships involving `ModelConfig` (e.g. with `Ablation studies for the Vienna experiment.  Usage (from vienna/):     python ex` and `Add a second relation R2(a_i) = b_{(i+50) mod 100} and check if linearity still`) actually correct?**
  _`ModelConfig` has 214 INFERRED edges - model-reasoned connections that need verification._
- **Are the 195 inferred relationships involving `Config` (e.g. with `Ablation studies for the Vienna experiment.  Usage (from vienna/):     python ex` and `Add a second relation R2(a_i) = b_{(i+50) mod 100} and check if linearity still`) actually correct?**
  _`Config` has 195 INFERRED edges - model-reasoned connections that need verification._
- **Are the 138 inferred relationships involving `TrainConfig` (e.g. with `Ablation studies for the Vienna experiment.  Usage (from vienna/):     python ex` and `Add a second relation R2(a_i) = b_{(i+50) mod 100} and check if linearity still`) actually correct?**
  _`TrainConfig` has 138 INFERRED edges - model-reasoned connections that need verification._
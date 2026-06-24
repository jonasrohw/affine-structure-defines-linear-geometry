from pathlib import Path

import torch

from factual_recall.synthetic_curriculum import (
    CurriculumConfig,
    StructuredVocab,
    SyntheticExperimentConfig,
    build_curriculum_sequences,
    load_synthetic_config,
    run_experiment1,
    run_experiment2,
    run_frequency_contrast,
    run_mixed_relations,
)
from factual_recall.config import TransformerConfig


def test_structured_attrs_are_unique_and_affine():
    vocab = StructuredVocab(
        n_entities=32,
        n_f=12,
        n_g=12,
        delta_f=3,
        delta_g=5,
        seed=1,
    )

    all_attrs = list(vocab.subject_attrs) + list(vocab.object_attrs)

    assert len(set(all_attrs)) == 2 * vocab.n_entities
    for subject_attr, object_attr in zip(vocab.subject_attrs, vocab.object_attrs):
        assert object_attr == (
            (subject_attr[0] + vocab.delta_f) % vocab.n_f,
            (subject_attr[1] + vocab.delta_g) % vocab.n_g,
        )


def test_curriculum_phase2_uses_only_novel_single_exposure_triples():
    vocab = StructuredVocab(n_entities=24, n_f=10, n_g=10, seed=2)
    cfg = CurriculumConfig(
        n_entities=24,
        n_f=10,
        n_g=10,
        phase1_k=5,
        phase1_attr_repeats=1,
        phase1_triple_repeats=3,
        phase2_triples=11,
    )

    phase1, phase2, phase1_indices, phase2_indices = build_curriculum_sequences(vocab, cfg)

    phase1_triple_subjects = [seq[1] - vocab.subject_offset for seq in phase1 if seq[2] == vocab.relation_id]
    phase2_triple_subjects = [seq[1] - vocab.subject_offset for seq in phase2]

    assert sorted(set(phase1_triple_subjects)) == phase1_indices
    assert phase2_triple_subjects == phase2_indices
    assert set(phase1_indices).isdisjoint(phase2_indices)
    assert len(phase2_triple_subjects) == len(set(phase2_triple_subjects))


def test_curriculum_phase2_repeat_support_preserves_unique_indices():
    vocab = StructuredVocab(n_entities=24, n_f=10, n_g=10, seed=2)
    cfg = CurriculumConfig(
        n_entities=24,
        n_f=10,
        n_g=10,
        phase1_k=5,
        phase1_attr_repeats=1,
        phase1_triple_repeats=3,
        phase2_triples=7,
        phase2_repeats=3,
    )

    _, phase2, _, phase2_indices = build_curriculum_sequences(vocab, cfg)
    phase2_triple_subjects = [seq[1] - vocab.subject_offset for seq in phase2]

    assert len(phase2) == 21
    assert sorted(set(phase2_triple_subjects)) == phase2_indices


def test_synthetic_config_loader_and_tiny_run(tmp_path):
    cfg = load_synthetic_config(Path("configs/runs/synthetic_exp1_smoke.toml"))
    cfg.curriculum.output_dir = tmp_path
    cfg.curriculum.phase1_attr_repeats = 1
    cfg.curriculum.phase1_triple_repeats = 2
    cfg.curriculum.phase2_triples = 8
    cfg.curriculum.snapshot_every_triples = 4
    cfg.curriculum.eval_limit = 8
    cfg.curriculum.affine_train_limit = 8
    cfg.model = TransformerConfig(
        n_layers=1,
        hidden_dim=32,
        n_heads=4,
        context_len=8,
        tie_embeddings=False,
    )
    cfg = SyntheticExperimentConfig(vocab=cfg.vocab, model=cfg.model, curriculum=cfg.curriculum)

    result = run_experiment1(cfg, torch.device("cpu"), quiet=True)

    assert Path(result["result_path"]).exists()
    assert len(result["history"]) >= 2
    assert result["phase2_sequences"] == 8
    assert result["phase2_unique_triples"] == 8
    assert result["history"][-1]["phase2_triples_seen"] == 8
    assert "heldout_mean_rank" in result["history"][-1]
    assert "embedding_affine_nn_accuracy" in result["history"][-1]


def test_synthetic_exp2_separates_structured_from_random(tmp_path):
    cfg = load_synthetic_config(Path("configs/runs/synthetic_exp2_smoke.toml"))
    cfg.curriculum.output_dir = tmp_path
    cfg.curriculum.phase1_k = 32
    cfg.curriculum.eval_limit = 32

    result = run_experiment2(cfg, quiet=True)
    structured = result["relations"]["R0_structured"]
    random = result["relations"]["R1_random"]

    assert Path(result["result_path"]).exists()
    assert structured["affine_nn_accuracy"] > random["affine_nn_accuracy"]
    assert structured["affine_cosine"] > random["affine_cosine"]


def test_frequency_contrast_low_freq_linear_high_freq_random(tmp_path):
    cfg = load_synthetic_config(Path("configs/runs/synthetic_frequency_contrast_smoke.toml"))
    cfg.curriculum.output_dir = tmp_path
    cfg.curriculum.n_entities = 48
    cfg.curriculum.n_f = 14
    cfg.curriculum.n_g = 14
    cfg.curriculum.phase1_attr_repeats = 1
    cfg.curriculum.phase1_triple_repeats = 8
    cfg.curriculum.phase2_triples = 24
    cfg.curriculum.eval_limit = 16
    cfg.curriculum.batch_size = 16
    cfg.vocab = StructuredVocab(
        n_entities=48,
        n_f=14,
        n_g=14,
        delta_f=cfg.curriculum.delta_f,
        delta_g=cfg.curriculum.delta_g,
        seed=cfg.curriculum.seed,
    )
    cfg.model = TransformerConfig(
        n_layers=1,
        hidden_dim=32,
        n_heads=4,
        context_len=8,
        tie_embeddings=False,
    )

    result = run_frequency_contrast(cfg, torch.device("cpu"), quiet=True)
    low = result["relations"]["low_frequency_structured"]
    high = result["relations"]["high_frequency_random"]

    assert Path(result["result_path"]).exists()
    assert result["low_frequency_repeats"] == 1
    assert result["high_frequency_repeats"] == 8
    assert low["embedding_affine_nn_accuracy"] > high["embedding_affine_nn_accuracy"]
    assert low["embedding_affine_cosine"] > high["embedding_affine_cosine"]


def test_mixed_relations_keep_structured_more_affine_than_random(tmp_path):
    cfg = load_synthetic_config(Path("configs/runs/synthetic_mixed_relations_smoke.toml"))
    cfg.curriculum.output_dir = tmp_path
    cfg.curriculum.n_entities = 48
    cfg.curriculum.n_f = 14
    cfg.curriculum.n_g = 14
    cfg.curriculum.phase1_attr_repeats = 1
    cfg.curriculum.phase1_triple_repeats = 8
    cfg.curriculum.phase2_triples = 32
    cfg.curriculum.eval_limit = 16
    cfg.curriculum.batch_size = 16
    cfg.vocab = StructuredVocab(
        n_entities=48,
        n_f=14,
        n_g=14,
        delta_f=cfg.curriculum.delta_f,
        delta_g=cfg.curriculum.delta_g,
        seed=cfg.curriculum.seed,
        n_relations=2,
    )
    cfg.model = TransformerConfig(
        n_layers=1,
        hidden_dim=32,
        n_heads=4,
        context_len=8,
        tie_embeddings=False,
    )

    result = run_mixed_relations(cfg, torch.device("cpu"), quiet=True)
    structured = result["relations"]["low_frequency_structured"]
    random = result["relations"]["high_frequency_random"]

    assert Path(result["result_path"]).exists()
    assert result["structured_repeats"] == 1
    assert result["random_repeats"] == 8
    assert structured["prompt_affine_nn_accuracy"] >= random["prompt_affine_nn_accuracy"]
    assert "same_subject_relation_hidden_cosine_train" in result["mechanistic"]


def test_mixed_relations_matched_frequency_config(tmp_path):
    cfg = load_synthetic_config(Path("configs/runs/synthetic_mixed_relations_matched.toml"))
    cfg.curriculum.output_dir = tmp_path
    cfg.curriculum.n_entities = 48
    cfg.curriculum.n_f = 14
    cfg.curriculum.n_g = 14
    cfg.curriculum.phase1_attr_repeats = 1
    cfg.curriculum.phase2_triples = 24
    cfg.curriculum.structured_relation_repeats = 4
    cfg.curriculum.random_relation_repeats = 4
    cfg.curriculum.eval_limit = 16
    cfg.curriculum.batch_size = 16
    cfg.vocab = StructuredVocab(
        n_entities=48,
        n_f=14,
        n_g=14,
        delta_f=cfg.curriculum.delta_f,
        delta_g=cfg.curriculum.delta_g,
        seed=cfg.curriculum.seed,
        n_relations=2,
    )
    cfg.model = TransformerConfig(
        n_layers=1,
        hidden_dim=32,
        n_heads=4,
        context_len=8,
        tie_embeddings=False,
    )

    result = run_mixed_relations(cfg, torch.device("cpu"), quiet=True)

    assert Path(result["result_path"]).exists()
    assert result["matched_frequency"] is True
    assert result["structured_repeats"] == result["random_repeats"] == 4
    assert set(result["relations"]) == {"structured", "random"}

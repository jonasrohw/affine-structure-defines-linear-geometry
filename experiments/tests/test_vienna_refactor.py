from pathlib import Path

import torch

from factual_recall.config import RelationSpec, TokenVocabulary
from factual_recall.config_loader import load_experiment_config
from factual_recall.datasets import TOKENS_PER_TRIPLE, generate_sequence, make_dataloader
from factual_recall.model import build_model
from factual_recall.training import evaluate, train


def test_vienna_token_layout_and_relations():
    vocab = TokenVocabulary(n_subjects=12, n_objects=12)

    assert vocab.sep_id == 0
    assert vocab.subject_id(0) == 1
    assert vocab.object_id(0) == 13
    assert vocab.relation_id(0) == 25
    assert vocab.filler_id(0) == 28
    assert vocab.vocab_size == 31
    assert vocab.apply_relation(0, 5) == 5
    assert vocab.apply_relation(1, 5) == 11


def test_custom_relation_specs_extend_dataset():
    vocab = TokenVocabulary(
        n_subjects=8,
        n_objects=8,
        relation_specs=(
            RelationSpec(name="identity", kind="identity"),
            RelationSpec(name="plus_two", kind="cyclic_shift", shift=2),
        ),
    )

    assert vocab.num_relations == 2
    assert vocab.apply_relation(1, 7) == 1
    assert vocab.relation_names() == ["identity", "plus_two"]


def test_sequence_generation_uses_configured_vocab():
    vocab = TokenVocabulary(n_subjects=12, n_objects=12)
    rng = __import__("random").Random(0)
    tokens = generate_sequence(28, vocab.train_indices, vocab, rng)

    assert len(tokens) == 4 * TOKENS_PER_TRIPLE
    assert tokens[0] == vocab.sep_id
    assert all(0 <= token < vocab.vocab_size for token in tokens)


def test_config_loader_and_model_forward():
    cfg = load_experiment_config(Path("configs/runs/vienna_smoke.toml"))
    model = build_model(cfg.model, cfg.vocab)
    loader = make_dataloader(cfg)
    x, _ = next(iter(loader))

    logits = model(x)

    assert logits.shape == (cfg.train.batch_size, cfg.train.seq_length - 1, cfg.vocab.vocab_size)


def test_tiny_train_and_evaluate_smoke():
    cfg = load_experiment_config(Path("configs/runs/vienna_smoke.toml"))
    device = torch.device("cpu")
    loader = make_dataloader(cfg)
    model = build_model(cfg.model, cfg.vocab)

    history = train(model, loader, cfg, device, val_loader=loader, verbose=False)
    metrics = evaluate(model, loader, device, cfg, max_batches=1)

    assert len(history["train_loss"]) == 1
    assert metrics["loss"] > 0

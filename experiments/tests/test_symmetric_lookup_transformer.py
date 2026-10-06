"""Tests for the GPT-style symmetric lookup experiment."""

import numpy as np
import torch

from factual_recall.symmetric_lookup import build_world
from factual_recall.symmetric_lookup_transformer import (
    LookupTokenVocabulary,
    TransformerLookupConfig,
    _world_config,
    attribute_prompts,
    make_model,
    relation_prompts,
)
from factual_recall.symmetric_lookup_transformer_raw_search import (
    reverse_attribute_prompts,
)


def _tiny_config(**overrides):
    values = dict(
        seed=7,
        n_entities=16,
        n_attributes=4,
        hidden_dim=16,
        n_layers=1,
        n_heads=4,
        graph_swaps_per_edge=5,
    )
    values.update(overrides)
    return TransformerLookupConfig(**values)


def test_sentence_prompts_use_one_shared_entity_vocabulary_and_explicit_answers():
    cfg = _tiny_config()
    vocab = LookupTokenVocabulary(cfg.n_entities, cfg.n_attributes)
    device = torch.device("cpu")

    attribute_x, attribute_y = attribute_prompts(
        vocab,
        np.array([2, 3]),
        np.array([0, 1]),
        np.array([0, 1]),
        device,
    )
    assert attribute_x.tolist() == [
        [vocab.bos_id, vocab.entity_id(2), vocab.attribute_id(0)],
        [vocab.bos_id, vocab.entity_id(3), vocab.attribute_id(1)],
    ]
    assert attribute_y.tolist() == [vocab.zero_id, vocab.one_id]

    relation_x, relation_y = relation_prompts(
        vocab,
        torch.tensor([0, 4]),
        torch.tensor([2, 3]),
        torch.tensor([8, 9]),
        torch.tensor([1.0, 0.0]),
    )
    assert relation_x.tolist() == [
        [vocab.bos_id, vocab.relation_id(0), vocab.entity_id(2), vocab.entity_id(8)],
        [vocab.bos_id, vocab.relation_id(4), vocab.entity_id(3), vocab.entity_id(9)],
    ]
    assert relation_y.tolist() == [vocab.yes_id, vocab.no_id]


def test_inverse_attribute_examples_are_ordinary_next_token_prompts():
    cfg = _tiny_config()
    vocab = LookupTokenVocabulary(cfg.n_entities, cfg.n_attributes)
    attribute_x, attribute_y = attribute_prompts(
        vocab,
        np.array([2, 3]),
        np.array([0, 1]),
        np.array([0, 1]),
        torch.device("cpu"),
    )

    inverse_x, inverse_y = reverse_attribute_prompts(
        vocab, attribute_x, attribute_y
    )

    assert inverse_x.tolist() == [
        [vocab.bos_id, vocab.attribute_id(0), vocab.zero_id],
        [vocab.bos_id, vocab.attribute_id(1), vocab.one_id],
    ]
    assert inverse_y.tolist() == [vocab.entity_id(2), vocab.entity_id(3)]


def test_model_is_an_unmodified_decoder_only_transformer_without_special_heads():
    cfg = _tiny_config(tie_embeddings=True)
    vocab = LookupTokenVocabulary(cfg.n_entities, cfg.n_attributes)
    model = make_model(cfg, vocab)

    assert model.n_layers == 1
    assert model.model_cfg.context_len == 4
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()
    assert not hasattr(model, "relation_heads")
    assert not hasattr(model, "attribute_decoder")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_random_relation_is_symmetric_and_degree_matched_before_tokenization():
    cfg = _tiny_config()
    world = build_world(_world_config(cfg))

    assert np.array_equal(world.labels, world.labels.transpose(0, 2, 1))
    assert np.all(world.labels.sum(axis=2) == cfg.n_entities // 2)

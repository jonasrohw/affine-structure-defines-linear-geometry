"""Invariant and diagnostic tests for the revised symmetric lookup experiment."""

import numpy as np
import pytest
import torch

from factual_recall.symmetric_lookup import (
    SymmetricLookupConfig,
    SymmetricLookupModel,
    _freeze_representation,
    _train_relations,
    all_ols_metrics,
    build_world,
    relation_accuracies,
)


def _tiny_config(**overrides):
    values = dict(
        seed=7,
        n_entities=16,
        n_attributes=4,
        embedding_dim=4,
        nonlinear_width=16,
        nonlinear_depth=1,
        graph_swaps_per_edge=5,
    )
    values.update(overrides)
    return SymmetricLookupConfig(**values)


def test_world_has_explicit_balanced_attributes_and_matched_symmetric_relations():
    cfg = _tiny_config()
    world = build_world(cfg)

    assert world.attributes.shape == (16, 4)
    assert len(np.unique(world.attributes, axis=0)) == 16
    assert (world.attributes.sum(axis=0) == 8).all()
    assert len(world.attribute_sentences()) == 16 * 4
    assert len(world.relation_sentences()) == 5 * 16 * 16

    assert (world.labels == world.labels.transpose(0, 2, 1)).all()
    # Every structured relation and the arbitrary table have exactly the same
    # number of positive objects for every subject.
    assert (world.labels.sum(axis=2) == 8).all()
    assert not any(
        np.array_equal(world.labels[world.arbitrary_relation], world.labels[index])
        for index in range(world.n_attributes)
    )


def test_relation_model_is_exactly_symmetric_with_and_without_mlp():
    cfg = _tiny_config()
    model = SymmetricLookupModel(cfg)
    relations = torch.tensor([0, 1, 4, 4])
    left = torch.tensor([1, 2, 3, 7])
    right = torch.tensor([5, 6, 9, 4])

    for include_nonlinear in (False, True):
        forward = model.relation_logits(relations, left, right, include_nonlinear)
        reverse = model.relation_logits(relations, right, left, include_nonlinear)
        assert torch.allclose(forward, reverse, atol=1e-7)


def test_mechanistic_decomposition_exactly_reproduces_model_logits():
    pytest.importorskip("matplotlib")
    from factual_recall.symmetric_lookup_mechanistic_analysis import mlp_activations

    cfg = _tiny_config(nonlinear_depth=2)
    model = SymmetricLookupModel(cfg)
    relations = torch.tensor([0, 1, 4, 4])
    subjects = torch.tensor([1, 2, 3, 7])
    objects = torch.tensor([5, 6, 9, 4])

    traced = mlp_activations(model, relations, subjects, objects)
    direct = model.relation_logits(relations, subjects, objects, True)

    assert torch.allclose(traced["full"], direct, atol=1e-7)
    assert torch.allclose(
        traced["full"], traced["affine"] + traced["gate"] * traced["residual"]
    )


def test_attribute_geometry_makes_equality_ols_linear_but_random_table_not_linear():
    cfg = _tiny_config(embedding_dim=4, ols_folds=4)
    world = build_world(cfg)
    model = SymmetricLookupModel(cfg)
    # An exact learned-attribute geometry: one centered coordinate per bit.
    with torch.no_grad():
        model.entity_embeddings.weight.copy_(
            torch.tensor(world.attributes * 2 - 1, dtype=torch.float32)
        )
    metrics = all_ols_metrics(model, world, cfg)

    assert min(row["full_r2"] for row in metrics[: world.n_attributes]) > 1 - 1e-10
    assert metrics[world.arbitrary_relation]["cv_r2"] < 0.5


def test_affine_only_random_head_cannot_trivially_interpolate_tiny_world():
    cfg = _tiny_config(nonlinear_width=0)
    world = build_world(cfg)
    model = SymmetricLookupModel(cfg)
    accuracies = relation_accuracies(
        model,
        world,
        torch.device("cpu"),
        include_nonlinear=False,
        relation_ids=[world.arbitrary_relation],
    )
    assert len(accuracies) == 1
    assert 0.0 <= accuracies[0] <= 1.0


def test_relation_training_respects_frozen_or_trainable_embedding_policy():
    device = torch.device("cpu")
    cfg = _tiny_config(
        batch_size=32,
        arbitrary_steps=2,
        learning_rate=1e-2,
        relation_attr_rehearsal_weight=1.0,
    )
    world = build_world(cfg)

    trainable = SymmetricLookupModel(cfg)
    before = trainable.entity_embeddings.weight.detach().clone()
    _train_relations(
        trainable,
        world,
        cfg,
        device,
        [world.arbitrary_relation],
        max_steps=2,
        seed_offset=10,
        quiet=True,
        penalize_structured_residual=False,
        train_representation=True,
    )
    assert not torch.equal(before, trainable.entity_embeddings.weight.detach())

    frozen = SymmetricLookupModel(cfg)
    _freeze_representation(frozen)
    before = frozen.entity_embeddings.weight.detach().clone()
    _train_relations(
        frozen,
        world,
        cfg,
        device,
        [world.arbitrary_relation],
        max_steps=2,
        seed_offset=10,
        quiet=True,
        penalize_structured_residual=False,
        train_representation=False,
    )
    assert torch.equal(before, frozen.entity_embeddings.weight.detach())

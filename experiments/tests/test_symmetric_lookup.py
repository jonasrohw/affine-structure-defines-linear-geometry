"""Balanced world and affine-probe invariants."""
import numpy as np
from factual_recall.symmetric_lookup import SymmetricLookupConfig, build_world, ols_relation_metrics

def _tiny_config():
    return SymmetricLookupConfig(seed=7, n_entities=16, n_attributes=4, graph_swaps_per_edge=5)

def test_world_has_explicit_balanced_attributes_and_matched_symmetric_relations():
    cfg = _tiny_config()
    world = build_world(cfg)

    assert world.attributes.shape == (16, 4)
    assert len(np.unique(world.attributes, axis=0)) == 16
    assert (world.attributes.sum(axis=0) == 8).all()

    assert (world.labels == world.labels.transpose(0, 2, 1)).all()
    # Every structured relation and the arbitrary table have exactly the same
    # number of positive objects for every subject.
    assert (world.labels.sum(axis=2) == 8).all()
    assert not any(
        np.array_equal(world.labels[world.arbitrary_relation], world.labels[index])
        for index in range(world.n_attributes)
    )


def test_attribute_geometry_makes_equality_affine():
    world = build_world(_tiny_config())
    embeddings = world.attributes * 2.0 - 1
    metrics = [ols_relation_metrics(embeddings, labels, 4, 7) for labels in world.labels]
    assert min(row["full_r2"] for row in metrics[:-1]) > 1 - 1e-10
    assert metrics[-1]["full_r2"] < 0.5

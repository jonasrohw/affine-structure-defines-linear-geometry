"""Invariant checks for the linear_shift experiment design."""

import numpy as np

from factual_recall.linear_shift import World


def test_category_world_matched_frequencies():
    world = World("category", seed=19)
    assert world.n_subjects == 192 and world.n_objects == 32
    assert len(world.probe_subjects) == 128
    assert len(world.heldout_subjects) == 64
    assert not set(world.probe_subjects) & set(world.heldout_subjects)

    # Every subject has a unique (f, g) pair.
    assert len(set(world.subject_attrs)) == world.n_subjects

    # Split is stratified: per f-value, 4 probe + 2 held-out subjects.
    for bucket in range(32):
        probe_in = [i for i in world.probe_subjects if i % 32 == bucket]
        held_in = [i for i in world.heldout_subjects if i % 32 == bucket]
        assert len(probe_in) == 4 and len(held_in) == 2

    # Matched object frequencies: under both relations, every object is the
    # target of exactly 4 probe subjects and exactly 2 held-out subjects.
    for relation in ("semantic", "random"):
        for subjects, expected in ((world.probe_subjects, 4), (world.heldout_subjects, 2)):
            counts = np.bincount(
                [world.target(relation, i) for i in subjects], minlength=world.n_objects
            )
            assert (counts == expected).all(), (relation, expected)

    # The semantic target is determined by the subject's f attribute; the
    # random target is a different assignment.
    assert all(world.target("semantic", i) == world.subject_attrs[i][0] for i in range(192))
    assert any(world.target("random", i) != world.target("semantic", i) for i in range(192))


def test_random_relation_is_seed_dependent_and_balanced():
    a, b = World("category", seed=19), World("category", seed=31)
    assert any(a.target("random", i) != b.target("random", i) for i in range(192))


def test_corpus_single_occurrence_counts():
    from factual_recall.linear_shift import LinearShiftConfig, build_corpus

    cfg = LinearShiftConfig(world="category", seed=19, n_triples=128, occurrences=1, attr_repeats=2)
    world = World("category", seed=19)
    sequences, trained = build_corpus(cfg, world)
    assert len(trained) == 128
    # Each relation triple occurs exactly once; attribute sentences twice.
    triple_seqs = [tuple(s) for s in sequences if s[2] in (world.r0, world.r1)]
    assert len(triple_seqs) == 256 and len(set(triple_seqs)) == 256
    assert len(sequences) == (192 + 32) * 2 + 256

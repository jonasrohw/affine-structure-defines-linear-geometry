"""Focused tests for transformation-level affine/Jacobian diagnostics."""

import torch

from factual_recall.linear_shift import World
from factual_recall.linear_shift_linearity_analysis import (
    affine_reconstruction_metrics,
    apply_affine_map,
    different_target_pairs,
    fit_affine_map,
    jacobian_metrics,
    midpoint_metrics,
    target_stratified_affine_folds,
)


def test_target_stratified_affine_split_uses_only_memorized_subjects():
    world = World("category", seed=19)
    fit, evaluation = target_stratified_affine_folds(
        world, world.probe_subjects, "random"
    )[0]
    assert len(fit) == 96
    assert len(evaluation) == 32
    assert not set(fit) & set(evaluation)
    assert set(fit) | set(evaluation) == set(world.probe_subjects)
    for target in range(world.n_objects):
        assert sum(world.target("random", subject) == target for subject in fit) == 3
        assert sum(world.target("random", subject) == target for subject in evaluation) == 1


def test_target_stratified_affine_folds_hold_out_every_memorized_subject_once():
    world = World("category", seed=19)
    folds = target_stratified_affine_folds(world, world.probe_subjects, "random")
    assert len(folds) == 4
    heldout = []
    for fit, evaluation in folds:
        assert len(fit) == 96 and len(evaluation) == 32
        assert not set(fit) & set(evaluation)
        heldout.extend(evaluation)
    assert sorted(heldout) == sorted(world.probe_subjects)


def test_affine_function_has_zero_reconstruction_midpoint_and_jacobian_defects():
    torch.manual_seed(0)
    x = torch.randn(20, 4)
    true_weight = torch.randn(4, 3)
    true_bias = torch.randn(3)

    def function(row):
        return row @ true_weight + true_bias

    y = torch.stack([function(row) for row in x])
    weight, bias = fit_affine_map(x[:12], y[:12])
    predicted = apply_affine_map(x[12:], weight, bias)
    reconstruction = affine_reconstruction_metrics(y[12:], predicted)
    pairs = [(index, (index + 1) % 8) for index in range(8)]
    midpoint = midpoint_metrics(function, x[12:], y[12:], pairs)
    jacobian = jacobian_metrics(function, x[12:], weight)
    assert reconstruction["r2"] > 1.0 - 1e-12
    assert midpoint["relative_frobenius_defect"] < 1e-6
    assert jacobian["pooled_relative_dispersion"] < 1e-12
    assert jacobian["affine_fit_relative_error"] < 1e-6


def test_nonlinear_function_has_midpoint_and_jacobian_defects():
    x = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.0, 2.0],
            [0.5, -1.5],
            [1.0, 0.5],
            [1.5, 2.0],
            [2.0, -0.5],
        ]
    )

    def function(row):
        return torch.stack((row[0].square(), row[0] * row[1], row[1].square()))

    y = torch.stack([function(row) for row in x])
    weight, _ = fit_affine_map(x[:4], y[:4])
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    midpoint = midpoint_metrics(function, x, y, pairs)
    jacobian = jacobian_metrics(function, x, weight)
    assert midpoint["relative_frobenius_defect"] > 0.1
    assert jacobian["pooled_relative_dispersion"] > 0.1


def test_different_target_pairs_never_pair_equal_targets():
    world = World("category", seed=19)
    subjects = world.probe_subjects[:32]
    pairs = different_target_pairs(world, subjects, "random")
    assert pairs
    assert len(pairs) <= len(subjects) * (len(subjects) - 1) // 2
    for left, right in pairs:
        assert world.target("random", subjects[left]) != world.target("random", subjects[right])

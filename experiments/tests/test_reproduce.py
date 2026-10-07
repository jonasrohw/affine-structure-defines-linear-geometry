"""Integration safeguards for the reproduction entry point."""
import json

import torch

from factual_recall.reproduce import cached_stage, configure_runtime
from factual_recall.linear_shift import LinearShiftConfig, World, build_corpus, train_one_epoch


def test_experiment2_schedule_matches_recovered_training_logs():
    from factual_recall.reproduce import experiment2_schedule

    # The 500-step setting for every seed and 100-step first continuation selected
    # different checkpoints. These are the command limits, not observed stop times.
    assert experiment2_schedule(19) == [
        ("joint", 12000, 1e-3, 500), ("continuation", 8000, 3e-4, 200),
        ("continuation2", 3000, 1e-4, 100)]
    assert experiment2_schedule(31) == [
        ("joint", 12000, 1e-3, 1000), ("continuation", 8000, 3e-4, 200),
        ("continuation2", 3000, 1e-4, 100)]
    assert experiment2_schedule(47) == [
        ("joint", 12000, 1e-3, 1000), ("continuation", 8000, 3e-4, 200)]


def test_resume_rejects_a_change_of_numerical_runtime(tmp_path):
    from factual_recall.reproduce import runtime_signature

    path = tmp_path / "stage.json"
    first = runtime_signature(torch.device("cpu"), "deterministic")
    second = runtime_signature(torch.device("cpu"), "beta")
    cached_stage(path, {"runtime": first}, lambda: {"value": 1}, False)
    assert cached_stage(path, {"runtime": second}, lambda: {"value": 2}, True)["value"] == 2


def test_resume_rejects_changed_inputs_and_corrupt_checkpoints(tmp_path):
    calls = []

    def operation():
        calls.append(True)
        return {"value": len(calls), "_state_dict": {"weight": torch.tensor([1.0])}}

    path = tmp_path / "stage.json"
    signature = {"seed": 19}
    assert cached_stage(path, signature, operation, True)["value"] == 1
    assert cached_stage(path, signature, operation, True)["value"] == 1
    path.with_suffix(".pt").write_bytes(b"interrupted write")
    assert cached_stage(path, signature, operation, True)["value"] == 2
    assert cached_stage(path, {"seed": 31}, operation, True)["value"] == 3
    assert torch.load(path.with_suffix(".pt"), weights_only=True)["weight"].item() == 1


def test_failed_stage_cannot_leave_a_valid_completion_marker(tmp_path):
    import pytest

    path = tmp_path / "stage.json"
    cached_stage(path, {}, lambda: {"value": 1}, False)

    def fail():
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        cached_stage(path, {}, fail, False)
    assert not path.with_suffix(".complete.json").exists()


def test_cpu_training_repeats_bit_for_bit_on_the_same_runtime():
    configure_runtime()
    cfg = LinearShiftConfig(attr_repeats=1, occurrences=1)
    world = World("category", cfg.seed)
    corpus, _ = build_corpus(cfg, world)
    first, first_steps, first_loss = train_one_epoch(cfg, world, corpus, torch.device("cpu"))
    second, second_steps, second_loss = train_one_epoch(cfg, world, corpus, torch.device("cpu"))
    assert first_steps == second_steps and first_loss == second_loss
    assert first.n_parameters() == 84288
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key


def test_reference_comparison_reads_generated_values(tmp_path):
    from factual_recall.paper_audit import experiment1_summary

    rows = []
    for frequency in (1, 4, 16, 64):
        for seed in (19, 31, 47):
            relations = {name: {"probe_accuracy": seed / 100,
                               "trained_check": {"recall": .25},
                               "heldout_check": {"recall": .5}}
                         for name in ("semantic", "random")}
            rows.append({"config": {"occurrences": frequency, "attr_repeats": 128},
                         "relations": relations})
    first = experiment1_summary(rows)
    rows[0]["relations"]["semantic"]["probe_accuracy"] += .3
    second = experiment1_summary(rows)
    assert abs(second["1"]["semantic"]["probe"]["mean"] - first["1"]["semantic"]["probe"]["mean"] - .1) < 1e-12
    # These reports remain machine-readable.
    json.dumps(second, allow_nan=False)


def test_joint_selection_requires_symmetry_and_preserves_passing_checkpoint():
    from factual_recall.symmetric_lookup_transformer_raw_search import joint_gate, joint_checkpoint_score

    assert not joint_gate(1., [1., 1.], .994, .99, .3, .995)
    assert joint_gate(.995, [.995, .995], .995, .95, .499, .995)
    assert not joint_gate(1., [1., 1.], 1., .99, .50, .995)
    passing = joint_checkpoint_score(True, True, True, .995, .96, .4)
    failing = joint_checkpoint_score(True, True, False, 1., .999, .3)
    assert passing > failing


def test_exact_baseline_detects_changed_missing_and_extra_results():
    from factual_recall.paper_audit import compare_snapshots

    expected = {'numbers': {'accuracy': 1.}, 'tensors': {'weight': 'abc'}, 'joint_gates': {'19': True}}
    assert all(row['matches'] for row in compare_snapshots(expected, expected))
    changed = {'numbers': {'accuracy': 1. - 1e-15, 'extra': 0}, 'tensors': {}, 'joint_gates': {'19': False}}
    assert sum(not row['matches'] for row in compare_snapshots(changed, expected)) == 4


def test_snapshot_checks_tensor_contents_and_all_required_artifacts(tmp_path):
    import pytest
    from factual_recall.paper_audit import measurement_snapshot, compare_snapshots

    required = ('experiment1/sweep.json', 'experiment1/mechanism.json',
                'experiment1/linearity.json', 'paper/data/lment_frequency/summary.json',
                'paper/data/lment_causality/summary.json')
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"value": 1.0}')
    (tmp_path / 'experiment2').mkdir()
    with pytest.raises(FileNotFoundError):
        measurement_snapshot(tmp_path)
    (tmp_path / 'experiment2/analysis.json').write_text(
        '{"seeds": [{"seed": 19, "passes_joint_gate": true}]}')
    checkpoint = tmp_path / 'experiment2/final.pt'
    torch.save({'weight': torch.tensor([1.0])}, checkpoint)
    first = measurement_snapshot(tmp_path)
    (tmp_path / 'experiment1/sweep.json').write_text('{"value": 1.0, "elapsed_sec": 123.0}')
    torch.save({'weight': torch.tensor([1.0])}, checkpoint)
    assert all(c['matches'] for c in compare_snapshots(first, measurement_snapshot(tmp_path)))
    torch.save({'weight': torch.nextafter(torch.tensor([1.0]), torch.tensor([2.0]))}, checkpoint)
    changes = [c for c in compare_snapshots(first, measurement_snapshot(tmp_path)) if not c['matches']]
    assert len(changes) == 1 and changes[0]['metric'].startswith('tensors/')

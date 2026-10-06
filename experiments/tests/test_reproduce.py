"""Integration safeguards for the reproduction entry point."""
import json

import torch

from factual_recall.reproduce import cached_stage, configure_runtime
from factual_recall.linear_shift import LinearShiftConfig, World, build_corpus, train_one_epoch


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

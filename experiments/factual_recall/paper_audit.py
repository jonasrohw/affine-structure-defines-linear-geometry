"""Export result tables and compare fresh measurements with reported evidence.

Reference files are never used to train a model or populate a generated figure.
They are read only here, after measurement, to detect numerical drift.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT
from .linear_shift_linearity_analysis import STAGES


def load(path):
    return json.loads(Path(path).read_text())


def experiment1_summary(rows):
    result = {}
    for frequency in (1, 4, 16, 64):
        group = [r for r in rows if r["config"]["occurrences"] == frequency
                 and r["config"]["attr_repeats"] == 128]
        result[str(frequency)] = {}
        for relation in ("semantic", "random"):
            result[str(frequency)][relation] = {}
            for metric, path in (("probe", ("probe_accuracy",)),
                                 ("trained", ("trained_check", "recall")),
                                 ("heldout", ("heldout_check", "recall"))):
                values = []
                for row in group:
                    value = row["relations"][relation]
                    for part in path:
                        value = value[part]
                    values.append(value)
                # The archived Experiment 1 analyses use population SD (ddof=0).
                result[str(frequency)][relation][metric] = {
                    "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=0))}
    return result


def flatten_numbers(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten_numbers(item, f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten_numbers(item, f"{prefix}/{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def audit_historical(output):
    checks = []

    def check(name, actual, expected, tolerance):
        matched = actual is not None and math.isfinite(actual) and abs(actual - expected) <= tolerance
        checks.append({"metric": name, "actual": actual, "expected": expected,
                       "absolute_tolerance": tolerance, "matches": matched})

    def compare(name, actual, expected, tolerance):
        measured = dict(flatten_numbers(actual))
        for key, value in flatten_numbers(expected):
            if math.isfinite(value):
                check(name + key, measured.get(key), value, tolerance)

    first = experiment1_summary(load(output / "experiment1/sweep.json"))
    # Table 1, in column order: semantic/random probe, trained recall, held-out recall.
    expected_rows = {
        "1": (1.000, .031, .029, .010, .016, .010, .000, .013),
        "4": (1.000, .016, .055, .034, .057, .031, .000, .013),
        "16": (.990, .036, .672, .167, .635, .031, .007, .007),
    }
    for frequency, expected in expected_rows.items():
        fields = [(metric, relation, "mean") for metric in ("probe", "trained", "heldout")
                  for relation in ("semantic", "random")]
        fields += [("probe", relation, "sd") for relation in ("semantic", "random")]
        for (metric, relation, statistic), value in zip(fields, expected, strict=True):
            check(f"table1/F{frequency}/{relation}/{metric}/{statistic}",
                  first[frequency][relation][metric][statistic], value, .0005)
    check("appendix/F64/random_trained_recall", first["64"]["random"]["trained"]["mean"], 1.0, 0)
    mech = load(output / "experiment1/mechanism.json")
    reference_mech = load(PROJECT_ROOT / "paper/reference/experiment1_mechanism.json")
    # Aggregates cover centroid probes, attention/MLP interventions and patch recovery.
    # Neuron indices are discovery metadata, not numerical manuscript claims.
    for key in ("baseline_full_vocab_accuracy", "all_heads_mean_ablated_accuracy",
                "geometry_centroid_accuracy", "attention_values_patch_recovery",
                "attention_patterns_patch_recovery", "relation_gate",
                "neuron_subject_specific_margin_fraction"):
        compare("experiment1/mechanism/" + key, mech["aggregate"]["goal1"][key],
                reference_mech["aggregate"]["goal1"][key], 5e-5)
    linear = load(output / "experiment1/linearity.json")["aggregate"]["semantic"]
    table = [(.993, .002, .080, .012, .742, .072, .035, .007, .085, .013),
             (.983, .004, .129, .015, .750, .066, .273, .010, .179, .005),
             (.983, .003, .131, .011, .742, .061, .095, .005, .137, .007),
             (.974, .006, .160, .017, .758, .057, .247, .027, .172, .009),
             (.993, .003, .081, .017, .760, .060, .199, .004, .228, .007),
             (.965, .010, .185, .027, .750, .056, .352, .011, .281, .013)]
    for stage, expected in zip(STAGES, table, strict=True):
        metrics = transformation_metrics(linear[stage])
        for index, value in enumerate(expected):
            check(f"transformation_table/{stage}/{index}", metrics[index], value, .0005)
    for statistic, expected in (("mean", .284), ("sd", .013)):
        check("full_lookup/tangent_jacobian/" + statistic,
              linear["full_lookup"]["jacobian"]["tangent_pooled_relative_dispersion"][statistic], expected, .0005)
    second = load(output / "experiment2/analysis.json")
    reference_second = load(PROJECT_ROOT / "paper/reference/experiment2.json")
    for actual, expected in zip(second["seeds"], reference_second["seeds"], strict=True):
        for key in ("behavior", "raw_embedding_ols", "component_ablations", "layer_ablations",
                    "logit_lens", "causal_trace", "num_parameters", "frozen_parameter_count"):
            compare(f"experiment2/seed{actual['seed']}/{key}", actual[key], expected[key], 5e-5)
        check(f"experiment2/seed{actual['seed']}/joint_gate", int(actual["passes_joint_gate"]), 1, 0)
    compare("lment/frequency", load(output / "paper/data/lment_frequency/summary.json"),
            load(PROJECT_ROOT / "paper/data/lment_frequency/summary.json"), 1e-10)
    causal = load(output / "paper/data/lment_causality/summary.json")
    for key, value, tolerance in (("n_relations", 56, 0), ("n_test_query_trials", 15874, 0),
                                  ("pearson_log1p_frequency", .393, .0005)):
        check("lment/causality/" + key, causal[key], value, tolerance)
    report = {"all_match": all(c["matches"] for c in checks),
              "checked": len(checks), "mismatches": sum(not c["matches"] for c in checks),
              "scope": "Fresh synthetic training/analysis; LMEnt aggregation from hash-verified snapshots. "
                       "Pretrained inference, corpus counting, and cited literature are external inputs.",
              "checks": checks}
    with (output / "audit.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    return report


def measurement_snapshot(output):
    """All numerical experiment outputs, plus checkpoint tensor contents.

    Paths, serialization metadata and wall time are not scientific results.
    Training histories, discovered neuron indices and permutation samples are.
    """
    import hashlib
    import torch

    output = Path(output)
    numbers, tensors = {}, {}
    for required in ("experiment1/sweep.json", "experiment1/mechanism.json",
                     "experiment1/linearity.json", "experiment2/analysis.json",
                     "paper/data/lment_frequency/summary.json", "paper/data/lment_causality/summary.json"):
        if not (output / required).is_file():
            raise FileNotFoundError(output / required)
    for directory in ("experiment1", "experiment2"):
        for path in sorted((output / directory).rglob("*.json")):
            if path.name.endswith(".complete.json"):
                continue
            for key, value in flatten_numbers(load(path)):
                # run_single records wall time both per run and in sweep.json.
                # It is operational metadata, not a scientific measurement.
                if key.endswith("/elapsed_sec"):
                    continue
                numbers[str(path.relative_to(output)) + key] = (
                    value if math.isfinite(value) else str(value))
        for path in sorted((output / directory).rglob("*.pt")):
            state = torch.load(path, map_location="cpu", weights_only=True)
            for key, tensor in sorted(state.items()):
                tensor = tensor.detach().cpu().contiguous()
                identity = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
                tensors[str(path.relative_to(output)) + "/" + key] = hashlib.sha256(
                    identity + tensor.numpy().tobytes()).hexdigest()
    for path in sorted((output / "paper/data").rglob("summary.json")):
        for key, value in flatten_numbers(load(path)):
            numbers[str(path.relative_to(output)) + key] = value if math.isfinite(value) else str(value)
    analysis = load(output / "experiment2/analysis.json")
    gates = {str(row["seed"]): row["passes_joint_gate"] for row in analysis["seeds"]}
    return {"numbers": numbers, "tensors": tensors, "joint_gates": gates}


def compare_snapshots(actual, expected):
    checks = []
    for section in ("numbers", "tensors", "joint_gates"):
        for key in sorted(actual[section].keys() | expected[section].keys()):
            a, b = actual[section].get(key), expected[section].get(key)
            checks.append({"metric": section + "/" + key, "actual": a,
                           "expected": b, "absolute_tolerance": 0,
                           "matches": key in actual[section] and key in expected[section] and a == b})
    return checks


def audit(output, reference="current"):
    historical = audit_historical(output)
    if reference == "historical":
        return historical
    (output / "historical_audit.json").write_text(json.dumps(historical, indent=2) + "\n")
    (output / "audit.csv").replace(output / "historical_audit.csv")
    baseline = PROJECT_ROOT / "paper/baselines/rtx4090-v1/measurements.json"
    if not baseline.exists():
        raise RuntimeError("RTX baseline is not certified yet; use --reference historical for discovery runs")
    actual = measurement_snapshot(output)
    checks = compare_snapshots(actual, load(baseline))
    for seed, passed in actual["joint_gates"].items():
        checks.append({"metric": "required_joint_gate/" + seed, "actual": passed,
                       "expected": True, "absolute_tolerance": 0, "matches": passed is True})
    # External LMEnt snapshots retain their original audit and tolerances.
    checks.extend(c for c in historical["checks"] if c["metric"].startswith("lment/"))
    with (output / "audit.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    return {"all_match": all(c["matches"] for c in checks), "checked": len(checks),
            "mismatches": sum(not c["matches"] for c in checks),
            "scope": "Exact RTX4090-v1 numerical outputs and tensor contents; historical audit saved separately.",
            "checks": checks}


def transformation_metrics(stage):
    return [stage[section][metric][statistic]
            for section, metric in (("affine_reconstruction", "r2"),
                                    ("affine_reconstruction", "centered_relative_l2_error"),
                                    ("affine_decision", "target_accuracy"),
                                    ("midpoint", "relative_frobenius_defect"),
                                    ("jacobian", "pooled_relative_dispersion"))
            for statistic in ("mean", "sd")]


def write_tables(output):
    """Export full-precision tables without requiring a manuscript template."""
    output = Path(output)
    first = experiment1_summary(load(output / "experiment1/sweep.json"))
    analysis = load(output / "experiment2/analysis.json")
    linear = load(output / "experiment1/linearity.json")["aggregate"]["semantic"]
    directory = output / "paper/tables"
    directory.mkdir(parents=True, exist_ok=True)

    def write_csv(name, rows):
        with (directory / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    rows = []
    for frequency, relations in first.items():
        for relation, metrics in relations.items():
            for metric, values in metrics.items():
                rows.append({"frequency": int(frequency), "relation": relation,
                             "metric": metric, **values})
    write_csv("experiment1.csv", rows)

    rows = []
    for row in analysis["seeds"]:
        behavior, ols = row["behavior"], row["raw_embedding_ols"]
        rows.append({"seed": row["seed"],
                     "attribute_accuracy": behavior["attribute_accuracy"],
                     "min_structured_accuracy": behavior["min_structured_accuracy"],
                     "random_accuracy": behavior["arbitrary_accuracy"],
                     "min_symmetry_agreement": behavior["min_symmetry_agreement"],
                     "min_structured_r2": ols["min_structured_full_r2"],
                     "random_r2": ols["arbitrary"]["full_r2"],
                     "random_cv_r2": ols["arbitrary"]["cv_r2"]})
    rows.append({"seed": "mean", **{key: float(np.mean([row[key] for row in rows]))
                                    for key in rows[0] if key != "seed"}})
    write_csv("experiment2.csv", rows)

    rows = []
    for stage in STAGES:
        for section, metric in (("affine_reconstruction", "r2"),
                                ("affine_reconstruction", "centered_relative_l2_error"),
                                ("affine_decision", "target_accuracy"),
                                ("midpoint", "relative_frobenius_defect"),
                                ("jacobian", "pooled_relative_dispersion")):
            values = linear[stage][section][metric]
            rows.append({"stage": stage, "section": section, "metric": metric,
                         "mean": values["mean"], "sd": values["sd"]})
    write_csv("transformations.csv", rows)
    (output / "paper/measurements.json").write_text(json.dumps({
        "experiment1": first, "experiment1_transformations": linear,
        "experiment2": analysis["summary"],
    }, indent=2) + "\n")

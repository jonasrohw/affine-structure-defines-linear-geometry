"""Generate manuscript tables and compare fresh measurements with reported evidence.

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


def audit(output):
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


def transformation_metrics(stage):
    return [stage[section][metric][statistic]
            for section, metric in (("affine_reconstruction", "r2"),
                                    ("affine_reconstruction", "centered_relative_l2_error"),
                                    ("affine_decision", "target_accuracy"),
                                    ("midpoint", "relative_frobenius_defect"),
                                    ("jacobian", "pooled_relative_dispersion"))
            for statistic in ("mean", "sd")]


def write_tables(output):
    first = experiment1_summary(load(output / "experiment1/sweep.json"))
    second = load(output / "experiment2/analysis.json")["seeds"]
    linear = load(output / "experiment1/linearity.json")["aggregate"]["semantic"]
    path = output / "paper/main.tex"
    manuscript = path.read_text()

    def replace_table(label, rows):
        nonlocal manuscript
        end_label = manuscript.index(r"\label{" + label + "}")
        start = manuscript.rfind(r"\begin{tabular}", 0, end_label)
        start_rows = manuscript.index(r"\midrule", start) + len(r"\midrule")
        end_rows = manuscript.index(r"\bottomrule", start_rows)
        manuscript = manuscript[:start_rows] + "\n" + "\n".join(rows) + "\n" + manuscript[end_rows:]

    rows = []
    for frequency in ("1", "4", "16"):
        values = [frequency]
        for metric in ("probe", "trained", "heldout"):
            for relation in ("semantic", "random"):
                data = first[frequency][relation][metric]
                values.append(f"${data['mean']:.3f}\\pm{data['sd']:.3f}$" if metric == "probe"
                              else f"{data['mean']:.3f}")
        rows.append(" & ".join(values) + r" \\")
    replace_table("tab:goal1", rows)
    for label, complete in (("tab:goal2", False), ("tab:goal2-complete", True)):
        rows, matrix = [], []
        for row in second:
            b, ols = row["behavior"], row["raw_embedding_ols"]
            values = ([b["attribute_accuracy"]] if complete else [])
            values += [b["min_structured_accuracy"], b["arbitrary_accuracy"]]
            values += [b["min_symmetry_agreement"]] if complete else []
            values += [ols["min_structured_full_r2"], ols["arbitrary"]["full_r2"]]
            values += [ols["arbitrary"]["cv_r2"]] if complete else []
            matrix.append(values)
            rows.append(str(row["seed"]) + " & " + " & ".join(f"{v:.5f}" for v in values) + r" \\")
        rows += [r"\midrule", "Mean & " + " & ".join(f"{v:.5f}" for v in np.mean(matrix, axis=0)) + r" \\"]
        replace_table(label, rows)
    names = ("Attention core", "Attention block", "MLP core", "MLP block", "Readout", "Full lookup")
    rows = []
    for name, stage in zip(names, STAGES, strict=True):
        values = transformation_metrics(linear[stage])
        rows.append(name + " & " + " & ".join(f"${values[i]:.3f}\\pm{values[i+1]:.3f}$"
                                               for i in range(0, 10, 2)) + r" \\")
    replace_table("tab:goal1-transformations", rows)
    path.write_text(manuscript)
    (output / "paper/measurements.json").write_text(json.dumps({
        "experiment1": first, "experiment1_transformations": linear,
        "experiment2": load(output / "experiment2/analysis.json")["summary"],
    }, indent=2) + "\n")

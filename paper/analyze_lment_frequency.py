"""Aggregate the archived triplets-dataset LMEnt frequency curves for the paper.

Run with numpy, pandas, and matplotlib. The default uses the local data snapshot;
--source-root PATH refreshes it after checking the original predictions, counts,
curve payload, and estimation/evaluation split. No model inference is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/lment_frequency"
AXES = {
    "subject": ("subject_document_count", "Subject"),
    "subject-object": ("subject_object_document_count", "Subject + object"),
    "subject-relation": ("subject_relation_document_count", "Subject + cue"),
}
KEYS = ["relation", "subject", "object"]
EXAMPLES = [
    "country official language", "organization headquarters",
    "landmark country", "country currency",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_facts(facts):
    assert not facts.duplicated(KEYS).any(), "Duplicate fact keys"
    assert facts.faithful.dtype == bool, "Expected Boolean outcomes"
    assert (facts.faithful == (
        facts.lm_prediction_token_id == facts.lre_prediction_token_id
    )).all(), "Faithfulness does not match token agreement"
    for column, _ in AXES.values():
        assert facts[column].notna().all()
        assert (facts[column] >= 0).all()
        assert (facts[column] % 1 == 0).all()
    for column in ["subject_object_document_count", "subject_relation_document_count"]:
        assert (facts[column] <= facts.subject_document_count).all()


def snapshot(source):
    run = source / "results/lment-lre-v2"
    predictions = pd.read_csv(run / "predictions.csv")
    counts = pd.read_csv(run / "frequency/fact_counts.csv")
    facts = predictions.merge(counts, on=KEYS, how="left", validate="one_to_one")
    check_facts(facts)
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(facts) == manifest["micro_n"]
    assert int(facts.faithful.sum()) == manifest["micro_correct"]
    assert facts.relation.nunique() == manifest["completed_relations"]
    assert np.isclose(facts.groupby("relation").faithful.mean().mean(),
                      manifest["macro_faithfulness"])
    result_paths = []
    for path in sorted((run / "relation_results").glob("*.json")):
        result_paths.append(path)
        result = json.loads(path.read_text())
        if result["status"] != "completed":
            continue
        train = {(r["subject"], r["object"]) for r in result["train_samples"]}
        group = facts[facts.relation == result["relation"]]
        assert not train.intersection(zip(group.subject, group.object))
    payload_path = source / "lre-frequency-site/data/curves.json"
    payload = json.loads(payload_path.read_text())
    assert payload["meta"]["evaluatedFacts"] == len(facts)
    assert len(payload["curves"]) == 3 * facts.relation.nunique()
    for curve in payload["curves"]:
        column = AXES[curve["metric"]][0]
        group = facts[facts.relation == curve["relation"]]
        actual = group.groupby(column).faithful.agg(["size", "sum"])
        expected = pd.DataFrame(curve["raw"]).set_index("frequency").sort_index()
        assert actual.index.tolist() == expected.index.tolist()
        assert actual["size"].tolist() == expected.nFacts.tolist()
        assert actual["sum"].tolist() == expected.nFaithful.tolist()
    columns = ["relation", "subject", "object", "subject_qid",
               "lm_prediction_token_id", "lre_prediction_token_id", "faithful",
               "layer", "beta"] + [c for c, _ in AXES.values()]
    DATA.mkdir(parents=True, exist_ok=True)
    facts[columns].to_csv(DATA / "facts.csv", index=False)
    paths = [run / "predictions.csv", run / "frequency/fact_counts.csv",
             run / "manifest.json", run / "frequency/manifest.json",
             run / "contexts/manifest.json", source / "work/lment-lre-v2/manifest.json",
             source / "scripts/build_lment_lre_curves.py",
             source / "scripts/count_lment_lre_frequencies.py", payload_path] + result_paths
    evaluation_code = source.parent / "lment-count-analysis/scripts/evaluate_lre_faithfulness.py"
    if evaluation_code.exists():
        paths.append(evaluation_code)
    provenance = {
        "source_root": str(source.resolve()),
        "input_sha256": {str(p.relative_to(source)) if p.is_relative_to(source)
                         else str(p): sha256(p) for p in paths},
        "snapshot_sha256": sha256(DATA / "facts.csv"),
        "evaluation_manifest": manifest,
        "dataset_manifest": json.loads((source / "work/lment-lre-v2/manifest.json").read_text()),
        "count_manifest": json.loads((run / "frequency/manifest.json").read_text()),
        "context_manifest": json.loads((run / "contexts/manifest.json").read_text()),
        "checks": ["one-to-one count join", "exact match to all 174 raw browser curves",
                   "no estimation/evaluation fact overlap", "evaluation totals reproduced"],
    }
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def contrasts(facts, minimum_tail=10):
    records = []
    for metric, (column, _) in AXES.items():
        for relation, group in facts.groupby("relation", sort=True):
            q1, q3 = group[column].quantile([0.25, 0.75])
            low, high = group[group[column] <= q1], group[group[column] >= q3]
            eligible = (q1 < q3 and min(len(low), len(high)) >= minimum_tail
                        and group.lm_prediction_token_id.nunique() >= 2)
            records.append({
                "metric": metric, "relation": relation, "n": len(group),
                "n_tokens": group.lm_prediction_token_id.nunique(),
                "q25": q1, "q75": q3, "n_low": len(low), "n_high": len(high),
                "correct_low": int(low.faithful.sum()),
                "correct_high": int(high.faithful.sum()),
                "low_frequency_max": int(low[column].max()),
                "high_frequency_min": int(high[column].min()),
                "faithfulness_low": low.faithful.mean(),
                "faithfulness_high": high.faithful.mean(),
                "delta_pp": 100 * (high.faithful.mean() - low.faithful.mean()),
                "eligible": eligible,
            })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.source_root:
        snapshot(args.source_root)
    provenance = json.loads((DATA / "provenance.json").read_text())
    assert sha256(DATA / "facts.csv") == provenance["snapshot_sha256"]
    facts = pd.read_csv(DATA / "facts.csv")
    check_facts(facts)
    contrast = contrasts(facts)
    contrast.to_csv(DATA / "relation_contrasts.csv", index=False)
    eligible = contrast[contrast.eligible]
    bins = [(0, 0, "0"), (1, 9, "1–9"), (10, 99, "10–99"),
            (100, 999, "100–999"), (1000, np.inf, "≥1,000")]
    records, stats = [], []
    for metric, (column, _) in AXES.items():
        for lower, upper, label in bins:
            group = facts[facts[column].between(lower, upper)]
            records.append(dict(metric=metric, bin=label, n=len(group),
                                correct=int(group.faithful.sum()),
                                faithfulness=group.faithful.mean(),
                                relations=group.relation.nunique()))
        group = eligible[eligible.metric == metric]
        # Pearson correlation of midranks equals Spearman's rho, including ties.
        rho = facts[column].rank().corr(facts.faithful.rank())
        stats.append(dict(metric=metric, pooled_spearman_rho=rho,
                          eligible_relations=len(group),
                          decreasing=int((group.delta_pp < -1e-10).sum()),
                          equal=int((group.delta_pp.abs() <= 1e-10).sum()),
                          increasing=int((group.delta_pp > 1e-10).sum()),
                          mean_delta_pp=group.delta_pp.mean()))
    binned = pd.DataFrame(records)
    binned.to_csv(DATA / "binned_faithfulness.csv", index=False)
    sensitivity = []
    for minimum in [5, 10, 20]:
        frame = contrasts(facts, minimum)
        for metric, group in frame[frame.eligible].groupby("metric"):
            sensitivity.append(dict(minimum_tail=minimum, metric=metric,
                                    eligible=len(group),
                                    decreasing=int((group.delta_pp < -1e-10).sum()),
                                    equal=int((group.delta_pp.abs() <= 1e-10).sum()),
                                    increasing=int((group.delta_pp > 1e-10).sum()),
                                    mean_delta_pp=group.delta_pp.mean()))
    pd.DataFrame(sensitivity).to_csv(DATA / "tail_size_sensitivity.csv", index=False)
    token_counts = facts.groupby("relation").lm_prediction_token_id.nunique()
    variable = facts[~facts.relation.isin(token_counts[token_counts == 1].index)]
    relation_means = facts.groupby("relation").agg(
        n=("faithful", "size"), faithfulness=("faithful", "mean"),
        mean_frequency=("subject_object_document_count", "mean"),
        n_tokens=("lm_prediction_token_id", "nunique"),
    )
    relation_means.to_csv(DATA / "relation_means.csv")
    relation_correlations = {}
    for name, group in {
        "all": relation_means,
        "variable_tokens": relation_means[relation_means.n_tokens >= 2],
        "variable_tokens_n_ge_20": relation_means[
            (relation_means.n_tokens >= 2) & (relation_means.n >= 20)],
    }.items():
        positive = group[group.mean_frequency > 0]
        relation_correlations[name] = {
            "n_relations": len(group),
            "pearson_log1p": np.log1p(group.mean_frequency).corr(group.faithfulness),
            "n_positive_frequency_relations": len(positive),
            "pearson_log_positive": np.log(positive.mean_frequency).corr(positive.faithfulness),
        }
    summary = dict(n_facts=len(facts), n_faithful=int(facts.faithful.sum()),
                   relations=facts.relation.nunique(),
                   micro_faithfulness=facts.faithful.mean(),
                   macro_faithfulness=facts.groupby("relation").faithful.mean().mean(),
                   single_token_relations=token_counts[token_counts == 1].index.tolist(),
                   variable_token_facts=len(variable),
                   variable_token_faithfulness=variable.faithful.mean(),
                   variable_token_rho={metric: variable[column].rank().corr(variable.faithful.rank())
                                       for metric, (column, _) in AXES.items()},
                   relation_level_correlations=relation_correlations,
                   metrics=stats)
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Generate the manuscript tables from the same data as the plots.
    table = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Frequency axis & Spearman $\rho$ & Lower & Equal & Higher \\", r"\midrule"]
    for row in stats:
        table.append(f"{AXES[row['metric']][1]} & ${row['pooled_spearman_rho']:.3f}$ & "
                     f"{row['decreasing']} & {row['equal']} & {row['increasing']} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (DATA / "aggregate_table.tex").write_text("\n".join(table) + "\n")
    table = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Relation & Low count & Faithful / $n$ & High count & Faithful / $n$ \\",
             r"\midrule"]
    for relation in EXAMPLES:
        row = eligible[(eligible.metric == "subject-object") & (eligible.relation == relation)].iloc[0]
        table.append(f"{relation.capitalize()} & $\\leq {row.low_frequency_max}$ & "
                     f"{row.correct_low}/{row.n_low} & $\\geq {row.high_frequency_min}$ & "
                     f"{row.correct_high}/{row.n_high} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (DATA / "examples_table.tex").write_text("\n".join(table) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), layout="constrained")
    colors = ["#176B8A", "#B84A3A", "#807143"]
    # Panel A uses actual held-out interventions; panel B retains the historical
    # faithfulness analysis. The two cohorts must not be treated as identical.
    causal_data = ROOT / "data/lment_causality"
    causal_provenance = json.loads((causal_data / "provenance.json").read_text())
    assert sha256(causal_data / "relation_means.csv") == causal_provenance["snapshot_sha256"]
    group = pd.read_csv(causal_data / "relation_means.csv")
    assert len(group) == group.relation_id.nunique() == 56
    assert group.value.between(0, 1).all() and group.mean_count.ge(0).all()
    assert (group.metric == "causal_pairwise_success").all()
    correlation = np.log1p(group.mean_count).corr(group.value)
    axes[0].scatter(1 + group.mean_count, group.value,
                    color=colors[0], s=20, alpha=.8, edgecolors="white", linewidths=.3)
    axes[0].set(xscale="log", ylim=(0, 1), yticks=np.linspace(0, 1, 6),
                ylabel="Causality (pairwise intervention success)",
                xlabel="1 + mean subject–object document count")
    axes[0].text(.97, .95, f"{len(group)} relations; Pearson r = {correlation:.3f}",
                 transform=axes[0].transAxes, fontsize=7, ha="right", va="top")
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].set_title("A  Frequency and causality across relations", loc="left", fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)
    rng = np.random.default_rng(42)
    for x, (metric, (_, label)) in enumerate(AXES.items()):
        values = eligible[eligible.metric == metric].delta_pp.to_numpy()
        jitter = rng.uniform(-0.19, 0.19, len(values))
        axes[1].scatter(x + jitter, values, s=18, alpha=.8, linewidths=.3,
                        edgecolors="white", c=np.where(values < 0, colors[1], colors[0]))
        axes[1].scatter(x, values.mean(), marker="D", c="black", s=25, zorder=4)
        row = stats[x]
        axes[1].text(x, 47, f"{row['decreasing']}↓  {row['equal']}=  {row['increasing']}↑",
                     ha="center", fontsize=7)
    axes[1].axhline(0, color="#777777", lw=.8, ls="--")
    axes[1].set(xticks=range(3), xticklabels=["Subject", "Subject\n+ object", "Subject\n+ cue"],
                ylim=(-45, 53), yticks=[-40, -20, 0, 20, 40],
                ylabel="Upper − lower quartile (percentage points)",
                xlabel="One point per eligible relation; ◆ mean")
    axes[1].set_title("B  Frequency differences within relations", loc="left", fontsize=9)
    axes[1].grid(axis="y", alpha=.2)
    (ROOT / "figures").mkdir(exist_ok=True)
    for extension in ["pdf", "png"]:
        fig.savefig(ROOT / f"figures/lment_frequency_aggregate.{extension}", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

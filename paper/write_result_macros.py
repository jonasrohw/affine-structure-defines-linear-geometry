"""Generate numerical manuscript prose directly from fresh measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from factual_recall.paper_audit import experiment1_summary


def write(results):
    def load(name):
        return json.loads((results / name).read_text())
    sweep = load('experiment1/sweep.json')
    exp1 = experiment1_summary(sweep)
    linear = load('experiment1/linearity.json')['aggregate']['semantic']['full_lookup']
    mech = load('experiment1/mechanism.json')['aggregate']['goal1']
    data = load('experiment2/analysis.json')
    rows, summary = data['seeds'], data['summary']
    macros = {}
    def number(name, value, digits=3):
        macros[name] = f'{value:.{digits}f}'
    def stats(name, values, digits=3):
        macros[name] = f'{values["mean"]:.{digits}f}\\pm{values["sd"]:.{digits}f}'
    for repeats, tag in [(2, 'Low'), (24, 'High')]:
        values = [r['relations']['semantic']['probe_accuracy'] for r in sweep
                  if r['config']['attr_repeats'] == repeats and r['config']['n_triples'] == 128]
        number('ExpOneTransition' + tag, np.mean(values), 2)
    number('ExpTwoAttribute', summary['attribute_accuracy']['mean'])
    for relation, tag in [('semantic', 'Semantic'), ('random', 'Random')]:
        for metric in ('probe', 'trained', 'heldout'):
            value = exp1['16'][relation][metric]
            number('ExpOne' + tag + metric.title(), value['mean'])
            stats('ExpOne' + tag + metric.title() + 'Stats', value)
    for name, section, key in [('Affine', 'affine_reconstruction', 'r2'),
                                ('Midpoint', 'midpoint', 'relative_frobenius_defect'),
                                ('Jacobian', 'jacobian', 'pooled_relative_dispersion'),
                                ('Tangent', 'jacobian', 'tangent_pooled_relative_dispersion')]:
        stats('ExpOne' + name, linear[section][key])
    for name, key in [('ValuePatch', 'attention_values_patch_recovery'),
                      ('PatternPatch', 'attention_patterns_patch_recovery'),
                      ('HeadsPresent', 'all_heads_present_accuracy')]:
        stats('ExpOne' + name, mech[key])
    stats('ExpOneRelationPatch', mech['relation_gate']['all_mlp_patch_recovery'])
    for name, key in [('TopOnly', 'with_only_top_k_variable'), ('TopAblated', 'after_top_k_ablation')]:
        stats('ExpOne' + name, mech['neuron_subject_specific_margin_fraction']['32'][key])
    for name, key in [('MinStructured', 'min_structured_accuracy'), ('Random', 'arbitrary_accuracy'),
                      ('Symmetry', 'min_symmetry_agreement'), ('StructuredRsq', 'min_structured_full_r2'),
                      ('RandomRsq', 'arbitrary_full_r2')]:
        number('ExpTwo' + name, summary[key]['mean'], 4 if name in ('MinStructured', 'Random', 'Symmetry') else 3)
        stats('ExpTwo' + name + 'Stats', summary[key], 5)
    structured = [fit['full_r2'] for row in rows for fit in row['raw_embedding_ols']['structured']]
    number('ExpTwoStructuredMeanRsq', np.mean(structured), 5)
    number('ExpTwoStructuredLowestRsq', min(structured), 5)
    def sequence(name, values, digits=3):
        formatted = [f'{v:.{digits}f}' for v in values]
        macros[name] = ', '.join(formatted[:-1]) + ', and ' + formatted[-1] if len(formatted)>1 else formatted[0]
    sequence('ExpTwoRandomRsqSeeds', [r['raw_embedding_ols']['arbitrary']['full_r2'] for r in rows])
    sequence('ExpTwoRandomErrors', [r['raw_embedding_ols']['arbitrary']['full_relative_error'] for r in rows])
    sequence('ExpTwoNullPercentiles', [r['permutation_null']['observed_percentile'] for r in rows], 1)
    number('ExpTwoNullMean', np.mean([r['permutation_null']['mean'] for r in rows]))
    for mode, tag in [('full', 'Full'), ('gelu_identity', 'Gelu'), ('uniform_attention', 'Uniform'),
                      ('no_mlp', 'NoMlp'), ('no_attention', 'NoAttention')]:
        for group, prefix in [('arbitrary', 'Random'), ('structured_mean', 'Structured')]:
            val = summary['component_ablations'][mode][group]
            stats('ExpTwo' + prefix + tag, val)
            stats('ExpTwo' + prefix + tag + 'Precise', val, 5)
    # Emit the complete curves rather than cherry-picking individual layers.
    lens = np.mean([r['logit_lens']['arbitrary'] for r in rows], axis=0)
    sequence('ExpTwoLens', lens)
    macros['ExpTwoLens'] = macros['ExpTwoLens'].rsplit(', and ', 1)[0] + f', and {lens[-1]:.4f}'
    for component, tag in [('attention', 'Attention'), ('mlp', 'Mlp')]:
        curve = np.mean([[v['arbitrary'] for v in r['layer_ablations'][component]] for r in rows], axis=0)
        macros['ExpTwo' + tag + 'Range'] = f'{min(curve):.3f}--{max(curve):.3f}'
    for position, tag in [('subject', 'Subject'), ('object', 'Object')]:
        curve = np.mean([r['causal_trace']['arbitrary'][position] for r in rows], axis=0)
        sequence('ExpTwo' + tag + 'Trace', curve)
    path = results / 'paper/result_values.tex'
    path.write_text('% Generated from this run; do not edit numerical values by hand.\n' +
                    ''.join('\\newcommand{\\' + key + '}{' + value + '}\n' for key, value in macros.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    write(parser.parse_args().results)

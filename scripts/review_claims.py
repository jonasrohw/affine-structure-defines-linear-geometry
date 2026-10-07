"""Record how adopting a new measured baseline affects the paper's claims."""
import argparse
import json
from pathlib import Path
import numpy as np
from factual_recall.paper_audit import experiment1_summary


def review(results, historical):
    current = json.loads((results/'experiment2/analysis.json').read_text())
    old = json.loads(historical.read_text())
    rows = current['seeds']
    def mechanism(data):
        seeds = data['seeds']
        return {
            'random_r2_mean': data['summary']['arbitrary_full_r2']['mean'],
            'null_percentiles': {str(r['seed']):r['permutation_null']['observed_percentile'] for r in seeds},
            'random_logit_lens_mean': np.mean([r['logit_lens']['arbitrary'] for r in seeds],axis=0).tolist(),
            'random_component_accuracy': {mode: data['summary']['component_ablations'][mode]['arbitrary']
                for mode in ('full','gelu_identity','uniform_attention','no_mlp','no_attention')},
            'random_layer_ablations': {component: np.mean([[v['arbitrary'] for v in r['layer_ablations'][component]]
                for r in seeds],axis=0).tolist() for component in ('attention','mlp')},
            'random_trace': {pos:np.mean([r['causal_trace']['arbitrary'][pos] for r in seeds],axis=0).tolist()
                             for pos in ('subject','object')},
        }
    first = experiment1_summary(json.loads((results/'experiment1/sweep.json').read_text()))
    linear = json.loads((results/'experiment1/linearity.json').read_text())['aggregate']['semantic']['full_lookup']
    checks = {
        'exp1_affine_transfer_perfect_at_frequency_one': first['1']['semantic']['probe']['mean'] == 1.,
        'exp1_semantic_heldout_recall_above_half': first['16']['semantic']['heldout']['mean'] > .5,
        'exp1_random_training_recall_perfect_at_frequency_64': first['64']['random']['trained']['mean'] == 1.,
        'exp1_lookup_approximately_affine_but_positive_defect_and_jacobian_dispersion': (
            linear['affine_reconstruction']['r2']['min'] > .95
            and linear['midpoint']['relative_frobenius_defect']['min'] > 0
            and linear['jacobian']['pooled_relative_dispersion']['min'] > 0),
        'all_behavior_symmetry_geometry_gates_pass': all(r['passes_joint_gate'] for r in rows),
        'each_structured_fit_explains_over_95_percent': all(r['raw_embedding_ols']['min_structured_full_r2']>=.95 for r in rows),
        'each_random_fit_explains_less_than_half': all(r['raw_embedding_ols']['arbitrary']['full_r2']<.5 for r in rows),
        'random_cv_r2_negative_every_seed': all(r['raw_embedding_ols']['arbitrary']['cv_r2']<0 for r in rows),
        'each_global_intervention_harms_both_relation_types_every_seed': all(
            r['component_ablations'][mode][group] < r['component_ablations']['full'][group]
            for r in rows for mode in ('gelu_identity','uniform_attention','no_mlp','no_attention')
            for group in ('arbitrary','structured_mean')),
        'each_layer_ablation_harms_random_recall_every_seed': all(
            v['arbitrary'] < r['behavior']['arbitrary_accuracy'] for r in rows
            for component in ('attention','mlp') for v in r['layer_ablations'][component]),
    }
    minimum = min(r['raw_embedding_ols']['min_structured_full_r2'] for r in rows)
    revisions = [
        {'claim': 'Seeds 31 and 47 exceed the 98th permutation-null percentile',
         'replacement': 'Report each measured percentile explicitly; the historical seed-specific statement does not hold.',
         'percentiles': {str(r['seed']): r['permutation_null']['observed_percentile'] for r in rows}},
        {'claim': 'Every structured fit has R-squared at least .995',
         'replacement': 'Report the measured minimum, correcting an inequality already overstated in the historical prose.',
         'measured_minimum': minimum},
        {'claim': 'Exact layer-localization and patch-recovery values describe the new runs',
         'replacement': 'Replace all curves and prose values; layer localization is specific to the reported solutions.'},
    ]
    return {'core_claim_checks': checks, 'claim_revisions': revisions,
            'historical':mechanism(old), 'rtx':mechanism(current),
            'method_change':'Joint checkpoint selection and early stopping now include the existing symmetry threshold.',
            'interpretation':'Numerical null percentiles and layer/patch curves are solution-specific. '
                             'Check core_claim_checks before retaining each qualitative claim; do not infer cross-hardware invariance.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--historical',type=Path,default=Path('paper/reference/experiment2.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=review(args.results,args.historical)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['core_claim_checks'],indent=2))

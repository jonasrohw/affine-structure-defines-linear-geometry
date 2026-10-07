#!/usr/bin/env python3
"""Compare two fresh full runs; write evidence, never automatically adopt a baseline."""
import argparse
import hashlib
import json
from pathlib import Path
from factual_recall.paper_audit import compare_snapshots, measurement_snapshot
from factual_recall.reproduce import source_fingerprint


def verify(first, second, destination):
    if destination.resolve() in (first.resolve(), second.resolve()):
        raise ValueError('Write verification evidence outside both run directories')
    if first.resolve() == second.resolve():
        raise ValueError('Two distinct fresh run directories are required')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (first, second)]
    for run, m in zip((first, second), manifests):
        assert (run/'paper/main.pdf').is_file(), 'The full run must finish PDF generation'
        for name, expected_hash in m['artifacts'].items():
            assert hashlib.sha256((run/name).read_bytes()).hexdigest() == expected_hash, name
        if m['status'] != 'completed':
            assert m['reference'] == 'historical' and m['status'] == 'failed'
            assert m['error'].startswith('Measurements differ from the selected paper baseline.')
        assert m['stage'] == 'all' and not m['smoke'] and m['seeds'] == [19, 31, 47]
        assert '--resume' not in m['argv'], 'Certification requires fresh training'
        assert m['deterministic_algorithms'] and m['device']['experiment2'] == 'cuda'
        assert m['runtime']['gpu'] == 'NVIDIA GeForce RTX 4090'
    assert manifests[0]['source_fingerprint'] == manifests[1]['source_fingerprint']
    assert manifests[0]['runtime'] == manifests[1]['runtime']
    assert manifests[0]['packages'] == manifests[1]['packages']
    snapshots = [measurement_snapshot(p) for p in (first, second)]
    for snapshot in snapshots:
        assert set(snapshot['joint_gates']) == {'19', '31', '47'}
        assert len({key.split('/')[1] for key in snapshot['tensors']}) == 3
    checks = compare_snapshots(*snapshots)
    mismatches = [c for c in checks if not c['matches']]
    passed = not mismatches and all(snapshots[0]['joint_gates'].values())
    destination.mkdir(parents=True, exist_ok=True)
    report = {'all_match': passed, 'checked': len(checks), 'mismatches': mismatches,
              'counts': {key: len(value) for key, value in snapshots[0].items()},
              'source_fingerprint': manifests[0]['source_fingerprint'],
              'verification_source_fingerprint': source_fingerprint(),
              'excluded_operational_field': 'elapsed_sec (wall-clock duration)',
              'runtime': manifests[0]['runtime'], 'packages': manifests[0]['packages'],
              'runs': [{'path': str(p), 'manifest_sha256': hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),
                        'elapsed_seconds': m['elapsed_seconds']} for p,m in zip((first,second), manifests)],
              'scope': 'Two independent full fresh runs, exact numerical results and tensor contents. '
                       'Certified only for this CPU/GPU, driver, locked runtime and source; no cross-hardware guarantee.'}
    (destination/'repeatability.json').write_text(json.dumps(report, indent=2)+'\n')
    if not passed:
        raise SystemExit(f'Repeatability failed: {len(mismatches)} differing outputs, gates {snapshots[0]["joint_gates"]}')
    (destination/'measurements.json').write_text(json.dumps(snapshots[0], indent=2, sort_keys=True)+'\n')
    print(f'Exact match: {report["counts"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.first, args.second, args.output)

#!/usr/bin/env python3
"""Run the paper container offline on the certified RTX host (stdlib only)."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command(image, arguments, source=False, python=False):
    if 'AMD Ryzen 9 7950X3D 16-Core Processor' not in Path('/proc/cpuinfo').read_text():
        raise SystemExit('Experiment 1 certification requires the Ryzen 9 7950X3D CPU.')
    driver = Path('/proc/driver/nvidia/version').read_text()
    if '610.57.04' not in driver:
        raise SystemExit('The RTX baseline requires NVIDIA driver 610.57.04.')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    if gpu != 'NVIDIA GeForce RTX 4090':
        raise SystemExit('The RTX baseline requires a single NVIDIA GeForce RTX 4090.')
    cmd = ['docker', 'run', '--rm', '--network', 'none']
    for name in ('nvidia0', 'nvidiactl', 'nvidia-uvm', 'nvidia-uvm-tools'):
        cmd += ['--device', '/dev/' + name]
    # Works without changing Docker's daemon or installing the NVIDIA toolkit.
    libraries = subprocess.check_output(['ldconfig', '-p'], text=True)
    for name in ('libcuda.so.1', 'libnvidia-ptxjitcompiler.so.1', 'libnvidia-nvvm.so.4'):
        direct = Path('/usr/lib') / name
        candidates = [line.split(' => ')[1] for line in libraries.splitlines()
                      if line.strip().startswith(name + ' ') and 'x86-64' in line]
        path = direct.resolve() if direct.exists() else Path(candidates[0]).resolve()
        cmd += ['-v', f'{path}:/usr/local/nvidia/lib64/{name}:ro']
    cmd += ['-e', 'LD_LIBRARY_PATH=/usr/local/nvidia/lib64',
            '-v', f'{ROOT / "results"}:/app/results']
    if source:
        for name in ('experiments', 'paper', 'scripts'):
            cmd += ['-v', f'{ROOT / name}:/app/{name}:ro']
    if python:
        cmd += ['--entrypoint', '/opt/beta-venv/bin/python']
    cmd += [image]
    if not python:
        cmd += ['--experiment2-device', 'cuda', '--runtime-profile', 'deterministic']
    return cmd + arguments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='factual-recall:rtx')
    parser.add_argument('--source', action='store_true', help='Diagnostic: mount current source read-only')
    parser.add_argument('--python', action='store_true', help='Run a Python command inside the same runtime')
    args, rest = parser.parse_known_args()
    (ROOT / 'results').mkdir(exist_ok=True)
    cmd = command(args.image, rest, args.source, args.python)
    print(json.dumps(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))

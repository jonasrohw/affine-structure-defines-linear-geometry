"""Train, measure, plot, and audit the current manuscript from one command."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

# Fix BLAS and OpenMP before importing the numerical libraries, including in workers.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("SOURCE_DATE_EPOCH", "1791244800")

import torch

from .config import PROJECT_ROOT
from .linear_shift import LinearShiftConfig, run_single
from . import linear_shift_mechanistic_analysis as mechanism
from . import linear_shift_linearity_analysis as linearity
from . import symmetric_lookup_transformer_raw_analysis as lookup_analysis
from .symmetric_lookup_transformer import TransformerLookupConfig
from .symmetric_lookup_transformer_raw_search import (
    train_joint_geometry,
    train_structured_geometry,
)

SEEDS = (19, 31, 47)
ATTRIBUTE_REPEATS = (2, 4, 8, 16, 24)


def configure_runtime():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=mechanism._jsonable) + "\n")
    temporary.replace(path)


def source_fingerprint():
    paths = [PROJECT_ROOT / "uv.lock", PROJECT_ROOT / ".python-version"]
    paths += sorted((PROJECT_ROOT / "experiments/factual_recall").glob("*.py"))
    payload = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in paths}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cached_stage(path, signature, operation, resume):
    """Reuse a stage only when inputs AND all its saved artifacts still match."""
    path = Path(path)
    record = path.with_suffix(".complete.json")
    signature = json.loads(json.dumps(signature, default=str))
    if resume and record.exists():
        saved = json.loads(record.read_text())
        if saved["signature"] == signature and all(
            (path.parent / name).is_file() and sha256(path.parent / name) == digest
            for name, digest in saved["artifacts"].items()
        ):
            print(f"Reusing {path}", flush=True)
            return json.loads(path.read_text())
    # Remove the completion marker before running so an interrupted stage cannot pass.
    record.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = operation()
    artifacts = {}
    state = result.pop("_state_dict", None) if isinstance(result, dict) else None
    if state is not None:
        checkpoint = path.with_suffix(".pt")
        torch.save(state, checkpoint)
        artifacts[checkpoint.name] = sha256(checkpoint)
    write_json(path, result)
    artifacts[path.name] = sha256(path)
    write_json(record, {"signature": signature, "artifacts": artifacts})
    return result


def experiment1(output, smoke, resume, fingerprint):
    seeds = SEEDS[:1] if smoke else SEEDS
    device = torch.device("cpu")
    rows = []
    configs = [LinearShiftConfig(seed=seed, occurrences=f) for seed in seeds
               for f in (1, 4, 16, 64)]
    configs += [LinearShiftConfig(seed=seed, occurrences=1, attr_repeats=a, n_triples=k)
                for seed in seeds for a in ATTRIBUTE_REPEATS for k in (0, 128)]
    for cfg in configs:
        cfg.output_dir = output / "experiment1"
        name = f"seed{cfg.seed}_f{cfg.occurrences}_a{cfg.attr_repeats}_k{cfg.n_triples}"
        print(f"Experiment 1: {name}", flush=True)
        row = cached_stage(cfg.output_dir / f"{name}.json",
                           {"source": fingerprint, "config": asdict(cfg)},
                           lambda: run_single(cfg, device), resume)
        rows.append(row)
    write_json(output / "experiment1/sweep.json", rows)
    # Even smoke mode exercises interventions and automatic differentiation on F=16.
    runs, linear_runs = [], []
    for seed in seeds:
        print(f"Experiment 1: interventions and Jacobians, seed {seed}", flush=True)
        runs.append(cached_stage(output / f"experiment1/mechanism_seed{seed}.json",
                    {"source": fingerprint, "seed": seed},
                    lambda: mechanism.analyze_goal_seed("goal1", seed, device), resume))
        linear_runs.append(cached_stage(output / f"experiment1/linearity_seed{seed}.json",
                    {"source": fingerprint, "seed": seed, "smoke": smoke},
                    lambda: linearity.analyze_seed(seed, device, relation="semantic", occurrences=16,
                                                  jacobian_subject_limit=1 if smoke else None), resume))
    result = {"runs": runs, "aggregate": mechanism.aggregate_results(runs)}
    write_json(output / "experiment1/mechanism.json", result)
    linear_result = {"runs": linear_runs, "aggregate": linearity.aggregate_runs(linear_runs)}
    write_json(output / "experiment1/linearity.json", linear_result)
    return rows, result, linear_result


def experiment2_seed(arguments):
    output, seed, smoke, resume, fingerprint = arguments
    configure_runtime()
    device = torch.device("cpu")
    directory = output / "experiment2" / f"seed{seed}"
    cfg = TransformerLookupConfig(seed=seed, output_dir=directory)
    if smoke:
        cfg = replace(cfg, phase1_steps=2, joint_steps=2, eval_every=2, batch_size=32,
                      attribute_batch_size=32)
    signature = {"source": fingerprint, "config": asdict(cfg), "reverse_attribute_weight": 4.0}
    structured = directory / "structured.json"
    cached_stage(structured, signature,
                 lambda: train_structured_geometry(cfg, device, 4.0), resume)
    previous = structured.with_suffix(".pt")
    history = []
    total_steps = 0
    schedules = [("joint", cfg.joint_steps, 1e-3, cfg.eval_every),
                 ("continuation", 2 if smoke else 200, 3e-4, 2 if smoke else 100)]
    if seed in (19, 31):
        schedules.append(("continuation2", 2 if smoke else 100, 1e-4, 2 if smoke else 100))
    result = None
    for name, steps, lr, interval in schedules:
        stage_cfg = replace(cfg, joint_steps=steps, learning_rate=lr, eval_every=interval)
        stage_path = directory / f"{name}.json"
        print(f"Experiment 2: seed {seed}, {name}, {steps} steps", flush=True)
        result = cached_stage(stage_path,
                    {"source": fingerprint, "config": asdict(stage_cfg),
                     "checkpoint_sha256": sha256(previous), "reverse_attribute_weight": 4.0},
                    lambda: train_joint_geometry(stage_cfg, previous, device, 4.0), resume)
        for row in result["history"]:
            history.append({**row, "step": row["step"] + total_steps})
        total_steps = history[-1]["step"]
        previous = stage_path.with_suffix(".pt")
    analysis = cached_stage(directory / "analysis.json",
                {"source": fingerprint, "checkpoint_sha256": sha256(previous), "smoke": smoke},
                lambda: lookup_analysis.analyze_seed(seed, device, 2 if smoke else 500, previous), resume)
    analysis["passes_joint_gate"] = result["passes_joint_gate"]
    return analysis, history


def experiment2(output, smoke, resume, fingerprint, workers):
    seeds = SEEDS[:1] if smoke else SEEDS
    arguments = [(output, seed, smoke, resume, fingerprint) for seed in seeds]
    if workers == 1 or len(seeds) == 1:
        values = [experiment2_seed(arg) for arg in arguments]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            values = list(pool.map(experiment2_seed, arguments))
    rows = [value[0] for value in values]
    histories = {seed: value[1] for seed, value in zip(seeds, values, strict=True)}
    result = {"seeds": rows, "summary": lookup_analysis.summarize(rows), "histories": histories}
    write_json(output / "experiment2/analysis.json", result)
    return result


def build_paper(output, smoke, compile_pdf):
    paper = output / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.tex", "*.sty", "*.bst", "*.bib"):
        for source in (PROJECT_ROOT / "paper").glob(pattern):
            shutil.copy2(source, paper / source.name)
    commands = [
        ["paper/analyze_lment_frequency.py", "--output-dir", str(paper)],
        ["paper/make_dataset_figure.py", "--output-dir", str(paper / "figures")],
    ]
    commands.append(["paper/make_paper_figures.py", "--results", str(output)])
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=True)
    if compile_pdf:
        if shutil.which("latexmk") is None:
            raise RuntimeError("PDF compilation requires latexmk and TeX Live; use Docker or --no-pdf.")
        with (paper / "build.log").open("w") as log:
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                           cwd=paper, check=True, stdout=log, stderr=subprocess.STDOUT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Exercise the pipeline; not a paper reproduction")
    parser.add_argument("--resume", action="store_true", help="Reuse stages with verified inputs and artifact hashes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=3, help="Independent CPU seed processes (each uses one thread)")
    parser.add_argument("--stage", choices=("all", "experiment1", "experiment2", "paper"), default="all")
    parser.add_argument("--no-pdf", action="store_true", help="Generate data, figures and tables without LaTeX")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = (args.output or PROJECT_ROOT / "results" / ("smoke" if args.smoke else "full")).resolve()
    if output == PROJECT_ROOT or output == PROJECT_ROOT / "paper":
        parser.error("--output must be a separate generated-output directory")
    configure_runtime()
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    start = time.monotonic()
    manifest = {
        "status": "running", "smoke": args.smoke, "stage": args.stage,
        "source_fingerprint": fingerprint, "lock_sha256": sha256(PROJECT_ROOT / "uv.lock"),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "torch": torch.__version__, "torch_build": torch.__config__.show(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "device": "cpu", "threads_per_worker": 1, "workers": args.workers,
        "deterministic_algorithms": True, "seeds": list(SEEDS[:1] if args.smoke else SEEDS),
        "argv": sys.argv,
    }
    manifest_path = output / ("manifest.json" if args.stage == "all" else f"manifest_{args.stage}.json")
    write_json(manifest_path, manifest)
    try:
        if args.stage in ("all", "experiment1"):
            experiment1(output, args.smoke, args.resume, fingerprint)
        if args.stage in ("all", "experiment2"):
            experiment2(output, args.smoke, args.resume, fingerprint, args.workers)
        if args.stage in ("all", "paper"):
            build_paper(output, args.smoke, not args.no_pdf)
            if not args.smoke:
                from .paper_audit import audit
                report = audit(output)
                write_json(output / "audit.json", report)
                if not report["all_match"]:
                    raise RuntimeError("Measurements differ from the archived paper. See audit.json; generated results were preserved.")
        manifest["status"] = "smoke_passed" if args.smoke else "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - start
        manifest["artifacts"] = {str(p.relative_to(output)): sha256(p)
                                 for p in sorted(output.rglob("*"))
                                 if p.is_file() and not p.name.startswith("manifest")}
        write_json(manifest_path, manifest)
    print(f"{manifest['status']}: {output}", flush=True)


if __name__ == "__main__":
    main()

"""Train, measure, plot, and audit the experiments from one command."""

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
import subprocess
import sys
import time

# Fix BLAS and OpenMP before importing the numerical libraries, including in workers.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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


def configure_runtime(profile="deterministic"):
    torch.set_num_threads(1)
    deterministic = profile == "deterministic"
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def experiment2_schedule(seed, smoke=False):
    """Command limits and evaluation intervals recovered from beta's stage logs.

    The original refinements allowed 8,000/3,000 steps and stopped at 200/100.
    Evaluation frequency changes checkpoint selection and the early-stop point.
    """
    if smoke:
        stages = [("joint", 2, 1e-3, 2), ("continuation", 2, 3e-4, 2)]
    else:
        stages = [("joint", 12000, 1e-3, 500 if seed == 19 else 1000),
                  ("continuation", 8000, 3e-4, 200)]
    if seed in (19, 31):
        stages.append(("continuation2", 2 if smoke else 3000, 1e-4, 2 if smoke else 100))
    return stages


def runtime_signature(device, profile):
    """Prevent resume from mixing checkpoints made by different runtimes."""
    beta_lock = Path("/opt/rtx-environment/uv.lock")
    if not beta_lock.exists():
        beta_lock = Path("/opt/beta-environment/uv.lock")
    return {"device": str(device), "profile": profile, "python": sys.version,
            "torch": torch.__version__, "torch_build": torch.__config__.show(),
            "cuda": torch.version.cuda, "machine": platform.machine(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "environment_lock_sha256": sha256(beta_lock if beta_lock.exists() else PROJECT_ROOT / "uv.lock"),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "nvidia_driver": (Path("/proc/driver/nvidia/version").read_text()
                              if device.type == "cuda" and Path("/proc/driver/nvidia/version").exists()
                              else None),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}


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
    runtime = runtime_signature(device, "deterministic" if torch.are_deterministic_algorithms_enabled() else "beta")
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
                           {"source": fingerprint, "config": asdict(cfg), "runtime": runtime},
                           lambda: run_single(cfg, device), resume)
        rows.append(row)
    write_json(output / "experiment1/sweep.json", rows)
    # Even smoke mode exercises interventions and automatic differentiation on F=16.
    runs, linear_runs = [], []
    for seed in seeds:
        print(f"Experiment 1: interventions and Jacobians, seed {seed}", flush=True)
        runs.append(cached_stage(output / f"experiment1/mechanism_seed{seed}.json",
                    {"source": fingerprint, "seed": seed, "runtime": runtime},
                    lambda: mechanism.analyze_goal_seed("goal1", seed, device), resume))
        linear_runs.append(cached_stage(output / f"experiment1/linearity_seed{seed}.json",
                    {"source": fingerprint, "seed": seed, "smoke": smoke, "runtime": runtime},
                    lambda: linearity.analyze_seed(seed, device, relation="semantic", occurrences=16,
                                                  jacobian_subject_limit=1 if smoke else None), resume))
    result = {"runs": runs, "aggregate": mechanism.aggregate_results(runs)}
    write_json(output / "experiment1/mechanism.json", result)
    linear_result = {"runs": linear_runs, "aggregate": linearity.aggregate_runs(linear_runs)}
    write_json(output / "experiment1/linearity.json", linear_result)
    return rows, result, linear_result


def experiment2_seed(arguments):
    output, seed, smoke, resume, fingerprint, device_name, profile = arguments
    configure_runtime(profile)
    device = torch.device(device_name)
    runtime = runtime_signature(device, profile)
    directory = output / "experiment2" / f"seed{seed}"
    cfg = TransformerLookupConfig(seed=seed, output_dir=directory,
                                  require_symmetry_for_stopping=profile != "beta")
    cfg = replace(cfg, joint_steps=cfg.phase1_steps)
    if smoke:
        cfg = replace(cfg, phase1_steps=2, joint_steps=2, eval_every=2, batch_size=32,
                      attribute_batch_size=32)
    signature = {"source": fingerprint, "config": asdict(cfg), "runtime": runtime,
                 "reverse_attribute_weight": 4.0}
    structured = directory / "structured.json"
    cached_stage(structured, signature,
                 lambda: train_structured_geometry(cfg, device, 4.0), resume)
    previous = structured.with_suffix(".pt")
    history = []
    total_steps = 0
    schedules = experiment2_schedule(seed, smoke)
    result = None
    for name, steps, lr, interval in schedules:
        stage_cfg = replace(cfg, phase1_steps=steps, joint_steps=steps,
                            learning_rate=lr, eval_every=interval)
        stage_path = directory / f"{name}.json"
        print(f"Experiment 2: seed {seed}, {name}, {steps} steps", flush=True)
        result = cached_stage(stage_path,
                    {"source": fingerprint, "config": asdict(stage_cfg), "runtime": runtime,
                     "checkpoint_sha256": sha256(previous), "reverse_attribute_weight": 4.0},
                    lambda: train_joint_geometry(stage_cfg, previous, device, 4.0), resume)
        for row in result["history"]:
            history.append({**row, "step": row["step"] + total_steps})
        total_steps = history[-1]["step"]
        previous = stage_path.with_suffix(".pt")
    analysis = cached_stage(directory / "analysis.json",
                {"source": fingerprint, "checkpoint_sha256": sha256(previous),
                 "smoke": smoke, "runtime": runtime},
                lambda: lookup_analysis.analyze_seed(seed, device, 2 if smoke else 500, previous), resume)
    analysis["passes_joint_gate"] = result["passes_joint_gate"]
    analysis["runtime"] = runtime
    return analysis, history


def experiment2(output, smoke, resume, fingerprint, workers, device="cpu", profile="deterministic"):
    seeds = SEEDS[:1] if smoke else SEEDS
    devices = ([f"cuda:{index % torch.cuda.device_count()}" for index in range(len(seeds))]
               if device == "cuda" else [device] * len(seeds))
    arguments = [(output, seed, smoke, resume, fingerprint, target, profile)
                 for seed, target in zip(seeds, devices, strict=True)]
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


def build_report(output):
    """Generate standalone results without manuscript source files."""
    paper = output / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    commands = [
        ["paper/analyze_lment_frequency.py", "--output-dir", str(paper)],
        ["paper/make_dataset_figure.py", "--output-dir", str(paper / "figures")],
        ["paper/make_paper_figures.py", "--results", str(output)],
        ["paper/write_result_macros.py", "--results", str(output)],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Exercise the pipeline; not a paper reproduction")
    parser.add_argument("--resume", action="store_true", help="Reuse stages with verified inputs and artifact hashes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=3, help="Independent seed processes (one numerical CPU thread each)")
    parser.add_argument("--experiment2-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--runtime-profile", choices=("deterministic", "beta"), default="deterministic",
                        help="beta preserves the recovered trainer's nondeterministic CUDA settings")
    parser.add_argument("--reference", choices=("current", "historical"), default="current",
                        help="Audit the certified RTX paper baseline or the untouched historical paper")
    parser.add_argument("--stage", choices=("all", "experiment1", "experiment2", "report", "paper"), default="all")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = (args.output or PROJECT_ROOT / "results" / ("smoke" if args.smoke else "full")).resolve()
    if output == PROJECT_ROOT or output == PROJECT_ROOT / "paper":
        parser.error("--output must be a separate generated-output directory")
    configure_runtime(args.runtime_profile)
    if args.experiment2_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use a CUDA PyTorch build and expose the GPU to Docker")
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    start = time.monotonic()
    manifest = {
        "status": "running", "smoke": args.smoke, "stage": args.stage,
        "source_fingerprint": fingerprint, "lock_sha256": sha256(PROJECT_ROOT / "uv.lock"),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "torch": torch.__version__, "torch_build": torch.__config__.show(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "device": {"experiment1": "cpu", "experiment2": args.experiment2_device},
        "threads_per_worker": 1, "workers": args.workers,
        "runtime": runtime_signature(torch.device(args.experiment2_device), args.runtime_profile),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "seeds": list(SEEDS[:1] if args.smoke else SEEDS),
        "argv": sys.argv,
        "reference": args.reference,
    }
    beta_lock = Path("/opt/beta-environment/uv.lock")
    if beta_lock.exists():
        manifest["beta_environment_lock_sha256"] = sha256(beta_lock)
    manifest_path = output / ("manifest.json" if args.stage == "all" else f"manifest_{args.stage}.json")
    write_json(manifest_path, manifest)
    try:
        if args.stage in ("all", "experiment1"):
            experiment1(output, args.smoke, args.resume, fingerprint)
        if args.stage in ("all", "experiment2"):
            experiment2(output, args.smoke, args.resume, fingerprint, args.workers,
                        args.experiment2_device, args.runtime_profile)
        if args.stage in ("all", "report", "paper"):
            build_report(output)
            if not args.smoke:
                from .paper_audit import audit
                report = audit(output, reference=args.reference)
                write_json(output / "audit.json", report)
                if not report["all_match"]:
                    raise RuntimeError("Measurements differ from the selected paper baseline. See audit.json; generated results were preserved.")
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

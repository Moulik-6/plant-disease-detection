#!/usr/bin/env python3
"""
Single-Image Latency Benchmarking Script for Plant Disease Classification Models.

Features:
  - Loads resnet50, efficientnet_v2, vit using load_model_and_classes
  - Sets PyTorch CPU thread count via --threads (default 4)
  - Uses torch.inference_mode()
  - 20 warm-up runs per model/ensemble
  - 10 interleaved rounds (resnet50, efficientnet_v2, vit, ensemble in turn,
    20 timed runs each, batch size 1) to eliminate drift bias
  - Measures ensemble as running all three models sequentially
  - Reports median, p10, and p90 per model and ensemble
  - Emits warning if any model's p90/p10 ratio exceeds 1.5 (noisy run)
  - Prints thread count, torch version, CPU name, and Windows active power plan
  - Saves results to ml/results/latency.json
"""

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import functools

# Ensure unbuffered, real-time logging
print = functools.partial(print, flush=True)

import numpy as np
import torch

# Force torch.load(weights_only=False) globally
_orig_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = safe_torch_load

# Setup sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "ml" else CURRENT_DIR

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from gradcam import load_model_and_classes
from src.utils import load_config


def resolve_path(rel_or_abs_path: str) -> str:
    """Resolves path relative to current working directory, project root, or ml dir."""
    if os.path.isabs(rel_or_abs_path) and os.path.exists(rel_or_abs_path):
        return rel_or_abs_path
    if os.path.exists(rel_or_abs_path):
        return os.path.abspath(rel_or_abs_path)

    cand_proj = os.path.join(PROJECT_ROOT, rel_or_abs_path)
    if os.path.exists(cand_proj):
        return os.path.abspath(cand_proj)

    cand_ml = os.path.join(CURRENT_DIR, rel_or_abs_path)
    if os.path.exists(cand_ml):
        return os.path.abspath(cand_ml)

    return rel_or_abs_path


def get_cpu_name() -> str:
    """Returns descriptive CPU name across Windows, Linux, and macOS."""
    if sys.platform.startswith("win"):
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val and str(val).strip():
                return str(val).strip()
        except Exception:
            pass

    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
            if out.strip():
                return out.strip()
        except Exception:
            pass

    cpu = platform.processor() or platform.machine() or "Unknown CPU"
    return str(cpu).strip()


def get_power_plan() -> str:
    """Retrieves the active Windows power plan or returns not applicable on non-Windows."""
    if not sys.platform.startswith("win"):
        return "not applicable (non-Windows)"
    try:
        out = subprocess.check_output(
            ["powercfg", "/getactivescheme"],
            text=True,
            stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception as e:
        return f"Unavailable ({e})"


# Alias for backward compatibility
get_windows_power_plan = get_power_plan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rigorous single-image latency benchmarking for plant disease classification models"
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of CPU threads to configure for PyTorch (default: 4)"
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="ml/checkpoints",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--results-dir", type=str, default="ml/results",
        help="Directory to save latency.json"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for latency.json output (defaults to <results-dir>/latency.json)"
    )
    parser.add_argument(
        "--rounds", type=int, default=10,
        help="Number of interleaved benchmark rounds (default: 10)"
    )
    parser.add_argument(
        "--runs-per-round", type=int, default=20,
        help="Timed runs per model per round (default: 20)"
    )
    parser.add_argument(
        "--warmup", type=int, default=20,
        help="Number of warm-up runs per model/ensemble (default: 20)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to benchmark on (cuda / cpu). Defaults to auto-detection."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Thread count configuration
    torch.set_num_threads(args.threads)
    configured_threads = torch.get_num_threads()

    # 2. Device configuration & system metadata
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_str = str(device)
    torch_version_str = torch.__version__
    cpu_name_str = get_cpu_name()
    power_plan_str = get_power_plan()

    print("=" * 97)
    print(" LATENCY BENCHMARK HARDWARE & EXECUTION ENVIRONMENT")
    print(f"  Device:             {device_str}")
    if device.type == "cuda":
        print(f"  GPU Name:           {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch Version:    {torch_version_str}")
    print(f"  Configured Threads: {configured_threads}")
    print(f"  CPU Model:          {cpu_name_str}")
    print(f"  Power Plan:         {power_plan_str}")
    print(f"  Warm-up Runs:       {args.warmup}")
    print(f"  Benchmark Rounds:   {args.rounds} (with {args.runs_per_round} runs/round = {args.rounds * args.runs_per_round} total timed runs)")
    print("=" * 97)

    results_dir = resolve_path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    output_path = (
        resolve_path(args.output)
        if args.output
        else os.path.join(results_dir, "latency.json")
    )

    # 3. Model configs and checkpoint loading
    model_names = ["resnet50", "efficientnet_v2", "vit"]
    config_paths = {
        "resnet50": resolve_path("ml/configs/config.yaml"),
        "efficientnet_v2": resolve_path("ml/configs/efficientnet_v2.yaml"),
        "vit": resolve_path("ml/configs/vit.yaml")
    }
    ckpt_paths = {
        name: resolve_path(os.path.join(args.checkpoints_dir, name, "best_model.pth"))
        for name in model_names
    }

    loaded_models: Dict[str, torch.nn.Module] = {}
    print("\n[*] Loading models with load_model_and_classes...")
    for name in model_names:
        cpath = ckpt_paths[name]
        cfg_path = config_paths[name]
        if not os.path.exists(cpath):
            raise FileNotFoundError(f"Checkpoint for '{name}' not found at: {cpath}")
        print(f"    - Loading '{name}' from {cpath}...")
        model, _, arch, _, _ = load_model_and_classes(cfg_path, cpath, device)
        model.eval()
        loaded_models[name] = model

    m_resnet = loaded_models["resnet50"]
    m_eff = loaded_models["efficientnet_v2"]
    m_vit = loaded_models["vit"]

    # 4. Dummy input (batch size 1, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # 5. Warm-up runs with torch.inference_mode()
    print(f"\n[*] Executing {args.warmup} warm-up runs per model/ensemble...")
    with torch.inference_mode():
        for name in model_names:
            m = loaded_models[name]
            for _ in range(args.warmup):
                _ = m(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Ensemble warm-up
        for _ in range(args.warmup):
            _ = m_resnet(dummy_input)
            _ = m_eff(dummy_input)
            _ = m_vit(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

    print("    [PASS] Warm-up complete.")

    # 6. Interleaved benchmark rounds
    # In each round: resnet50, efficientnet_v2, vit in turn, then ensemble
    latencies: Dict[str, List[float]] = {
        "resnet50": [],
        "efficientnet_v2": [],
        "vit": [],
        "ensemble": []
    }

    print(f"\n[*] Running {args.rounds} interleaved rounds ({args.runs_per_round} timed runs each)...")
    with torch.inference_mode():
        for r in range(1, args.rounds + 1):
            t_round_start = time.perf_counter()

            # (a) ResNet-50
            for _ in range(args.runs_per_round):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = m_resnet(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies["resnet50"].append((t1 - t0) * 1000.0)

            # (b) EfficientNet-V2
            for _ in range(args.runs_per_round):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = m_eff(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies["efficientnet_v2"].append((t1 - t0) * 1000.0)

            # (c) ViT
            for _ in range(args.runs_per_round):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = m_vit(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies["vit"].append((t1 - t0) * 1000.0)

            # (d) Ensemble: measured time of running all three sequentially
            for _ in range(args.runs_per_round):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = m_resnet(dummy_input)
                _ = m_eff(dummy_input)
                _ = m_vit(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies["ensemble"].append((t1 - t0) * 1000.0)

            t_round_elapsed = time.perf_counter() - t_round_start
            print(f"    - Round {r:02d}/{args.rounds:02d} completed in {t_round_elapsed:.2f}s")

    # 7. Compute statistics (median, p10, p90, ratio)
    bench_results: Dict[str, Dict[str, Any]] = {}
    all_keys = ["resnet50", "efficientnet_v2", "vit", "ensemble"]
    noisy_warnings = []

    for key in all_keys:
        samples = np.array(latencies[key])
        med = float(np.median(samples))
        p10 = float(np.percentile(samples, 10))
        p90 = float(np.percentile(samples, 90))
        ratio = float(p90 / p10) if p10 > 0 else 0.0
        is_noisy = bool(ratio > 1.5)

        bench_results[key] = {
            "median_ms": med,
            "p10_ms": p10,
            "p90_ms": p90,
            "p90_p10_ratio": ratio,
            "noisy_run": is_noisy,
            "sample_count": len(samples)
        }

        if is_noisy:
            noisy_warnings.append(
                f"[WARNING] Model '{key}' has a p90/p10 ratio of {ratio:.2f} > 1.5 (noisy run / high variance)!"
            )

    # 8. Print Results Table
    print("\n" + "=" * 97)
    print(" LATENCY BENCHMARK RESULTS SUMMARY (BATCH SIZE 1)")
    print("=" * 97)
    table_headers = ["Model", "Median (ms)", "p10 (ms)", "p90 (ms)", "p90/p10 Ratio", "Stability"]
    row_fmt = "| {:<17} | {:>13} | {:>11} | {:>11} | {:>15} | {:>11} |"
    divider = "+-------------------+---------------+-------------+-------------+-----------------+-------------+"

    print(divider)
    print(row_fmt.format(*table_headers))
    print(divider)

    for key in all_keys:
        res = bench_results[key]
        med_s = f"{res['median_ms']:.2f}"
        p10_s = f"{res['p10_ms']:.2f}"
        p90_s = f"{res['p90_ms']:.2f}"
        ratio_s = f"{res['p90_p10_ratio']:.2f}"
        status_s = "NOISY (>1.5)" if res["noisy_run"] else "STABLE"
        print(row_fmt.format(key, med_s, p10_s, p90_s, ratio_s, status_s))

    print(divider + "\n")

    # 9. Print warnings if any
    if noisy_warnings:
        for w in noisy_warnings:
            print(w)
        print()
    else:
        print("[*] All models maintained stable execution (p90/p10 <= 1.5).\n")

    # 10. Save latency.json
    env_payload = {
        "device": device_str,
        "torch_version": torch_version_str,
        "torch_num_threads": configured_threads,
        "cpu_name": cpu_name_str,
        "power_plan": power_plan_str
    }
    if device.type == "cuda":
        env_payload["gpu_name"] = torch.cuda.get_device_name(0)

    payload = {
        "environment": env_payload,
        "benchmark_config": {
            "threads": configured_threads,
            "warmup_runs": args.warmup,
            "rounds": args.rounds,
            "runs_per_round": args.runs_per_round,
            "total_timed_runs_per_model": args.rounds * args.runs_per_round,
            "batch_size": 1
        },
        "models": bench_results,
        # Direct top-level access
        **bench_results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[*] Benchmark latency results successfully saved to: {output_path}")


if __name__ == "__main__":
    main()

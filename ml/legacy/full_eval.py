#!/usr/bin/env python3
"""
Full Evaluation & Benchmarking Script for Plant Disease Classification Models.

Extends ml/evaluate.py, ml/gradcam.py (load_model_and_classes), and ml/src/dataset.py.
Evaluates:
  - ResNet-50
  - EfficientNet-V2 (S)
  - Vision Transformer (ViT-B/16)
  - Probability-Averaging Ensemble

Outputs:
  - ml/results/test_files.csv (path, true label)
  - ml/results/probs_<name>.npy
  - ml/results/per_class_<name>.csv & ml/results/per_class_metrics.csv
  - ml/results/confusion_matrix_<name>.png & ml/results/confusion_matrix.png
  - ml/results/metrics.json
  - Comparison table printed to console
"""

import argparse
import csv
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import functools

# Ensure all prints flush immediately
print = functools.partial(print, flush=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Force torch.load(weights_only=False) globally as required
# ---------------------------------------------------------------------------
_orig_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = safe_torch_load

# ---------------------------------------------------------------------------
# Setup sys.path and imports
# ---------------------------------------------------------------------------
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _repo_root

from ml.analysis.gradcam import load_model_and_classes
from ml.src.dataset import get_dataloaders
from ml.src.utils import load_config


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
    """Returns a descriptive CPU model name across Windows, Linux, and macOS."""
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
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
            if out.strip():
                return out.strip()
        except Exception:
            pass

    cpu = platform.processor() or platform.machine() or "Unknown CPU"
    return str(cpu).strip()


def get_weights_only_size_mb(checkpoint_path: str) -> float:
    """
    Computes weights-only size in MB (state_dict only).
    Measures the serialized state_dict buffer size.
    """
    resolved_path = resolve_path(checkpoint_path)
    ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    size_mb = len(buffer.getvalue()) / (1024.0 * 1024.0)
    return float(size_mb)


def measure_single_image_latency(
    model: torch.nn.Module,
    device: torch.device,
    img_size: int = 224,
    warmup_runs: int = 10,
    timed_runs: int = 50
) -> float:
    """
    Measures median single-image inference latency with batch size 1.
    Performs 10 warm-up runs followed by 50 timed runs.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(timed_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return float(np.median(latencies))


def measure_ensemble_latency(
    models: List[torch.nn.Module],
    device: torch.device,
    img_size: int = 224,
    warmup_runs: int = 10,
    timed_runs: int = 50
) -> float:
    """
    Measures median single-image latency for probability-averaging ensemble with batch size 1.
    """
    for m in models:
        m.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            outs = [F.softmax(m(dummy_input), dim=1) for m in models]
            _ = torch.stack(outs).mean(dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(timed_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = [F.softmax(m(dummy_input), dim=1) for m in models]
            _ = torch.stack(outs).mean(dim=0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return float(np.median(latencies))


def compute_metrics_from_probs(
    probs: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str]
) -> Tuple[Dict[str, float], Dict[str, Any], np.ndarray]:
    """
    Computes accuracy, macro precision/recall/F1, per-class metrics, and confusion matrix.
    """
    y_pred = np.argmax(probs, axis=1)
    all_labels = list(range(len(class_names)))
    present_labels = np.unique(np.concatenate([y_true, y_pred]))

    acc = float(accuracy_score(y_true, y_pred))

    if len(present_labels) < len(class_names):
        # When evaluating a subset (e.g. with --limit), compute macro metrics over present classes
        macro_p = float(precision_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0))
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            target_names=[class_names[i] for i in present_labels],
            output_dict=True,
            zero_division=0
        )
    else:
        macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=all_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    summary = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1
    }
    return summary, report_dict, cm


def save_per_class_csv(report_dict: dict, class_names: List[str], output_csv_path: str):
    """Saves per-class precision, recall, f1-score, and support to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for cname in class_names:
            c_data = report_dict.get(cname, {})
            writer.writerow([
                cname,
                f"{c_data.get('precision', 0.0):.4f}",
                f"{c_data.get('recall', 0.0):.4f}",
                f"{c_data.get('f1-score', 0.0):.4f}",
                int(c_data.get("support", 0))
            ])


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_png_path: str,
    title: str = "Confusion Matrix"
):
    """Plots and saves a high-resolution confusion matrix heatmap PNG."""
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    n_classes = len(class_names)
    figsize = (max(14, int(n_classes * 0.45)), max(12, int(n_classes * 0.40)))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Sample Count", rotation=270, labelpad=15, fontsize=10)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=90, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    if n_classes <= 40:
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                if val > 0:
                    ax.text(
                        j, i, format(val, "d"),
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=6
                    )

    fig.tight_layout()
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_comparison_table(
    summary_data: Dict[str, Dict[str, Any]],
    device_str: str,
    torch_version_str: str,
    torch_num_threads: int,
    cpu_name_str: str,
    num_samples: int
):
    """Prints a formatted ASCII comparison table of evaluation results."""
    print("\n" + "=" * 97)
    print(" EVALUATION ENVIRONMENT & HARDWARE CONFIGURATION")
    print(f"  Device:        {device_str}")
    print(f"  PyTorch:       {torch_version_str}")
    print(f"  Torch Threads: {torch_num_threads}")
    print(f"  CPU:           {cpu_name_str}")
    print(f"  Test Samples:  {num_samples}")
    print("=" * 97)

    table_headers = ["Model", "Accuracy (%)", "Macro Prec", "Macro Rec", "Macro F1", "Weights (MB)", "Latency (ms)"]
    row_fmt = "| {:<17} | {:>12} | {:>10} | {:>9} | {:>8} | {:>12} | {:>12} |"
    divider = "+-------------------+--------------+------------+-----------+----------+--------------+--------------+"

    print(divider)
    print(row_fmt.format(*table_headers))
    print(divider)

    for name, m in summary_data.items():
        acc_str = f"{m['accuracy'] * 100:.2f}%"
        prec_str = f"{m['macro_precision']:.4f}"
        rec_str = f"{m['macro_recall']:.4f}"
        f1_str = f"{m['macro_f1']:.4f}"
        weights_str = f"{m['weights_size_mb']:.2f}"
        lat_val = m.get("median_latency_ms")
        lat_str = f"{lat_val:.2f}" if lat_val is not None else "null"
        print(row_fmt.format(name, acc_str, prec_str, rec_str, f1_str, weights_str, lat_str))

    print(divider + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full evaluation and comparison of Plant Disease Classification models"
    )
    parser.add_argument(
        "--config", type=str, default="ml/legacy/config.yaml",
        help="Path to base config file"
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="ml/checkpoints",
        help="Directory containing model checkpoint folders"
    )
    parser.add_argument(
        "--results-dir", type=str, default="ml/results",
        help="Output directory for predictions, metrics, and figures"
    )
    parser.add_argument(
        "--output-metrics", type=str, default=None,
        help="Optional custom output path for metrics.json (defaults to <results-dir>/metrics.json)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Dataset directory override"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Inference device (cuda / cpu). Defaults to auto-detection."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for test evaluation loader"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of test samples for smoke testing (e.g. 50)"
    )
    parser.add_argument(
        "--skip-latency", action="store_true",
        help="Skip single-image latency benchmarking phase and ensemble latency check"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Device configuration & system info
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_str = str(device)
    torch_version_str = torch.__version__
    torch_num_threads = torch.get_num_threads()
    cpu_name_str = get_cpu_name()

    print("[*] Evaluation initializing...")
    print(f"[*] Device:        {device_str}")
    print(f"[*] PyTorch:       {torch_version_str}")
    print(f"[*] Torch Threads: {torch_num_threads}")
    print(f"[*] CPU:           {cpu_name_str}")

    # When --limit is used, write all outputs to ml/results_smoke/ instead of ml/results/
    if args.limit is not None:
        target_results_dir = "ml/results_smoke" if args.results_dir == "ml/results" else args.results_dir
    else:
        target_results_dir = args.results_dir

    results_dir = resolve_path(target_results_dir)
    os.makedirs(results_dir, exist_ok=True)
    if args.limit is not None:
        print(f"[*] Smoke-test mode active (--limit {args.limit}): writing outputs to {results_dir}")

    # 2. Config mapping and checkpoint mapping for the three models
    model_names = ["resnet50", "efficientnet_v2", "vit"]
    config_paths = {
        "resnet50": resolve_path("ml/legacy/config.yaml"),
        "efficientnet_v2": resolve_path("ml/legacy/efficientnet_v2.yaml"),
        "vit": resolve_path("ml/legacy/vit.yaml")
    }
    ckpt_paths = {
        name: resolve_path(os.path.join(args.checkpoints_dir, name, "best_model.pth"))
        for name in model_names
    }

    # Verify checkpoint existence
    for name, cpath in ckpt_paths.items():
        if not os.path.exists(cpath):
            raise FileNotFoundError(f"Checkpoint for '{name}' not found at: {cpath}")

    # 3. Load base config and initialize dataset with num_workers=0
    base_config = load_config(config_paths["resnet50"])
    if "dataset" not in base_config:
        base_config["dataset"] = {}
    
    # Force num_workers=0 as strictly required
    base_config["dataset"]["num_workers"] = 0
    if args.batch_size:
        base_config["dataset"]["batch_size"] = args.batch_size

    print("[*] Loading test dataset (using test split only for reporting)...")
    _, _, test_loader, dataloader_class_to_idx = get_dataloaders(
        base_config, data_dir_override=args.data_dir
    )

    # Enforce limit if specified for smoke tests
    test_dataset = test_loader.dataset
    if args.limit is not None and args.limit > 0:
        total_avail = len(test_dataset.indices)
        limit_n = min(args.limit, total_avail)
        print(f"[*] Applying smoke-test limit: {limit_n} samples (out of {total_avail}).")
        test_dataset.indices = test_dataset.indices[:limit_n]
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    num_test_samples = len(test_dataset)
    print(f"[*] Test set ready with {num_test_samples} samples.")

    # 4. Load models with load_model_and_classes and verify class_to_idx
    loaded_models: Dict[str, torch.nn.Module] = {}
    loaded_class_to_idx: Dict[str, Dict[str, int]] = {}
    weights_sizes_mb: Dict[str, float] = {}
    latencies_ms: Dict[str, float] = {}

    for name in model_names:
        ckpt_file = ckpt_paths[name]
        cfg_file = config_paths[name]
        print(f"[*] Loading model '{name}' from {ckpt_file}...")

        # Load model using load_model_and_classes
        model, cfg, arch, c_to_i, idx_to_c = load_model_and_classes(cfg_file, ckpt_file, device)
        model.eval()
        loaded_models[name] = model
        loaded_class_to_idx[name] = c_to_i

        # Weights-only size in MB (state_dict only)
        w_size = get_weights_only_size_mb(ckpt_file)
        weights_sizes_mb[name] = w_size
        print(f"    - State-dict weights size: {w_size:.2f} MB")

    # 5. Assert class_to_idx is identical across all three and matches get_dataloaders
    print("[*] Asserting class_to_idx across all three models and get_dataloaders...")
    for name in model_names:
        assert loaded_class_to_idx[name] == dataloader_class_to_idx, (
            f"Class mapping mismatch between model '{name}' and get_dataloaders!"
        )
    assert loaded_class_to_idx["resnet50"] == loaded_class_to_idx["efficientnet_v2"], (
        "Class mapping mismatch between resnet50 and efficientnet_v2!"
    )
    assert loaded_class_to_idx["resnet50"] == loaded_class_to_idx["vit"], (
        "Class mapping mismatch between resnet50 and vit!"
    )
    print("    [PASS] class_to_idx is identical across all models and matches get_dataloaders.")

    class_to_idx = dataloader_class_to_idx
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # 6. Extract and save ml/results/test_files.csv (path, true label)
    test_files_csv_path = os.path.join(results_dir, "test_files.csv")
    print(f"[*] Saving test file records to {test_files_csv_path}...")
    
    test_records: List[Tuple[str, str, int]] = []
    for sub_idx in range(len(test_dataset)):
        dataset_idx = test_dataset.indices[sub_idx]
        img_path, true_label_idx = test_dataset.dataset.samples[dataset_idx]
        true_label_name = idx_to_class[true_label_idx]
        test_records.append((img_path, true_label_name, true_label_idx))

    with open(test_files_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label"])
        for img_path, true_label_name, _ in test_records:
            writer.writerow([img_path, true_label_name])

    y_true = np.array([lbl_idx for _, _, lbl_idx in test_records], dtype=np.int64)

    # 7. Run test loader once per model and save softmax probabilities
    saved_prob_files: Dict[str, str] = {}
    for name in model_names:
        model = loaded_models[name]
        prob_out_path = os.path.join(results_dir, f"probs_{name}.npy")
        saved_prob_files[name] = prob_out_path

        print(f"[*] Running test inference for '{name}'...")
        all_probs_list = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_probs_list.append(probs)

        model_probs = np.concatenate(all_probs_list, axis=0)
        np.save(prob_out_path, model_probs)
        print(f"    - Probabilities saved to: {prob_out_path} (shape: {model_probs.shape})")

    # 8. Build probability-averaging ensemble
    print("[*] Building probability-averaging ensemble from saved probability files...")
    p_resnet = np.load(saved_prob_files["resnet50"])
    p_eff = np.load(saved_prob_files["efficientnet_v2"])
    p_vit = np.load(saved_prob_files["vit"])

    ensemble_probs = (p_resnet + p_eff + p_vit) / 3.0
    ensemble_prob_path = os.path.join(results_dir, "probs_ensemble.npy")
    np.save(ensemble_prob_path, ensemble_probs)
    saved_prob_files["ensemble"] = ensemble_prob_path
    print(f"    - Ensemble probabilities saved to: {ensemble_prob_path}")

    # Compute ensemble weights size
    ensemble_weights_mb = sum(weights_sizes_mb[m] for m in model_names)
    weights_sizes_mb["ensemble"] = ensemble_weights_mb
    print(f"    - Ensemble weights size: {ensemble_weights_mb:.2f} MB")

    # 9. Compute metrics, per-class metrics CSV, and confusion matrix PNG from saved probabilities
    all_eval_names = model_names + ["ensemble"]
    summary_results: Dict[str, Dict[str, Any]] = {}
    combined_per_class_rows = []

    for name in all_eval_names:
        print(f"[*] Computing evaluation metrics for '{name}' from {saved_prob_files[name]}...")
        probs = np.load(saved_prob_files[name])
        summary, report_dict, cm = compute_metrics_from_probs(probs, y_true, class_names)

        summary["weights_size_mb"] = weights_sizes_mb[name]
        summary["median_latency_ms"] = None
        summary["device"] = device_str
        summary["torch_version"] = torch_version_str
        summary["torch_num_threads"] = torch_num_threads
        summary["cpu_name"] = cpu_name_str
        summary_results[name] = summary

        # Save per-class metrics CSV for this model
        per_class_csv = os.path.join(results_dir, f"per_class_{name}.csv")
        save_per_class_csv(report_dict, class_names, per_class_csv)
        print(f"    - Per-class CSV saved to: {per_class_csv}")

        # Accumulate for combined per-class CSV
        for cname in class_names:
            c_data = report_dict.get(cname, {})
            combined_per_class_rows.append([
                name,
                cname,
                f"{c_data.get('precision', 0.0):.4f}",
                f"{c_data.get('recall', 0.0):.4f}",
                f"{c_data.get('f1-score', 0.0):.4f}",
                int(c_data.get("support", 0))
            ])

        # Save confusion matrix PNG for this model
        cm_png_path = os.path.join(results_dir, f"confusion_matrix_{name}.png")
        plot_and_save_confusion_matrix(
            cm, class_names, cm_png_path, title=f"Confusion Matrix - {name}"
        )
        print(f"    - Confusion matrix PNG saved to: {cm_png_path}")

        # Also save default confusion_matrix.png for the ensemble (or primary model)
        if name == "ensemble":
            default_cm_path = os.path.join(results_dir, "confusion_matrix.png")
            plot_and_save_confusion_matrix(
                cm, class_names, default_cm_path, title="Confusion Matrix - Probability-Averaging Ensemble"
            )

    # Save combined per-class metrics CSV
    combined_csv_path = os.path.join(results_dir, "per_class_metrics.csv")
    with open(combined_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "class", "precision", "recall", "f1_score", "support"])
        writer.writerows(combined_per_class_rows)
    print(f"[*] Combined per-class metrics CSV saved to: {combined_csv_path}")

    # 10. Measure single-image inference latency: one model at a time, then the ensemble
    if not args.skip_latency:
        print("\n[*] Measuring single-image inference latency (10 warm-up, 50 timed runs, batch size 1)...")
        for name in model_names:
            print(f"[*] Measuring latency for '{name}'...")
            lat = measure_single_image_latency(loaded_models[name], device)
            latencies_ms[name] = lat
            summary_results[name]["median_latency_ms"] = lat
            print(f"    - Median single-image latency ({name}): {lat:.2f} ms")

        print("[*] Measuring latency for 'ensemble'...")
        ensemble_lat = measure_ensemble_latency(
            [loaded_models["resnet50"], loaded_models["efficientnet_v2"], loaded_models["vit"]],
            device
        )
        latencies_ms["ensemble"] = ensemble_lat
        summary_results["ensemble"]["median_latency_ms"] = ensemble_lat
        print(f"    - Median single-image latency (ensemble): {ensemble_lat:.2f} ms")

        # Warning if ensemble latency is less than the maximum single-model latency
        max_single_lat = max(latencies_ms[name] for name in model_names)
        if latencies_ms["ensemble"] < max_single_lat:
            print(
                f"\n[WARNING] Ensemble latency ({latencies_ms['ensemble']:.2f} ms) is less than "
                f"the maximum single-model latency ({max_single_lat:.2f} ms)! "
                f"Ensemble latency should roughly equal the sum of individual models.\n"
            )
    else:
        print("\n[*] Skipping latency measurement phase (--skip-latency specified).")

    # 11. Print comparison table
    print_comparison_table(
        summary_results,
        device_str=device_str,
        torch_version_str=torch_version_str,
        torch_num_threads=torch_num_threads,
        cpu_name_str=cpu_name_str,
        num_samples=num_test_samples
    )

    # 12. Save metrics.json
    metrics_json_path = (
        resolve_path(args.output_metrics)
        if args.output_metrics
        else os.path.join(results_dir, "metrics.json")
    )
    
    metrics_payload = {
        "metadata": {
            "device": device_str,
            "torch_version": torch_version_str,
            "torch_num_threads": torch_num_threads,
            "cpu_name": cpu_name_str,
            "num_test_samples": num_test_samples,
            "limit": args.limit
        },
        "device": device_str,
        "torch_version": torch_version_str,
        "torch_num_threads": torch_num_threads,
        "cpu_name": cpu_name_str,
        "num_test_samples": num_test_samples,
        "models": summary_results,
        # Direct model keys for easy top-level indexing
        **summary_results
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[*] Evaluation metrics saved to: {metrics_json_path}")


if __name__ == "__main__":
    main()

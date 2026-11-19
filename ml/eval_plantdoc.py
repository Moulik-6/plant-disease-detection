#!/usr/bin/env python3
"""
PlantDoc Out-of-Domain Evaluation & Cross-Dataset Generalisation Benchmark.

Evaluates PlantVillage-trained models (ResNet-50, EfficientNetV2-S, ViT-B/16, and Ensemble)
on the realistic, in-the-wild PlantDoc dataset across mapped crop-disease categories.

Features:
  - Robust directory walking using os.scandir preserving raw file paths (supports odd characters).
  - Configurable class mapping with confidence ratings (ml/configs/plantdoc_map.yaml).
  - Strict assertions ensuring mapped targets match PlantVillage class_to_idx.
  - Identical image preprocessing (Resize to 224x224 and ImageNet normalization).
  - Full statistical breakdown:
      * Top-1 accuracy over all 38 classes
      * Top-1 accuracy restricted to mapped classes (masking unmapped probabilities)
      * Top-5 accuracy
      * Macro-F1 over present classes
      * High-confidence subset evaluation (excluding medium confidence mappings)
      * Confidence calibration (mean confidence on correct vs. error, accuracy & coverage at >=0.7 and >=0.9)
      * Top 15 most frequent confusion pairs
      * Per-class recall CSV and mapped confusion matrix PNGs
  - Vectorized perceptual hash overlap audit (--check-overlap) against PlantVillage train split.
  - Structured output serialization into files.csv, probs_<name>.npy, and metrics.json.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import functools

# Ensure unbuffered, real-time logging
print = functools.partial(print, flush=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# ---------------------------------------------------------------------------
# Force torch.load(weights_only=False) globally
# ---------------------------------------------------------------------------
_orig_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = safe_torch_load

# ---------------------------------------------------------------------------
# Path resolution and module imports
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "ml" else CURRENT_DIR

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from gradcam import load_model_and_classes
from src.dataset import get_transforms
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


# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------
class PlantDocDataset(Dataset):
    """
    Dataset wrapper for PlantDoc images preserving raw file paths and applying
    identical PlantVillage evaluation transforms.
    """
    def __init__(self, samples: List[Dict[str, Any]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        item = self.samples[idx]
        img_path = item["path"]
        try:
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
            tensor = self.transform(rgb_img)
            return tensor, item["target_idx"], idx
        except Exception as e:
            # Fallback for transient read issues
            print(f"[ERROR] Failed to load image at {img_path}: {e}")
            dummy_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
            return dummy_tensor, item["target_idx"], idx


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="PlantDoc out-of-domain evaluation & cross-dataset generalisation benchmark"
    )
    parser.add_argument(
        "--plantdoc-dir", type=str, required=True,
        help="Root folder of PlantDoc dataset containing train/ and test/"
    )
    parser.add_argument(
        "--splits", type=str, default="train,test",
        help="Comma-separated splits to evaluate (default: 'train,test')"
    )
    parser.add_argument(
        "--map", type=str, default="ml/configs/plantdoc_map.yaml",
        help="Path to PlantDoc to PlantVillage class mapping YAML (default: ml/configs/plantdoc_map.yaml)"
    )
    parser.add_argument(
        "--results-dir", type=str, default="ml/results/plantdoc",
        help="Output directory for evaluation results (default: ml/results/plantdoc)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Smoke test: use N random images with a fixed seed and write to ml/results_smoke/plantdoc"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for DataLoader (default: 64)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=(0 if sys.platform.startswith("win") else 4),
        help="Number of DataLoader workers (default: 0 on Windows, 4 on Linux/macOS)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Inference device (cuda / cpu). Auto-detected if omitted."
    )
    parser.add_argument(
        "--check-overlap", action="store_true",
        help="Compute perceptual hashes and report Hamming distance overlap against PlantVillage train split"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
):
    """Generates and saves a clean confusion matrix heatmap for mapped classes."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n_classes = len(class_names)
    fig_size = max(10, n_classes * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Normalize by row (true class support)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 105)
    print(" PLANTDOC OUT-OF-DOMAIN GENERALISATION & ROBUSTNESS BENCHMARK")
    print("=" * 105)

    # 1. Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference Device: {device}")

    # Output directory configuration
    if args.limit is not None and args.limit > 0:
        results_dir = (
            resolve_path("ml/results_smoke/plantdoc")
            if args.results_dir == "ml/results/plantdoc"
            else resolve_path(args.results_dir)
        )
        print(f"[*] Smoke-test mode active (--limit {args.limit}): writing outputs to {results_dir}")
    else:
        results_dir = resolve_path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load model checkpoints and verify class_to_idx
    model_names = ["resnet50", "efficientnet_v2", "vit"]
    config_paths = {
        "resnet50": resolve_path("ml/configs/config.yaml"),
        "efficientnet_v2": resolve_path("ml/configs/efficientnet_v2.yaml"),
        "vit": resolve_path("ml/configs/vit.yaml")
    }
    ckpt_paths = {
        name: resolve_path(f"ml/checkpoints/{name}/best_model.pth")
        for name in model_names
    }

    loaded_models: Dict[str, torch.nn.Module] = {}
    loaded_class_to_idx: Dict[str, Dict[str, int]] = {}

    print("\n[*] Loading checkpoints via load_model_and_classes...")
    for name in model_names:
        cpath = ckpt_paths[name]
        cfg_path = config_paths[name]
        if not os.path.exists(cpath):
            raise FileNotFoundError(f"Checkpoint for '{name}' not found at: {cpath}")
        model, cfg, arch, c_to_i, idx_to_c = load_model_and_classes(cfg_path, cpath, device)
        model.eval()
        loaded_models[name] = model
        loaded_class_to_idx[name] = c_to_i
        print(f"    - Loaded '{name}' ({arch}): {len(c_to_i)} classes.")

    # Assert class_to_idx consistency across all three models
    assert loaded_class_to_idx["resnet50"] == loaded_class_to_idx["efficientnet_v2"], (
        "class_to_idx mismatch between resnet50 and efficientnet_v2!"
    )
    assert loaded_class_to_idx["resnet50"] == loaded_class_to_idx["vit"], (
        "class_to_idx mismatch between resnet50 and vit!"
    )
    class_to_idx = loaded_class_to_idx["resnet50"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    all_pv_classes = [idx_to_class[i] for i in range(len(class_to_idx))]
    print(f"    [PASS] class_to_idx is identical across all models ({len(class_to_idx)} classes).")

    # 3. Load PlantDoc mapping configuration
    map_file_path = resolve_path(args.map)
    print(f"\n[*] Loading PlantDoc mapping from: {map_file_path}")
    if not os.path.exists(map_file_path):
        raise FileNotFoundError(f"PlantDoc mapping file not found: {map_file_path}")

    with open(map_file_path, "r", encoding="utf-8") as f:
        map_yaml = yaml.safe_load(f)
    mapping: Dict[str, Dict[str, str]] = map_yaml.get("mapping", {})

    # Assert every mapped target exists in class_to_idx
    missing_targets = []
    mapped_target_names = set()
    for p_class, info in mapping.items():
        target = info.get("target")
        mapped_target_names.add(target)
        if target not in class_to_idx:
            missing_targets.append((p_class, target))

    if missing_targets:
        print("\n[ERROR] The following mapped targets do NOT exist in PlantVillage class_to_idx:")
        for p_class, tgt in missing_targets:
            print(f"  - '{p_class}' -> '{tgt}'")
        raise AssertionError(f"Found {len(missing_targets)} invalid mapped targets in {args.map}!")

    print(f"    [PASS] All {len(mapping)} PlantDoc categories map to valid PlantVillage classes.")

    # Identify the 10 PlantVillage classes with no PlantDoc counterpart
    unmapped_pv_classes = sorted(list(set(all_pv_classes) - mapped_target_names))
    print(f"\n[*] PlantVillage classes without PlantDoc counterpart ({len(unmapped_pv_classes)} classes):")
    for c in unmapped_pv_classes:
        print(f"    - {c}")

    # Set of mapped class indices for restricted prediction
    mapped_class_indices = np.array(sorted([class_to_idx[t] for t in mapped_target_names]), dtype=np.int64)

    # 4. Resolve identical evaluation transform from ml/src/dataset.py
    base_cfg = load_config(config_paths["resnet50"])
    _, eval_transform = get_transforms(base_cfg)

    # 5. Dataset discovery using os.scandir preserving raw file paths
    plantdoc_root = resolve_path(args.plantdoc_dir)
    if not os.path.exists(plantdoc_root):
        raise FileNotFoundError(f"PlantDoc root directory does not exist: {plantdoc_root}")

    target_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"\n[*] Scanning PlantDoc directory across splits: {target_splits}...")

    valid_samples: List[Dict[str, Any]] = []
    split_class_counts: Dict[str, Dict[str, int]] = {s: {} for s in target_splits}
    excluded_folders: Set[str] = set()
    excluded_files_count = 0
    unreadable_files_count = 0
    allowed_exts = {".jpg", ".jpeg", ".png"}

    for split in target_splits:
        split_dir = os.path.join(plantdoc_root, split)
        if not os.path.exists(split_dir):
            print(f"    [WARN] Split folder does not exist: {split_dir}. Skipping.")
            continue

        with os.scandir(split_dir) as split_it:
            for folder_entry in split_it:
                if not folder_entry.is_dir():
                    continue

                folder_name = folder_entry.name
                if folder_name not in mapping:
                    excluded_folders.add(folder_name)
                    # Count excluded files
                    try:
                        with os.scandir(folder_entry.path) as fit:
                            for fe in fit:
                                if fe.is_file() and Path(fe.name).suffix.lower() in allowed_exts:
                                    excluded_files_count += 1
                    except Exception:
                        pass
                    continue

                map_info = mapping[folder_name]
                target_name = map_info["target"]
                confidence = map_info.get("confidence", "high")
                target_id = class_to_idx[target_name]

                if folder_name not in split_class_counts[split]:
                    split_class_counts[split][folder_name] = 0

                with os.scandir(folder_entry.path) as file_it:
                    for file_entry in file_it:
                        if not file_entry.is_file():
                            continue

                        # Check extension (case-insensitive)
                        ext = Path(file_entry.name).suffix.lower()
                        if ext not in allowed_exts:
                            continue

                        raw_path = file_entry.path

                        # Verify image readability
                        try:
                            with Image.open(raw_path) as test_im:
                                test_im.verify()
                            # Quick load check
                            with Image.open(raw_path) as test_im:
                                _ = test_im.convert("RGB")
                        except Exception:
                            unreadable_files_count += 1
                            continue

                        split_class_counts[split][folder_name] += 1
                        valid_samples.append({
                            "path": raw_path,
                            "split": split,
                            "plantdoc_class": folder_name,
                            "mapped_class": target_name,
                            "target_idx": target_id,
                            "confidence_level": confidence
                        })

    total_mapped_images = len(valid_samples)
    print("\n" + "=" * 105)
    print(" PLANTDOC DATASET DISCOVERY SUMMARY")
    print("=" * 105)
    print(f"  Total Valid Mapped Images:     {total_mapped_images}")
    print(f"  Unreadable / Corrupt Images:   {unreadable_files_count}")
    print(f"  Excluded Folders (Not Mapped): {len(excluded_folders)} ({sorted(list(excluded_folders))})")
    print(f"  Excluded Images Count:         {excluded_files_count}")
    print(f"  PlantVillage Classes Unmapped: {len(unmapped_pv_classes)}")
    print("=" * 105)

    if total_mapped_images == 0:
        raise RuntimeError("No valid mapped PlantDoc images were found. Aborting evaluation.")

    # 6. Smoke-test sample subsampling if requested
    if args.limit is not None and args.limit > 0:
        random.seed(42)
        if len(valid_samples) > args.limit:
            print(f"[*] Applying smoke-test limit ({args.limit} samples with seed 42)...")
            valid_samples = random.sample(valid_samples, args.limit)
            total_mapped_images = len(valid_samples)

    # 7. Save files.csv (path, split, plantdoc_class, mapped_class, confidence_level)
    files_csv_path = os.path.join(results_dir, "files.csv")
    print(f"[*] Saving file records to: {files_csv_path}...")
    with open(files_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "split", "plantdoc_class", "mapped_class", "confidence_level"])
        for s in valid_samples:
            writer.writerow([s["path"], s["split"], s["plantdoc_class"], s["mapped_class"], s["confidence_level"]])

    # 8. Construct DataLoader
    dataset = PlantDocDataset(valid_samples, transform=eval_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # Extract ground truth target array
    y_true = np.array([s["target_idx"] for s in valid_samples], dtype=np.int64)
    confidence_levels = [s["confidence_level"] for s in valid_samples]
    high_conf_mask = np.array([c == "high" for c in confidence_levels], dtype=bool)

    # 9. Run inference per model and extract softmax probabilities
    saved_probs: Dict[str, np.ndarray] = {}
    print("\n[*] Running model inference on PlantDoc images...")

    for name in model_names:
        print(f"[*] Executing forward pass for '{name}'...")
        model = loaded_models[name]
        prob_list = []

        with torch.inference_mode():
            for images, _, _ in dataloader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                prob_list.append(probs)

        model_probs = np.concatenate(prob_list, axis=0)
        prob_out_path = os.path.join(results_dir, f"probs_{name}.npy")
        np.save(prob_out_path, model_probs)
        saved_probs[name] = model_probs
        print(f"    - Saved '{name}' probabilities to: {prob_out_path} (shape: {model_probs.shape})")

    # 10. Probability-averaging ensemble
    print("[*] Building probability-averaging ensemble...")
    ensemble_probs = (saved_probs["resnet50"] + saved_probs["efficientnet_v2"] + saved_probs["vit"]) / 3.0
    ensemble_out_path = os.path.join(results_dir, "probs_ensemble.npy")
    np.save(ensemble_out_path, ensemble_probs)
    saved_probs["ensemble"] = ensemble_probs
    print(f"    - Saved ensemble probabilities to: {ensemble_out_path}")

    # 11. Compute detailed evaluation metrics
    all_eval_models = model_names + ["ensemble"]
    metrics_summary: Dict[str, Any] = {}
    present_classes = sorted(list(set(y_true)))

    print("\n" + "=" * 105)
    print(" COMPUTING MULTI-CRITERIA METRICS & ACCURACY BREAKDOWNS")
    print("=" * 105)

    for name in all_eval_models:
        probs = saved_probs[name]

        # (a) Top-1 accuracy over all 38 outputs
        preds_38 = np.argmax(probs, axis=1)
        top1_all = float(accuracy_score(y_true, preds_38))

        # (b) Top-1 accuracy restricted to mapped classes only
        # Mask probabilities of unmapped classes with -1.0 before argmax
        restricted_probs = probs.copy()
        unmapped_mask = np.ones(probs.shape[1], dtype=bool)
        unmapped_mask[mapped_class_indices] = False
        restricted_probs[:, unmapped_mask] = -1.0
        preds_restricted = np.argmax(restricted_probs, axis=1)
        top1_restricted = float(accuracy_score(y_true, preds_restricted))

        # (c) Top-5 accuracy
        top5_indices = np.argsort(probs, axis=1)[:, -5:]
        top5_correct = np.any(top5_indices == y_true[:, None], axis=1)
        top5_acc = float(np.mean(top5_correct))

        # (d) Macro-F1 over classes present
        macro_f1_all = float(f1_score(y_true, preds_38, labels=present_classes, average="macro", zero_division=0))
        macro_f1_restricted = float(f1_score(y_true, preds_restricted, labels=present_classes, average="macro", zero_division=0))

        # (e) All of the above excluding the two classes marked confidence: medium
        high_conf_count = int(np.sum(high_conf_mask))
        if high_conf_count > 0:
            y_true_hc = y_true[high_conf_mask]
            preds_38_hc = preds_38[high_conf_mask]
            preds_rest_hc = preds_restricted[high_conf_mask]
            top5_hc = top5_correct[high_conf_mask]
            present_hc = sorted(list(set(y_true_hc)))

            hc_top1_all = float(accuracy_score(y_true_hc, preds_38_hc))
            hc_top1_rest = float(accuracy_score(y_true_hc, preds_rest_hc))
            hc_top5 = float(np.mean(top5_hc))
            hc_macro_f1 = float(f1_score(y_true_hc, preds_38_hc, labels=present_hc, average="macro", zero_division=0))
        else:
            hc_top1_all, hc_top1_rest, hc_top5, hc_macro_f1 = 0.0, 0.0, 0.0, 0.0

        # (f) Per-class recall CSV and confusion matrix PNG for mapped classes
        per_class_recall_rows = []
        for c_idx in mapped_class_indices:
            c_name = idx_to_class[c_idx]
            c_mask = (y_true == c_idx)
            support = int(np.sum(c_mask))
            if support > 0:
                c_correct = int(np.sum(preds_38[c_mask] == c_idx))
                rec = c_correct / support
            else:
                c_correct = 0
                rec = 0.0
            per_class_recall_rows.append({
                "class_name": c_name,
                "class_idx": int(c_idx),
                "support": support,
                "correct": c_correct,
                "recall": float(rec)
            })

        # Save per-class recall CSV
        recall_csv_path = os.path.join(results_dir, f"per_class_recall_{name}.csv")
        with open(recall_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "class_idx", "support", "correct", "recall"])
            for r in per_class_recall_rows:
                writer.writerow([r["class_name"], r["class_idx"], r["support"], r["correct"], f"{r['recall']:.4f}"])

        # Confusion Matrix PNG over mapped classes
        mapped_labels_order = list(mapped_class_indices)
        mapped_names_order = [idx_to_class[i] for i in mapped_labels_order]
        cm_mapped = confusion_matrix(y_true, preds_restricted, labels=mapped_labels_order)

        cm_png_path = os.path.join(results_dir, f"confusion_matrix_{name}.png")
        plot_confusion_matrix(
            cm_mapped,
            mapped_names_order,
            cm_png_path,
            title=f"PlantDoc Confusion Matrix - {name} (predictions restricted to mapped classes)"
        )
        if name == "ensemble":
            default_cm_path = os.path.join(results_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                cm_mapped,
                mapped_names_order,
                default_cm_path,
                title="PlantDoc Confusion Matrix - Ensemble (predictions restricted to mapped classes)"
            )

        # (g) Confidence calibration on correct vs wrong predictions & thresholds
        confidences = np.max(probs, axis=1)
        is_correct = (preds_38 == y_true)

        mean_conf_correct = float(np.mean(confidences[is_correct])) if np.sum(is_correct) > 0 else 0.0
        mean_conf_wrong = float(np.mean(confidences[~is_correct])) if np.sum(~is_correct) > 0 else 0.0

        mask_07 = (confidences >= 0.7)
        cov_07 = float(np.mean(mask_07))
        acc_07 = float(np.mean(is_correct[mask_07])) if np.sum(mask_07) > 0 else 0.0

        mask_09 = (confidences >= 0.9)
        cov_09 = float(np.mean(mask_09))
        acc_09 = float(np.mean(is_correct[mask_09])) if np.sum(mask_09) > 0 else 0.0

        # (h) 15 most frequent confusion pairs (true -> predicted, count)
        error_mask = ~is_correct
        confusion_pairs_count: Dict[Tuple[str, str], int] = {}
        for t_idx, p_idx in zip(y_true[error_mask], preds_38[error_mask]):
            pair = (idx_to_class[t_idx], idx_to_class[p_idx])
            confusion_pairs_count[pair] = confusion_pairs_count.get(pair, 0) + 1

        sorted_confusion_pairs = sorted(
            confusion_pairs_count.items(), key=lambda x: x[1], reverse=True
        )[:15]

        top_15_confusion_list = [
            {
                "true_class": pair[0],
                "predicted_class": pair[1],
                "error_count": cnt,
                "pct_of_all_errors": float(cnt / max(1, np.sum(error_mask)) * 100.0)
            }
            for pair, cnt in sorted_confusion_pairs
        ]

        metrics_summary[name] = {
            "top1_accuracy_all_38": top1_all,
            "top1_accuracy_restricted": top1_restricted,
            "top5_accuracy": top5_acc,
            "macro_f1_all_38": macro_f1_all,
            "macro_f1_restricted": macro_f1_restricted,
            "high_confidence_subset": {
                "sample_count": high_conf_count,
                "top1_accuracy_all": hc_top1_all,
                "top1_accuracy_restricted": hc_top1_rest,
                "top5_accuracy": hc_top5,
                "macro_f1": hc_macro_f1
            },
            "confidence_calibration": {
                "mean_confidence_correct": mean_conf_correct,
                "mean_confidence_wrong": mean_conf_wrong,
                "threshold_0_7": {
                    "accuracy": acc_07,
                    "coverage": cov_07
                },
                "threshold_0_9": {
                    "accuracy": acc_09,
                    "coverage": cov_09
                }
            },
            "top_15_confusion_pairs": top_15_confusion_list,
            "per_class_recall": per_class_recall_rows
        }

    # 12. Optional perceptual hash overlap audit (--check-overlap)
    overlap_results = None
    if args.check_overlap:
        print("\n" + "=" * 105)
        print(" RUNNING PERCEPTUAL HASH OVERLAP AUDIT (--check-overlap)")
        print("=" * 105)
        hashes_path = resolve_path("ml/results/hashes.npy")
        if not os.path.exists(hashes_path):
            print(f"[WARN] Hashes cache '{hashes_path}' not found! Skipping overlap check.")
        else:
            print(f"[*] Loading PlantVillage train hashes from: {hashes_path}...")
            pv_data = np.load(hashes_path, allow_pickle=True)
            pv_dict = pv_data.item() if pv_data.shape == () else dict(pv_data)
            pv_train_hashes = pv_dict.get("train_hashes")  # (N_train, 64) bool

            if pv_train_hashes is None:
                print("[WARN] 'train_hashes' not found in hashes.npy. Skipping overlap check.")
            else:
                import imagehash

                def compute_phash_single(path_str):
                    try:
                        with Image.open(path_str) as p_img:
                            return imagehash.phash(p_img, hash_size=8).hash.flatten()
                    except Exception:
                        return np.zeros(64, dtype=bool)

                print(f"[*] Computing perceptual hashes for {len(valid_samples)} PlantDoc images...")
                t_hash_start = time.time()
                with ThreadPoolExecutor(max_workers=args.num_workers or 4) as ex:
                    pdoc_hashes_list = list(ex.map(compute_phash_single, [s["path"] for s in valid_samples]))
                pdoc_hashes = np.array(pdoc_hashes_list, dtype=bool)
                print(f"    - Completed PlantDoc hashes in {time.time() - t_hash_start:.2f}s.")

                # Vectorized distance search
                train_bits_f = pv_train_hashes.astype(np.float32) * 2.0 - 1.0
                pdoc_bits_f = pdoc_hashes.astype(np.float32) * 2.0 - 1.0

                pdoc_tensor = torch.from_numpy(pdoc_bits_f)
                train_tensor_t = torch.from_numpy(train_bits_f.T)

                chunk_size = 500
                min_dists = []
                with torch.inference_mode():
                    for start in range(0, len(pdoc_hashes), chunk_size):
                        chunk = pdoc_tensor[start:start + chunk_size]
                        dots = torch.matmul(chunk, train_tensor_t)
                        max_dots, _ = torch.max(dots, dim=1)
                        dists = ((64.0 - max_dots) / 2.0).round().long()
                        min_dists.append(dists.numpy())

                min_dists_arr = np.concatenate(min_dists, axis=0)
                leak_count = int(np.sum(min_dists_arr <= 4))
                leak_pct = float(leak_count / len(min_dists_arr) * 100.0)

                print(f"[*] Overlap Check Results:")
                print(f"    - PlantDoc samples with PV Train neighbor at Hamming dist <= 4: {leak_count} / {len(min_dists_arr)} ({leak_pct:.2f}%)")
                print(f"    - Overlap Independence Verified: {'YES' if leak_pct < 1.0 else 'NO (Minor overlap detected)'}")

                overlap_results = {
                    "total_plantdoc_samples": len(min_dists_arr),
                    "pv_train_neighbors_dist_le_4": leak_count,
                    "pv_train_neighbors_pct": leak_pct,
                    "independent": bool(leak_pct < 1.0)
                }

    # 13. Print Comparison Table
    print("\n" + "=" * 105)
    print(" PLANTDOC OUT-OF-DOMAIN BENCHMARK RESULTS COMPARISON TABLE")
    print("=" * 105)
    tbl_hdr = [
        "Model",
        "Top-1 (38)",
        "Top-1 (Restr)",
        "Top-5",
        "Macro-F1",
        "HighConf Top-1",
        "Conf>=0.7 Acc(Cov)",
        "Conf>=0.9 Acc(Cov)"
    ]
    tbl_fmt = "| {:<17} | {:>10} | {:>13} | {:>7} | {:>8} | {:>14} | {:>18} | {:>18} |"
    tbl_div = "+-------------------+------------+---------------+---------+----------+----------------+--------------------+--------------------+"

    print(tbl_div)
    print(tbl_fmt.format(*tbl_hdr))
    print(tbl_div)

    for name in all_eval_models:
        m = metrics_summary[name]
        c7 = m["confidence_calibration"]["threshold_0_7"]
        c9 = m["confidence_calibration"]["threshold_0_9"]
        c7_str = f"{c7['accuracy']*100:.1f}% ({c7['coverage']*100:.0f}%)"
        c9_str = f"{c9['accuracy']*100:.1f}% ({c9['coverage']*100:.0f}%)"
        hc_str = f"{m['high_confidence_subset']['top1_accuracy_all']*100:.2f}%"

        print(tbl_fmt.format(
            name,
            f"{m['top1_accuracy_all_38']*100:.2f}%",
            f"{m['top1_accuracy_restricted']*100:.2f}%",
            f"{m['top5_accuracy']*100:.2f}%",
            f"{m['macro_f1_all_38']:.4f}",
            hc_str,
            c7_str,
            c9_str
        ))
    print(tbl_div)

    # 14. Save metrics.json
    final_metrics_payload = {
        "metadata": {
            "dataset": "PlantDoc",
            "plantdoc_root": plantdoc_root,
            "splits_evaluated": target_splits,
            "total_samples": total_mapped_images,
            "device": str(device),
            "torch_version": torch.__version__,
            "limit": args.limit
        },
        "dataset_summary": {
            "images_per_split_and_class": split_class_counts,
            "total_mapped_images": total_mapped_images,
            "excluded_folders_count": len(excluded_folders),
            "excluded_folders": sorted(list(excluded_folders)),
            "excluded_files_count": excluded_files_count,
            "unreadable_files_count": unreadable_files_count,
            "unmapped_plantvillage_classes_count": len(unmapped_pv_classes),
            "unmapped_plantvillage_classes": unmapped_pv_classes
        },
        "models": metrics_summary,
        # Direct top-level access
        **metrics_summary
    }

    if overlap_results:
        final_metrics_payload["overlap_check"] = overlap_results

    metrics_out_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)

    print(f"\n[*] Full PlantDoc evaluation metrics saved to: {metrics_out_path}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Model Evaluation and Comparison Pipeline for Plant Disease Detection.

Evaluates trained model runs (and optional ensemble) across standard manifest-driven
test sets:
  - pv_clean_test (split column == pv_test and legacy_split == test, source plantvillage)
  - pv_test (split column == pv_test)
  - pv_legacy_test (legacy_split == test, source plantvillage)
  - field_test (split column == field_test)
  - field_val (split column == field_val)
  - other_pool (split column == other_pool, unmapped/unknown plants and diseases)

For each run and set, saves:
  - probs_<set>.npy (predicted probability matrix)
  - predictions_<set>.csv (per-image predictions, confidence, correctness, metadata)
  - eval/metrics.json (comprehensive metrics, confidence intervals, error analyses)

For class-labelled sets, computes:
  - top-1 over all 38 outputs
  - top-1 with predictions restricted to the classes present in that set
  - top-5 accuracy over all 38 outputs
  - macro-F1 over classes present in ground truth
  - per-class recall and support
  - 15 most frequent confusion pairs
  - share of predictions going to each class (top 10)
  - mean confidence when correct versus wrong
  - expected calibration error (ECE, 15 bins)
  - accuracy and coverage at confidence >= 0.7 and >= 0.9
  - cluster bootstrap 95% confidence interval for top-1 accuracy (1000 resamples, seed 42,
    resampling by group_id so near-duplicates are not treated as independent)

For field_test, reports all of the above separately for:
  - source == plantdoc
  - source == plantwild
  - confidence_level == high
  - confidence_level == medium

For other_pool:
  - share of images with confidence >= 0.7 and >= 0.9
  - mean max-probability
  - top 10 predicted classes (no accuracy)

CLI:
  python ml/evaluate.py (--run runs/<exp>/<timestamp> | --checkpoint <path> --name <label>) [...] [--sets pv_clean_test,field_test] [--ensemble] [--precision fp32|bf16] [--out <dir>]
"""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torchvision import transforms
import yaml

# Ensure repository root is on sys.path
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ml.analysis.gradcam import load_model_and_classes
from ml.src.datasets import (
    DEFAULT_SOURCE_ROOTS,
    DEFAULT_UNIFIED_CLASSES_YAML,
    ManifestDataset,
    load_unified_class_to_idx,
)
from ml.src.model_zoo import build_any_model, load_checkpoint_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

ALLOWED_SETS: Set[str] = {
    "pv_clean_test",
    "pv_test",
    "pv_legacy_test",
    "field_test",
    "field_val",
    "other_pool",
}


def compute_ece(confidences: np.ndarray, correct_mask: np.ndarray, num_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE) with 15 equally spaced probability bins in [0, 1].
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(correct_mask[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float(ece)


def compute_cluster_bootstrap_ci(
    group_ids: List[str],
    correct_mask: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes a cluster bootstrap 95% confidence interval for top-1 accuracy (1000 resamples,
    seed 42, resampling by group_id so near-duplicates are not counted as independent).
    """
    n = len(correct_mask)
    if n == 0:
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_95": [0.0, 0.0],
            "n_resamples": n_resamples,
            "n_clusters": 0,
            "seed": seed,
        }

    # Ensure non-empty group IDs (fallback to sample index if empty string)
    safe_groups = [g if (g and str(g).strip()) else f"indiv_{i}" for i, g in enumerate(group_ids)]
    unique_groups, group_indices = np.unique(safe_groups, return_inverse=True)
    k_clusters = len(unique_groups)

    # Precompute cluster-level sums of correct indicators and cluster sizes
    cluster_correct = np.bincount(group_indices, weights=correct_mask.astype(np.float64))
    cluster_sizes = np.bincount(group_indices)

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(k_clusters, size=(n_resamples, k_clusters), replace=True)

    boot_correct = cluster_correct[sampled_indices].sum(axis=1)
    boot_totals = cluster_sizes[sampled_indices].sum(axis=1)
    boot_accs = boot_correct / boot_totals

    ci_lower = float(np.percentile(boot_accs, 2.5))
    ci_upper = float(np.percentile(boot_accs, 97.5))
    point_estimate = float(np.mean(correct_mask))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_95": [ci_lower, ci_upper],
        "n_resamples": n_resamples,
        "n_clusters": k_clusters,
        "seed": seed,
    }


def compute_class_labelled_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    group_ids: List[str],
    idx_to_class: Dict[int, str],
) -> Dict[str, Any]:
    """
    Computes diagnostic metrics for class-labelled sets (pv_test, pv_legacy_test, field_test, field_val).
    """
    n_samples = len(targets)
    if n_samples == 0:
        return {
            "num_samples": 0,
            "top1_all": 0.0,
            "top1_restricted": 0.0,
            "top5": 0.0,
            "macro_f1": 0.0,
            "classes_covered": 0,
            "per_class": {},
            "top_confusion_pairs": [],
            "prediction_distribution_top10": [],
            "mean_confidence_correct": 0.0,
            "mean_confidence_wrong": 0.0,
            "ece": 0.0,
            "accuracy_coverage": {
                "conf_ge_0.7": {"coverage": 0.0, "accuracy": 0.0},
                "conf_ge_0.9": {"coverage": 0.0, "accuracy": 0.0},
            },
            "bootstrap_ci_95": {
                "point_estimate": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "ci_95": [0.0, 0.0],
            },
        }

    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    correct_mask = (preds == targets)

    # 1. Top-1 over all 38 outputs
    top1_all = float(np.mean(correct_mask))

    # 2. Top-1 with predictions restricted to the classes present in that set
    classes_present = np.unique(targets)
    classes_present = classes_present[classes_present >= 0]
    classes_present = np.sort(classes_present)

    if len(classes_present) > 0:
        sub_probs = probs[:, classes_present]
        sub_argmax = np.argmax(sub_probs, axis=1)
        restricted_preds = classes_present[sub_argmax]
        top1_restricted = float(np.mean(restricted_preds == targets))
    else:
        top1_restricted = 0.0

    # 3. Top-5 accuracy over all 38 outputs
    if probs.shape[1] >= 5:
        top5_indices = np.argsort(probs, axis=1)[:, -5:]
        top5_correct = np.array([t in top5 for t, top5 in zip(targets, top5_indices)], dtype=bool)
        top5_acc = float(np.mean(top5_correct))
    else:
        top5_acc = top1_all

    # 4. Macro-F1 over classes present in ground truth
    if len(classes_present) > 0:
        macro_f1 = float(f1_score(targets, preds, labels=classes_present, average="macro", zero_division=0))
    else:
        macro_f1 = 0.0

    # 5. Per-class recall and support
    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    for c in classes_present:
        c_mask = (targets == c)
        c_sup = int(np.sum(c_mask))
        c_rec = float(np.mean(preds[c_mask] == c)) if c_sup > 0 else 0.0
        cls_name = idx_to_class.get(int(c), str(c))
        per_class_metrics[cls_name] = {"recall": c_rec, "support": c_sup}

    # 6. 15 most frequent confusion pairs
    confusion_counts = Counter()
    for t, p in zip(targets, preds):
        if t != p:
            t_name = idx_to_class.get(int(t), str(t))
            p_name = idx_to_class.get(int(p), str(p))
            confusion_counts[(t_name, p_name)] += 1

    top_15_confusion = [
        {"true_class": t_cls, "pred_class": p_cls, "count": int(cnt)}
        for (t_cls, p_cls), cnt in confusion_counts.most_common(15)
    ]

    # 7. Share of predictions going to each class (top 10)
    pred_counts = Counter(preds)
    top_10_pred_shares = [
        {
            "class": idx_to_class.get(int(c), str(c)),
            "count": int(cnt),
            "share": float(cnt / n_samples),
        }
        for c, cnt in pred_counts.most_common(10)
    ]

    # 8. Mean confidence when correct versus wrong
    mean_conf_correct = float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else 0.0
    mean_conf_wrong = float(np.mean(confidences[~correct_mask])) if np.any(~correct_mask) else 0.0

    # 9. Expected calibration error (15 bins)
    ece = compute_ece(confidences, correct_mask, num_bins=15)

    # 10. Accuracy and coverage at confidence >= 0.7 and >= 0.9
    mask_07 = (confidences >= 0.7)
    cov_07 = float(np.mean(mask_07))
    acc_07 = float(np.mean(correct_mask[mask_07])) if np.any(mask_07) else 0.0

    mask_09 = (confidences >= 0.9)
    cov_09 = float(np.mean(mask_09))
    acc_09 = float(np.mean(correct_mask[mask_09])) if np.any(mask_09) else 0.0

    # 11. Cluster bootstrap 95% confidence interval for top-1 accuracy
    boot_ci = compute_cluster_bootstrap_ci(group_ids, correct_mask, n_resamples=1000, seed=42)

    return {
        "num_samples": n_samples,
        "top1_all": top1_all,
        "top1_restricted": top1_restricted,
        "top5": top5_acc,
        "macro_f1": macro_f1,
        "classes_covered": int(len(classes_present)),
        "per_class": per_class_metrics,
        "top_confusion_pairs": top_15_confusion,
        "prediction_distribution_top10": top_10_pred_shares,
        "mean_confidence_correct": mean_conf_correct,
        "mean_confidence_wrong": mean_conf_wrong,
        "ece": ece,
        "accuracy_coverage": {
            "conf_ge_0.7": {"coverage": cov_07, "accuracy": acc_07},
            "conf_ge_0.9": {"coverage": cov_09, "accuracy": acc_09},
        },
        "bootstrap_ci_95": boot_ci,
    }


def compute_other_pool_metrics(probs: np.ndarray, idx_to_class: Dict[int, str]) -> Dict[str, Any]:
    """
    Computes diagnostic metrics for other_pool (unknown plants and diseases).
    Reports no accuracy, only confidence shares, mean max-probability, and top 10 predicted classes.
    """
    n_samples = len(probs)
    if n_samples == 0:
        return {
            "num_samples": 0,
            "share_confidence_ge_0.7": 0.0,
            "share_confidence_ge_0.9": 0.0,
            "mean_max_prob": 0.0,
            "top_10_predicted_classes": [],
        }

    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    share_07 = float(np.mean(confidences >= 0.7))
    share_09 = float(np.mean(confidences >= 0.9))
    mean_max_prob = float(np.mean(confidences))

    pred_counts = Counter(preds)
    top_10_preds = [
        {
            "class": idx_to_class.get(int(c), str(c)),
            "count": int(cnt),
            "share": float(cnt / n_samples),
        }
        for c, cnt in pred_counts.most_common(10)
    ]

    return {
        "num_samples": n_samples,
        "share_confidence_ge_0.7": share_07,
        "share_confidence_ge_0.9": share_09,
        "mean_max_prob": mean_max_prob,
        "top_10_predicted_classes": top_10_preds,
    }


def build_eval_transform(run_config: Dict[str, Any]) -> transforms.Compose:
    """
    Builds the augmentation-independent evaluation transform from run config:
    - preprocess: squash | crop
    - img_size: 224 (default)
    - ImageNet normalization
    """
    img_size = run_config.get("img_size")
    if img_size is None:
        img_size = run_config.get("dataset", {}).get("img_size", 224)
    img_size = int(img_size)

    preprocess = run_config.get("preprocess")
    if preprocess is None:
        preprocess = run_config.get("dataset", {}).get("preprocess", "squash")
    preprocess = str(preprocess).lower().strip()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if preprocess == "crop":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:  # squash (default)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def build_dataset_for_split(
    set_name: str,
    manifest_path: Union[str, Path],
    transform: Callable,
    class_to_idx: Dict[str, int],
    source_roots: Optional[Dict[str, str]] = None,
    use_cache: bool = True,
) -> ManifestDataset:
    """
    Instantiates a ManifestDataset for a requested split, reads manifest.csv to attach full
    sample metadata (confidence_level, pv_dup, group_id), and enforces the rule:
      'rows with pv_dup=1 stay in the set only if their split says so.'
    """
    manifest_p = Path(manifest_path).resolve()

    # Determine split configuration
    if set_name == "pv_clean_test":
        split_column = "split"
        splits = ["pv_test"]
        source_filter = ["plantvillage"]
        filter_unmapped = True
    elif set_name == "pv_test":
        split_column = "split"
        splits = ["pv_test"]
        source_filter = None
        filter_unmapped = True
    elif set_name == "field_test":
        split_column = "split"
        splits = ["field_test"]
        source_filter = None
        filter_unmapped = True
    elif set_name == "field_val":
        split_column = "split"
        splits = ["field_val"]
        source_filter = None
        filter_unmapped = True
    elif set_name == "other_pool":
        split_column = "split"
        splits = ["other_pool"]
        source_filter = None
        filter_unmapped = False  # other_pool samples have unified_label 'other'
    elif set_name == "pv_legacy_test":
        split_column = "legacy_split"
        splits = ["test"]
        source_filter = ["plantvillage"]
        filter_unmapped = True
    else:
        raise ValueError(f"Unknown evaluation set: '{set_name}'. Allowed: {sorted(list(ALLOWED_SETS))}")

    ds = ManifestDataset(
        manifest_path=manifest_p,
        splits=splits,
        transform=transform,
        class_to_idx=class_to_idx,
        source_roots=source_roots,
        use_cache=use_cache,
        filter_unmapped=filter_unmapped,
        split_column=split_column,
        source_filter=source_filter,
    )

    # Read full manifest rows to retrieve confidence_level, pv_dup, and verify splits
    manifest_rows: Dict[int, Dict[str, str]] = {}
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader):
                manifest_rows[idx] = r

    # Filter and enrich dataset samples
    valid_samples = []
    valid_labels = []
    valid_sources = []
    valid_is_field = []

    for s in ds.samples:
        r_idx = s.get("manifest_row_idx")
        row = manifest_rows.get(r_idx, {}) if r_idx is not None else {}

        # For pv_clean_test, legacy_split must be 'test'
        if set_name == "pv_clean_test":
            if row.get("legacy_split", "").strip().lower() != "test":
                continue

        pv_dup_raw = str(row.get("pv_dup", "0")).strip()
        is_pv_dup = pv_dup_raw in ("1", "true", "True")

        # Rule: rows with pv_dup=1 stay in the set only if their split says so
        if is_pv_dup:
            row_split = row.get("split", "").strip()
            expected_split = "pv_test" if set_name in ("pv_legacy_test", "pv_clean_test") else set_name
            if row_split != expected_split:
                continue

        s["confidence_level"] = row.get("confidence_level", "").strip().lower()
        s["pv_dup"] = 1 if is_pv_dup else 0
        s["orig_label"] = row.get("orig_label", "").strip()
        s["orig_split"] = row.get("orig_split", "").strip()
        s["legacy_split"] = row.get("legacy_split", "").strip()

        valid_samples.append(s)
        valid_labels.append(s["label_idx"])
        valid_sources.append(s["source"])
        valid_is_field.append(s["is_field"])

    ds.samples = valid_samples
    ds.labels = valid_labels
    ds.sources = valid_sources
    ds.is_field = valid_is_field

    if set_name == "pv_clean_test":
        n_images = len(ds.samples)
        class_counts = Counter(s.get("unified_label") for s in ds.samples if s.get("unified_label"))
        if class_to_idx:
            counts = [class_counts.get(c, 0) for c in class_to_idx.keys()]
        else:
            counts = list(class_counts.values()) if class_counts else [0]
        min_count = int(min(counts)) if counts else 0
        med_count = float(np.median(counts)) if counts else 0.0
        med_str = f"{int(med_count)}" if med_count.is_integer() else f"{med_count:.1f}"
        print(f"pv_clean_test: {n_images} images, per-class min: {min_count}, median: {med_str}", flush=True)
        logger.info("pv_clean_test: %d images, per-class min: %d, median: %s", n_images, min_count, med_str)

    return ds


def run_inference(
    model: nn.Module,
    dataset: ManifestDataset,
    device: torch.device,
    use_cuda: bool,
    use_channels_last: bool = False,
    precision: str = "fp32",
    batch_size: int = 32,
    num_workers: int = 0,
) -> np.ndarray:
    """
    Runs model inference over dataset, returning float32 probabilities of shape (N, 38).
    Autocast is used only when precision is 'bf16'.
    """
    if len(dataset) == 0:
        num_classes = len(dataset.class_to_idx)
        return np.zeros((0, num_classes), dtype=np.float32)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    model.eval()
    all_probs: List[np.ndarray] = []
    use_bf16 = (precision.lower().strip() == "bf16")

    with torch.no_grad():
        for images, _, _ in loader:
            if images.dim() == 4 and use_cuda and use_channels_last:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            else:
                images = images.to(device, non_blocking=True)

            if use_bf16:
                with torch.autocast(
                    device_type="cuda" if use_cuda else "cpu",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    logits = model(images)
                    probs = torch.softmax(logits, dim=-1)
            else:
                logits = model(images)
                probs = torch.softmax(logits, dim=-1)

            all_probs.append(probs.cpu().float().numpy())

    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(dataset.class_to_idx)), dtype=np.float32)


def save_predictions_csv(
    csv_path: Path,
    dataset: ManifestDataset,
    probs: np.ndarray,
    idx_to_class: Dict[int, str],
) -> None:
    """
    Saves per-image prediction records to CSV:
    path, source, true_label, true_idx, pred_label, pred_idx, confidence, correct, group_id, confidence_level, pv_dup.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "source",
        "true_label",
        "true_idx",
        "pred_label",
        "pred_idx",
        "confidence",
        "correct",
        "group_id",
        "confidence_level",
        "pv_dup",
    ]

    preds = np.argmax(probs, axis=1) if len(probs) > 0 else np.array([], dtype=int)
    confidences = np.max(probs, axis=1) if len(probs) > 0 else np.array([], dtype=float)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, sample in enumerate(dataset.samples):
            pred_idx = int(preds[i])
            pred_label = idx_to_class.get(pred_idx, str(pred_idx))
            conf = float(confidences[i])
            true_idx = int(sample.get("label_idx", -1))
            true_label = sample.get("unified_label", "")

            # For unmapped/other_pool, correctness is not defined
            correct_val = int(pred_idx == true_idx) if true_idx >= 0 else ""

            row = {
                "path": sample.get("path", ""),
                "source": sample.get("source", ""),
                "true_label": true_label,
                "true_idx": true_idx,
                "pred_label": pred_label,
                "pred_idx": pred_idx,
                "confidence": f"{conf:.6f}",
                "correct": correct_val,
                "group_id": sample.get("group_id", ""),
                "confidence_level": sample.get("confidence_level", ""),
                "pv_dup": sample.get("pv_dup", 0),
            }
            writer.writerow(row)


def print_comparison_table(table_rows: List[Dict[str, Any]]) -> None:
    """
    Prints a comparison table to the console:
      Columns: Run / Model, pv top-1, field_test Top-1, doc top-1, wild top-1,
               field_test Macro-F1, field_test ECE, other_pool Conf>=0.9.
    """
    headers = [
        "Run / Model",
        "pv top-1",
        "field_test Top-1",
        "doc top-1",
        "wild top-1",
        "field_test Macro-F1",
        "field_test ECE",
        "other_pool Conf>=0.9",
    ]

    def fmt_val(val: Optional[float]) -> str:
        if val is None:
            return "-"
        return f"{val:.4f}"

    rows_data = []
    for r in table_rows:
        pv_top1_str = r.get("pv_top1")
        if pv_top1_str is None or pv_top1_str == "":
            if r.get("pv_clean_test_top1") is not None:
                pv_top1_str = f"{r['pv_clean_test_top1']:.4f} (clean)"
            elif r.get("pv_test_top1") is not None:
                pv_top1_str = f"{r['pv_test_top1']:.4f} (test)"
            elif r.get("pv_legacy_test_top1") is not None:
                pv_top1_str = f"{r['pv_legacy_test_top1']:.4f} (legacy)"
            else:
                pv_top1_str = "-"

        rows_data.append([
            r.get("name", "Unknown"),
            pv_top1_str,
            fmt_val(r.get("field_test_top1")),
            fmt_val(r.get("doc_top1")),
            fmt_val(r.get("wild_top1")),
            fmt_val(r.get("field_test_macro_f1")),
            fmt_val(r.get("field_test_ece")),
            fmt_val(r.get("other_pool_conf_09")),
        ])

    col_widths = [len(h) for h in headers]
    for row in rows_data:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_str = "| " + " | ".join(h.ljust(col_widths[i]) if i == 0 else h.rjust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print("\n" + sep)
    print(header_str)
    print(sep)
    for row in rows_data:
        row_str = "| " + " | ".join(val.ljust(col_widths[i]) if i == 0 else val.rjust(col_widths[i]) for i, val in enumerate(row)) + " |"
        print(row_str)
    print(sep + "\n")


def evaluate_dataset_probs(
    set_name: str,
    probs: np.ndarray,
    dataset: ManifestDataset,
    idx_to_class: Dict[int, str],
) -> Dict[str, Any]:
    """
    Dispatches metrics computation for a specific dataset split and its probability matrix.
    """
    if set_name == "other_pool":
        return compute_other_pool_metrics(probs=probs, idx_to_class=idx_to_class)

    targets = np.array([s["label_idx"] for s in dataset.samples], dtype=int)
    group_ids = [s.get("group_id", "") for s in dataset.samples]

    metrics = compute_class_labelled_metrics(
        targets=targets,
        probs=probs,
        group_ids=group_ids,
        idx_to_class=idx_to_class,
    )

    # For field_test, report breakdowns by source and confidence_level
    if set_name == "field_test" and len(targets) > 0:
        sources_arr = np.array([s.get("source", "").lower().strip() for s in dataset.samples])
        conf_arr = np.array([s.get("confidence_level", "").lower().strip() for s in dataset.samples])

        by_source = {}
        for src in ["plantdoc", "plantwild"]:
            s_mask = (sources_arr == src)
            if np.any(s_mask):
                by_source[src] = compute_class_labelled_metrics(
                    targets=targets[s_mask],
                    probs=probs[s_mask],
                    group_ids=[group_ids[i] for i in range(len(group_ids)) if s_mask[i]],
                    idx_to_class=idx_to_class,
                )
            else:
                by_source[src] = {"num_samples": 0}

        by_conf = {}
        for clevel in ["high", "medium"]:
            c_mask = (conf_arr == clevel)
            if np.any(c_mask):
                by_conf[clevel] = compute_class_labelled_metrics(
                    targets=targets[c_mask],
                    probs=probs[c_mask],
                    group_ids=[group_ids[i] for i in range(len(group_ids)) if c_mask[i]],
                    idx_to_class=idx_to_class,
                )
            else:
                by_conf[clevel] = {"num_samples": 0}

        metrics["by_source"] = by_source
        metrics["by_confidence_level"] = by_conf

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Evaluation and Cross-Run Comparison for Plant Disease Detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single run on default sets (pv_clean_test, field_test):
  python ml/evaluate.py --run runs/a_pv_only/20261120_120000

  # Evaluate a legacy PlantVillage checkpoint:
  python ml/evaluate.py --checkpoint ml/checkpoints/resnet50/best_model.pth --name resnet50

  # Compare a trained run against the baseline checkpoint and compute ensemble:
  python ml/evaluate.py \\
      --run runs/a_pv_only/20261120_120000 \\
      --checkpoint ml/checkpoints/resnet50/best_model.pth --name resnet50 \\
      --sets pv_clean_test,field_test \\
      --ensemble
        """,
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        nargs="+",
        default=[],
        help="Path to run directory containing best.pth and config.yaml (repeatable for multiple runs).",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        nargs="+",
        default=[],
        help="Path to model checkpoint (e.g. ml/checkpoints/resnet50/best_model.pth). Alternative to --run.",
    )
    parser.add_argument(
        "--name",
        dest="names",
        action="append",
        nargs="+",
        default=[],
        help="Label for the checkpoint model (e.g. resnet50). Used with --checkpoint to save output under runs/baseline_<label>/eval/.",
    )
    parser.add_argument(
        "--sets",
        type=str,
        default="pv_clean_test,field_test",
        help="Comma-separated dataset splits to evaluate (allowed: pv_clean_test, pv_test, pv_legacy_test, field_test, field_val, other_pool). Default: pv_clean_test,field_test.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Average probabilities of all runs that share a set and report ensemble metrics as an extra row.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional destination directory for ensemble artifacts and consolidated comparison outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count (default: 0 on Windows, 4 on Linux).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (default: 'cuda' if available else 'cpu').",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Path to manifest CSV file (default: data/manifest.csv).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
        help="Inference precision: 'fp32' or 'bf16' (default: 'fp32'). Autocast is enabled only when 'bf16'.",
    )

    args = parser.parse_args()

    # Flatten run directory arguments
    run_paths: List[Path] = []
    if args.runs:
        for item in args.runs:
            if isinstance(item, list):
                for sub in item:
                    run_paths.append(Path(sub).resolve())
            else:
                run_paths.append(Path(item).resolve())

    # Deduplicate while preserving order
    seen_paths = set()
    unique_run_paths: List[Path] = []
    for p in run_paths:
        if p not in seen_paths:
            seen_paths.add(p)
            unique_run_paths.append(p)

    # Flatten checkpoint and name arguments
    ckpt_paths: List[Path] = []
    if args.checkpoints:
        for item in args.checkpoints:
            if isinstance(item, list):
                for sub in item:
                    ckpt_paths.append(Path(sub).resolve())
            else:
                ckpt_paths.append(Path(item).resolve())

    name_labels: List[str] = []
    if args.names:
        for item in args.names:
            if isinstance(item, list):
                for sub in item:
                    name_labels.append(str(sub).strip())
            else:
                name_labels.append(str(item).strip())

    if not unique_run_paths and not ckpt_paths:
        parser.error("Must provide either --run <run_dir> or --checkpoint <path> --name <label>.")

    # Build evaluation targets
    eval_targets: List[Dict[str, Any]] = []

    for run_dir in unique_run_paths:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        best_pth = run_dir / "best.pth"
        if not best_pth.exists():
            raise FileNotFoundError(f"best.pth checkpoint missing in run directory: {run_dir}")

        run_cfg_path = run_dir / "config.yaml"
        eval_targets.append({
            "type": "run",
            "run_dir": run_dir,
            "checkpoint_path": best_pth,
            "config_path": run_cfg_path if run_cfg_path.exists() else Path("ml/legacy/config.yaml"),
            "eval_dir": run_dir / "eval",
            "name": f"{run_dir.parent.name}/{run_dir.name}",
            "identifier": str(run_dir),
            "label": run_dir.name,
        })

    for i, ckpt in enumerate(ckpt_paths):
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
        label = name_labels[i] if i < len(name_labels) else (
            ckpt.parent.name if ckpt.parent.name not in ("checkpoints", "") else ckpt.stem
        )
        baseline_name = f"baseline_{label}"
        eval_targets.append({
            "type": "checkpoint",
            "run_dir": None,
            "checkpoint_path": ckpt,
            "config_path": Path("ml/legacy/config.yaml"),
            "eval_dir": Path("runs") / baseline_name / "eval",
            "name": baseline_name,
            "identifier": baseline_name,
            "label": label,
        })

    # Validate requested sets
    requested_sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    for s in requested_sets:
        if s not in ALLOWED_SETS:
            raise ValueError(f"Invalid evaluation set '{s}'. Allowed: {sorted(list(ALLOWED_SETS))}")

    if not requested_sets:
        raise ValueError("No valid evaluation sets specified.")

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")

    if args.num_workers is None:
        is_windows = (sys.platform == "win32") or (os.name == "nt")
        num_workers = 0 if is_windows else 4
    else:
        num_workers = int(args.num_workers)

    logger.info("Evaluation device: %s | num_workers: %d | sets: %s", device, num_workers, requested_sets)

    # Dictionary storing probabilities per set per run/checkpoint: probs_by_set[set_name][id_str] = np.ndarray
    probs_by_set: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in requested_sets}
    datasets_by_set: Dict[str, ManifestDataset] = {}
    class_mapping_ref: Optional[Dict[str, int]] = None
    idx_mapping_ref: Optional[Dict[int, str]] = None

    comparison_rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Evaluate Individual Runs and Checkpoints
    # -------------------------------------------------------------------------
    for target in eval_targets:
        target_name = target["name"]
        target_id = target["identifier"]
        ckpt_path = target["checkpoint_path"]
        run_cfg_path = target["config_path"]
        run_eval_dir = target["eval_dir"]

        run_cfg: Dict[str, Any] = {}
        if target["type"] == "run" and run_cfg_path and run_cfg_path.exists():
            with open(run_cfg_path, "r", encoding="utf-8") as f:
                run_cfg = yaml.safe_load(f) or {}

        logger.info("Loading model and checkpoint for '%s' from: %s", target_name, ckpt_path)
        fallback_cfg = "ml/legacy/config.yaml"

        ckpt_raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        target_arch = (
            ckpt_raw.get("architecture")
            or ckpt_raw.get("config", {}).get("model", {}).get("architecture")
            or run_cfg.get("model", {}).get("architecture", "")
        )
        target_arch = str(target_arch).lower().strip()

        if target_arch in ("fusion_concat", "fusion_gated"):
            model, loaded_cfg, arch, class_to_idx, idx_to_class = load_checkpoint_model(
                checkpoint_path=str(ckpt_path),
                config=run_cfg,
                device=device,
            )
        else:
            model, loaded_cfg, arch, class_to_idx, idx_to_class = load_model_and_classes(
                config_path=str(run_cfg_path if (run_cfg_path and run_cfg_path.exists()) else fallback_cfg),
                checkpoint_path=str(ckpt_path),
                device=device,
            )

        # Merge configs for complete transform resolution (uses checkpoint's own config for preprocessing)
        merged_cfg = {**loaded_cfg, **run_cfg}
        eval_transform = build_eval_transform(merged_cfg)

        if class_mapping_ref is None:
            class_mapping_ref = class_to_idx
            idx_mapping_ref = idx_to_class

        # Determine channels_last: false for efficientnet and fusion architectures, true otherwise
        use_channels_last = not (arch.lower().startswith("efficientnet") or arch.lower().startswith("fusion"))
        if use_cuda and use_channels_last:
            model = model.to(memory_format=torch.channels_last)

        run_eval_dir.mkdir(parents=True, exist_ok=True)

        target_type = target.get("type")
        split_col = run_cfg.get("dataset", {}).get("split_column") or run_cfg.get("split_column")
        is_legacy_target = (target_type == "checkpoint") or (split_col == "legacy_split")

        raw_init_ckpt = (
            run_cfg.get("init_checkpoint")
            or run_cfg.get("model", {}).get("init_checkpoint")
        )
        if not raw_init_ckpt and target.get("type") == "run":
            raw_init_ckpt = (
                merged_cfg.get("init_checkpoint")
                or merged_cfg.get("model", {}).get("init_checkpoint")
            )
        has_init_checkpoint = bool(
            target.get("type") == "run"
            and raw_init_ckpt
            and str(raw_init_ckpt).strip().lower() not in ("", "none", "null", "false")
        )

        run_metrics_output: Dict[str, Any] = {
            "target": target_name,
            "architecture": arch,
            "checkpoint": str(ckpt_path),
            "precision": args.precision,
            "timestamp": datetime.now().isoformat(),
            "skipped_sets": {},
            "sets": {},
        }
        if target["type"] == "run":
            run_metrics_output["run_dir"] = str(target["run_dir"])
        else:
            run_metrics_output["name"] = target_name
            run_metrics_output["label"] = target["label"]

        # Track row metrics for comparison table
        row_entry: Dict[str, Any] = {
            "name": target_name,
            "pv_clean_test_top1": None,
            "pv_test_top1": None,
            "pv_legacy_test_top1": None,
            "pv_top1": "-",
            "field_test_top1": None,
            "doc_top1": None,
            "wild_top1": None,
            "field_test_macro_f1": None,
            "field_test_ece": None,
            "other_pool_conf_09": None,
        }

        manifest_file = args.manifest or merged_cfg.get("manifest_path", "data/manifest.csv")
        source_roots = merged_cfg.get("sources", DEFAULT_SOURCE_ROOTS)
        use_cache = merged_cfg.get("use_cache", True)

        skipped_sets: Dict[str, str] = {}

        for set_name in requested_sets:
            # Leakage Guard: Check for cross-split evaluation leaks
            if has_init_checkpoint and set_name in ("pv_test", "pv_legacy_test"):
                msg = (
                    f"Target was initialised from a model that saw the legacy training split "
                    f"(init_checkpoint: {raw_init_ckpt}); skipped {set_name} to prevent train/test data leakage."
                )
                banner = (
                    "\n" + "!" * 78 + "\n"
                    f"WARNING [Leakage Guard]: Target '{target_name}' was initialised from a model that saw\n"
                    f"the legacy training split (init_checkpoint: {raw_init_ckpt}).\n"
                    f"Skipping evaluation set '{set_name}' to prevent train/test data leakage.\n"
                    + "!" * 78 + "\n"
                )
                print(banner, file=sys.stderr)
                logger.warning("LEAKAGE GUARD: Target '%s' skipped set '%s': %s", target_name, set_name, msg)
                skipped_sets[set_name] = msg
                continue

            if is_legacy_target and set_name == "pv_test":
                msg = "Target uses legacy PlantVillage split; skipped pv_test to prevent train/test data leakage."
                banner = (
                    "\n" + "!" * 78 + "\n"
                    f"WARNING [Leakage Guard]: Target '{target_name}' uses legacy PlantVillage split.\n"
                    f"Skipping evaluation set 'pv_test' to prevent train/test data leakage.\n"
                    + "!" * 78 + "\n"
                )
                print(banner, file=sys.stderr)
                logger.warning("LEAKAGE GUARD: Target '%s' skipped set 'pv_test': %s", target_name, msg)
                skipped_sets[set_name] = msg
                continue

            if (not is_legacy_target) and set_name == "pv_legacy_test":
                msg = "Target uses new manifest split; skipped pv_legacy_test to prevent train/test data leakage."
                banner = (
                    "\n" + "!" * 78 + "\n"
                    f"WARNING [Leakage Guard]: Target '{target_name}' uses new manifest split.\n"
                    f"Skipping evaluation set 'pv_legacy_test' to prevent train/test data leakage.\n"
                    + "!" * 78 + "\n"
                )
                print(banner, file=sys.stderr)
                logger.warning("LEAKAGE GUARD: Target '%s' skipped set 'pv_legacy_test': %s", target_name, msg)
                skipped_sets[set_name] = msg
                continue

            logger.info("Evaluating [%s] on split '%s'...", target_name, set_name)

            ds = build_dataset_for_split(
                set_name=set_name,
                manifest_path=manifest_file,
                transform=eval_transform,
                class_to_idx=class_to_idx,
                source_roots=source_roots,
                use_cache=use_cache,
            )
            datasets_by_set[set_name] = ds

            probs = run_inference(
                model=model,
                dataset=ds,
                device=device,
                use_cuda=use_cuda,
                use_channels_last=use_channels_last,
                precision=args.precision,
                batch_size=args.batch_size,
                num_workers=num_workers,
            )

            # Record probabilities for ensemble
            probs_by_set[set_name][target_id] = probs

            # Save probs_<set>.npy
            npy_path = run_eval_dir / f"probs_{set_name}.npy"
            np.save(npy_path, probs)

            # Save predictions_<set>.csv
            csv_path = run_eval_dir / f"predictions_{set_name}.csv"
            save_predictions_csv(csv_path, dataset=ds, probs=probs, idx_to_class=idx_to_class)

            # Compute split metrics
            set_metrics = evaluate_dataset_probs(
                set_name=set_name,
                probs=probs,
                dataset=ds,
                idx_to_class=idx_to_class,
            )
            run_metrics_output["sets"][set_name] = set_metrics

            # Update comparison row
            if set_name == "pv_clean_test":
                row_entry["pv_clean_test_top1"] = set_metrics.get("top1_all")
            elif set_name == "pv_test":
                row_entry["pv_test_top1"] = set_metrics.get("top1_all")
            elif set_name == "pv_legacy_test":
                row_entry["pv_legacy_test_top1"] = set_metrics.get("top1_all")
            elif set_name == "field_test":
                row_entry["field_test_top1"] = set_metrics.get("top1_all")
                row_entry["field_test_macro_f1"] = set_metrics.get("macro_f1")
                row_entry["field_test_ece"] = set_metrics.get("ece")
                by_source = set_metrics.get("by_source", {})
                doc_m = by_source.get("plantdoc", {})
                if doc_m.get("num_samples", 0) > 0:
                    row_entry["doc_top1"] = doc_m.get("top1_all")
                wild_m = by_source.get("plantwild", {})
                if wild_m.get("num_samples", 0) > 0:
                    row_entry["wild_top1"] = wild_m.get("top1_all")
            elif set_name == "other_pool":
                row_entry["other_pool_conf_09"] = set_metrics.get("share_confidence_ge_0.9")

        target["evaluated_sets"] = [s for s in requested_sets if s not in skipped_sets]
        run_metrics_output["skipped_sets"] = skipped_sets

        if row_entry.get("pv_clean_test_top1") is not None:
            row_entry["pv_top1"] = f"{row_entry['pv_clean_test_top1']:.4f} (clean)"
        elif row_entry.get("pv_test_top1") is not None:
            row_entry["pv_top1"] = f"{row_entry['pv_test_top1']:.4f} (test)"
        elif row_entry.get("pv_legacy_test_top1") is not None:
            row_entry["pv_top1"] = f"{row_entry['pv_legacy_test_top1']:.4f} (legacy)"
        else:
            row_entry["pv_top1"] = "-"

        # Save metrics.json inside target directory under eval/
        metrics_json_path = run_eval_dir / "metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(run_metrics_output, f, indent=2)

        logger.info("Saved evaluation outputs to: %s", run_eval_dir)
        comparison_rows.append(row_entry)

    # -------------------------------------------------------------------------
    # 2. Ensemble Evaluation (if requested)
    # -------------------------------------------------------------------------
    if args.ensemble:
        ensemble_name = "Ensemble"
        logger.info("Computing ensemble evaluation across %d target(s)...", len(eval_targets))

        if args.out:
            ensemble_dir = Path(args.out) / "eval"
        else:
            ensemble_dir = Path("runs/ensemble/eval")
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        ensemble_row: Dict[str, Any] = {
            "name": ensemble_name,
            "pv_clean_test_top1": None,
            "pv_test_top1": None,
            "pv_legacy_test_top1": None,
            "pv_top1": "-",
            "field_test_top1": None,
            "doc_top1": None,
            "wild_top1": None,
            "field_test_macro_f1": None,
            "field_test_ece": None,
            "other_pool_conf_09": None,
        }

        ensemble_metrics_output: Dict[str, Any] = {
            "ensemble_targets": [t["name"] for t in eval_targets],
            "precision": args.precision,
            "timestamp": datetime.now().isoformat(),
            "sets": {},
        }

        for set_name in requested_sets:
            # Candidate targets that actually evaluated this set
            candidate_targets = [
                t for t in eval_targets
                if set_name in t.get("evaluated_sets", []) and t["identifier"] in probs_by_set.get(set_name, {})
            ]
            if not candidate_targets:
                print(f"Ensemble members for '{set_name}': None")
                logger.warning("No evaluated targets available for ensemble on set '%s'.", set_name)
                continue

            # Require the same list of evaluated sets for every member of a set
            groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
            for t in candidate_targets:
                sig = tuple(sorted(t.get("evaluated_sets", [])))
                groups.setdefault(sig, []).append(t)

            sorted_groups = sorted(
                groups.items(),
                key=lambda item: (len(item[1]), "pv_clean_test" in item[0], "pv_test" in item[0], len(item[0])),
                reverse=True,
            )
            chosen_sig, chosen_targets = sorted_groups[0]
            included_names = [t["name"] for t in chosen_targets]

            # Print which members were included in each set
            print(f"Ensemble members for '{set_name}': {', '.join(included_names)}")
            logger.info("Ensemble members for '%s': %s (signature: %s)", set_name, included_names, list(chosen_sig))

            if len(groups) > 1:
                excluded_names = [t["name"] for t in candidate_targets if t not in chosen_targets]
                logger.warning(
                    "Ensemble for set '%s': excluded targets %s because their evaluated sets differed from selected signature %s.",
                    set_name,
                    excluded_names,
                    list(chosen_sig),
                )

            # Average probabilities across the selected members for this set
            run_prob_list = [probs_by_set[set_name][t["identifier"]] for t in chosen_targets]
            ensemble_probs = np.mean(run_prob_list, axis=0)
            ds = datasets_by_set[set_name]

            # Save ensemble probs and predictions
            npy_path = ensemble_dir / f"probs_{set_name}.npy"
            np.save(npy_path, ensemble_probs)

            csv_path = ensemble_dir / f"predictions_{set_name}.csv"
            save_predictions_csv(csv_path, dataset=ds, probs=ensemble_probs, idx_to_class=idx_mapping_ref)

            ens_set_metrics = evaluate_dataset_probs(
                set_name=set_name,
                probs=ensemble_probs,
                dataset=ds,
                idx_to_class=idx_mapping_ref,
            )
            ens_set_metrics["members"] = included_names
            ens_set_metrics["evaluated_sets_signature"] = list(chosen_sig)
            ensemble_metrics_output["sets"][set_name] = ens_set_metrics

            if set_name == "pv_clean_test":
                ensemble_row["pv_clean_test_top1"] = ens_set_metrics.get("top1_all")
            elif set_name == "pv_test":
                ensemble_row["pv_test_top1"] = ens_set_metrics.get("top1_all")
            elif set_name == "pv_legacy_test":
                ensemble_row["pv_legacy_test_top1"] = ens_set_metrics.get("top1_all")
            elif set_name == "field_test":
                ensemble_row["field_test_top1"] = ens_set_metrics.get("top1_all")
                ensemble_row["field_test_macro_f1"] = ens_set_metrics.get("macro_f1")
                ensemble_row["field_test_ece"] = ens_set_metrics.get("ece")
                by_source = ens_set_metrics.get("by_source", {})
                doc_m = by_source.get("plantdoc", {})
                if doc_m.get("num_samples", 0) > 0:
                    ensemble_row["doc_top1"] = doc_m.get("top1_all")
                wild_m = by_source.get("plantwild", {})
                if wild_m.get("num_samples", 0) > 0:
                    ensemble_row["wild_top1"] = wild_m.get("top1_all")
            elif set_name == "other_pool":
                ensemble_row["other_pool_conf_09"] = ens_set_metrics.get("share_confidence_ge_0.9")

        if ensemble_row.get("pv_clean_test_top1") is not None:
            ensemble_row["pv_top1"] = f"{ensemble_row['pv_clean_test_top1']:.4f} (clean)"
        elif ensemble_row.get("pv_test_top1") is not None:
            ensemble_row["pv_top1"] = f"{ensemble_row['pv_test_top1']:.4f} (test)"
        elif ensemble_row.get("pv_legacy_test_top1") is not None:
            ensemble_row["pv_top1"] = f"{ensemble_row['pv_legacy_test_top1']:.4f} (legacy)"
        else:
            ensemble_row["pv_top1"] = "-"

        # Save ensemble metrics.json
        with open(ensemble_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(ensemble_metrics_output, f, indent=2)

        comparison_rows.append(ensemble_row)
        logger.info("Saved ensemble evaluation outputs to: %s", ensemble_dir)

    # -------------------------------------------------------------------------
    # 3. Print Comparison Table
    # -------------------------------------------------------------------------
    print_comparison_table(comparison_rows)

    # If --out is provided, write consolidated comparison report
    if args.out:
        out_p = Path(args.out)
        out_p.mkdir(parents=True, exist_ok=True)
        summary_file = out_p / "comparison_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({"comparison_table": comparison_rows}, f, indent=2)
        logger.info("Consolidated comparison summary written to: %s", summary_file)


if __name__ == "__main__":
    main()

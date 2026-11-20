#!/usr/bin/env python3
"""
PlantDoc In-Depth Analysis Script.

Computes comprehensive diagnostic metrics on PlantDoc out-of-domain evaluation results:
  (1) Top 10 most-predicted classes and prediction shares, marking classes with no PlantDoc counterpart.
  (2) Crop-level accuracy (crop prefix) and conditional exact class accuracy given correct crop.
  (3) Healthy vs. diseased binary classification accuracy, false disease rate, and false healthy rate.
  (4) Top-1 accuracy over all 38 classes split by split (train/test) and confidence level (high/medium).
  (5) 95% bootstrap confidence interval (1,000 resamples, seed 42) for top-1 accuracy.
  (6) Optional duplicate check inside PlantDoc (perceptual hashing Hamming distance <= 4)
      and 24-image contact sheets if --plantdoc-dir is provided.

Runs efficiently on a laptop CPU using files.csv, probs_<name>.npy, and checkpoint class_to_idx.
"""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def load_inputs(results_dir: str, checkpoint_path: str):
    """Loads files.csv, checkpoint class_to_idx, and model probabilities."""
    # 1. Load checkpoint to retrieve class_to_idx mapping
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    if "class_to_idx" not in ckpt:
        raise KeyError(f"'class_to_idx' not found in checkpoint: {checkpoint_path}")

    class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
    idx_to_class: Dict[int, str] = {v: k for k, v in class_to_idx.items()}

    # 2. Load files.csv
    files_csv_path = os.path.join(results_dir, "files.csv")
    if not os.path.exists(files_csv_path):
        raise FileNotFoundError(f"files.csv not found at: {files_csv_path}")

    records: List[Dict[str, str]] = []
    with open(files_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        raise ValueError(f"No records found in {files_csv_path}")

    # Ground truth targets
    y_true = np.array([class_to_idx[r["mapped_class"]] for r in records], dtype=np.int64)

    # 3. Load model probabilities
    models = ["resnet50", "efficientnet_v2", "vit", "ensemble"]
    probs: Dict[str, np.ndarray] = {}
    for model_name in models:
        prob_path = os.path.join(results_dir, f"probs_{model_name}.npy")
        if not os.path.exists(prob_path):
            raise FileNotFoundError(f"Probability file missing for '{model_name}': {prob_path}")
        p = np.load(prob_path)
        if p.shape[0] != len(records):
            raise ValueError(
                f"Row mismatch in {prob_path}: expected {len(records)} rows, got {p.shape[0]}"
            )
        probs[model_name] = p

    return records, class_to_idx, idx_to_class, y_true, probs


def run_model_analysis(
    records: List[Dict[str, str]],
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    y_true: np.ndarray,
    probs: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Any], List[str]]:
    """Computes analyses (1) through (5) for all models."""
    n_samples = len(records)
    mapped_classes_in_csv = set(r["mapped_class"] for r in records)
    all_pv_classes = set(class_to_idx.keys())
    unmapped_pv_classes = sorted(list(all_pv_classes - mapped_classes_in_csv))

    splits = np.array([r["split"] for r in records])
    conf_levels = np.array([r["confidence_level"] for r in records])
    true_crops = np.array([r["mapped_class"].split("___")[0] for r in records])
    true_healthy = np.array(["___healthy" in r["mapped_class"] for r in records], dtype=bool)

    train_mask = (splits == "train")
    test_mask = (splits == "test")
    high_mask = (conf_levels == "high")
    med_mask = (conf_levels == "medium")

    results_by_model: Dict[str, Any] = {}

    for model_name, p in probs.items():
        preds = np.argmax(p, axis=1)
        correct_mask = (preds == y_true)
        top1_acc = float(np.mean(correct_mask))

        # (1) Top 10 most-predicted classes with share and counterpart status
        counter = Counter(preds)
        top10_list = []
        for cls_idx, cnt in counter.most_common(10):
            cname = idx_to_class[cls_idx]
            has_counterpart = cname in mapped_classes_in_csv
            share = float(cnt / n_samples)
            top10_list.append({
                "class": cname,
                "count": int(cnt),
                "share": share,
                "has_plantdoc_counterpart": has_counterpart
            })

        # (2) Crop-level accuracy and conditional exact accuracy given crop
        pred_crops = np.array([idx_to_class[idx].split("___")[0] for idx in preds])
        crop_correct_mask = (pred_crops == true_crops)
        crop_accuracy = float(np.mean(crop_correct_mask))
        correct_crop_cnt = int(np.sum(crop_correct_mask))

        if correct_crop_cnt > 0:
            exact_given_crop = float(np.mean(preds[crop_correct_mask] == y_true[crop_correct_mask]))
        else:
            exact_given_crop = 0.0

        # (3) Healthy vs. diseased binary accuracy and error rates
        pred_healthy = np.array(["___healthy" in idx_to_class[idx] for idx in preds], dtype=bool)
        binary_acc = float(np.mean(pred_healthy == true_healthy))

        # False disease rate (healthy predicted diseased)
        healthy_count = int(np.sum(true_healthy))
        if healthy_count > 0:
            rate_healthy_pred_diseased = float(np.mean(~pred_healthy[true_healthy]))
        else:
            rate_healthy_pred_diseased = 0.0

        # False healthy rate (diseased predicted healthy)
        diseased_count = int(np.sum(~true_healthy))
        if diseased_count > 0:
            rate_diseased_pred_healthy = float(np.mean(pred_healthy[~true_healthy]))
        else:
            rate_diseased_pred_healthy = 0.0

        # (4) Top-1 accuracy split by split (train/test) and confidence (high/medium)
        acc_train = float(np.mean(preds[train_mask] == y_true[train_mask])) if np.sum(train_mask) > 0 else 0.0
        acc_test = float(np.mean(preds[test_mask] == y_true[test_mask])) if np.sum(test_mask) > 0 else 0.0
        acc_high = float(np.mean(preds[high_mask] == y_true[high_mask])) if np.sum(high_mask) > 0 else 0.0
        acc_med = float(np.mean(preds[med_mask] == y_true[med_mask])) if np.sum(med_mask) > 0 else 0.0

        # (5) 95% bootstrap confidence interval (1000 resamples, seed 42)
        rng = np.random.RandomState(42)
        boot_indices = rng.randint(0, n_samples, size=(1000, n_samples))
        correct_indicators = correct_mask.astype(np.float64)
        boot_accs = correct_indicators[boot_indices].mean(axis=1)
        ci_lower = float(np.percentile(boot_accs, 2.5))
        ci_upper = float(np.percentile(boot_accs, 97.5))

        results_by_model[model_name] = {
            "top1_accuracy_all_38": top1_acc,
            "bootstrap_ci_95": {
                "point_estimate": top1_acc,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_resamples": 1000,
                "confidence_level": 0.95,
                "seed": 42
            },
            "crop_level_accuracy": {
                "crop_accuracy": crop_accuracy,
                "exact_class_accuracy_given_correct_crop": exact_given_crop,
                "correct_crop_count": correct_crop_cnt,
                "total_samples": n_samples
            },
            "healthy_vs_diseased": {
                "binary_accuracy": binary_acc,
                "rate_healthy_predicted_diseased": rate_healthy_pred_diseased,
                "rate_diseased_predicted_healthy": rate_diseased_pred_healthy,
                "healthy_count": healthy_count,
                "diseased_count": diseased_count
            },
            "accuracy_by_split": {
                "train": {"accuracy": acc_train, "count": int(np.sum(train_mask))},
                "test": {"accuracy": acc_test, "count": int(np.sum(test_mask))}
            },
            "accuracy_by_confidence": {
                "high": {"accuracy": acc_high, "count": int(np.sum(high_mask))},
                "medium": {"accuracy": acc_med, "count": int(np.sum(med_mask))}
            },
            "top_10_predicted_classes": top10_list
        }

    return results_by_model, unmapped_pv_classes


def run_plantdoc_duplicate_check_and_contact_sheets(
    plantdoc_dir: str,
    records: List[Dict[str, str]],
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    y_true: np.ndarray,
    probs: Dict[str, np.ndarray],
    out_dir: str
) -> Dict[str, Any]:
    """
    (6) Duplicate check inside PlantDoc using 64-bit perceptual hashes (Hamming distance <= 4),
    and saves two 24-image contact sheets (Strawberry___Leaf_scorch false alarms and correct predictions).
    """
    import imagehash
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 90)
    print(" RUNNING PLANTDOC INTERNAL DUPLICATE CHECK & CONTACT SHEETS (--plantdoc-dir)")
    print("=" * 90)

    # Resolve local file paths
    resolved_paths: List[Optional[str]] = []
    for r in records:
        cand1 = os.path.join(plantdoc_dir, r["split"], r["plantdoc_class"], os.path.basename(r["path"]))
        cand2 = os.path.join(plantdoc_dir, r["split"], os.path.basename(r["path"]))
        cand3 = os.path.join(plantdoc_dir, os.path.basename(r["path"]))
        cand4 = r["path"]

        matched = None
        for cand in [cand1, cand2, cand3, cand4]:
            if os.path.exists(cand):
                matched = os.path.abspath(cand)
                break
        resolved_paths.append(matched)

    missing_cnt = sum(1 for p in resolved_paths if p is None)
    if missing_cnt > 0:
        print(f"[WARN] {missing_cnt}/{len(records)} image paths could not be located under {plantdoc_dir}.")

    # Compute pHash for all readable images
    hashes: List[Optional[np.ndarray]] = []
    for p in resolved_paths:
        if p and os.path.exists(p):
            try:
                with Image.open(p) as img:
                    h = imagehash.phash(img, hash_size=8).hash.flatten()
                    hashes.append(h)
            except Exception:
                hashes.append(None)
        else:
            hashes.append(None)

    # Split train vs test indices
    train_indices = [i for i, r in enumerate(records) if r["split"] == "train" and hashes[i] is not None]
    test_indices = [i for i, r in enumerate(records) if r["split"] == "test" and hashes[i] is not None]

    duplicate_pairs: List[Dict[str, Any]] = []

    if train_indices and test_indices:
        train_hashes = np.array([hashes[i] for i in train_indices], dtype=bool)
        test_hashes = np.array([hashes[i] for i in test_indices], dtype=bool)

        bits_train = train_hashes.astype(np.float32) * 2.0 - 1.0
        bits_test = test_hashes.astype(np.float32) * 2.0 - 1.0

        # Pairwise Hamming distance = (64 - similarity) / 2
        sim = np.matmul(bits_test, bits_train.T)
        dist = (64.0 - sim) / 2.0

        near_dup_mask = (dist <= 4.0)
        test_match_idx, train_match_idx = np.where(near_dup_mask)

        for t_idx, tr_idx in zip(test_match_idx, train_match_idx):
            orig_test_i = test_indices[t_idx]
            orig_train_i = train_indices[tr_idx]
            d = int(dist[t_idx, tr_idx])
            duplicate_pairs.append({
                "test_index": orig_test_i,
                "train_index": orig_train_i,
                "distance": d,
                "test_path": resolved_paths[orig_test_i],
                "train_path": resolved_paths[orig_train_i],
                "test_plantdoc_class": records[orig_test_i]["plantdoc_class"],
                "train_plantdoc_class": records[orig_train_i]["plantdoc_class"],
                "test_mapped_class": records[orig_test_i]["mapped_class"],
                "train_mapped_class": records[orig_train_i]["mapped_class"]
            })

    print(f"[*] Found {len(duplicate_pairs)} train/test pairs at Hamming distance <= 4.")

    # Generate Contact Sheets
    # Contact sheet 1: Images predicted as Strawberry___Leaf_scorch whose true class is different
    # Use ensemble predictions for contact sheets
    ens_preds = np.argmax(probs["ensemble"], axis=1)
    scorch_idx = class_to_idx.get("Strawberry___Leaf_scorch")

    scorch_candidates = [
        i for i in range(len(records))
        if ens_preds[i] == scorch_idx and y_true[i] != scorch_idx and resolved_paths[i] is not None
    ]

    rng = random.Random(42)
    selected_scorch = scorch_candidates if len(scorch_candidates) <= 24 else rng.sample(scorch_candidates, 24)

    sheet1_path = os.path.join(out_dir, "contact_sheet_strawberry_scorch.png")
    if selected_scorch:
        fig, axes = plt.subplots(4, 6, figsize=(22, 15))
        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx < len(selected_scorch):
                s_idx = selected_scorch[ax_idx]
                img_path = resolved_paths[s_idx]
                try:
                    with Image.open(img_path) as im:
                        im_thumb = im.convert("RGB").resize((224, 224))
                        ax.imshow(im_thumb)
                    t_cls = records[s_idx]["mapped_class"]
                    p_cls = idx_to_class[ens_preds[s_idx]]
                    ax.set_title(f"True: {t_cls[:25]}\nPred: {p_cls[:25]}", fontsize=7.5)
                except Exception:
                    ax.text(0.5, 0.5, "Load Error", ha="center", va="center")
            ax.axis("off")
        plt.suptitle("PlantDoc Samples Mispredicted as Strawberry___Leaf_scorch (Ensemble)", fontsize=14, y=0.99)
        plt.tight_layout()
        plt.savefig(sheet1_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[*] Saved Strawberry___Leaf_scorch contact sheet: {sheet1_path}")

    # Contact sheet 2: Randomly chosen correct predictions
    correct_candidates = [
        i for i in range(len(records))
        if ens_preds[i] == y_true[i] and resolved_paths[i] is not None
    ]
    selected_correct = correct_candidates if len(correct_candidates) <= 24 else rng.sample(correct_candidates, 24)

    sheet2_path = os.path.join(out_dir, "contact_sheet_correct.png")
    if selected_correct:
        fig, axes = plt.subplots(4, 6, figsize=(22, 15))
        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx < len(selected_correct):
                c_idx = selected_correct[ax_idx]
                img_path = resolved_paths[c_idx]
                try:
                    with Image.open(img_path) as im:
                        im_thumb = im.convert("RGB").resize((224, 224))
                        ax.imshow(im_thumb)
                    cls_name = records[c_idx]["mapped_class"]
                    ax.set_title(f"Correct: {cls_name[:30]}", fontsize=8)
                except Exception:
                    ax.text(0.5, 0.5, "Load Error", ha="center", va="center")
            ax.axis("off")
        plt.suptitle("PlantDoc Randomly Sampled Correct Predictions (Ensemble)", fontsize=14, y=0.99)
        plt.tight_layout()
        plt.savefig(sheet2_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[*] Saved correct predictions contact sheet: {sheet2_path}")

    return {
        "num_train_test_pairs_le_4": len(duplicate_pairs),
        "pairs": duplicate_pairs,
        "contact_sheet_strawberry_scorch": sheet1_path if selected_scorch else None,
        "contact_sheet_correct": sheet2_path if selected_correct else None
    }


def print_summary_table(
    records: List[Dict[str, str]],
    unmapped_pv_classes: List[str],
    results_by_model: Dict[str, Any]
):
    """Prints a structured, formatted summary table to console."""
    total_n = len(records)
    n_train = sum(1 for r in records if r["split"] == "train")
    n_test = sum(1 for r in records if r["split"] == "test")
    n_high = sum(1 for r in records if r["confidence_level"] == "high")
    n_med = sum(1 for r in records if r["confidence_level"] == "medium")
    n_healthy = sum(1 for r in records if "___healthy" in r["mapped_class"])
    n_diseased = total_n - n_healthy

    print("\n" + "=" * 115)
    print(" PLANTDOC COMPREHENSIVE OUT-OF-DOMAIN ANALYSIS SUMMARY")
    print("=" * 115)
    print(f"  Dataset Samples:     {total_n:,} total (Train: {n_train:,}, Test: {n_test:,})")
    print(f"  Confidence Split:    High: {n_high:,} ({n_high/total_n*100:.1f}%), Medium: {n_med:,} ({n_med/total_n*100:.1f}%)")
    print(f"  Condition Split:     Healthy: {n_healthy:,} ({n_healthy/total_n*100:.1f}%), Diseased: {n_diseased:,} ({n_diseased/total_n*100:.1f}%)")
    print(f"  PlantVillage Classes: 38 total (28 with PlantDoc counterpart, {len(unmapped_pv_classes)} unmapped)")
    print("=" * 115)

    # 1. Main Performance Metrics Table
    print("\n--- MODEL PERFORMANCE & DIAGNOSTIC ACCURACY COMPARISON ---")
    hdr = (
        f"| {'Model':<15} | {'Top-1 Acc (95% CI)':<23} | {'Crop Acc':<10} | "
        f"{'Exact|Crop':<11} | {'Binary H/D':<10} | {'Train Acc':<10} | {'Test Acc':<9} | {'High Conf':<10} | {'Med Conf':<9} |"
    )
    div = "+-----------------+-------------------------+------------+-------------+------------+------------+-----------+------------+-----------+"
    print(div)
    print(hdr)
    print(div)

    for mname, res in results_by_model.items():
        ci = res["bootstrap_ci_95"]
        ci_str = f"{res['top1_accuracy_all_38']*100:.2f}% [{ci['ci_lower']*100:.1f}-{ci['ci_upper']*100:.1f}%]"
        crop_acc_str = f"{res['crop_level_accuracy']['crop_accuracy']*100:.2f}%"
        exact_given_crop_str = f"{res['crop_level_accuracy']['exact_class_accuracy_given_correct_crop']*100:.2f}%"
        bin_acc_str = f"{res['healthy_vs_diseased']['binary_accuracy']*100:.2f}%"
        train_acc_str = f"{res['accuracy_by_split']['train']['accuracy']*100:.2f}%"
        test_acc_str = f"{res['accuracy_by_split']['test']['accuracy']*100:.2f}%"
        high_acc_str = f"{res['accuracy_by_confidence']['high']['accuracy']*100:.2f}%"
        med_acc_str = f"{res['accuracy_by_confidence']['medium']['accuracy']*100:.2f}%"

        print(
            f"| {mname:<15} | {ci_str:<23} | {crop_acc_str:>10} | "
            f"{exact_given_crop_str:>11} | {bin_acc_str:>10} | {train_acc_str:>10} | {test_acc_str:>9} | {high_acc_str:>10} | {med_acc_str:>9} |"
        )
    print(div)

    # 2. Healthy vs. Diseased Error Breakdown
    print("\n--- HEALTHY vs. DISEASED ERROR DIRECTIONALITY ---")
    h_hdr = f"| {'Model':<15} | {'Binary Acc':<11} | {'Healthy -> Diseased (FP rate)':<30} | {'Diseased -> Healthy (FN rate)':<30} |"
    h_div = "+-----------------+-------------+--------------------------------+--------------------------------+"
    print(h_div)
    print(h_hdr)
    print(h_div)
    for mname, res in results_by_model.items():
        hvd = res["healthy_vs_diseased"]
        bin_str = f"{hvd['binary_accuracy']*100:.2f}%"
        h_to_d_str = f"{hvd['rate_healthy_predicted_diseased']*100:.2f}% ({int(hvd['rate_healthy_predicted_diseased']*hvd['healthy_count'])}/{hvd['healthy_count']})"
        d_to_h_str = f"{hvd['rate_diseased_predicted_healthy']*100:.2f}% ({int(hvd['rate_diseased_predicted_healthy']*hvd['diseased_count'])}/{hvd['diseased_count']})"
        print(f"| {mname:<15} | {bin_str:>11} | {h_to_d_str:>30} | {d_to_h_str:>30} |")
    print(h_div)

    # 3. Top 10 Predictions Table per Model
    print("\n--- TOP 10 PREDICTED CLASSES PER MODEL (* indicates NO PlantDoc counterpart) ---")
    for mname, res in results_by_model.items():
        print(f"\nModel: {mname.upper()}")
        print("  Rank | Class Name                                      | Predictions | Share   | PlantDoc Counterpart?")
        print("  -----+-------------------------------------------------+-------------+---------+----------------------")
        for rank, item in enumerate(res["top_10_predicted_classes"], start=1):
            mark = "" if item["has_plantdoc_counterpart"] else " * [NO COUNTERPART]"
            status = "Yes" if item["has_plantdoc_counterpart"] else "No *"
            print(
                f"  {rank:>4} | {item['class']:<47} | {item['count']:>11,} | {item['share']*100:>6.2f}% | {status}"
            )
    print("\n" + "=" * 115 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive diagnostic analysis of PlantDoc out-of-domain evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="ml/results/plantdoc",
        help="Directory containing PlantDoc evaluation outputs (default: ml/results/plantdoc)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for analysis artifacts (default: <results-dir>/analysis)"
    )
    parser.add_argument(
        "--plantdoc-dir",
        type=str,
        default=None,
        help="Optional root directory of PlantDoc dataset for internal duplicate check & contact sheets"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml/checkpoints/resnet50/best_model.pth",
        help="Path to checkpoint containing class_to_idx (default: ml/checkpoints/resnet50/best_model.pth)"
    )

    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(results_dir, "analysis")
    checkpoint_path = os.path.abspath(args.checkpoint)

    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Starting PlantDoc Analysis...")
    print(f"    - Results directory: {results_dir}")
    print(f"    - Output directory:  {out_dir}")
    print(f"    - Checkpoint:        {checkpoint_path}")
    if args.plantdoc_dir:
        print(f"    - PlantDoc Dataset:  {os.path.abspath(args.plantdoc_dir)}")

    # 1. Load inputs
    records, class_to_idx, idx_to_class, y_true, probs = load_inputs(results_dir, checkpoint_path)
    print(f"[*] Loaded {len(records):,} records and {len(class_to_idx)} classes from checkpoint.")

    # 2. Run model analyses (1) - (5)
    results_by_model, unmapped_pv_classes = run_model_analysis(
        records, class_to_idx, idx_to_class, y_true, probs
    )

    # 3. Optional duplicate check & contact sheets (6)
    duplicate_results = None
    if args.plantdoc_dir:
        duplicate_results = run_plantdoc_duplicate_check_and_contact_sheets(
            args.plantdoc_dir, records, class_to_idx, idx_to_class, y_true, probs, out_dir
        )

    # 4. Print summary tables
    print_summary_table(records, unmapped_pv_classes, results_by_model)

    # 5. Construct analysis.json payload
    analysis_payload = {
        "metadata": {
            "results_dir": results_dir,
            "out_dir": out_dir,
            "total_samples": len(records),
            "models_evaluated": list(probs.keys()),
            "plantvillage_classes_count": len(class_to_idx),
            "unmapped_plantvillage_classes_count": len(unmapped_pv_classes),
            "unmapped_plantvillage_classes": unmapped_pv_classes
        },
        "models": results_by_model,
        "duplicate_check": duplicate_results
    }

    # Save to <out>/analysis.json
    out_json_path = os.path.join(out_dir, "analysis.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)
    print(f"[*] Analysis payload saved to: {out_json_path}")

    # Also save to <results-dir>/analysis.json to satisfy both locations
    results_json_path = os.path.join(results_dir, "analysis.json")
    if os.path.abspath(results_json_path) != os.path.abspath(out_json_path):
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(analysis_payload, f, indent=2)
        print(f"[*] Analysis payload copy saved to: {results_json_path}")


if __name__ == "__main__":
    main()

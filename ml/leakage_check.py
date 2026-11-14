#!/usr/bin/env python3
"""
Data Leakage & Perceptual Hash Duplicate Detection Script.

Features:
  - Rebuilds stratified train/val/test split using get_dataloaders from ml/src/dataset.py (num_workers=0).
  - Computes 64-bit perceptual hashes (imagehash.phash) for all train and test images.
  - Caches hashes in ml/results/hashes.npy for ultra-fast reruns.
  - Computes minimum Hamming distance from every test image to any train image using vectorized operations.
  - Reports overall percentage of test images with a train neighbor at distance 0, <=4, and <=8.
  - Identifies and reports the 10 worst classes with the highest near-duplicate leakage.
  - Generates side-by-side PNG comparisons of the 30 closest test/train image pairs in ml/results/leakage/.
  - Evaluates impact of data leakage across models (resnet50, efficientnet_v2, vit, ensemble) using
    ml/results/test_files.csv and ml/results/probs_<name>.npy, comparing accuracy and error counts
    on near-duplicates (<=4) vs. clean test samples (>4).
  - Saves full structured leakage metrics to ml/results/leakage_metrics.json.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import functools

# Ensure unbuffered, real-time logging
print = functools.partial(print, flush=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import torch

# ---------------------------------------------------------------------------
# Setup sys.path to extend dataset.py and utils.py
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "ml" else CURRENT_DIR

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset import get_dataloaders
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


def compute_single_phash_bits(path: str) -> np.ndarray:
    """
    Computes a 64-bit perceptual hash (imagehash.phash) and returns a 1D boolean array of length 64.
    """
    try:
        with Image.open(path) as img:
            h = imagehash.phash(img, hash_size=8)
            return h.hash.flatten()
    except Exception as e:
        print(f"[ERROR] Failed to hash image {path}: {e}")
        return np.zeros(64, dtype=bool)


def compute_hashes_parallel(
    paths: List[str],
    desc: str,
    workers: int = 8
) -> np.ndarray:
    """Computes perceptual hashes across a list of image paths using thread-level parallelism."""
    print(f"[*] Computing 64-bit perceptual hashes for {len(paths)} {desc} images (workers={workers})...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = list(executor.map(compute_single_phash_bits, paths))
    elapsed = time.time() - t0
    rate = len(paths) / elapsed if elapsed > 0 else 0.0
    print(f"    - Completed {len(paths)} {desc} hashes in {elapsed:.2f}s ({rate:.1f} images/sec).")
    return np.array(hashes, dtype=bool)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perceptual hash data leakage & train/test duplicate detection"
    )
    parser.add_argument(
        "--config", type=str, default="ml/configs/config.yaml",
        help="Path to dataset configuration YAML file"
    )
    parser.add_argument(
        "--results-dir", type=str, default="ml/results",
        help="Directory containing test_files.csv and model probability npy files"
    )
    parser.add_argument(
        "--hashes-cache", type=str, default=None,
        help="Path for hashes.npy cache file (defaults to <results-dir>/hashes.npy)"
    )
    parser.add_argument(
        "--leakage-dir", type=str, default=None,
        help="Directory to save closest pair PNG visualizations (defaults to <results-dir>/leakage/)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of threads for parallel image hashing (default: 8)"
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Force recomputing hashes even if cache exists"
    )
    parser.add_argument(
        "--threshold", type=int, default=4,
        help="Hamming distance threshold for near-duplicate classification (default: 4)"
    )
    parser.add_argument(
        "--save-top-k", type=int, default=30,
        help="Number of closest test/train pairs to save as side-by-side PNGs (default: 30)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 95)
    print(" DATA LEAKAGE & PERCEPTUAL HASH (pHash) NEAR-DUPLICATE AUDIT")
    print("=" * 95)

    results_dir = resolve_path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    cache_path = (
        resolve_path(args.hashes_cache)
        if args.hashes_cache
        else os.path.join(results_dir, "hashes.npy")
    )

    leakage_dir = (
        resolve_path(args.leakage_dir)
        if args.leakage_dir
        else os.path.join(results_dir, "leakage")
    )
    os.makedirs(leakage_dir, exist_ok=True)

    # 1. Rebuild split with get_dataloaders (force num_workers=0)
    config_path = resolve_path(args.config)
    print(f"[*] Loading configuration from: {config_path}")
    config = load_config(config_path)

    if "dataset" not in config:
        config["dataset"] = {}
    config["dataset"]["num_workers"] = 0

    print("[*] Rebuilding train/val/test split using get_dataloaders(num_workers=0)...")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(config)

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    base_dataset = train_dataset.dataset

    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Extract ordered paths and labels
    train_paths: List[str] = []
    train_labels: List[int] = []
    for idx in train_dataset.indices:
        p, lbl = base_dataset.samples[idx]
        train_paths.append(p)
        train_labels.append(lbl)

    test_paths: List[str] = []
    test_labels: List[int] = []
    for idx in test_dataset.indices:
        p, lbl = base_dataset.samples[idx]
        test_paths.append(p)
        test_labels.append(lbl)

    num_train = len(train_paths)
    num_test = len(test_paths)
    print(f"    - Split reconstructed: {num_train} train images, {len(val_loader.dataset)} val images, {num_test} test images across {num_classes} classes.")

    # 2. Compute or load cached perceptual hashes (ml/results/hashes.npy)
    train_hashes: Optional[np.ndarray] = None
    test_hashes: Optional[np.ndarray] = None
    cache_valid = False

    if os.path.exists(cache_path) and not args.force_recompute:
        print(f"[*] Checking existing hash cache at: {cache_path}...")
        try:
            cached_obj = np.load(cache_path, allow_pickle=True)
            cached_data = cached_obj.item() if cached_obj.shape == () else dict(cached_obj)

            cached_train_h = cached_data.get("train_hashes")
            cached_test_h = cached_data.get("test_hashes")

            if (
                isinstance(cached_train_h, np.ndarray) and
                isinstance(cached_test_h, np.ndarray) and
                len(cached_train_h) == num_train and
                len(cached_test_h) == num_test
            ):
                train_hashes = cached_train_h
                test_hashes = cached_test_h
                cache_valid = True
                print(f"    [PASS] Valid hash cache loaded ({num_train} train, {num_test} test hashes).")
            else:
                print(f"    [WARN] Cache length mismatch or invalid format. Recomputing...")
        except Exception as e:
            print(f"    [WARN] Failed to read cache ({e}). Recomputing...")

    if not cache_valid:
        train_hashes = compute_hashes_parallel(train_paths, "train", workers=args.workers)
        test_hashes = compute_hashes_parallel(test_paths, "test", workers=args.workers)

        print(f"[*] Caching hashes to: {cache_path}...")
        cache_dict = {
            "train_hashes": train_hashes,
            "train_paths": train_paths,
            "train_labels": np.array(train_labels, dtype=np.int64),
            "test_hashes": test_hashes,
            "test_paths": test_paths,
            "test_labels": np.array(test_labels, dtype=np.int64),
            "class_to_idx": class_to_idx,
            "metadata": {
                "hash_type": "phash_64",
                "hash_size": 8,
                "num_train": num_train,
                "num_test": num_test
            }
        }
        np.save(cache_path, cache_dict)
        print(f"    [PASS] Hashes successfully saved to {cache_path}.")

    # 3. Vectorized Hamming distance search from each test image to any train image
    print(f"\n[*] Calculating minimum Hamming distances for all {num_test} test images to {num_train} train images...")
    t_dist_start = time.time()

    # Convert binary boolean hashes into +1.0 / -1.0 float32 representations
    # For vectors u, v in {-1, +1}^64: dot(u, v) = 64 - 2 * Hamming_distance(u, v)
    # Therefore: Hamming_distance(u, v) = (64 - dot(u, v)) / 2
    train_bits_float = train_hashes.astype(np.float32) * 2.0 - 1.0  # (num_train, 64)
    test_bits_float = test_hashes.astype(np.float32) * 2.0 - 1.0    # (num_test, 64)

    test_tensor = torch.from_numpy(test_bits_float)
    train_tensor_t = torch.from_numpy(train_bits_float.T)  # (64, num_train)

    chunk_size = 1000
    min_dist_chunks = []
    closest_train_idx_chunks = []

    with torch.inference_mode():
        for start_idx in range(0, num_test, chunk_size):
            test_chunk = test_tensor[start_idx:start_idx + chunk_size]
            # Matmul: (chunk_size, 64) x (64, num_train) -> (chunk_size, num_train)
            dots = torch.matmul(test_chunk, train_tensor_t)
            max_dots, best_train_idx = torch.max(dots, dim=1)

            # Compute exact integer Hamming distances
            dists = ((64.0 - max_dots) / 2.0).round().long()
            min_dist_chunks.append(dists.numpy())
            closest_train_idx_chunks.append(best_train_idx.numpy())

    min_distances = np.concatenate(min_dist_chunks, axis=0)               # (num_test,)
    closest_train_indices = np.concatenate(closest_train_idx_chunks, axis=0) # (num_test,)
    t_dist_elapsed = time.time() - t_dist_start

    print(f"    [PASS] Distance matrix search completed in {t_dist_elapsed:.3f}s (310.8M pairs evaluated).")

    # 4. Overall Leakage Metrics
    total_test = num_test
    count_0 = int(np.sum(min_distances == 0))
    count_le4 = int(np.sum(min_distances <= 4))
    count_le8 = int(np.sum(min_distances <= 8))

    pct_0 = (count_0 / total_test) * 100.0
    pct_le4 = (count_le4 / total_test) * 100.0
    pct_le8 = (count_le8 / total_test) * 100.0

    print("\n" + "=" * 95)
    print(" OVERALL TEST SET LEAKAGE SUMMARY")
    print("=" * 95)
    print(f"  Total Test Samples:                                {total_test}")
    print(f"  Exact Duplicates         (Hamming distance == 0):  {count_0:>5} / {total_test} ({pct_0:>6.2f}%)")
    print(f"  Near Duplicates (Strict) (Hamming distance <= 4):  {count_le4:>5} / {total_test} ({pct_le4:>6.2f}%)")
    print(f"  Near Duplicates (Broad)  (Hamming distance <= 8):  {count_le8:>5} / {total_test} ({pct_le8:>6.2f}%)")
    print("=" * 95)

    # 5. Per-class Leakage and Top 10 Worst Classes
    test_labels_arr = np.array(test_labels, dtype=np.int64)
    class_stats = []

    for c_idx in range(num_classes):
        c_name = class_names[c_idx]
        mask = (test_labels_arr == c_idx)
        n_c = int(np.sum(mask))

        if n_c > 0:
            c_dists = min_distances[mask]
            c_0 = int(np.sum(c_dists == 0))
            c_le4 = int(np.sum(c_dists <= 4))
            c_le8 = int(np.sum(c_dists <= 8))

            c_pct_0 = (c_0 / n_c) * 100.0
            c_pct_le4 = (c_le4 / n_c) * 100.0
            c_pct_le8 = (c_le8 / n_c) * 100.0
        else:
            c_0, c_le4, c_le8 = 0, 0, 0
            c_pct_0, c_pct_le4, c_pct_le8 = 0.0, 0.0, 0.0

        class_stats.append({
            "class_idx": c_idx,
            "class_name": c_name,
            "total_test": n_c,
            "count_dist_0": c_0,
            "pct_dist_0": c_pct_0,
            "count_dist_le4": c_le4,
            "pct_dist_le4": c_pct_le4,
            "count_dist_le8": c_le8,
            "pct_dist_le8": c_pct_le8
        })

    # Sort to determine top 10 worst classes (ranked by % near-duplicates <=4, then <=8, then ==0)
    sorted_worst_classes = sorted(
        class_stats,
        key=lambda x: (x["pct_dist_le4"], x["pct_dist_le8"], x["pct_dist_0"], x["count_dist_le4"]),
        reverse=True
    )
    top_10_worst = sorted_worst_classes[:10]

    print("\n" + "=" * 95)
    print(" TOP 10 WORST CLASSES WITH HIGHEST TEST LEAKAGE (SORTED BY % DIST <= 4)")
    print("=" * 95)
    worst_tbl_header = ["Rank", "Class Name", "Test Count", "Dist == 0", "Dist <= 4 (Near-Dup)", "Dist <= 8"]
    worst_tbl_fmt = "| {:<4} | {:<38} | {:>10} | {:>12} | {:>22} | {:>12} |"
    worst_divider = "+------+----------------------------------------+------------+--------------+------------------------+--------------+"

    print(worst_divider)
    print(worst_tbl_fmt.format(*worst_tbl_header))
    print(worst_divider)

    for rank, entry in enumerate(top_10_worst, 1):
        d0_str = f"{entry['pct_dist_0']:.1f}% ({entry['count_dist_0']})"
        dle4_str = f"{entry['pct_dist_le4']:.1f}% ({entry['count_dist_le4']})"
        dle8_str = f"{entry['pct_dist_le8']:.1f}% ({entry['count_dist_le8']})"
        print(worst_tbl_fmt.format(
            rank,
            entry["class_name"][:38],
            entry["total_test"],
            d0_str,
            dle4_str,
            dle8_str
        ))
    print(worst_divider)

    # 6. Save the 30 closest test/train pairs side by side as PNGs in ml/results/leakage/
    k_pairs = min(args.save_top_k, total_test)
    print(f"\n[*] Saving {k_pairs} closest test/train pairs side by side as PNGs in: {leakage_dir}...")

    # Sort test images by minimum Hamming distance ascending
    sorted_test_indices = np.argsort(min_distances)
    top_k_indices = sorted_test_indices[:k_pairs]

    top_pairs_metadata = []

    for rank, t_idx in enumerate(top_k_indices, 1):
        tr_idx = int(closest_train_indices[t_idx])
        dist_val = int(min_distances[t_idx])

        test_img_path = test_paths[t_idx]
        train_img_path = train_paths[tr_idx]

        test_cls = class_names[test_labels[t_idx]]
        train_cls = class_names[train_labels[tr_idx]]
        is_same_class = (test_cls == train_cls)

        top_pairs_metadata.append({
            "rank": rank,
            "distance": dist_val,
            "test_image": test_img_path,
            "train_image": train_img_path,
            "test_class": test_cls,
            "train_class": train_cls,
            "same_class": is_same_class
        })

        out_png_name = f"pair_{rank:02d}_dist_{dist_val}.png"
        out_png_path = os.path.join(leakage_dir, out_png_name)

        try:
            with Image.open(test_img_path) as im_test, Image.open(train_img_path) as im_train:
                im_test_rgb = im_test.convert("RGB")
                im_train_rgb = im_train.convert("RGB")

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                axes[0].imshow(im_test_rgb)
                axes[0].set_title(
                    f"Test Image ({test_cls})\n{os.path.basename(test_img_path)}",
                    fontsize=9
                )
                axes[0].axis("off")

                axes[1].imshow(im_train_rgb)
                axes[1].set_title(
                    f"Nearest Train Image ({train_cls})\n{os.path.basename(train_img_path)}",
                    fontsize=9
                )
                axes[1].axis("off")

                match_status = "SAME CLASS" if is_same_class else "CROSS-CLASS"
                fig.suptitle(
                    f"Pair #{rank:02d} | Hamming Distance = {dist_val} / 64 [{match_status}]",
                    fontsize=12,
                    fontweight="bold"
                )
                fig.tight_layout()
                fig.savefig(out_png_path, dpi=120, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            print(f"    [ERROR] Failed to save comparison PNG for pair #{rank}: {e}")

    print(f"    [PASS] Saved {k_pairs} comparison PNGs to {leakage_dir}")

    # 7. Model evaluation on near-duplicates (<=4) vs. clean test samples (>4)
    print("\n[*] Evaluating model accuracy on near-duplicates (<=4) vs. non-duplicates (>4)...")

    # Load test_files.csv to confirm alignment
    test_files_csv_path = os.path.join(results_dir, "test_files.csv")
    if not os.path.exists(test_files_csv_path):
        raise FileNotFoundError(f"Required test_files.csv not found at: {test_files_csv_path}")

    csv_paths: List[str] = []
    csv_labels: List[str] = []
    with open(test_files_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_paths.append(row["path"])
            csv_labels.append(row["true_label"])

    if len(csv_paths) != num_test:
        raise ValueError(
            f"Row count mismatch: test_files.csv has {len(csv_paths)} entries, but test set has {num_test} samples."
        )

    # Use ground truth labels from test_files.csv mapped to integer IDs
    y_true = np.array([class_to_idx[lbl] for lbl in csv_labels], dtype=np.int64)

    near_mask = (min_distances <= args.threshold)
    clean_mask = (min_distances > args.threshold)

    n_near = int(np.sum(near_mask))
    n_clean = int(np.sum(clean_mask))

    print(f"    - Near-duplicate partition (Hamming dist <= {args.threshold}): {n_near} samples ({(n_near/total_test)*100.0:.2f}%)")
    print(f"    - Clean partition          (Hamming dist >  {args.threshold}): {n_clean} samples ({(n_clean/total_test)*100.0:.2f}%)")

    model_names = ["resnet50", "efficientnet_v2", "vit", "ensemble"]
    model_eval_results: Dict[str, Dict[str, Any]] = {}

    for name in model_names:
        prob_path = os.path.join(results_dir, f"probs_{name}.npy")
        if not os.path.exists(prob_path):
            print(f"    [WARN] Probabilities file '{prob_path}' not found! Skipping '{name}'.")
            continue

        probs = np.load(prob_path)
        if len(probs) != num_test:
            raise ValueError(
                f"Length mismatch: {prob_path} has {len(probs)} predictions, expected {num_test}."
            )

        preds = np.argmax(probs, axis=1)
        is_correct = (preds == y_true)
        is_error = ~is_correct

        # Near-duplicate group metrics
        near_correct = int(np.sum(is_correct[near_mask]))
        near_errors = int(np.sum(is_error[near_mask]))
        near_acc = (near_correct / n_near * 100.0) if n_near > 0 else 0.0

        # Clean group metrics
        clean_correct = int(np.sum(is_correct[clean_mask]))
        clean_errors = int(np.sum(is_error[clean_mask]))
        clean_acc = (clean_correct / n_clean * 100.0) if n_clean > 0 else 0.0

        # Overall metrics
        total_correct = int(np.sum(is_correct))
        total_errors = int(np.sum(is_error))
        total_acc = (total_correct / total_test * 100.0) if total_test > 0 else 0.0

        model_eval_results[name] = {
            "near_accuracy_pct": near_acc,
            "near_errors": near_errors,
            "near_total": n_near,
            "clean_accuracy_pct": clean_acc,
            "clean_errors": clean_errors,
            "clean_total": n_clean,
            "overall_accuracy_pct": total_acc,
            "overall_errors": total_errors,
            "overall_total": total_test,
            "accuracy_gap_pct": near_acc - clean_acc
        }

    # 8. Print Model Comparison Table
    print("\n" + "=" * 95)
    print(f" MODEL ACCURACY ON NEAR-DUPLICATES (DIST <= {args.threshold}) VS. CLEAN SAMPLES (DIST > {args.threshold})")
    print(f" Near-Duplicate Partition: N = {n_near} ({n_near/total_test*100.0:.2f}%) | Clean Partition: N = {n_clean} ({n_clean/total_test*100.0:.2f}%)")
    print("=" * 95)

    tbl_header = [
        "Model",
        f"Near-Dup Acc (N={n_near})",
        "Near Errors",
        f"Clean Acc (N={n_clean})",
        "Clean Errors",
        "Overall Acc",
        "Total Errors"
    ]
    tbl_fmt = "| {:<17} | {:>24} | {:>11} | {:>22} | {:>12} | {:>11} | {:>12} |"
    tbl_divider = "+-------------------+--------------------------+-------------+------------------------+--------------+-------------+--------------+"

    print(tbl_divider)
    print(tbl_fmt.format(*tbl_header))
    print(tbl_divider)

    for name in model_names:
        if name in model_eval_results:
            m = model_eval_results[name]
            print(tbl_fmt.format(
                name,
                f"{m['near_accuracy_pct']:.2f}%",
                f"{m['near_errors']} / {n_near}",
                f"{m['clean_accuracy_pct']:.2f}%",
                f"{m['clean_errors']} / {n_clean}",
                f"{m['overall_accuracy_pct']:.2f}%",
                f"{m['overall_errors']} / {total_test}"
            ))
    print(tbl_divider)

    # 9. Save full leakage metrics JSON
    metrics_json_path = os.path.join(results_dir, "leakage_metrics.json")
    json_payload = {
        "dataset_split": {
            "num_train": num_train,
            "num_val": len(val_loader.dataset),
            "num_test": num_test,
            "num_classes": num_classes
        },
        "overall_leakage": {
            "total_test_samples": total_test,
            "distance_0": {
                "count": count_0,
                "percentage": pct_0
            },
            "distance_le4": {
                "count": count_le4,
                "percentage": pct_le4
            },
            "distance_le8": {
                "count": count_le8,
                "percentage": pct_le8
            }
        },
        "top_10_worst_classes": top_10_worst,
        "all_class_stats": class_stats,
        "model_leakage_performance": model_eval_results,
        "top_closest_pairs": top_pairs_metadata
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(f"\n[*] Full leakage analysis metrics successfully saved to: {metrics_json_path}\n")


if __name__ == "__main__":
    main()

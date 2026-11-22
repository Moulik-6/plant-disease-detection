#!/usr/bin/env python3
"""
Unified Manifest Builder & Cross-Dataset Near-Duplicate Group Splitter.

Scans:
  1. PlantVillage (data/plantvillage_dataset/color)
  2. PlantDoc (data/plantdoc/PlantDoc-Dataset/train and test, mapped via ml/configs/mappings/plantdoc_map.yaml)
  3. PlantWild v1 (data/plantwild/plantwild/images, mapped via ml/configs/mappings/plantwild_map.yaml)

Produces:
  - data/manifest.csv:
      Columns: path, source, orig_label, unified_label, confidence_level, orig_split, legacy_split, group_id, split, pv_dup
  - data/manifest_report.json:
      Counts per source, class, and split; duplicate cluster metrics; cross-dataset duplicate pairs;
      official split file listings; URL record group merge audit; 38-class per-split counts;
      split integrity verification; and SHA-256 checksum of manifest.csv.

Key Design & Soundness Highlights:
  - PlantWild Ingestion: Real adapter for PlantWild v1 using `ml/data/sources/plantwild.py`. Unmapped folders
    receive unified_label="other" and are placed in the "other_pool" split.
  - URL Record Deduplication: Uses `data/plantwild/plantwild/url_record.json`. URL-based union only between
    images sharing the same orig_label (never cross-label).
  - URL Merge Audit: Formally computes and reports the number of PlantWild groups merged by the URL record,
    and counts URL clusters that would have crossed labels without unioning them.
  - Joint Field Clustering: PlantDoc and PlantWild are clustered jointly with 64-bit pHash Union-Find
    (threshold <= 4) strictly within identical unified_label.
  - PlantVillage-Field Duplicate Isolation: Field images at Hamming distance <= 4 to PlantVillage images
    are flagged with pv_dup=1 and forced into field_train, exempt from stratified val/test assignment.
  - Group-Stratified Splitting:
      * PlantVillage: 70/15/15 by group (pv_train, pv_val, pv_test), seed=42.
      * Field (PlantDoc + PlantWild): 70/15/15 by group (field_train, field_val, field_test), seed=42.
      * Unmapped classes ("other"): Segregated exclusively into "other_pool", never placed in field_test.
      * Tiny classes (<5 groups): Allocated entirely to train to preserve statistical validity.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import torch
import yaml

# Ensure repository root is on sys.path
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ml.src.dataset import stratified_split_indices
from ml.data.sources.plantwild import PlantWildAdapter

from torchvision.datasets import ImageFolder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_manifest")


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) Data Structure
# ---------------------------------------------------------------------------
class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression and rank optimization."""

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i: int) -> int:
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1


# ---------------------------------------------------------------------------
# Perceptual Hashing & Vectorized Distance Computations
# ---------------------------------------------------------------------------
def compute_single_phash(img_path: str) -> np.ndarray:
    """Computes a 64-bit perceptual hash (imagehash.phash) as a 1D boolean array."""
    import imagehash
    try:
        with Image.open(img_path) as img:
            h = imagehash.phash(img, hash_size=8)
            return h.hash.flatten()
    except Exception as e:
        logger.warning("Failed to compute pHash for %s: %s", img_path, e)
        return np.zeros(64, dtype=bool)


def compute_hashes_parallel(paths: List[str], workers: int = 8, desc: str = "images") -> np.ndarray:
    """Computes perceptual hashes across a list of image paths using thread-level parallelism."""
    from concurrent.futures import ThreadPoolExecutor

    logger.info("Computing 64-bit perceptual hashes for %d %s (workers=%d)...", len(paths), desc, workers)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = list(executor.map(compute_single_phash, paths))
    elapsed = time.time() - t0
    rate = len(paths) / elapsed if elapsed > 0 else 0.0
    logger.info("Computed %d %s hashes in %.2fs (%.1f img/s).", len(paths), desc, elapsed, rate)
    return np.array(hashes, dtype=bool)


def cluster_near_duplicates_vectorized(
    hashes: np.ndarray,
    labels: Optional[List[str]] = None,
    threshold: int = 4,
    chunk_size: int = 2000
) -> List[int]:
    """
    Finds connected components among image hashes with Hamming distance <= threshold.
    Vectorized using PyTorch matrix multiplication on float32 +/-1.0 representations:
        dot(u, v) = 64 - 2 * Hamming_distance(u, v)
        Hamming_distance <= threshold  <=>  dot(u, v) >= 64 - 2 * threshold

    When `labels` is provided, union operations are strictly restricted to pairs
    sharing the exact same label (no cross-label pHash clustering).
    """
    n_samples = len(hashes)
    uf = UnionFind(n_samples)

    if n_samples <= 1:
        return [0] * n_samples

    bits_float = hashes.astype(np.float32) * 2.0 - 1.0
    bits_tensor = torch.from_numpy(bits_float)
    bits_tensor_t = bits_tensor.T  # (64, N)

    min_dot_threshold = 64 - 2 * threshold

    logger.info(
        "Clustering %d images (threshold distance <= %d, min dot-product >= %d, chunk_size=%d)...",
        n_samples, threshold, min_dot_threshold, chunk_size
    )

    t0 = time.time()
    with torch.inference_mode():
        for r_start in range(0, n_samples, chunk_size):
            r_end = min(r_start + chunk_size, n_samples)
            chunk = bits_tensor[r_start:r_end]  # (chunk_len, 64)
            dots = torch.matmul(chunk, bits_tensor_t)  # (chunk_len, N)

            row_idx, col_idx = torch.where(dots >= min_dot_threshold)
            row_arr = (row_idx + r_start).cpu().numpy()
            col_arr = col_idx.cpu().numpy()

            upper_mask = col_arr > row_arr
            for u, v in zip(row_arr[upper_mask], col_arr[upper_mask]):
                # phash union only between images with the same unified_label
                if labels is None or labels[int(u)] == labels[int(v)]:
                    uf.union(int(u), int(v))

    logger.info("Clustering completed in %.2fs.", time.time() - t0)
    return [uf.find(i) for i in range(n_samples)]


def count_cross_dataset_duplicate_pairs(
    hashes_a: np.ndarray,
    hashes_b: np.ndarray,
    threshold: int = 4,
    chunk_size: int = 2000,
    desc: str = "A-B"
) -> int:
    """
    Counts pairs of images between dataset A and dataset B with Hamming distance <= threshold.
    """
    n_a = len(hashes_a)
    n_b = len(hashes_b)

    if n_a == 0 or n_b == 0:
        return 0

    min_dot_threshold = 64 - 2 * threshold
    bits_a = torch.from_numpy(hashes_a.astype(np.float32) * 2.0 - 1.0)
    bits_b_t = torch.from_numpy((hashes_b.astype(np.float32) * 2.0 - 1.0).T)

    logger.info("Evaluating cross-dataset near-duplicate pairs for %s (%d x %d comparisons)...", desc, n_a, n_b)
    pair_count = 0
    with torch.inference_mode():
        for r_start in range(0, n_a, chunk_size):
            r_end = min(r_start + chunk_size, n_a)
            chunk = bits_a[r_start:r_end]
            dots = torch.matmul(chunk, bits_b_t)
            pair_count += int((dots >= min_dot_threshold).sum().item())

    logger.info("Found %d near-duplicate pairs (Hamming distance <= %d) for %s.", pair_count, threshold, desc)
    return pair_count


def audit_pv_field_duplicates_and_flag_matches(
    pv_hashes: np.ndarray,
    pdoc_hashes: np.ndarray,
    pwild_hashes: np.ndarray,
    threshold: int = 4,
    chunk_size: int = 2000
) -> Tuple[int, int, np.ndarray]:
    """
    Evaluates near-duplicate pairs between PlantVillage and field images (PlantDoc + PlantWild),
    and flags which field images have at least one PlantVillage image at Hamming distance <= threshold.

    Returns:
      (pv_pdoc_pairs, pv_pwild_pairs, field_has_pv_match)
    """
    n_pv = len(pv_hashes)
    n_pdoc = len(pdoc_hashes)
    n_pwild = len(pwild_hashes)
    field_hashes = np.concatenate([pdoc_hashes, pwild_hashes], axis=0) if (n_pdoc and n_pwild) else (pdoc_hashes if n_pdoc else pwild_hashes)
    n_field = len(field_hashes)

    if n_pv == 0 or n_field == 0:
        return 0, 0, np.zeros(n_field, dtype=bool)

    min_dot_threshold = 64 - 2 * threshold
    bits_pv = torch.from_numpy(pv_hashes.astype(np.float32) * 2.0 - 1.0)
    bits_pv_t = bits_pv.T  # (64, N_pv)
    bits_field = torch.from_numpy(field_hashes.astype(np.float32) * 2.0 - 1.0)  # (N_field, 64)

    field_has_pv_match = np.zeros(n_field, dtype=bool)
    pv_pdoc_pairs = 0
    pv_pwild_pairs = 0

    logger.info("Evaluating PlantVillage <-> Field duplicate pairs & flagging PV duplicates (%d x %d comparisons)...", n_field, n_pv)
    with torch.inference_mode():
        for r_start in range(0, n_field, chunk_size):
            r_end = min(r_start + chunk_size, n_field)
            chunk = bits_field[r_start:r_end]
            dots = torch.matmul(chunk, bits_pv_t)  # (chunk_len, N_pv)

            match_mask = dots >= min_dot_threshold  # (chunk_len, N_pv)
            chunk_matches = match_mask.any(dim=1).cpu().numpy()
            field_has_pv_match[r_start:r_end] = chunk_matches

            if r_start < n_pdoc:
                pdoc_slice_end = min(r_end, n_pdoc)
                pv_pdoc_pairs += int(match_mask[:pdoc_slice_end - r_start].sum().item())
            if r_end > n_pdoc:
                pwild_slice_start = max(r_start, n_pdoc)
                pwild_offset = pwild_slice_start - r_start
                pv_pwild_pairs += int(match_mask[pwild_offset:].sum().item())

    logger.info("PlantVillage <-> Field audit: %d PV-Doc pairs, %d PV-Wild pairs (%d total).",
                pv_pdoc_pairs, pv_pwild_pairs, pv_pdoc_pairs + pv_pwild_pairs)
    logger.info("Flagged %d field images with PlantVillage match at Hamming distance <= %d.",
                int(field_has_pv_match.sum()), threshold)

    return pv_pdoc_pairs, pv_pwild_pairs, field_has_pv_match


# ---------------------------------------------------------------------------
# Dataset Discovery
# ---------------------------------------------------------------------------
def scan_plantvillage(
    pv_root: str,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Scans PlantVillage dataset directory (data/plantvillage_dataset/color).
    Uses ImageFolder to deterministically discover files and imports stratified_split_indices
    from ml/src/dataset.py (seed=42) to establish the authoritative legacy_split.
    """
    pv_path = Path(pv_root).resolve()
    if not pv_path.exists():
        raise FileNotFoundError(f"PlantVillage dataset directory not found: {pv_path}")

    logger.info("Scanning PlantVillage at: %s", pv_path)
    dataset = ImageFolder(str(pv_path))
    total_pv_images = len(dataset.samples)
    logger.info("Discovered %d PlantVillage images across %d classes.", total_pv_images, len(dataset.classes))

    logger.info("Computing legacy PlantVillage stratified split using ml/src/dataset.py (seed=%d)...", seed)
    train_idx, val_idx, test_idx = stratified_split_indices(dataset, seed=seed)

    legacy_split_map: Dict[int, str] = {}
    for idx in train_idx:
        legacy_split_map[idx] = "train"
    for idx in val_idx:
        legacy_split_map[idx] = "val"
    for idx in test_idx:
        legacy_split_map[idx] = "test"

    records = []
    for idx, (img_abs_path, _) in enumerate(dataset.samples):
        img_p = Path(img_abs_path)
        rel_path = img_p.relative_to(pv_path).as_posix()
        orig_label = img_p.parent.name
        unified_label = orig_label  # PlantVillage labels are the reference standard

        records.append({
            "absolute_path": str(img_p),
            "path": rel_path,
            "source": "plantvillage",
            "orig_label": orig_label,
            "unified_label": unified_label,
            "confidence_level": "",      # PlantDoc/PlantWild only
            "orig_split": "",            # PlantDoc train/test only
            "legacy_split": legacy_split_map[idx],
        })

    return records


def scan_plantdoc(
    plantdoc_root: str,
    map_yaml_path: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scans PlantDoc dataset (data/plantdoc/PlantDoc-Dataset/train and test),
    mapping original class directory names to unified PlantVillage labels via ml/configs/mappings/plantdoc_map.yaml.
    """
    doc_path = Path(plantdoc_root).resolve()
    map_path = Path(map_yaml_path).resolve()

    if not doc_path.exists():
        raise FileNotFoundError(f"PlantDoc root directory not found: {doc_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"PlantDoc mapping configuration not found: {map_path}")

    logger.info("Loading PlantDoc mapping from: %s", map_path)
    with open(map_path, "r", encoding="utf-8") as f:
        map_config = yaml.safe_load(f)

    mapping: Dict[str, Dict[str, str]] = map_config.get("mapping", {})
    records = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    excluded_folders: Set[str] = set()

    for split in ["train", "test"]:
        split_dir = doc_path / split
        if not split_dir.exists():
            logger.warning("PlantDoc split directory '%s' does not exist. Skipping.", split_dir)
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            orig_label = class_dir.name
            if orig_label not in mapping:
                excluded_folders.add(orig_label)
                continue

            map_info = mapping[orig_label]
            unified_label = map_info["target"]
            confidence = map_info.get("confidence", "high")

            for img_file in sorted(class_dir.iterdir()):
                if img_file.is_file() and img_file.suffix in valid_extensions:
                    rel_path = img_file.relative_to(doc_path).as_posix()
                    records.append({
                        "absolute_path": str(img_file),
                        "path": rel_path,
                        "source": "plantdoc",
                        "orig_label": orig_label,
                        "unified_label": unified_label,
                        "confidence_level": str(confidence),
                        "orig_split": split,
                        "legacy_split": "",  # Empty for non-PlantVillage sources
                    })

    metadata = {
        "excluded_folders": sorted(list(excluded_folders)),
        "mapped_categories_count": len(mapping),
        "total_mapped_records": len(records)
    }
    logger.info(
        "Discovered %d valid mapped PlantDoc records across train and test splits (excluded %d unmapped categories).",
        len(records), len(excluded_folders)
    )

    return records, metadata


def scan_plantwild(
    plantwild_root: str,
    map_yaml_path: str,
    url_record_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scans PlantWild v1 dataset using PlantWildAdapter.
    Class folders are mapped via ml/configs/mappings/plantwild_map.yaml. Folders not in the mapping
    receive unified_label='other' and are preserved in the manifest. Official split lists
    are discovered and recorded. Associated source URLs are extracted from url_record.json.
    """
    adapter = PlantWildAdapter(
        data_dir=plantwild_root,
        mapping_file=map_yaml_path,
        url_record_file=url_record_path
    )
    return adapter.scan()


# ---------------------------------------------------------------------------
# Stratified Group Split Assignment
# ---------------------------------------------------------------------------
def assign_group_splits_pv(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Assigns PlantVillage splits (pv_train, pv_val, pv_test) strictly by group_id,
    stratified by unified_label (seed=42). Classes with fewer than 5 groups go to pv_train only.
    """
    groups_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups_by_id[r["group_id"]].append(r)

    # Every group has exactly one unified_label (asserted in clustering integrity check)
    group_labels: Dict[str, str] = {
        gid: group_records[0]["unified_label"]
        for gid, group_records in groups_by_id.items()
    }

    groups_by_label: Dict[str, List[str]] = defaultdict(list)
    for gid, label in group_labels.items():
        groups_by_label[label].append(gid)

    group_split_assignment: Dict[str, str] = {}
    tiny_classes_report: List[Dict[str, Any]] = []

    for label in sorted(groups_by_label.keys()):
        gids = sorted(groups_by_label[label])
        rng = random.Random(seed)
        rng.shuffle(gids)

        n_groups = len(gids)
        total_images = sum(len(groups_by_id[g]) for g in gids)

        if n_groups < 5:
            for gid in gids:
                group_split_assignment[gid] = "pv_train"
            tiny_classes_report.append({
                "source": "plantvillage",
                "unified_label": label,
                "group_count": n_groups,
                "image_count": total_images,
                "assigned_split": "pv_train",
                "reason": "Fewer than 5 groups; assigned to pv_train only."
            })
        else:
            n_train = int(n_groups * train_ratio)
            n_val = int(n_groups * val_ratio)
            n_test = n_groups - n_train - n_val

            if n_val == 0 and n_groups >= 3:
                n_val = 1
                n_train -= 1
            if n_test == 0 and n_groups >= 3:
                n_test = 1
                n_train -= 1

            for gid in gids[:n_train]:
                group_split_assignment[gid] = "pv_train"
            for gid in gids[n_train:n_train + n_val]:
                group_split_assignment[gid] = "pv_val"
            for gid in gids[n_train + n_val:]:
                group_split_assignment[gid] = "pv_test"

    for r in records:
        r["split"] = group_split_assignment[r["group_id"]]

    return records, tiny_classes_report


def assign_group_splits_field(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Assigns field splits across combined PlantDoc and PlantWild records.
    - Groups with unified_label 'other' are assigned strictly to 'other_pool' (never field_test).
    - Groups containing at least one pv_dup image are forced into field_train, exempt from val/test stratification.
    - Remaining groups (38 classes) are split 70/15/15 into field_train, field_val, field_test (seed=42).
    - Classes with fewer than 5 remaining groups are assigned entirely to field_train.
    """
    groups_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups_by_id[r["group_id"]].append(r)

    # Every group has exactly one unified_label (asserted in clustering integrity check)
    group_labels: Dict[str, str] = {
        gid: group_records[0]["unified_label"]
        for gid, group_records in groups_by_id.items()
    }

    group_split_assignment: Dict[str, str] = {}
    tiny_classes_report: List[Dict[str, Any]] = []

    # 1. Handle 'other' groups -> 'other_pool'
    for gid, label in group_labels.items():
        if label == "other":
            group_split_assignment[gid] = "other_pool"

    # 2. Force groups containing at least one pv_dup image into field_train
    # (exempt from the stratified val/test assignment)
    forced_pv_dup_groups = 0
    forced_pv_dup_images = 0

    for gid, group_records in groups_by_id.items():
        label = group_labels[gid]
        if label != "other" and any(r.get("pv_dup", 0) == 1 for r in group_records):
            group_split_assignment[gid] = "field_train"
            forced_pv_dup_groups += 1
            forced_pv_dup_images += len(group_records)

    logger.info(
        "PV-Field Duplicate Isolation: Forced %d groups (%d images) with pv_dup=1 into field_train (exempt from val/test).",
        forced_pv_dup_groups, forced_pv_dup_images
    )

    # 3. Stratify the remaining groups of the 38 classes across field_train, field_val, field_test
    remaining_groups_by_label: Dict[str, List[str]] = defaultdict(list)
    for gid, label in group_labels.items():
        if gid not in group_split_assignment:
            remaining_groups_by_label[label].append(gid)

    for label in sorted(remaining_groups_by_label.keys()):
        gids = sorted(remaining_groups_by_label[label])
        rng = random.Random(seed)
        rng.shuffle(gids)

        n_groups = len(gids)
        total_images = sum(len(groups_by_id[g]) for g in gids)

        if n_groups < 5:
            for gid in gids:
                group_split_assignment[gid] = "field_train"
            tiny_classes_report.append({
                "source": "field (plantdoc+plantwild)",
                "unified_label": label,
                "group_count": n_groups,
                "image_count": total_images,
                "assigned_split": "field_train",
                "reason": "Fewer than 5 remaining groups; assigned to field_train only."
            })
        else:
            n_train = int(n_groups * train_ratio)
            n_val = int(n_groups * val_ratio)
            n_test = n_groups - n_train - n_val

            if n_val == 0 and n_groups >= 3:
                n_val = 1
                n_train -= 1
            if n_test == 0 and n_groups >= 3:
                n_test = 1
                n_train -= 1

            for gid in gids[:n_train]:
                group_split_assignment[gid] = "field_train"
            for gid in gids[n_train:n_train + n_val]:
                group_split_assignment[gid] = "field_val"
            for gid in gids[n_train + n_val:]:
                group_split_assignment[gid] = "field_test"

    for r in records:
        r["split"] = group_split_assignment[r["group_id"]]

    forced_stats = {
        "forced_groups_count": forced_pv_dup_groups,
        "forced_images_count": forced_pv_dup_images
    }
    return records, tiny_classes_report, forced_stats


# ---------------------------------------------------------------------------
# Core Manifest Generation Pipeline
# ---------------------------------------------------------------------------
def build_manifest(
    pv_dir: str,
    plantdoc_dir: str,
    plantwild_dir: str,
    pdoc_map_file: str,
    pwild_map_file: str,
    output_manifest: str,
    output_report: str,
    url_record_file: Optional[str] = "data/plantwild/plantwild/url_record.json",
    cache_pv_hashes: Optional[str] = None,
    cache_pdoc_hashes: Optional[str] = None,
    cache_pwild_hashes: Optional[str] = None,
    workers: int = 8,
    threshold: int = 4,
    seed: int = 42,
    allow_big_clusters: bool = False
) -> None:
    """Orchestrates multi-source ingestion, joint field clustering, URL deduplication, and split assignment."""
    logger.info("=" * 80)
    logger.info("BUILDING UNIFIED PLANT DISEASE MANIFEST (PlantVillage + PlantDoc + PlantWild)")
    logger.info("=" * 80)

    # 1. Scan Datasets
    pv_records = scan_plantvillage(pv_dir, seed=seed)
    pdoc_records, pdoc_meta = scan_plantdoc(plantdoc_dir, pdoc_map_file)
    pwild_records, pwild_meta = scan_plantwild(plantwild_dir, pwild_map_file, url_record_path=url_record_file)

    logger.info("Discovered sample counts:")
    logger.info("  - PlantVillage: %d", len(pv_records))
    logger.info("  - PlantDoc:     %d", len(pdoc_records))
    logger.info("  - PlantWild:    %d (%d mapped, %d other, %d with source URLs)",
                len(pwild_records), pwild_meta["mapped_images_count"],
                pwild_meta["other_images_count"], pwild_meta.get("url_matches_count", 0))

    # -----------------------------------------------------------------------
    # 2. Compute / Load Perceptual Hashes
    # -----------------------------------------------------------------------
    def load_or_compute_hashes(records: List[Dict[str, Any]], cache_file: Optional[str], desc: str) -> np.ndarray:
        if cache_file and os.path.exists(cache_file):
            logger.info("Loading cached %s hashes from: %s", desc, cache_file)
            loaded = np.load(cache_file, allow_pickle=True)
            if isinstance(loaded, np.ndarray) and len(loaded) == len(records):
                return loaded
            elif isinstance(loaded, dict) or loaded.shape == ():
                d = loaded.item() if loaded.shape == () else dict(loaded)
                if "hashes" in d and len(d["hashes"]) == len(records):
                    return d["hashes"]

        paths = [r["absolute_path"] for r in records]
        hashes = compute_hashes_parallel(paths, workers=workers, desc=desc)
        if cache_file:
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
            np.save(cache_file, hashes)
            logger.info("Saved %s hashes cache to: %s", desc, cache_file)
        return hashes

    pv_hashes = load_or_compute_hashes(pv_records, cache_pv_hashes, "PlantVillage")
    pdoc_hashes = load_or_compute_hashes(pdoc_records, cache_pdoc_hashes, "PlantDoc")
    pwild_hashes = load_or_compute_hashes(pwild_records, cache_pwild_hashes, "PlantWild")

    # -----------------------------------------------------------------------
    # 3. Cross-Dataset Near-Duplicate Pair Auditing & PV-Field Match Flagging
    # -----------------------------------------------------------------------
    field_hashes = np.concatenate([pdoc_hashes, pwild_hashes], axis=0) if (len(pdoc_hashes) and len(pwild_hashes)) else (pdoc_hashes if len(pdoc_hashes) else pwild_hashes)

    # (a) PlantDoc - PlantWild Duplicate Pairs
    pdoc_pwild_dup_pairs = count_cross_dataset_duplicate_pairs(
        pdoc_hashes, pwild_hashes, threshold=threshold, desc="PlantDoc <-> PlantWild"
    )

    # (b) PlantVillage - Field Duplicate Pairs & PV-Field Match Flagging
    pv_pdoc_dup_pairs, pv_pwild_dup_pairs, field_has_pv_match = audit_pv_field_duplicates_and_flag_matches(
        pv_hashes=pv_hashes,
        pdoc_hashes=pdoc_hashes,
        pwild_hashes=pwild_hashes,
        threshold=threshold
    )
    pv_field_dup_pairs = pv_pdoc_dup_pairs + pv_pwild_dup_pairs

    logger.info("=" * 80)
    logger.info("CROSS-DATASET NEAR-DUPLICATE AUDIT SUMMARY:")
    logger.info("  - PlantDoc <-> PlantWild duplicate pairs: %d", pdoc_pwild_dup_pairs)
    logger.info("  - PlantVillage <-> Field duplicate pairs: %d (PlantDoc: %d, PlantWild: %d)",
                pv_field_dup_pairs, pv_pdoc_dup_pairs, pv_pwild_dup_pairs)
    logger.info("=" * 80)

    # Flag pv_dup (0/1): field image whose unified_label is not "other" and has PV match
    for r in pv_records:
        r["pv_dup"] = 0

    field_records = pdoc_records + pwild_records
    for idx, r in enumerate(field_records):
        if r["unified_label"] != "other" and bool(field_has_pv_match[idx]):
            r["pv_dup"] = 1
        else:
            r["pv_dup"] = 0

    total_pv_dup_field_images = sum(r["pv_dup"] for r in field_records)
    logger.info(
        "Flagged %d field images as pv_dup=1 (unified_label != 'other' and Hamming distance <= %d to PlantVillage).",
        total_pv_dup_field_images, threshold
    )

    # -----------------------------------------------------------------------
    # 4. Clustering & Group Assignment
    # -----------------------------------------------------------------------
    # PlantVillage: Clustered internally (phash union strictly restricted to same unified_label)
    logger.info("Clustering PlantVillage duplicates (phash union only within same unified_label)...")
    pv_labels = [r["unified_label"] for r in pv_records]
    pv_root_ids = cluster_near_duplicates_vectorized(pv_hashes, labels=pv_labels, threshold=threshold)
    pv_cluster_members: Dict[int, List[int]] = defaultdict(list)
    for idx, r_id in enumerate(pv_root_ids):
        pv_cluster_members[r_id].append(idx)

    sorted_pv_roots = sorted(
        pv_cluster_members.keys(),
        key=lambda r: min(pv_records[i]["path"] for i in pv_cluster_members[r])
    )
    pv_root_to_group = {r_id: f"pv_{g_num:06d}" for g_num, r_id in enumerate(sorted_pv_roots, start=1)}
    for idx, r in enumerate(pv_records):
        r["group_id"] = pv_root_to_group[pv_root_ids[idx]]

    # Field Data: Clustered JOINTLY across PlantDoc and PlantWild
    logger.info("Clustering Field duplicates JOINTLY across PlantDoc and PlantWild (%d total images)...", len(field_hashes))
    n_field = len(field_records)
    n_pdoc = len(pdoc_records)
    pwild_field_indices = range(n_pdoc, n_field)

    field_uf = UnionFind(n_field)

    # 4a. pHash Vectorized Union-Find
    if n_field > 1:
        bits_float = field_hashes.astype(np.float32) * 2.0 - 1.0
        bits_tensor = torch.from_numpy(bits_float)
        bits_tensor_t = bits_tensor.T
        min_dot_threshold = 64 - 2 * threshold
        chunk_size = 2000

        with torch.inference_mode():
            for r_start in range(0, n_field, chunk_size):
                r_end = min(r_start + chunk_size, n_field)
                chunk = bits_tensor[r_start:r_end]
                dots = torch.matmul(chunk, bits_tensor_t)

                row_idx, col_idx = torch.where(dots >= min_dot_threshold)
                row_arr = (row_idx + r_start).cpu().numpy()
                col_arr = col_idx.cpu().numpy()

                upper_mask = col_arr > row_arr
                for u, v in zip(row_arr[upper_mask], col_arr[upper_mask]):
                    # phash union strictly restricted to images with the same unified_label
                    if field_records[int(u)]["unified_label"] == field_records[int(v)]["unified_label"]:
                        field_uf.union(int(u), int(v))

    # 4b. Record PlantWild group count BEFORE URL Record Merge
    pwild_groups_before = set(field_uf.find(i) for i in pwild_field_indices)
    num_pwild_groups_before = len(pwild_groups_before)

    # 4c. URL Record Merge: Force images sharing identical normalized source URLs into same group
    # URL-based union ONLY between images with the same orig_label. Never union across labels by URL.
    logger.info("Applying URL record deduplication from: %s", url_record_file)
    url_to_field_indices: Dict[str, List[int]] = defaultdict(list)
    images_with_url = 0

    for idx in pwild_field_indices:
        norm_url = field_records[idx].get("normalized_url")
        if norm_url:
            images_with_url += 1
            url_to_field_indices[norm_url].append(idx)

    multi_image_url_clusters = 0
    cross_label_url_clusters = 0

    for norm_url, indices in url_to_field_indices.items():
        if len(indices) > 1:
            multi_image_url_clusters += 1
            orig_labels = set(field_records[i]["orig_label"] for i in indices)
            if len(orig_labels) > 1:
                cross_label_url_clusters += 1

            # Group indices by orig_label: union ONLY within the same orig_label
            by_orig_label: Dict[str, List[int]] = defaultdict(list)
            for i in indices:
                by_orig_label[field_records[i]["orig_label"]].append(i)

            for same_label_indices in by_orig_label.values():
                if len(same_label_indices) > 1:
                    first_idx = same_label_indices[0]
                    for other_idx in same_label_indices[1:]:
                        field_uf.union(first_idx, other_idx)

    # 4d. Record PlantWild group count AFTER URL Record Merge
    pwild_groups_after = set(field_uf.find(i) for i in pwild_field_indices)
    num_pwild_groups_after = len(pwild_groups_after)
    pwild_groups_merged = num_pwild_groups_before - num_pwild_groups_after

    logger.info(
        "PlantWild URL Merge Audit: %d groups before -> %d groups after (%d PlantWild groups merged by shared URLs).",
        num_pwild_groups_before, num_pwild_groups_after, pwild_groups_merged
    )
    logger.info(
        "Cross-Label URL Audit: %d URL clusters would have crossed orig_labels (count only, no cross-label union performed).",
        cross_label_url_clusters
    )

    # Finalize Field group_ids
    field_root_ids = [field_uf.find(i) for i in range(n_field)]
    field_cluster_members: Dict[int, List[int]] = defaultdict(list)
    for idx, r_id in enumerate(field_root_ids):
        field_cluster_members[r_id].append(idx)

    sorted_field_roots = sorted(
        field_cluster_members.keys(),
        key=lambda r: min(field_records[i]["path"] for i in field_cluster_members[r])
    )
    field_root_to_group = {r_id: f"field_{g_num:06d}" for g_num, r_id in enumerate(sorted_field_roots, start=1)}
    for idx, r in enumerate(field_records):
        r["group_id"] = field_root_to_group[field_root_ids[idx]]

    # 4e. Assert that every duplicate group has exactly one unified_label
    logger.info("Asserting that every duplicate group has exactly one unified_label...")
    group_to_unified_labels: Dict[str, Set[str]] = defaultdict(set)
    for r in pv_records + field_records:
        group_to_unified_labels[r["group_id"]].add(r["unified_label"])

    multi_label_groups = {
        gid: sorted(list(labels))
        for gid, labels in group_to_unified_labels.items()
        if len(labels) != 1
    }
    if multi_label_groups:
        offending_lines = "\n".join(
            f"  - Group '{gid}': unified_labels={labels}"
            for gid, labels in sorted(multi_label_groups.items())[:20]
        )
        extra = f"\n  ... and {len(multi_label_groups) - 20} more." if len(multi_label_groups) > 20 else ""
        error_msg = (
            f"ABORT: Group label integrity assertion failed! Found {len(multi_label_groups)} duplicate group(s) "
            f"containing multiple unified_labels:\n"
            f"{offending_lines}{extra}"
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    else:
        logger.info("[PASS] Verified: Every duplicate group across all sources has exactly one unified_label.")

    # -----------------------------------------------------------------------
    # 5. Group-Based Split Assignment
    # -----------------------------------------------------------------------
    logger.info("Assigning PlantVillage group splits (pv_train, pv_val, pv_test)...")
    pv_records, pv_tiny = assign_group_splits_pv(pv_records, seed=seed)

    logger.info("Assigning Field group splits (field_train, field_val, field_test, and other_pool)...")
    field_records, field_tiny, pv_dup_forced_stats = assign_group_splits_field(field_records, seed=seed)

    all_records = pv_records + field_records
    all_tiny_reports = pv_tiny + field_tiny

    # -----------------------------------------------------------------------
    # 6. Verification: Verify No group_id Crosses Splits & other_pool Exclusivity
    # -----------------------------------------------------------------------
    group_to_splits: Dict[str, Set[str]] = defaultdict(set)
    all_clusters_by_gid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_records:
        group_to_splits[r["group_id"]].add(r["split"])
        all_clusters_by_gid[r["group_id"]].append(r)

    violating_groups = {gid: list(splits) for gid, splits in group_to_splits.items() if len(splits) > 1}
    split_integrity_passed = len(violating_groups) == 0

    if not split_integrity_passed:
        logger.error("FATAL: Group integrity check failed! %d groups cross split boundaries.", len(violating_groups))
        raise AssertionError("Data leakage detected: group_id crosses split partitions!")
    else:
        logger.info("[PASS] Verified that zero group_ids cross split boundaries (100%% group isolation).")

    # Assert that other_pool contains only images with unified_label "other"
    non_other_in_pool = [
        r for r in all_records
        if r["split"] == "other_pool" and r["unified_label"] != "other"
    ]
    if non_other_in_pool:
        error_msg = (
            f"ABORT: other_pool integrity assertion failed! Found {len(non_other_in_pool)} image(s) with non-'other' "
            f"unified_label in other_pool. Examples: {[r['path'] for r in non_other_in_pool[:5]]}"
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    else:
        logger.info("[PASS] Verified: other_pool contains exclusively images with unified_label='other'.")

    # Cluster Size and Multi-label Audit
    cluster_sizes = [len(m) for m in all_clusters_by_gid.values()]
    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0

    # 20 largest clusters with size and label
    sorted_clusters = sorted(all_clusters_by_gid.items(), key=lambda kv: len(kv[1]), reverse=True)
    twenty_largest_clusters: List[Dict[str, Any]] = []
    for gid, members in sorted_clusters[:20]:
        twenty_largest_clusters.append({
            "group_id": gid,
            "size": len(members),
            "unified_label": members[0]["unified_label"],
            "sources": sorted(list(set(r["source"] for r in members))),
        })

    # Print the five largest clusters with sizes to the console
    print("=" * 60)
    print("FIVE LARGEST CLUSTERS:")
    for rank, (gid, members) in enumerate(sorted_clusters[:5], 1):
        lbl = members[0]["unified_label"]
        print(f"  [{rank}] Group '{gid}': size={len(members)}, label='{lbl}'")
    print("=" * 60)

    num_clusters_more_than_one_label = sum(
        1 for members in all_clusters_by_gid.values() if len(set(r["unified_label"] for r in members)) > 1
    )

    logger.info("Cluster Audit: max cluster size = %d | clusters spanning >1 label = %d",
                max_cluster_size, num_clusters_more_than_one_label)

    # Abort with a clear message if any cluster exceeds 300 images unless --allow-big-clusters is passed
    big_clusters = [(gid, len(m)) for gid, m in all_clusters_by_gid.items() if len(m) > 300]
    if big_clusters and not allow_big_clusters:
        violator_lines = "\n".join(f"  - Group '{gid}': {sz} images" for gid, sz in big_clusters[:10])
        extra = f"\n  ... and {len(big_clusters) - 10} more." if len(big_clusters) > 10 else ""
        error_msg = (
            f"ABORT: Cluster size threshold exceeded! Found {len(big_clusters)} cluster(s) with > 300 images "
            f"(maximum cluster size: {max_cluster_size}):\n"
            f"{violator_lines}{extra}\n"
            f"To allow clusters exceeding 300 images, re-run with --allow-big-clusters."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # -----------------------------------------------------------------------
    # 7. Write data/manifest.csv
    # -----------------------------------------------------------------------
    out_manifest_path = Path(output_manifest).resolve()
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    csv_columns = [
        "path",
        "source",
        "orig_label",
        "unified_label",
        "confidence_level",
        "orig_split",
        "legacy_split",
        "group_id",
        "split",
        "pv_dup"
    ]

    logger.info("Writing manifest CSV to: %s", out_manifest_path)
    with open(out_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for r in all_records:
            writer.writerow({col: r[col] for col in csv_columns})

    hasher = hashlib.sha256()
    with open(out_manifest_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    manifest_sha256 = hasher.hexdigest()
    logger.info("Computed SHA-256 for manifest.csv: %s", manifest_sha256)

    # -----------------------------------------------------------------------
    # 8. Compile Diagnostic Metrics & Write data/manifest_report.json
    # -----------------------------------------------------------------------
    out_report_path = Path(output_report).resolve()
    out_report_path.parent.mkdir(parents=True, exist_ok=True)

    counts_by_source = dict(Counter(r["source"] for r in all_records))
    counts_by_split = dict(Counter(r["split"] for r in all_records))

    pv_group_sizes = [len(m) for m in pv_cluster_members.values()]
    pv_dup_clusters = sum(1 for sz in pv_group_sizes if sz > 1)
    pv_dup_images = sum(sz for sz in pv_group_sizes if sz > 1)

    field_group_sizes = [len(m) for m in field_cluster_members.values()]
    field_dup_clusters = sum(1 for sz in field_group_sizes if sz > 1)
    field_dup_images = sum(sz for sz in field_group_sizes if sz > 1)

    # 38 PlantVillage Classes Breakdown per split
    pv_classes_38 = sorted(list(set(r["unified_label"] for r in pv_records)))

    per_class_counts_38: Dict[str, Dict[str, int]] = {}
    standard_splits = ["pv_train", "pv_val", "pv_test", "field_train", "field_val", "field_test"]

    for cls_name in pv_classes_38:
        per_class_counts_38[cls_name] = {s: 0 for s in standard_splits}
        per_class_counts_38[cls_name]["total_pv"] = 0
        per_class_counts_38[cls_name]["total_field"] = 0
        per_class_counts_38[cls_name]["grand_total"] = 0

    for r in all_records:
        u_label = r["unified_label"]
        if u_label in per_class_counts_38:
            sp = r["split"]
            if sp in per_class_counts_38[u_label]:
                per_class_counts_38[u_label][sp] += 1
            if r["source"] == "plantvillage":
                per_class_counts_38[u_label]["total_pv"] += 1
            elif r["source"] in {"plantdoc", "plantwild"}:
                per_class_counts_38[u_label]["total_field"] += 1
            per_class_counts_38[u_label]["grand_total"] += 1

    nested_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in all_records:
        nested_counts[r["source"]][r["unified_label"]][r["split"]] += 1

    counts_by_source_class_split: Dict[str, Any] = {}
    for src in sorted(nested_counts.keys()):
        counts_by_source_class_split[src] = {}
        for cls_name in sorted(nested_counts[src].keys()):
            split_dict = dict(nested_counts[src][cls_name])
            total_cls = sum(split_dict.values())
            counts_by_source_class_split[src][cls_name] = {
                "total": total_cls,
                "splits": split_dict
            }

    url_file_resolved = Path(url_record_file).resolve() if url_record_file else None
    url_record_exists = url_file_resolved.exists() if url_file_resolved else False

    report_payload = {
        "manifest_path": str(out_manifest_path),
        "manifest_sha256": manifest_sha256,
        "total_records": len(all_records),
        "maximum_cluster_size": max_cluster_size,
        "clusters_containing_more_than_one_label_count": num_clusters_more_than_one_label,
        "ten_largest_clusters": twenty_largest_clusters[:10],
        "twenty_largest_clusters": twenty_largest_clusters,
        "counts_by_source": counts_by_source,
        "counts_by_split": counts_by_split,
        "cross_dataset_duplicate_pairs": {
            "plantdoc_plantwild_pairs": pdoc_pwild_dup_pairs,
            "plantvillage_field_pairs": pv_field_dup_pairs,
            "breakdown": {
                "plantvillage_plantdoc_pairs": pv_pdoc_dup_pairs,
                "plantvillage_plantwild_pairs": pv_pwild_dup_pairs
            }
        },
        "pv_dup_isolation": {
            "total_pv_dup_field_images": total_pv_dup_field_images,
            "forced_groups_to_field_train": pv_dup_forced_stats["forced_groups_count"],
            "forced_images_to_field_train": pv_dup_forced_stats["forced_images_count"]
        },
        "url_record_deduplication": {
            "url_record_path": str(url_file_resolved) if url_file_resolved else None,
            "url_record_found": url_record_exists,
            "total_plantwild_images_with_url": images_with_url,
            "unique_normalized_urls_count": len(url_to_field_indices),
            "multi_image_url_clusters_count": multi_image_url_clusters,
            "cross_label_url_clusters_count": cross_label_url_clusters,
            "plantwild_groups_before_url_merge": num_pwild_groups_before,
            "plantwild_groups_after_url_merge": num_pwild_groups_after,
            "plantwild_groups_merged": pwild_groups_merged
        },
        "duplicate_clusters": {
            "maximum_cluster_size": max_cluster_size,
            "clusters_containing_more_than_one_label_count": num_clusters_more_than_one_label,
            "ten_largest_clusters": twenty_largest_clusters[:10],
            "twenty_largest_clusters": twenty_largest_clusters,
            "plantvillage": {
                "total_images": len(pv_records),
                "total_groups": len(pv_cluster_members),
                "duplicate_clusters_count": pv_dup_clusters,
                "singleton_groups_count": len(pv_cluster_members) - pv_dup_clusters,
                "images_in_duplicate_clusters": pv_dup_images,
                "max_cluster_size": max(pv_group_sizes) if pv_group_sizes else 0
            },
            "field_joint_plantdoc_plantwild": {
                "total_images": len(field_records),
                "total_groups": len(field_cluster_members),
                "duplicate_clusters_count": field_dup_clusters,
                "singleton_groups_count": len(field_cluster_members) - field_dup_clusters,
                "images_in_duplicate_clusters": field_dup_images,
                "max_cluster_size": max(field_group_sizes) if field_group_sizes else 0
            },
            "total_duplicate_clusters": pv_dup_clusters + field_dup_clusters
        },
        "official_plantwild_split_files": pwild_meta.get("official_split_files", []),
        "check_no_group_id_crosses_splits": {
            "passed": split_integrity_passed,
            "violating_groups_count": len(violating_groups),
            "violating_groups": violating_groups
        },
        "check_other_pool_only_contains_other": {
            "passed": len(non_other_in_pool) == 0,
            "non_other_images_count": len(non_other_in_pool)
        },
        "check_every_group_has_single_unified_label": {
            "passed": len(multi_label_groups) == 0,
            "multi_label_groups_count": len(multi_label_groups)
        },
        "tiny_classes_assigned_to_train_only": all_tiny_reports,
        "per_class_counts_38_classes": per_class_counts_38,
        "counts_by_source_class_and_split": counts_by_source_class_split
    }

    logger.info("Saving manifest diagnostic report to: %s", out_report_path)
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.info("=" * 80)
    logger.info("MANIFEST BUILD SUCCESSFULLY COMPLETED")
    logger.info("  - Total Images:    %d", len(all_records))
    logger.info("  - PlantVillage:    %d", len(pv_records))
    logger.info("  - Field (Doc+Wild):%d (Other Pool: %d)", len(field_records), counts_by_split.get("other_pool", 0))
    logger.info("  - PW Groups Merged via URL: %d", pwild_groups_merged)
    logger.info("  - PV-Field Duplicates Forced to field_train: %d groups (%d images)",
                pv_dup_forced_stats["forced_groups_count"], pv_dup_forced_stats["forced_images_count"])
    logger.info("  - Manifest SHA256: %s", manifest_sha256)
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Builds unified plant disease manifest across PlantVillage, PlantDoc, and PlantWild v1."
    )
    parser.add_argument(
        "--pv-dir",
        type=str,
        default="data/plantvillage_dataset/color",
        help="Path to PlantVillage dataset root (default: data/plantvillage_dataset/color)"
    )
    parser.add_argument(
        "--plantdoc-dir",
        type=str,
        default="data/plantdoc/PlantDoc-Dataset",
        help="Path to PlantDoc dataset root (default: data/plantdoc/PlantDoc-Dataset)"
    )
    parser.add_argument(
        "--plantwild-dir",
        type=str,
        default="data/plantwild/plantwild",
        help="Path to PlantWild v1 dataset root (default: data/plantwild/plantwild)"
    )
    parser.add_argument(
        "--pdoc-map",
        type=str,
        default="ml/configs/mappings/plantdoc_map.yaml",
        help="Path to PlantDoc class mapping YAML (default: ml/configs/mappings/plantdoc_map.yaml)"
    )
    parser.add_argument(
        "--pwild-map",
        type=str,
        default="ml/configs/mappings/plantwild_map.yaml",
        help="Path to PlantWild class mapping YAML (default: ml/configs/mappings/plantwild_map.yaml)"
    )
    parser.add_argument(
        "--url-record",
        type=str,
        default="data/plantwild/plantwild/url_record.json",
        help="Path to PlantWild url_record.json (default: data/plantwild/plantwild/url_record.json)"
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default="data/manifest.csv",
        help="Path for output manifest CSV (default: data/manifest.csv)"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="data/manifest_report.json",
        help="Path for output manifest diagnostic report JSON (default: data/manifest_report.json)"
    )
    parser.add_argument(
        "--cache-pv-hashes",
        type=str,
        default=None,
        help="Optional path to cache/load PlantVillage perceptual hashes"
    )
    parser.add_argument(
        "--cache-pdoc-hashes",
        type=str,
        default=None,
        help="Optional path to cache/load PlantDoc perceptual hashes"
    )
    parser.add_argument(
        "--cache-pwild-hashes",
        type=str,
        default=None,
        help="Optional path to cache/load PlantWild perceptual hashes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting reproducibility (default: 42)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Hamming distance threshold for near-duplicate clustering (default: 4)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of threads for perceptual hashing (default: 8)"
    )
    parser.add_argument(
        "--allow-big-clusters",
        action="store_true",
        default=False,
        help="Allow clusters with > 300 images without aborting (default: False)"
    )

    args = parser.parse_args()

    build_manifest(
        pv_dir=args.pv_dir,
        plantdoc_dir=args.plantdoc_dir,
        plantwild_dir=args.plantwild_dir,
        pdoc_map_file=args.pdoc_map,
        pwild_map_file=args.pwild_map,
        output_manifest=args.output_manifest,
        output_report=args.output_report,
        url_record_file=args.url_record,
        cache_pv_hashes=args.cache_pv_hashes,
        cache_pdoc_hashes=args.cache_pdoc_hashes,
        cache_pwild_hashes=args.cache_pwild_hashes,
        workers=args.workers,
        threshold=args.threshold,
        seed=args.seed,
        allow_big_clusters=args.allow_big_clusters
    )


if __name__ == "__main__":
    main()

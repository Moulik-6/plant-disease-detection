#!/usr/bin/env python3
"""
Manifest-driven PyTorch Dataset for Plant Disease Detection.

Implements `ManifestDataset` which parses `data/manifest.csv`, filters by
dataset split names (e.g., 'pv_train', 'field_train', 'field_val'), resolves
cached 512px images or original source paths, and returns (image, label_idx, row_idx).
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml

logger = logging.getLogger("datasets")

# Default source roots relative to project root
DEFAULT_SOURCE_ROOTS: Dict[str, str] = {
    "plantvillage": "data/plantvillage_dataset/color",
    "plantdoc": "data/plantdoc/PlantDoc-Dataset",
    "plantwild": "data/plantwild/plantwild",
}

DEFAULT_UNIFIED_CLASSES_YAML = "ml/configs/labels/unified_classes.yaml"


def load_unified_class_to_idx(classes_yaml_path: str = DEFAULT_UNIFIED_CLASSES_YAML) -> Dict[str, int]:
    """Loads unified class_to_idx mapping from YAML file."""
    p = Path(classes_yaml_path)
    if not p.exists():
        # Fallback relative to current working dir or repo ml/
        cand = Path(__file__).resolve().parent.parent / "configs" / "labels" / "unified_classes.yaml"
        if cand.exists():
            p = cand
        else:
            raise FileNotFoundError(f"Unified classes config not found at: {classes_yaml_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "class_to_idx" in data:
        return {str(k): int(v) for k, v in data["class_to_idx"].items()}
    elif "classes" in data:
        return {str(cls_name): idx for idx, cls_name in enumerate(data["classes"])}
    else:
        raise ValueError(f"No 'class_to_idx' or 'classes' found in {classes_yaml_path}")


class ManifestDataset(Dataset):
    """
    PyTorch Dataset reading records from `data/manifest.csv`.

    Features:
      - Filters records by split names (e.g. ['pv_train', 'field_train'])
      - Utilizes `cache_path` when present and existing on disk (fast 512px cached copies)
      - Resolves source roots for original full-resolution files when cache is absent
      - Excludes unmapped/non-target labels (e.g. 'other') when filter_unmapped=True
      - Returns: (image_tensor, label_idx, row_idx)
    """

    def __init__(
        self,
        manifest_path: Union[str, Path] = "data/manifest.csv",
        splits: Optional[Union[str, List[str], Set[str]]] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        source_roots: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
        filter_unmapped: bool = True,
        split_column: str = "split",
        source_filter: Optional[Union[str, List[str], Set[str]]] = None,
        exclude_sources: Optional[Union[str, List[str], Set[str]]] = None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path).resolve()
        self.transform = transform
        self.use_cache = use_cache
        self.filter_unmapped = filter_unmapped
        self.split_column = split_column

        # 1. Resolve class_to_idx mapping
        if class_to_idx is None:
            self.class_to_idx = load_unified_class_to_idx()
        else:
            self.class_to_idx = dict(class_to_idx)

        # Inverse mapping: idx -> class name
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(len(self.class_to_idx))]

        # 2. Resolve source roots
        self.source_roots: Dict[str, Path] = {}
        roots = source_roots or DEFAULT_SOURCE_ROOTS
        for k, v in roots.items():
            self.source_roots[k.lower().strip()] = Path(v).resolve()

        # 3. Resolve splits and source filters
        if splits is None:
            self.allowed_splits: Optional[Set[str]] = None
        elif isinstance(splits, str):
            self.allowed_splits = {splits.strip()}
        else:
            self.allowed_splits = {s.strip() for s in splits}

        if source_filter is None:
            self.allowed_sources: Optional[Set[str]] = None
        elif isinstance(source_filter, str):
            self.allowed_sources = {source_filter.lower().strip()}
        else:
            self.allowed_sources = {s.lower().strip() for s in source_filter}

        if exclude_sources is None:
            self.excluded_sources: Optional[Set[str]] = None
        elif isinstance(exclude_sources, str):
            self.excluded_sources = {exclude_sources.lower().strip()}
        else:
            self.excluded_sources = {s.lower().strip() for s in exclude_sources if s}

        # 4. Load and filter manifest
        self.samples: List[Dict[str, Any]] = []
        self.labels: List[int] = []
        self.sources: List[str] = []
        self.is_field: List[bool] = []
        self.excluded_by_split: Dict[str, int] = {}
        self.num_excluded: int = 0

        if not self.manifest_path.exists():
            logger.warning("Manifest CSV not found on disk at: %s", self.manifest_path)
            return

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for orig_row_idx, row in enumerate(reader):
                split_name = row.get(self.split_column, "").strip()
                if self.allowed_splits is not None and split_name not in self.allowed_splits:
                    continue

                source_name = row.get("source", "").lower().strip()
                if self.allowed_sources is not None and source_name not in self.allowed_sources:
                    continue

                if self.excluded_sources and source_name in self.excluded_sources:
                    self.excluded_by_split[split_name] = self.excluded_by_split.get(split_name, 0) + 1
                    continue

                unified_label = row.get("unified_label", "").strip()
                if unified_label not in self.class_to_idx:
                    if self.filter_unmapped:
                        continue
                    label_idx = -1
                else:
                    label_idx = self.class_to_idx[unified_label]

                source_name = row.get("source", "").lower().strip()
                is_field_img = (source_name in {"plantdoc", "plantwild"}) or split_name.startswith("field_")

                sample_entry = {
                    "manifest_row_idx": orig_row_idx,
                    "path": row.get("path", "").strip(),
                    "source": source_name,
                    "unified_label": unified_label,
                    "label_idx": label_idx,
                    "split": split_name,
                    "cache_path": row.get("cache_path", "").strip(),
                    "group_id": row.get("group_id", "").strip(),
                    "is_field": is_field_img,
                }

                self.samples.append(sample_entry)
                self.labels.append(label_idx)
                self.sources.append(source_name)
                self.is_field.append(is_field_img)

        self.num_excluded = sum(self.excluded_by_split.values())
        if self.num_excluded > 0:
            for sp_name, cnt in sorted(self.excluded_by_split.items()):
                logger.info(
                    "ManifestDataset: removed %d images from split '%s' (exclude_sources=%s)",
                    cnt, sp_name, sorted(list(self.excluded_sources))
                )

        logger.info(
            "ManifestDataset initialized with %d samples (splits: %s)",
            len(self.samples),
            sorted(list(self.allowed_splits)) if self.allowed_splits else "all",
        )

    def _resolve_image_path(self, sample: Dict[str, Any]) -> Path:
        """
        Determines the disk path to load the image from:
          1. Uses cache_path if enabled and present on disk
          2. Otherwise uses source root + relative path or direct path
        """
        cache_str = sample.get("cache_path", "")
        if self.use_cache and cache_str:
            cp = Path(cache_str)
            if cp.exists():
                return cp.resolve()

        # Fallback to source root
        source = sample.get("source", "")
        rel_path = Path(sample["path"])

        if rel_path.exists():
            return rel_path.resolve()

        if source in self.source_roots:
            candidate = self.source_roots[source] / rel_path
            if candidate.exists():
                return candidate.resolve()
            return candidate

        return rel_path.resolve()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            (image, label_idx, row_idx)
        """
        sample = self.samples[idx]
        img_path = self._resolve_image_path(sample)

        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to open image at '{img_path}': {e}") from e

        if self.transform is not None:
            img = self.transform(img)

        label_idx = sample["label_idx"]
        return img, label_idx, idx

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        splits: Union[str, List[str], Set[str]],
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        split_column: Optional[str] = None,
        source_filter: Optional[Union[str, List[str]]] = None,
        exclude_sources: Optional[Union[str, List[str], Set[str]]] = None,
    ) -> "ManifestDataset":
        """Instantiates ManifestDataset directly from an experiment config dictionary."""
        dataset_cfg = config.get("dataset", {})
        manifest_path = config.get("manifest_path", "data/manifest.csv")
        source_roots = config.get("sources", DEFAULT_SOURCE_ROOTS)
        use_cache = config.get("use_cache", True)

        if split_column is None:
            split_column = dataset_cfg.get("split_column", "split")
        if source_filter is None:
            source_filter = dataset_cfg.get("source_filter", None)
        if exclude_sources is None:
            exclude_sources = dataset_cfg.get("exclude_sources", None)

        if class_to_idx is None:
            unified_yaml = config.get("unified_classes", DEFAULT_UNIFIED_CLASSES_YAML)
            class_to_idx = load_unified_class_to_idx(unified_yaml)

        return cls(
            manifest_path=manifest_path,
            splits=splits,
            transform=transform,
            class_to_idx=class_to_idx,
            source_roots=source_roots,
            use_cache=use_cache,
            filter_unmapped=True,
            split_column=split_column,
            source_filter=source_filter,
            exclude_sources=exclude_sources,
        )

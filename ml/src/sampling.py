#!/usr/bin/env python3
"""
Sampling Strategies for Balanced Multi-Domain Training.

Implements `FieldFractionWeightedSampler` and `build_weighted_sampler` using
PyTorch's `WeightedRandomSampler` to enforce:
  1. A target field-image fraction per batch (sampling.field_fraction)
  2. Optional inverse-sqrt class frequency balancing (sampling.class_balanced)
"""

from collections import Counter
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger("sampling")


def calculate_sample_weights(
    labels: Sequence[int],
    is_field: Sequence[bool],
    field_fraction: Optional[float] = None,
    class_balanced: bool = False,
) -> torch.DoubleTensor:
    """
    Computes per-sample sampling weights satisfying:
      - Field probability mass equals `field_fraction` (if both field and non-field exist)
      - Within each domain, samples are optionally weighted by inverse-sqrt class frequency

    Args:
        labels: Sequence of integer class labels for each sample
        is_field: Sequence of booleans (True if field image, False otherwise)
        field_fraction: Target proportion of field images per batch in (0, 1), or None
        class_balanced: If True, scale sample weights by 1 / sqrt(N_class)

    Returns:
        torch.DoubleTensor of normalized sampling weights (sums to 1.0)
    """
    n_samples = len(labels)
    if n_samples == 0:
        return torch.DoubleTensor([])

    if len(is_field) != n_samples:
        raise ValueError(f"Mismatch: len(labels)={n_samples} vs len(is_field)={len(is_field)}")

    field_indices = [i for i, f in enumerate(is_field) if f]
    pv_indices = [i for i, f in enumerate(is_field) if not f]

    n_field = len(field_indices)
    n_pv = len(pv_indices)

    # 1. Base weights from class balancing (inverse-sqrt class frequency)
    if class_balanced:
        if n_field > 0 and n_pv > 0 and field_fraction is not None:
            # Domain-specific class frequencies so domain class skews are balanced internally
            field_class_counts = Counter(labels[i] for i in field_indices)
            pv_class_counts = Counter(labels[i] for i in pv_indices)

            raw_weights = np.zeros(n_samples, dtype=np.float64)
            for i in field_indices:
                cnt = field_class_counts[labels[i]]
                raw_weights[i] = 1.0 / math.sqrt(max(1, cnt))

            for i in pv_indices:
                cnt = pv_class_counts[labels[i]]
                raw_weights[i] = 1.0 / math.sqrt(max(1, cnt))
        else:
            # Overall class frequencies across the dataset
            global_counts = Counter(labels)
            raw_weights = np.array(
                [1.0 / math.sqrt(max(1, global_counts[lbl])) for lbl in labels],
                dtype=np.float64
            )
    else:
        raw_weights = np.ones(n_samples, dtype=np.float64)

    # 2. Partition weights by domain if field_fraction is specified
    weights = np.zeros(n_samples, dtype=np.float64)

    if field_fraction is not None and 0.0 < field_fraction < 1.0:
        if n_field > 0 and n_pv > 0:
            sum_field_raw = np.sum(raw_weights[field_indices])
            sum_pv_raw = np.sum(raw_weights[pv_indices])

            if sum_field_raw > 0:
                weights[field_indices] = float(field_fraction) * (raw_weights[field_indices] / sum_field_raw)
            if sum_pv_raw > 0:
                weights[pv_indices] = float(1.0 - field_fraction) * (raw_weights[pv_indices] / sum_pv_raw)

            logger.info(
                "Configured field-fraction sampling: %.2f field (%d imgs) / %.2f non-field (%d imgs)",
                field_fraction, n_field, 1.0 - field_fraction, n_pv
            )
        elif n_field > 0:
            logger.warning("field_fraction=%.2f requested but dataset has 0 non-field images.", field_fraction)
            weights[field_indices] = raw_weights[field_indices] / np.sum(raw_weights[field_indices])
        else:
            logger.warning("field_fraction=%.2f requested but dataset has 0 field images.", field_fraction)
            weights[pv_indices] = raw_weights[pv_indices] / np.sum(raw_weights[pv_indices])
    else:
        # Uniform or pure class-balanced weighting across all samples
        total_sum = np.sum(raw_weights)
        weights = raw_weights / total_sum if total_sum > 0 else raw_weights

    # Ensure total probability sums to 1.0
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights /= w_sum

    return torch.as_tensor(weights, dtype=torch.double)


class FieldFractionWeightedSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler with built-in domain partitioning and inverse-sqrt class balancing.

    Ensures that batches drawn with replacement maintain the desired field_fraction
    (e.g. 0.5 for 50/50 mix, or 0.8 for 20% PV replay).
    """

    def __init__(
        self,
        dataset: Any,
        field_fraction: Optional[float] = None,
        class_balanced: bool = False,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        # Extract labels and is_field flags
        if hasattr(dataset, "labels") and hasattr(dataset, "is_field"):
            labels = dataset.labels
            is_field = dataset.is_field
        elif hasattr(dataset, "samples") and len(dataset.samples) > 0 and isinstance(dataset.samples[0], dict):
            labels = [s["label_idx"] for s in dataset.samples]
            is_field = [s.get("is_field", False) for s in dataset.samples]
        else:
            # Fallback inspection of tuples
            labels = [dataset[i][1] for i in range(len(dataset))]
            is_field = [False] * len(dataset)

        weights = calculate_sample_weights(
            labels=labels,
            is_field=is_field,
            field_fraction=field_fraction,
            class_balanced=class_balanced,
        )

        if num_samples is None:
            num_samples = len(dataset)

        super().__init__(weights=weights, num_samples=num_samples, replacement=replacement)


def build_weighted_sampler(
    dataset: Dataset,
    config: Optional[Dict[str, Any]] = None,
    field_fraction: Optional[float] = None,
    class_balanced: Optional[bool] = None,
    num_samples: Optional[int] = None,
) -> Optional[WeightedRandomSampler]:
    """
    Factory function to create a WeightedRandomSampler from config parameters.

    If neither field_fraction nor class_balanced is active, returns None so standard
    shuffled DataLoader can be used without overhead.
    """
    cfg_sampling = config.get("sampling", {}) if config else {}

    if field_fraction is None:
        field_fraction = cfg_sampling.get("field_fraction", None)

    if class_balanced is None:
        class_balanced = cfg_sampling.get("class_balanced", False)

    if num_samples is None:
        num_samples = cfg_sampling.get("num_samples", None)

    # If neither condition requires custom sampling, return None
    if field_fraction is None and not class_balanced:
        return None

    return FieldFractionWeightedSampler(
        dataset=dataset,
        field_fraction=field_fraction,
        class_balanced=bool(class_balanced),
        num_samples=num_samples,
        replacement=True,
    )

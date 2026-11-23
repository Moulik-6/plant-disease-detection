import math
import numpy as np
import torch
import pytest

from ml.src.sampling import calculate_sample_weights, FieldFractionWeightedSampler

def test_calculate_sample_weights_field_fraction():
    """Verify that calculate_sample_weights allocates the exact target probability mass to field images."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Synthetic dataset: 800 laboratory (PV), 200 field images across 5 classes
    n_samples = 1000
    labels = np.random.randint(0, 5, size=n_samples).tolist()
    is_field = [False] * 800 + [True] * 200

    target_fraction = 0.40
    weights = calculate_sample_weights(
        labels=labels,
        is_field=is_field,
        field_fraction=target_fraction,
        class_balanced=False,
    )

    weights_np = weights.numpy()
    assert math.isclose(weights_np.sum(), 1.0, rel_tol=1e-5), "Weights must sum to 1.0"

    field_mass = weights_np[np.array(is_field)].sum()
    assert math.isclose(field_mass, target_fraction, rel_tol=1e-5), (
        f"Expected field probability mass {target_fraction}, got {field_mass}"
    )

    pv_mass = weights_np[~np.array(is_field)].sum()
    assert math.isclose(pv_mass, 1.0 - target_fraction, rel_tol=1e-5), (
        f"Expected non-field probability mass {1.0 - target_fraction}, got {pv_mass}"
    )

    # Empirical test: sample 25,000 items with replacement and check observed fraction
    draws = torch.multinomial(weights, num_samples=25000, replacement=True)
    is_field_tensor = torch.tensor(is_field, dtype=torch.bool)
    empirical_field_frac = is_field_tensor[draws].float().mean().item()

    assert abs(empirical_field_frac - target_fraction) < 0.02, (
        f"Empirical field fraction {empirical_field_frac:.4f} deviated from target {target_fraction}"
    )

def test_sampler_class_balancing():
    """Verify that class-balanced sampling scales inversely with class frequency."""
    labels = [0] * 900 + [1] * 100
    is_field = [False] * 1000

    weights = calculate_sample_weights(
        labels=labels,
        is_field=is_field,
        field_fraction=None,
        class_balanced=True,
    )

    # Class 0 has 900 items, Class 1 has 100 items
    # Raw weights: class 0 = 1 / sqrt(900) = 1/30; class 1 = 1 / sqrt(100) = 1/10
    # Ratio of single sample weight: class 1 / class 0 = 30 / 10 = 3.0
    w0 = weights[0].item()
    w1 = weights[900].item()
    assert math.isclose(w1 / w0, 3.0, rel_tol=1e-4), f"Expected weight ratio 3.0, got {w1 / w0}"

import math
import numpy as np
import pytest
from sklearn.metrics import f1_score

from ml.evaluate import compute_ece, compute_class_labelled_metrics

def test_hand_computed_ece():
    """Verify ECE calculation against exact hand-calculated analytical examples."""
    # Test case 1: Single bin active
    # 4 predictions with conf=0.9; 3 correct, 1 wrong
    # Accuracy = 3/4 = 0.75, Avg Conf = 0.90
    # |0.75 - 0.90| = 0.15
    confidences = np.array([0.9, 0.9, 0.9, 0.9])
    correct_mask = np.array([True, True, True, False])

    ece = compute_ece(confidences, correct_mask, num_bins=15)
    assert math.isclose(ece, 0.15, abs_tol=1e-5), f"Expected ECE 0.15, got {ece}"

    # Test case 2: Perfectly calibrated predictions
    # 2 predictions with conf=1.0, both correct; 2 predictions with conf=0.5, 1 correct
    conf_cal = np.array([1.0, 1.0, 0.5, 0.5])
    correct_cal = np.array([True, True, True, False])
    # Bin [1.0]: acc=1.0, conf=1.0 -> diff=0.0
    # Bin (0.4667, 0.5333]: acc=0.5, conf=0.5 -> diff=0.0
    ece_cal = compute_ece(conf_cal, correct_cal, num_bins=15)
    assert math.isclose(ece_cal, 0.0, abs_tol=1e-5), f"Expected perfect ECE 0.0, got {ece_cal}"

    # Test case 3: Two distinct bins
    # Two samples with conf=0.2 (0 correct), two with conf=0.8 (2 correct)
    # Bin 1: acc=0.0, conf=0.2 -> diff=0.2, weight=0.5 -> 0.10
    # Bin 2: acc=1.0, conf=0.8 -> diff=0.2, weight=0.5 -> 0.10
    # Total ECE = 0.20
    conf_two = np.array([0.2, 0.2, 0.8, 0.8])
    correct_two = np.array([False, False, True, True])
    ece_two = compute_ece(conf_two, correct_two, num_bins=15)
    assert math.isclose(ece_two, 0.20, abs_tol=1e-5), f"Expected ECE 0.20, got {ece_two}"


def test_hand_computed_macro_f1_over_present_classes():
    """Verify macro-F1 over present classes against exact hand calculation."""
    # Synthetic ground truth has only classes 0 and 1
    targets = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])

    # Class 0: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=2/3
    # Class 1: TP=2, FP=1, FN=0 -> Prec=2/3, Rec=1.0, F1=4/5
    # Macro-F1 = (2/3 + 4/5) / 2 = 11/15 ≈ 0.7333333
    expected_f1 = 11.0 / 15.0

    present_classes = np.array([0, 1])
    computed_f1 = f1_score(targets, preds, labels=present_classes, average="macro", zero_division=0)
    assert math.isclose(computed_f1, expected_f1, rel_tol=1e-5), (
        f"Expected macro-F1 {expected_f1:.6f}, got {computed_f1:.6f}"
    )

    # Test through compute_class_labelled_metrics
    probs = np.zeros((4, 38), dtype=np.float32)
    for i, p in enumerate(preds):
        probs[i, p] = 0.95
        probs[i, (p + 1) % 38] = 0.05

    metrics = compute_class_labelled_metrics(
        targets=targets,
        probs=probs,
        group_ids=["g0", "g1", "g2", "g3"],
        idx_to_class={i: f"class_{i}" for i in range(38)}
    )

    assert math.isclose(metrics["macro_f1"], expected_f1, rel_tol=1e-4)
    assert metrics["num_samples"] == 4
    assert metrics["top1_all"] == 0.75  # 3 of 4 correct

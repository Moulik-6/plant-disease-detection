#!/usr/bin/env python3
"""
Post-Hoc Probability Calibration and Out-of-Domain Unknown Handling.

Pipeline steps:
  1. Temperature Scaling: Fits a single scalar temperature T on field_val by
     minimising negative log-likelihood (NLL) of probabilities converted to logits,
     using L-BFGS in log-space. Evaluates NLL, ECE (15 bins), and top-1 accuracy
     before and after calibration on field_val AND field_test.
  2. Group-Aware Split: Splits other_pool deterministically by group_id into
     unknown_val (50%) and unknown_test (50%) using seed 42 (clusters never overlap).
  3. Rejection Threshold Selection (Validation Only):
     Uses maximum calibrated softmax probability as confidence score. Selects
     thresholds strictly on field_val and unknown_val across three operating points:
       (a) Maximum Youden's J between known (field_val) and unknown (unknown_val)
       (b) Highest threshold keeping >= 90% of field_val images
       (c) Threshold achieving >= 90% accuracy among accepted field_val images at maximum coverage
  4. Test Evaluation:
     Evaluates field_test and unknown_test at each operating point:
       - Known test coverage
       - Accuracy among accepted known test images
       - Unknown test rejection rate
       - AUROC of confidence score for known vs. unknown
     Computes risk-coverage curve (100% down to 20% in steps of 10%) with
     group_id cluster bootstrap 95% confidence intervals (1000 resamples, seed 42).
  5. Outputs:
     - Formatted summary tables to stdout
     - JSON artifact containing all metrics to results/calibration/<name>.json
     - Reliability diagram and Risk-Coverage curve PNGs via matplotlib (Agg backend)

Strict Guard:
  Never uses field_test or unknown_test to fit T or choose any threshold.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrate")


def resolve_eval_dir(eval_dir_str: str) -> Path:
    """Resolves an eval directory containing probs_<set>.npy and predictions_<set>.csv."""
    p = Path(eval_dir_str).resolve()
    if p.is_dir() and (p / "probs_field_val.npy").exists():
        return p
    if (p / "eval" / "probs_field_val.npy").exists():
        return p / "eval"
    raise FileNotFoundError(f"Valid eval directory not found at: {eval_dir_str}")


def load_eval_data(
    eval_dirs: List[Path],
    set_name: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Loads probs_<set>.npy and predictions_<set>.csv across member eval dirs.
    Asserts identical row count, row order (by path), and probability matrix shape.
    Averages probabilities when multiple eval dirs are provided.
    """
    all_probs: List[np.ndarray] = []
    canonical_meta: Optional[List[Dict[str, Any]]] = None
    canonical_paths: Optional[List[str]] = None

    for ed in eval_dirs:
        prob_file = ed / f"probs_{set_name}.npy"
        csv_file = ed / f"predictions_{set_name}.csv"

        if not prob_file.exists():
            raise FileNotFoundError(f"Missing probability file: {prob_file}")
        if not csv_file.exists():
            raise FileNotFoundError(f"Missing predictions file: {csv_file}")

        probs = np.load(prob_file)
        if probs.ndim != 2:
            raise ValueError(f"Expected 2D probability matrix in {prob_file}, got shape {probs.shape}")

        rows: List[Dict[str, Any]] = []
        paths: List[str] = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                p_norm = r.get("path", "").strip().replace("\\", "/")
                paths.append(p_norm)
                rows.append({
                    "path": p_norm,
                    "source": r.get("source", "").strip(),
                    "true_idx": int(r.get("true_idx", -1)),
                    "pred_idx": int(r.get("pred_idx", -1)),
                    "correct": int(r.get("correct", 0)) if r.get("correct") not in ("", None) else int(int(r.get("pred_idx", -2)) == int(r.get("true_idx", -1))),
                    "group_id": r.get("group_id", "").strip() or p_norm,
                })

        if len(probs) != len(rows):
            raise ValueError(f"Row count mismatch in {ed} for set '{set_name}': {len(probs)} probs vs {len(rows)} CSV rows")

        if canonical_paths is None:
            canonical_paths = paths
            canonical_meta = rows
        else:
            if len(paths) != len(canonical_paths):
                raise AssertionError(f"Row count mismatch between eval dirs for set '{set_name}'")
            if paths != canonical_paths:
                raise AssertionError(f"Row order mismatch (paths differ) between eval dirs for set '{set_name}'")

        all_probs.append(probs)

    # Assert shape alignment
    first_shape = all_probs[0].shape
    for i, p_arr in enumerate(all_probs[1:], start=1):
        if p_arr.shape != first_shape:
            raise AssertionError(f"Probability shape mismatch for set '{set_name}': {first_shape} vs {p_arr.shape}")

    # Average probabilities across ensemble members
    ensemble_probs = np.mean(all_probs, axis=0)
    return ensemble_probs, canonical_meta or []


def compute_nll(probs: np.ndarray, targets: np.ndarray) -> float:
    """Computes Negative Log-Likelihood given predicted probabilities and targets."""
    n = len(targets)
    if n == 0:
        return 0.0
    valid_mask = (targets >= 0) & (targets < probs.shape[1])
    if not np.any(valid_mask):
        return 0.0
    p_sub = probs[valid_mask]
    t_sub = targets[valid_mask]
    p_correct = np.clip(p_sub[np.arange(len(t_sub)), t_sub], 1e-12, 1.0)
    return float(-np.mean(np.log(p_correct)))


def compute_ece(confidences: np.ndarray, correct_mask: np.ndarray, num_bins: int = 15) -> float:
    """Computes Expected Calibration Error with equal-width probability bins."""
    n = len(confidences)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        b_low = bin_edges[i]
        b_high = bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (confidences >= b_low) & (confidences <= b_high)
        else:
            mask = (confidences >= b_low) & (confidences < b_high)
        b_size = np.sum(mask)
        if b_size > 0:
            acc = np.mean(correct_mask[mask])
            conf = np.mean(confidences[mask])
            ece += (b_size / n) * np.abs(acc - conf)
    return float(ece)


def probs_to_logits(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Converts a probability matrix into centered logits via natural logarithm."""
    clipped = np.clip(probs, eps, 1.0)
    logits = np.log(clipped)
    # Subtract max per row for numerical stability
    logits -= np.max(logits, axis=1, keepdims=True)
    return logits


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Fits one scalar temperature T > 0 by minimising negative log-likelihood on validation data
    using L-BFGS in log-space (s = log T).
    """
    n = len(targets)
    valid_mask = (targets >= 0) & (targets < logits.shape[1])
    v_logits = logits[valid_mask]
    v_targets = targets[valid_mask]
    n_valid = len(v_targets)

    def nll_objective(s_arr: np.ndarray) -> float:
        s = s_arr[0]
        T = np.exp(s)
        scaled = v_logits / T
        lse = logsumexp(scaled, axis=1)
        correct_logits = scaled[np.arange(n_valid), v_targets]
        loss = -np.mean(correct_logits - lse)
        return float(loss)

    res = minimize(
        nll_objective,
        x0=np.array([0.0], dtype=np.float64),
        method="L-BFGS-B",
        options={"maxiter": 100, "ftol": 1e-10},
    )
    opt_s = float(res.x[0])
    T_opt = float(np.exp(opt_s))
    return T_opt


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scales logits by temperature and applies softmax."""
    T = max(temperature, 1e-6)
    scaled = logits / T
    return softmax(scaled, axis=1)


def split_other_pool_by_group(
    other_pool_probs: np.ndarray,
    other_pool_meta: List[Dict[str, Any]],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministically splits other_pool 50/50 by group_id into unknown_val and unknown_test.
    Guarantees that a group_id is never present in both partitions.
    """
    raw_groups = [r.get("group_id", "") or r.get("path", "") for r in other_pool_meta]
    safe_groups = [g if (g and str(g).strip()) else f"other_{i}" for i, g in enumerate(raw_groups)]
    unique_groups = np.unique(safe_groups)

    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)

    split_idx = len(shuffled_groups) // 2
    val_groups = set(shuffled_groups[:split_idx])
    test_groups = set(shuffled_groups[split_idx:])
    assert val_groups.isdisjoint(test_groups), "Group overlap between unknown_val and unknown_test"

    is_val = np.array([g in val_groups for g in safe_groups], dtype=bool)
    is_test = np.array([g in test_groups for g in safe_groups], dtype=bool)

    u_val_probs = other_pool_probs[is_val]
    u_test_probs = other_pool_probs[is_test]
    u_val_meta = [row for row, m in zip(other_pool_meta, is_val) if m]
    u_test_meta = [row for row, m in zip(other_pool_meta, is_test) if m]

    return u_val_probs, u_test_probs, u_val_meta, u_test_meta


def select_rejection_thresholds(
    cal_val_known_probs: np.ndarray,
    val_known_targets: np.ndarray,
    cal_val_unknown_probs: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Selects rejection thresholds strictly on field_val (known) and unknown_val (unknown).
    Operating points:
      (a) max_youden_j: Maximises Youden's J = TPR (known coverage) + TNR (unknown rejection) - 1
      (b) keep_90_known: Highest threshold that maintains >= 90% coverage of field_val
      (c) acc_90_max_cov: Threshold where accuracy among accepted field_val reaches >= 90% at maximum coverage
    """
    c_known = np.max(cal_val_known_probs, axis=1)
    c_unknown = np.max(cal_val_unknown_probs, axis=1)
    preds_known = np.argmax(cal_val_known_probs, axis=1)
    correct_known = (preds_known == val_known_targets)
    n_known = len(c_known)
    n_unknown = len(c_unknown)

    # Candidate thresholds
    all_confs = np.unique(np.concatenate([[0.0, 1.0], c_known, c_unknown]))

    # (a) Maximum Youden's J
    best_j = -1e9
    best_tau_a = 0.0
    for tau in all_confs:
        tpr = np.mean(c_known >= tau)
        tnr = np.mean(c_unknown < tau)
        j = tpr + tnr - 1.0
        if j > best_j:
            best_j = j
            best_tau_a = float(tau)

    val_a_tpr = float(np.mean(c_known >= best_tau_a))
    val_a_tnr = float(np.mean(c_unknown < best_tau_a))

    # (b) Highest threshold keeping >= 90% field_val
    valid_b = [float(tau) for tau in all_confs if np.mean(c_known >= tau) >= 0.90]
    best_tau_b = max(valid_b) if valid_b else 0.0
    val_b_cov = float(np.mean(c_known >= best_tau_b))
    val_b_acc = float(np.mean(correct_known[c_known >= best_tau_b])) if np.any(c_known >= best_tau_b) else 0.0

    # (c) Accuracy >= 90% at maximum coverage
    valid_c: List[Tuple[float, float, float]] = []  # (coverage, accuracy, tau)
    for tau in all_confs:
        accepted = (c_known >= tau)
        if np.any(accepted):
            acc = float(np.mean(correct_known[accepted]))
            cov = float(np.mean(accepted))
            if acc >= 0.90:
                valid_c.append((cov, acc, float(tau)))

    if valid_c:
        # Highest coverage, then lowest tau
        max_cov = max(item[0] for item in valid_c)
        candidates_c = [item for item in valid_c if np.isclose(item[0], max_cov)]
        best_c = min(candidates_c, key=lambda x: x[2])
        best_tau_c = best_c[2]
        val_c_cov = best_c[0]
        val_c_acc = best_c[1]
    else:
        # If no threshold reaches 90% accuracy, select the threshold maximizing accuracy
        best_tau_c = 1.0
        acc_tuples = []
        for tau in all_confs:
            accepted = (c_known >= tau)
            if np.any(accepted):
                acc_tuples.append((float(np.mean(correct_known[accepted])), float(np.mean(accepted)), float(tau)))
        if acc_tuples:
            best_acc_tuple = max(acc_tuples, key=lambda x: (x[0], x[1]))
            best_tau_c = best_acc_tuple[2]
            val_c_cov = best_acc_tuple[1]
            val_c_acc = best_acc_tuple[0]
        else:
            val_c_cov = 0.0
            val_c_acc = 0.0

    return {
        "youden_j": {
            "name": "Maximum Youden J (Known vs Unknown)",
            "threshold": best_tau_a,
            "val_metric_name": "Youden J",
            "val_metric_val": float(best_j),
            "val_known_coverage": val_a_tpr,
            "val_unknown_rejection": val_a_tnr,
        },
        "keep_90_known": {
            "name": "Keep >= 90% Known Images",
            "threshold": best_tau_b,
            "val_metric_name": "Coverage",
            "val_metric_val": val_b_cov,
            "val_known_coverage": val_b_cov,
            "val_accepted_accuracy": val_b_acc,
        },
        "acc_90_max_cov": {
            "name": "Reach >= 90% Accuracy at Max Coverage",
            "threshold": best_tau_c,
            "val_metric_name": "Accuracy",
            "val_metric_val": val_c_acc,
            "val_known_coverage": val_c_cov,
            "val_accepted_accuracy": val_c_acc,
        },
    }


def evaluate_operating_points_on_test(
    thresholds: Dict[str, Dict[str, Any]],
    cal_test_known_probs: np.ndarray,
    test_known_targets: np.ndarray,
    cal_test_unknown_probs: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates the validation-chosen operating points on held-out field_test and unknown_test."""
    c_known = np.max(cal_test_known_probs, axis=1)
    c_unknown = np.max(cal_test_unknown_probs, axis=1)
    preds_known = np.argmax(cal_test_known_probs, axis=1)
    correct_known = (preds_known == test_known_targets)

    evaluated_points: Dict[str, Dict[str, Any]] = {}

    for key, op in thresholds.items():
        tau = op["threshold"]
        accepted_k = (c_known >= tau)
        n_accepted = int(np.sum(accepted_k))

        cov_k = float(np.mean(accepted_k))
        acc_k = float(np.mean(correct_known[accepted_k])) if n_accepted > 0 else 0.0
        rej_u = float(np.mean(c_unknown < tau))

        evaluated_points[key] = {
            **op,
            "test_known_coverage": cov_k,
            "test_accepted_accuracy": acc_k,
            "test_unknown_rejection": rej_u,
            "test_accepted_samples": n_accepted,
        }

    return evaluated_points


def compute_risk_coverage_curve(
    cal_test_known_probs: np.ndarray,
    test_known_targets: np.ndarray,
    test_known_meta: List[Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Computes risk-coverage curve from 100% down to 20% coverage in steps of 10%
    with cluster bootstrap 95% confidence intervals by group_id on field_test.
    """
    c_known = np.max(cal_test_known_probs, axis=1)
    preds_known = np.argmax(cal_test_known_probs, axis=1)
    correct_known = (preds_known == test_known_targets)
    n = len(c_known)

    raw_groups = [r.get("group_id", "") or r.get("path", "") for r in test_known_meta]
    safe_groups = [g if (g and str(g).strip()) else f"k_{i}" for i, g in enumerate(raw_groups)]
    unique_groups, group_indices = np.unique(safe_groups, return_inverse=True)
    k_clusters = len(unique_groups)

    rng = np.random.default_rng(seed)
    sampled_clusters = rng.choice(k_clusters, size=(n_resamples, k_clusters), replace=True)

    target_coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    table_rows: List[Dict[str, Any]] = []

    for cov in target_coverages:
        # Determine threshold matching coverage
        tau = float(np.percentile(c_known, (1.0 - cov) * 100.0))
        accepted = (c_known >= tau)
        n_acc = int(np.sum(accepted))

        emp_cov = float(np.mean(accepted))
        emp_acc = float(np.mean(correct_known[accepted])) if n_acc > 0 else 0.0
        emp_risk = 1.0 - emp_acc

        cluster_acc = np.bincount(group_indices, weights=accepted.astype(np.float64), minlength=k_clusters)
        cluster_corr = np.bincount(group_indices, weights=(accepted & correct_known).astype(np.float64), minlength=k_clusters)

        boot_acc_tot = cluster_acc[sampled_clusters].sum(axis=1)
        boot_corr_tot = cluster_corr[sampled_clusters].sum(axis=1)
        boot_acc = np.divide(boot_corr_tot, boot_acc_tot, out=np.zeros_like(boot_corr_tot, dtype=float), where=boot_acc_tot > 0)
        boot_risk = 1.0 - boot_acc

        risk_ci = [float(np.percentile(boot_risk, 2.5)), float(np.percentile(boot_risk, 97.5))]
        acc_ci = [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))]

        table_rows.append({
            "target_coverage": float(cov),
            "actual_coverage": emp_cov,
            "threshold": tau,
            "accepted_samples": n_acc,
            "accuracy": emp_acc,
            "accuracy_ci_95": acc_ci,
            "risk": emp_risk,
            "risk_ci_95": risk_ci,
        })

    return table_rows


def plot_reliability_diagram(
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
    targets: np.ndarray,
    out_png: Path,
    name: str,
    num_bins: int = 15,
) -> None:
    """Plots and saves a reliability diagram comparing pre- and post-calibration probabilities."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(6.5, 6.0), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for probs, label, color in [
        (uncal_probs, "Uncalibrated", "#d62728"),
        (cal_probs, "Temperature Calibrated", "#1f77b4"),
    ]:
        conf = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        correct = (preds == targets)

        bin_accs = []
        bin_confs = []
        counts = []

        for i in range(num_bins):
            b_l = bin_edges[i]
            b_h = bin_edges[i + 1]
            mask = (conf >= b_l) & (conf <= b_h if i == num_bins - 1 else conf < b_h)
            cnt = np.sum(mask)
            counts.append(cnt)
            if cnt > 0:
                bin_accs.append(float(np.mean(correct[mask])))
                bin_confs.append(float(np.mean(conf[mask])))
            else:
                bin_accs.append(np.nan)
                bin_confs.append(bin_centers[i])

        ax_main.plot(bin_confs, bin_accs, marker="o", lw=1.8, label=label, color=color)
        if label == "Temperature Calibrated":
            ax_hist.bar(bin_centers, counts, width=1.0 / num_bins * 0.8, color=color, alpha=0.7)

    ax_main.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")
    ax_main.set_ylabel("Empirical Accuracy", fontsize=11)
    ax_main.set_ylim(-0.02, 1.02)
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(loc="upper left", frameon=True)
    ax_main.set_title(f"Reliability Diagram (field_test) - {name}", fontsize=12, fontweight="bold")

    ax_hist.set_xlabel("Confidence (Max Softmax Probability)", fontsize=11)
    ax_hist.set_ylabel("Sample Count", fontsize=10)
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_risk_coverage_curve(
    risk_table: List[Dict[str, Any]],
    operating_points: Dict[str, Dict[str, Any]],
    out_png: Path,
    name: str,
) -> None:
    """Plots and saves the Risk-Coverage curve with 95% cluster bootstrap confidence intervals."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    coverages = [r["actual_coverage"] * 100.0 for r in risk_table]
    risks = [r["risk"] * 100.0 for r in risk_table]
    ci_low = [r["risk_ci_95"][0] * 100.0 for r in risk_table]
    ci_high = [r["risk_ci_95"][1] * 100.0 for r in risk_table]

    ax.plot(coverages, risks, marker="o", color="#1f77b4", lw=2.0, label="Risk (1 - Accuracy)")
    ax.fill_between(coverages, ci_low, ci_high, color="#1f77b4", alpha=0.2, label="95% CI (Group Bootstrap)")

    # Highlight operating points
    markers = {"youden_j": ("s", "#2ca02c"), "keep_90_known": ("^", "#ff7f0e"), "acc_90_max_cov": ("*", "#d62728")}
    for key, (sym, col) in markers.items():
        if key in operating_points:
            op = operating_points[key]
            cov_pt = op["test_known_coverage"] * 100.0
            risk_pt = (1.0 - op["test_accepted_accuracy"]) * 100.0
            ax.scatter([cov_pt], [risk_pt], color=col, marker=sym, s=90, zorder=5, label=f"{op['name']}")

    ax.set_xlabel("Coverage on Known Field Test (%)", fontsize=11)
    ax.set_ylabel("Selective Risk / Error Rate (%)", fontsize=11)
    ax.set_title(f"Selective Classification: Risk-Coverage Curve ({name})", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def print_summary_tables(
    calibration_summary: Dict[str, Any],
    operating_points: Dict[str, Dict[str, Any]],
    risk_table: List[Dict[str, Any]],
    auroc: float,
) -> None:
    """Prints all formatted results tables to stdout."""
    print("\n" + "=" * 90)
    print(" 1. TEMPERATURE SCALING & CALIBRATION METRICS")
    print("=" * 90)
    t_val = calibration_summary["temperature"]
    print(f"Optimal Temperature (fitted strictly on field_val): T = {t_val:.4f}\n")

    cal_hdr = f"{'Partition':<14} | {'NLL (Before)':>12} | {'NLL (After)':>12} | {'ECE (Before)':>12} | {'ECE (After)':>12} | {'Top-1 Acc':>10}"
    sep = "-" * len(cal_hdr)
    print(cal_hdr)
    print(sep)
    for part in ["field_val", "field_test"]:
        p_data = calibration_summary[part]
        row = (
            f"{part:<14} | "
            f"{p_data['nll_before']:>12.4f} | "
            f"{p_data['nll_after']:>12.4f} | "
            f"{p_data['ece_before']:>12.4f} | "
            f"{p_data['ece_after']:>12.4f} | "
            f"{p_data['accuracy']*100:>9.2f}%"
        )
        print(row)
    print(sep)

    print("\n" + "=" * 105)
    print(f" 2. OUT-OF-DOMAIN UNKNOWN HANDLING (Known-vs-Unknown AUROC on Test = {auroc:.4f})")
    print("=" * 105)
    op_hdr = f"{'Operating Point':<32} | {'Threshold':>9} | {'Val Known Cov':>13} | {'Test Known Cov':>14} | {'Test Acc (Acc.)':>15} | {'Test Unk. Rej.':>14}"
    sep_op = "-" * len(op_hdr)
    print(op_hdr)
    print(sep_op)
    for key, op in operating_points.items():
        row = (
            f"{op['name']:<32} | "
            f"{op['threshold']:>9.4f} | "
            f"{op.get('val_known_coverage', 0.0)*100:>12.1f}% | "
            f"{op['test_known_coverage']*100:>13.1f}% | "
            f"{op['test_accepted_accuracy']*100:>14.2f}% | "
            f"{op['test_unknown_rejection']*100:>13.1f}%"
        )
        print(row)
    print(sep_op)
    print("Note: Thresholds chosen strictly on field_val and unknown_val. Evaluated on field_test and unknown_test.")

    print("\n" + "=" * 95)
    print(" 3. RISK-COVERAGE CURVE (field_test with group_id Cluster Bootstrap 95% CIs)")
    print("=" * 95)
    rc_hdr = f"{'Target Cov.':>11} | {'Actual Cov.':>11} | {'Threshold':>9} | {'Accepted N':>10} | {'Accepted Acc. (95% CI)':>23} | {'Selective Risk (95% CI)':>23}"
    sep_rc = "-" * len(rc_hdr)
    print(rc_hdr)
    print(sep_rc)
    for r in risk_table:
        acc_ci_str = f"{r['accuracy']*100:.1f}% [{r['accuracy_ci_95'][0]*100:.1f}%, {r['accuracy_ci_95'][1]*100:.1f}%]"
        risk_ci_str = f"{r['risk']*100:.1f}% [{r['risk_ci_95'][0]*100:.1f}%, {r['risk_ci_95'][1]*100:.1f}%]"
        row = (
            f"{r['target_coverage']*100:>10.0f}% | "
            f"{r['actual_coverage']*100:>10.1f}% | "
            f"{r['threshold']:>9.4f} | "
            f"{r['accepted_samples']:>10d} | "
            f"{acc_ci_str:>23} | "
            f"{risk_ci_str:>23}"
        )
        print(row)
    print(sep_rc + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Temperature calibration and unknown rejection evaluation for plant disease detection."
    )
    parser.add_argument(
        "--eval-dir",
        action="extend",
        nargs="+",
        required=True,
        help="One or more eval directories containing probs_<set>.npy and predictions_<set>.csv.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Identifier name for output files (defaults to run folder name).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/calibration",
        help="Directory to save JSON metrics and PNG figures (default: results/calibration).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting and cluster bootstrap (default: 42).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of cluster bootstrap resamples (default: 1000).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Flatten and resolve eval dirs
    raw_dirs = args.eval_dir if isinstance(args.eval_dir, list) else [args.eval_dir]
    eval_dirs = [resolve_eval_dir(d) for d in raw_dirs]

    if not args.name:
        first_ed = eval_dirs[0]
        name = first_ed.parent.name if first_ed.name == "eval" else first_ed.name
    else:
        name = args.name

    logger.info("Starting calibration pipeline for '%s' across %d eval dirs", name, len(eval_dirs))

    # 1. Load data
    val_probs, val_meta = load_eval_data(eval_dirs, "field_val")
    test_probs, test_meta = load_eval_data(eval_dirs, "field_test")
    other_probs, other_meta = load_eval_data(eval_dirs, "other_pool")

    val_targets = np.array([r["true_idx"] for r in val_meta], dtype=int)
    test_targets = np.array([r["true_idx"] for r in test_meta], dtype=int)

    # 2. Step 1: Fit temperature T on field_val ONLY
    val_logits = probs_to_logits(val_probs)
    test_logits = probs_to_logits(test_probs)
    other_logits = probs_to_logits(other_probs)

    T_opt = fit_temperature(val_logits, val_targets)
    logger.info("Optimized temperature fitted on field_val: T = %.4f", T_opt)

    # Calibrate probabilities
    val_cal_probs = apply_temperature(val_logits, T_opt)
    test_cal_probs = apply_temperature(test_logits, T_opt)
    other_cal_probs = apply_temperature(other_logits, T_opt)

    # Pre/post calibration metrics
    val_correct = (np.argmax(val_probs, axis=1) == val_targets)
    test_correct = (np.argmax(test_probs, axis=1) == test_targets)

    cal_summary = {
        "temperature": T_opt,
        "field_val": {
            "n_samples": len(val_targets),
            "nll_before": compute_nll(val_probs, val_targets),
            "nll_after": compute_nll(val_cal_probs, val_targets),
            "ece_before": compute_ece(np.max(val_probs, axis=1), val_correct),
            "ece_after": compute_ece(np.max(val_cal_probs, axis=1), val_correct),
            "accuracy": float(np.mean(val_correct)),
        },
        "field_test": {
            "n_samples": len(test_targets),
            "nll_before": compute_nll(test_probs, test_targets),
            "nll_after": compute_nll(test_cal_probs, test_targets),
            "ece_before": compute_ece(np.max(test_probs, axis=1), test_correct),
            "ece_after": compute_ece(np.max(test_cal_probs, axis=1), test_correct),
            "accuracy": float(np.mean(test_correct)),
        },
    }

    # 3. Step 2: Split other_pool deterministically by group_id
    u_val_probs, u_test_probs, u_val_meta, u_test_meta = split_other_pool_by_group(
        other_cal_probs, other_meta, seed=args.seed
    )
    logger.info(
        "other_pool split: unknown_val=%d samples, unknown_test=%d samples",
        len(u_val_probs), len(u_test_probs)
    )

    # 4. Step 3: Select rejection thresholds strictly on field_val and unknown_val
    val_thresholds = select_rejection_thresholds(
        cal_val_known_probs=val_cal_probs,
        val_known_targets=val_targets,
        cal_val_unknown_probs=u_val_probs,
    )

    # 5. Step 4: Evaluate operating points on held-out field_test and unknown_test
    test_eval_points = evaluate_operating_points_on_test(
        thresholds=val_thresholds,
        cal_test_known_probs=test_cal_probs,
        test_known_targets=test_targets,
        cal_test_unknown_probs=u_test_probs,
    )

    # Known vs Unknown AUROC on test
    test_known_conf = np.max(test_cal_probs, axis=1)
    test_unknown_conf = np.max(u_test_probs, axis=1)
    y_test_domain = np.concatenate([np.ones_like(test_known_conf), np.zeros_like(test_unknown_conf)])
    scores_test_domain = np.concatenate([test_known_conf, test_unknown_conf])
    auroc = float(roc_auc_score(y_test_domain, scores_test_domain))

    # Risk-coverage curve with group_id cluster bootstrap
    risk_table = compute_risk_coverage_curve(
        cal_test_known_probs=test_cal_probs,
        test_known_targets=test_targets,
        test_known_meta=test_meta,
        n_resamples=args.n_boot,
        seed=args.seed,
    )

    # Print summary tables
    print_summary_tables(
        calibration_summary=cal_summary,
        operating_points=test_eval_points,
        risk_table=risk_table,
        auroc=auroc,
    )

    # 6. Step 5: Save outputs
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "name": name,
        "eval_dirs": [str(d) for d in eval_dirs],
        "temperature": T_opt,
        "calibration": cal_summary,
        "operating_points": test_eval_points,
        "auroc_known_vs_unknown": auroc,
        "other_pool_split": {
            "unknown_val_samples": len(u_val_probs),
            "unknown_test_samples": len(u_test_probs),
        },
        "risk_coverage_curve": risk_table,
    }

    json_file = out_dir / f"{name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    logger.info("Saved calibration JSON to: %s", json_file)

    rel_png = out_dir / f"{name}_reliability.png"
    plot_reliability_diagram(
        uncal_probs=test_probs,
        cal_probs=test_cal_probs,
        targets=test_targets,
        out_png=rel_png,
        name=name,
    )
    logger.info("Saved reliability diagram to: %s", rel_png)

    rc_png = out_dir / f"{name}_risk_coverage.png"
    plot_risk_coverage_curve(
        risk_table=risk_table,
        operating_points=test_eval_points,
        out_png=rc_png,
        name=name,
    )
    logger.info("Saved risk-coverage plot to: %s", rc_png)


if __name__ == "__main__":
    main()

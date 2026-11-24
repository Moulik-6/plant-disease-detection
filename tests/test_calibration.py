import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.special import softmax

from ml.analysis.calibrate import (
    apply_temperature,
    compute_ece,
    compute_nll,
    fit_temperature,
    probs_to_logits,
    select_rejection_thresholds,
    split_other_pool_by_group,
)


def test_temperature_scaling_recovery():
    """
    (1) Verify that fit_temperature accurately recovers a known temperature T*
    from uncalibrated probabilities, lowers NLL, and preserves top-1 accuracy.
    """
    rng = np.random.default_rng(42)
    n, c = 2000, 10
    true_T = 1.6

    # Generate latent logits with predictive signal
    z = rng.normal(0, 1.8, size=(n, c))
    true_probs = softmax(z, axis=1)
    targets = np.array([rng.choice(c, p=p) for p in true_probs], dtype=int)

    # Overconfident uncalibrated logits
    uncal_logits = z * true_T
    uncal_probs = softmax(uncal_logits, axis=1)

    # Convert to logits and fit temperature
    logits = probs_to_logits(uncal_probs)
    recovered_T = fit_temperature(logits, targets)

    # Check recovery within tolerance
    assert abs(recovered_T - true_T) < 0.10, f"Expected T ~ {true_T}, got {recovered_T:.4f}"

    # Evaluate NLL and Accuracy
    cal_probs = apply_temperature(logits, recovered_T)
    nll_before = compute_nll(uncal_probs, targets)
    nll_after = compute_nll(cal_probs, targets)

    assert nll_after < nll_before, f"NLL should decrease: before={nll_before:.4f}, after={nll_after:.4f}"

    acc_before = float(np.mean(np.argmax(uncal_probs, axis=1) == targets))
    acc_after = float(np.mean(np.argmax(cal_probs, axis=1) == targets))
    assert np.isclose(acc_before, acc_after), "Temperature scaling must strictly preserve top-1 accuracy"


def test_threshold_selection_isolation():
    """
    (2) Verify that operating point thresholds are chosen solely from validation arrays,
    never touch test data, and satisfy their operational guarantees.
    """
    rng = np.random.default_rng(123)
    n_k, n_u, c = 300, 150, 5

    # Synthetic validation data: known has higher confidences and high accuracy on top confidences
    val_k_confs = rng.beta(5, 2, size=n_k)
    val_u_confs = rng.beta(2, 5, size=n_u)

    val_targets = rng.integers(0, c, size=n_k)
    # Higher confidence samples are mostly correct
    is_correct = (val_k_confs > 0.6) | (rng.random(size=n_k) < 0.3)
    val_preds = np.copy(val_targets)
    val_preds[~is_correct] = (val_targets[~is_correct] + 1) % c

    val_k_probs = np.zeros((n_k, c))
    for i in range(n_k):
        val_k_probs[i, val_preds[i]] = val_k_confs[i]
        rem = (1.0 - val_k_confs[i]) / (c - 1)
        for j in range(c):
            if j != val_preds[i]:
                val_k_probs[i, j] = rem

    val_u_probs = np.zeros((n_u, c))
    for i in range(n_u):
        pred_u = rng.integers(0, c)
        val_u_probs[i, pred_u] = val_u_confs[i]
        rem = (1.0 - val_u_confs[i]) / (c - 1)
        for j in range(c):
            if j != pred_u:
                val_u_probs[i, j] = rem

    # Select rejection thresholds using only validation data
    thresholds = select_rejection_thresholds(
        cal_val_known_probs=val_k_probs,
        val_known_targets=val_targets,
        cal_val_unknown_probs=val_u_probs,
    )

    tau_a = thresholds["youden_j"]["threshold"]
    tau_b = thresholds["keep_90_known"]["threshold"]
    tau_c = thresholds["acc_90_max_cov"]["threshold"]

    # (a) Check Youden J optimality on validation
    c_k = np.max(val_k_probs, axis=1)
    c_u = np.max(val_u_probs, axis=1)
    j_opt = thresholds["youden_j"]["val_metric_val"]
    for test_tau in np.linspace(0.0, 1.0, 101):
        tpr = np.mean(c_k >= test_tau)
        tnr = np.mean(c_u < test_tau)
        j_cand = tpr + tnr - 1.0
        assert j_cand <= j_opt + 1e-5, f"Candidate threshold {test_tau} achieved J={j_cand} > optimal J={j_opt}"

    # (b) Check keep >= 90% known on validation
    cov_b = np.mean(c_k >= tau_b)
    assert cov_b >= 0.90, f"Operating point (b) coverage must be >= 0.90, got {cov_b:.4f}"

    # (c) Check accuracy >= 90% at maximum coverage
    accepted_c = (c_k >= tau_c)
    if np.any(accepted_c):
        acc_c = np.mean((val_preds == val_targets)[accepted_c])
        if thresholds["acc_90_max_cov"]["val_metric_val"] >= 0.90:
            assert acc_c >= 0.90, f"Operating point (c) accuracy must be >= 0.90, got {acc_c:.4f}"


def test_other_pool_deterministic_split():
    """
    (3) Verify deterministic 50/50 group_id split of other_pool without cluster leakage.
    """
    n_samples = 400
    n_groups = 50
    probs = np.random.uniform(0.1, 0.9, size=(n_samples, 10))
    probs /= probs.sum(axis=1, keepdims=True)

    meta = []
    for i in range(n_samples):
        g = f"group_{i % n_groups}"
        meta.append({"path": f"img_{i}.jpg", "group_id": g, "true_idx": -1, "pred_idx": 0, "correct": 0})

    u_val_probs, u_test_probs, u_val_meta, u_test_meta = split_other_pool_by_group(probs, meta, seed=42)

    val_groups = {r["group_id"] for r in u_val_meta}
    test_groups = {r["group_id"] for r in u_test_meta}

    assert val_groups.isdisjoint(test_groups), "Groups must be mutually exclusive across val and test splits"
    assert len(val_groups) == n_groups // 2
    assert len(test_groups) == n_groups - (n_groups // 2)

    # Determinism
    v_p2, t_p2, v_m2, t_m2 = split_other_pool_by_group(probs, meta, seed=42)
    assert np.array_equal(u_val_probs, v_p2)
    assert np.array_equal(u_test_probs, t_p2)


def test_calibrate_cli_end_to_end(tmp_path):
    """
    (4) Verify end-to-end execution of ml/analysis/calibrate.py via CLI.
    """
    eval_dir = tmp_path / "run_demo" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "results_calibration"

    rng = np.random.default_rng(99)
    classes = 5

    for set_name, n in [("field_val", 120), ("field_test", 150), ("other_pool", 80)]:
        probs = rng.uniform(0.1, 1.0, size=(n, classes))
        probs /= probs.sum(axis=1, keepdims=True)
        np.save(eval_dir / f"probs_{set_name}.npy", probs)

        targets = rng.integers(0, classes, size=n) if set_name != "other_pool" else np.full(n, -1)
        preds = np.argmax(probs, axis=1)

        csv_file = eval_dir / f"predictions_{set_name}.csv"
        fieldnames = ["path", "source", "true_idx", "pred_idx", "correct", "group_id"]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n):
                writer.writerow({
                    "path": f"data/{set_name}/img_{i:04d}.jpg",
                    "source": "plantdoc" if i % 2 == 0 else "plantwild",
                    "true_idx": targets[i],
                    "pred_idx": preds[i],
                    "correct": int(preds[i] == targets[i]) if set_name != "other_pool" else 0,
                    "group_id": f"grp_{set_name}_{i // 3}",
                })

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "ml" / "analysis" / "calibrate.py"),
        "--eval-dir", str(eval_dir),
        "--name", "demo_run",
        "--out-dir", str(out_dir),
        "--seed", "42",
        "--n-boot", "100",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), timeout=30)
    assert res.returncode == 0, f"calibrate.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Verify outputs
    json_path = out_dir / "demo_run.json"
    rel_png = out_dir / "demo_run_reliability.png"
    rc_png = out_dir / "demo_run_risk_coverage.png"

    assert json_path.exists(), "JSON output missing"
    assert rel_png.exists(), "Reliability diagram missing"
    assert rc_png.exists(), "Risk-coverage curve missing"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "temperature" in data
    assert data["temperature"] > 0
    assert "operating_points" in data
    assert "youden_j" in data["operating_points"]
    assert "keep_90_known" in data["operating_points"]
    assert "acc_90_max_cov" in data["operating_points"]
    assert "auroc_known_vs_unknown" in data
    assert "risk_coverage_curve" in data
    assert len(data["risk_coverage_curve"]) == 9

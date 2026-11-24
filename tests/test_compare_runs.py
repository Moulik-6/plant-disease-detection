import csv
import json
import subprocess
import sys
from pathlib import Path
import pytest
import numpy as np

from ml.analysis.compare_runs import compare_pair, resolve_predictions_file


def create_synthetic_predictions_csv(csv_path: Path, num_samples: int = 100):
    """
    Creates two synthetic prediction sets:
    - Model A: 60% accuracy (60 correct, 40 wrong)
    - Model B: 80% accuracy (80 correct, 20 wrong)
    - On 20 images, Model A is wrong and Model B is right.
    - On all other 80 images, Model A and Model B make identical predictions (60 correct, 20 wrong).
    - Known top-1 margin: B - A = +0.20.
    - Discordant: A right/B wrong = 0, B right/A wrong = 20.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path", "source", "true_label", "true_idx", "pred_label",
        "pred_idx", "confidence", "correct", "group_id", "confidence_level", "pv_dup"
    ]

    # 3 classes: 0, 1, 2
    # 50 clusters: 2 samples per cluster
    classes = ["apple_scab", "black_rot", "cedar_rust"]
    rows_a = []
    rows_b = []

    for i in range(num_samples):
        group_id = f"cluster_{i // 2}"
        target_idx = i % 3
        target_label = classes[target_idx]
        img_path = f"data/field/img_{i:04d}.jpg"

        if i < 60:
            # Both correct
            pred_a = target_idx
            pred_b = target_idx
            correct_a = 1
            correct_b = 1
        elif i < 80:
            # Model A wrong, Model B right
            pred_a = (target_idx + 1) % 3
            pred_b = target_idx
            correct_a = 0
            correct_b = 1
        else:
            # Both wrong
            pred_a = (target_idx + 1) % 3
            pred_b = (target_idx + 1) % 3
            correct_a = 0
            correct_b = 0

        rows_a.append({
            "path": img_path,
            "source": "plantdoc",
            "true_label": target_label,
            "true_idx": target_idx,
            "pred_label": classes[pred_a],
            "pred_idx": pred_a,
            "confidence": 0.85,
            "correct": correct_a,
            "group_id": group_id,
            "confidence_level": "high",
            "pv_dup": 0,
        })

        rows_b.append({
            "path": img_path,
            "source": "plantdoc",
            "true_label": target_label,
            "true_idx": target_idx,
            "pred_label": classes[pred_b],
            "pred_idx": pred_b,
            "confidence": 0.92,
            "correct": correct_b,
            "group_id": group_id,
            "confidence_level": "high",
            "pv_dup": 0,
        })

    return rows_a, rows_b


def test_compare_runs_synthetic(tmp_path):
    run_a_dir = tmp_path / "run_a" / "eval"
    run_b_dir = tmp_path / "run_b" / "eval"
    run_a_dir.mkdir(parents=True, exist_ok=True)
    run_b_dir.mkdir(parents=True, exist_ok=True)
    csv_a = run_a_dir / "predictions_field_test.csv"
    csv_b = run_b_dir / "predictions_field_test.csv"

    rows_a, rows_b = create_synthetic_predictions_csv(csv_a, num_samples=100)

    with open(csv_a, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_a[0].keys()))
        writer.writeheader()
        writer.writerows(rows_a)

    with open(csv_b, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
        writer.writeheader()
        writer.writerows(rows_b)

    # 1. Test compare_pair directly
    res = compare_pair(
        run_a_path=str(tmp_path / "run_a"),
        run_b_path=str(tmp_path / "run_b"),
        set_name="field_test",
        n_boot=1000,
        seed=42,
    )

    assert res["num_samples"] == 100
    assert res["num_clusters"] == 50
    assert np.isclose(res["top1"]["model_a"], 0.60)
    assert np.isclose(res["top1"]["model_b"], 0.80)
    assert np.isclose(res["top1"]["diff"], 0.20)

    # Known 95% CI must bracket +0.20
    assert res["top1"]["ci_lower"] <= 0.20 <= res["top1"]["ci_upper"]
    assert res["top1"]["p_positive"] == 1.0

    # Discordant counts
    assert res["discordant"]["a_right_b_wrong"] == 0
    assert res["discordant"]["b_right_a_wrong"] == 20
    assert res["discordant"]["both_right"] == 60
    assert res["discordant"]["both_wrong"] == 20

    # Macro-F1: Model B must be strictly better
    assert res["macro_f1"]["diff"] > 0
    assert res["macro_f1"]["ci_lower"] > 0
    assert res["macro_f1"]["p_positive"] == 1.0

    # 2. Test CLI execution
    out_json = tmp_path / "comparisons.json"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "ml" / "analysis" / "compare_runs.py"),
        "--pair", str(tmp_path / "run_a"), str(tmp_path / "run_b"),
        "--set", "field_test",
        "--n-boot", "1000",
        "--seed", "42",
        "--out", str(out_json),
    ]

    cli_res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), timeout=30)
    assert cli_res.returncode == 0, f"CLI failed: {cli_res.stderr}\n{cli_res.stdout}"
    assert "PAIRED CLUSTER BOOTSTRAP MODEL COMPARISON" in cli_res.stdout
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "comparisons" in data
    assert len(data["comparisons"]) == 1
    saved_res = data["comparisons"][0]
    assert np.isclose(saved_res["top1"]["diff"], 0.20)

"""
Unit and integration tests for ml/analysis/error_analysis.py.
Verifies confusion pairs ranking, error share calculations, per-class/source recall,
weakest class filtering, contact sheet generation, and fusion Grad-CAM skip logic.
"""

import csv
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from PIL import Image

from ml.analysis.error_analysis import (
    compute_confusion_pairs,
    compute_recalls,
    get_weakest_classes,
    load_predictions,
    load_manifest,
    render_contact_sheet,
    run_gradcam_diagnostics,
)
from ml.src.fusion import FusionModel


@pytest.fixture
def synthetic_predictions_dir(tmp_path):
    """
    Creates an evaluation directory with a synthetic predictions_field_test.csv
    with controlled confusion patterns:
      - 5 x (Tomato___Late_blight -> Tomato___Early_blight)
      - 3 x (Potato___Late_blight -> Tomato___Late_blight)
      - 2 x (Apple___Cedar_apple_rust -> Apple___Apple_scab)
      - 15 x correct predictions for Apple___healthy
      - 10 x correct predictions for Tomato___Late_blight
      Total samples: 35. Total errors: 10. Accuracy: 25/35 = 71.43%.
    """
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_file = eval_dir / "predictions_field_test.csv"

    rows = []
    # Error pair 1: 5 errors
    for i in range(5):
        rows.append({
            "path": f"plantdoc/tomato_lb_{i}.jpg",
            "source": "plantdoc",
            "true_label": "Tomato___Late_blight",
            "true_idx": 0,
            "pred_label": "Tomato___Early_blight",
            "pred_idx": 1,
            "confidence": f"{0.75 + i * 0.02:.4f}",
            "correct": 0,
            "group_id": f"grp_err1_{i}",
        })

    # Error pair 2: 3 errors
    for i in range(3):
        rows.append({
            "path": f"plantwild/potato_lb_{i}.jpg",
            "source": "plantwild",
            "true_label": "Potato___Late_blight",
            "true_idx": 2,
            "pred_label": "Tomato___Late_blight",
            "pred_idx": 0,
            "confidence": f"{0.60 + i * 0.05:.4f}",
            "correct": 0,
            "group_id": f"grp_err2_{i}",
        })

    # Error pair 3: 2 errors
    for i in range(2):
        rows.append({
            "path": f"plantdoc/apple_rust_{i}.jpg",
            "source": "plantdoc",
            "true_label": "Apple___Cedar_apple_rust",
            "true_idx": 3,
            "pred_label": "Apple___Apple_scab",
            "pred_idx": 4,
            "confidence": f"{0.82 + i * 0.03:.4f}",
            "correct": 0,
            "group_id": f"grp_err3_{i}",
        })

    # Correct: 15 Apple___healthy (source plantwild)
    for i in range(15):
        rows.append({
            "path": f"plantwild/apple_healthy_{i}.jpg",
            "source": "plantwild",
            "true_label": "Apple___healthy",
            "true_idx": 5,
            "pred_label": "Apple___healthy",
            "pred_idx": 5,
            "confidence": "0.9500",
            "correct": 1,
            "group_id": f"grp_cor1_{i}",
        })

    # Correct: 10 Tomato___Late_blight (source plantdoc)
    for i in range(10):
        rows.append({
            "path": f"plantdoc/tomato_lb_cor_{i}.jpg",
            "source": "plantdoc",
            "true_label": "Tomato___Late_blight",
            "true_idx": 0,
            "pred_label": "Tomato___Late_blight",
            "pred_idx": 0,
            "confidence": "0.9100",
            "correct": 1,
            "group_id": f"grp_cor2_{i}",
        })

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "source", "true_label", "true_idx", "pred_label", "pred_idx", "confidence", "correct", "group_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Also create a dummy manifest.csv
    manifest_file = tmp_path / "manifest.csv"
    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        m_fieldnames = ["path", "source", "unified_label", "split", "cache_path"]
        writer = csv.DictWriter(f, fieldnames=m_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "path": r["path"],
                "source": r["source"],
                "unified_label": r["true_label"],
                "split": "test",
                "cache_path": "",
            })

    return {
        "eval_dir": eval_dir,
        "csv_file": csv_file,
        "manifest_file": manifest_file,
        "rows": rows,
    }


def test_confusion_pairs_calculation(synthetic_predictions_dir):
    """Verifies that confusion pairs are accurately counted and error shares sum to 100%."""
    records = load_predictions(synthetic_predictions_dir["eval_dir"], "field_test")
    top_pairs, total_errors = compute_confusion_pairs(records, top_k=15)

    assert total_errors == 10
    assert len(top_pairs) == 3

    # Pair #1: Tomato___Late_blight -> Tomato___Early_blight (5 errors, 50%)
    assert top_pairs[0]["rank"] == 1
    assert top_pairs[0]["true_label"] == "Tomato___Late_blight"
    assert top_pairs[0]["pred_label"] == "Tomato___Early_blight"
    assert top_pairs[0]["count"] == 5
    assert np.isclose(top_pairs[0]["share"], 0.50)
    assert np.isclose(top_pairs[0]["share_pct"], 50.0)

    # Pair #2: Potato___Late_blight -> Tomato___Late_blight (3 errors, 30%)
    assert top_pairs[1]["rank"] == 2
    assert top_pairs[1]["true_label"] == "Potato___Late_blight"
    assert top_pairs[1]["pred_label"] == "Tomato___Late_blight"
    assert top_pairs[1]["count"] == 3
    assert np.isclose(top_pairs[1]["share"], 0.30)
    assert np.isclose(top_pairs[1]["cum_share_pct"], 80.0)

    # Pair #3: Apple___Cedar_apple_rust -> Apple___Apple_scab (2 errors, 20%)
    assert top_pairs[2]["rank"] == 3
    assert top_pairs[2]["count"] == 2
    assert np.isclose(top_pairs[2]["cum_share_pct"], 100.0)


def test_recalls_and_weakest_classes(synthetic_predictions_dir):
    """Verifies per-class and per-source recall, and filtering of weakest classes (min_samples=15)."""
    records = load_predictions(synthetic_predictions_dir["eval_dir"], "field_test")
    per_class, per_source = compute_recalls(records)

    # Check Tomato___Late_blight: 10 correct out of 15 total (recall = 10/15 = 66.67%)
    tomato_stats = next(c for c in per_class if c["class_name"] == "Tomato___Late_blight")
    assert tomato_stats["total"] == 15
    assert tomato_stats["correct"] == 10
    assert tomato_stats["errors"] == 5
    assert np.isclose(tomato_stats["recall"], 10.0 / 15.0)

    # Check Apple___healthy: 15 correct out of 15 total (recall = 100%)
    apple_stats = next(c for c in per_class if c["class_name"] == "Apple___healthy")
    assert apple_stats["total"] == 15
    assert apple_stats["correct"] == 15
    assert np.isclose(apple_stats["recall"], 1.0)

    # Check per-source:
    # plantdoc: 5 errors + 2 errors + 10 correct = 17 total, 10 correct (10/17 = 58.82%)
    doc_stats = next(s for s in per_source if s["source"] == "plantdoc")
    assert doc_stats["total"] == 17
    assert doc_stats["correct"] == 10
    assert np.isclose(doc_stats["recall"], 10.0 / 17.0)

    # plantwild: 3 errors + 15 correct = 18 total, 15 correct (15/18 = 83.33%)
    wild_stats = next(s for s in per_source if s["source"] == "plantwild")
    assert wild_stats["total"] == 18
    assert wild_stats["correct"] == 15
    assert np.isclose(wild_stats["recall"], 15.0 / 18.0)

    # Weakest classes with at least 15 test images:
    # Only Tomato___Late_blight (n=15, 66.67%) and Apple___healthy (n=15, 100%) qualify.
    # Potato (n=3) and Apple rust (n=2) are excluded due to n < 15.
    weakest = get_weakest_classes(per_class, min_samples=15, top_k=10)
    assert len(weakest) == 2
    assert weakest[0]["class_name"] == "Tomato___Late_blight"
    assert weakest[1]["class_name"] == "Apple___healthy"


def test_contact_sheet_rendering(tmp_path):
    """Verifies that render_contact_sheet generates a valid PNG file."""
    out_img = tmp_path / "test_sheet.png"
    dummy_items = [
        {"true_label": "Class_A", "pred_label": "Class_B", "confidence": 0.88, "resolved_path": None},
        {"true_label": "Class_C", "pred_label": "Class_D", "confidence": 0.94, "resolved_path": None},
    ]

    render_contact_sheet(dummy_items, out_img, n_rows=1, n_cols=2, sheet_title="Test Sheet")
    assert out_img.is_file()
    assert out_img.stat().st_size > 0

    with Image.open(out_img) as img:
        assert img.format == "PNG"


def test_full_pipeline_cli_execution(synthetic_predictions_dir, tmp_path):
    """Executes the error_analysis.py script end-to-end and validates generated artifacts."""
    out_dir = tmp_path / "output_analysis"
    cmd = [
        sys.executable,
        "ml/analysis/error_analysis.py",
        "--eval-dir", str(synthetic_predictions_dir["eval_dir"]),
        "--set", "field_test",
        "--out", str(out_dir),
        "--manifest", str(synthetic_predictions_dir["manifest_file"]),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "ERROR ANALYSIS SUMMARY: FIELD_TEST" in res.stdout
    assert "Tomato___Late_blight" in res.stdout
    assert "Tomato___Early_blight" in res.stdout

    # Verify generated artifacts
    assert (out_dir / "summary.md").is_file()
    assert (out_dir / "per_class_recall.csv").is_file()
    assert (out_dir / "per_source_recall.csv").is_file()
    assert (out_dir / "most_confident_errors.png").is_file()

    # Verify at least one pair contact sheet exists
    pair_sheets = list(out_dir.glob("pair_*.png"))
    assert len(pair_sheets) >= 1

    # Verify summary.md contents
    summary_text = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "# Error Analysis Report: `field_test`" in summary_text
    assert "Top 15 Most Frequent Confusion Pairs" in summary_text
    assert "Tomato___Late_blight" in summary_text


def test_gradcam_fusion_skipping(tmp_path):
    """Verifies that run_gradcam_diagnostics detects fusion model and cleanly skips."""
    run_dir = tmp_path / "run_fusion"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pth"

    import torch
    fusion_model = FusionModel(num_classes=38, mode="concat", pretrained=False)
    torch.save({
        "model_state": fusion_model.state_dict(),
        "state_dict": fusion_model.state_dict(),
        "architecture": "fusion_concat",
        "config": {"model": {"architecture": "fusion_concat", "mode": "concat"}},
    }, str(ckpt_path))

    records = [{"path": "dummy.jpg", "correct": 0, "pred_idx": 0, "true_idx": 1}]
    result = run_gradcam_diagnostics(
        run_dir=run_dir,
        records=records,
        manifest_map={},
        out_dir=tmp_path / "gradcam_out",
    )
    # Should return None and not raise an error
    assert result is None

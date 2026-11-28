"""
Integration and Unit Tests for Plant Disease Detection API.
Verifies all endpoints, error handling, status threshold logic,
Grad-CAM explainability, and fusion model constraints.
"""

import io
import os
from pathlib import Path
from PIL import Image
import json
import numpy as np
import pytest
from starlette.testclient import TestClient
import torch
import torch.nn as nn
import yaml

import app.main as app_module
from app.main import app
from app.predictor import PlantDiseasePredictor


REPO_ROOT = Path(__file__).resolve().parents[2]
B_ENSEMBLE_CALIB = REPO_ROOT / "results" / "stage2" / "calibration" / "b_ensemble.json"
B_EFFNET_CALIB = REPO_ROOT / "results" / "stage2" / "calibration" / "b_effnet.json"


@pytest.fixture
def client(synthetic_checkpoints_dir, tmp_path, monkeypatch):
    """Configures environment and creates a TestClient with ResNet50 checkpoint."""
    test_db = str(tmp_path / "test_app.db")

    monkeypatch.setenv("APP_CHECKPOINTS", synthetic_checkpoints_dir["resnet_path"])
    monkeypatch.setenv("APP_CALIBRATION", synthetic_checkpoints_dir["calib_path"])
    monkeypatch.setenv("APP_DB", test_db)
    monkeypatch.setenv("APP_DEVICE", "cpu")
    monkeypatch.setenv("APP_STORE_IMAGES", "0")

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verifies /health reports ok status, mode, device, loaded models, and calibration thresholds."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] == "ensemble"
    assert data["device"] == "cpu"
    assert data["models_loaded"] >= 1
    assert "resnet50" in data["model_names"]
    assert "calibration" in data
    assert data["temperature"] == pytest.approx(0.9078)
    assert data["threshold"] == pytest.approx(0.6782)
    assert data["confident_threshold"] == pytest.approx(0.6782)
    assert data["uncertain_threshold"] == pytest.approx(0.4290)
    assert data["thresholds"]["confident"] == pytest.approx(0.6782)
    assert data["thresholds"]["uncertain"] == pytest.approx(0.4290)
    assert data["calibration"]["confident_threshold"] == pytest.approx(0.6782)
    assert data["calibration"]["uncertain_threshold"] == pytest.approx(0.4290)
    assert "youden_j" in data["calibration"]["operating_points"]
    assert "keep_90_known" in data["calibration"]["operating_points"]


def test_classes_endpoint(client):
    """Verifies /classes returns all 38 classes with parsed crop, condition, and health flag."""
    response = client.get("/classes")
    assert response.status_code == 200
    classes = response.json()
    assert isinstance(classes, list)
    assert len(classes) == 38

    # Verify schema structure of items
    for item in classes:
        assert "class_name" in item
        assert "crop" in item
        assert "condition" in item
        assert "healthy" in item
        assert isinstance(item["healthy"], bool)

    # Check known class
    tomato_lb = [c for c in classes if c["class_name"] == "Tomato___Late_blight"]
    assert len(tomato_lb) == 1
    assert tomato_lb[0]["crop"] == "Tomato"
    assert tomato_lb[0]["condition"] == "Late blight"
    assert tomato_lb[0]["healthy"] is False


def test_predict_valid_jpeg(client, sample_leaf_bytes):
    """Verifies /predict with valid JPEG returns top-1, top-3 differential, and latency."""
    files = {"image": ("leaf.jpg", sample_leaf_bytes, "image/jpeg")}
    response = client.post("/predict", files=files, data={"gradcam": "false"})
    assert response.status_code == 200
    data = response.json()

    assert "predicted_class" in data
    assert "crop" in data
    assert "condition" in data
    assert "healthy" in data
    assert "confidence" in data
    assert "status" in data
    assert data["status"] in ("confident", "uncertain", "not_recognised")

    # Top-3 Differential Diagnoses
    assert "top3" in data
    assert len(data["top3"]) == 3
    for diag in data["top3"]:
        assert "class_name" in diag
        assert "crop" in diag
        assert "condition" in diag
        assert "healthy" in diag
        assert "confidence" in diag

    assert "latency_ms" in data
    assert data["latency_ms"] > 0
    assert "caution" in data
    assert data["gradcam_image"] is None


def test_predict_with_gradcam(client, sample_leaf_bytes):
    """Verifies /predict with gradcam=true returns base64 data URI heatmap overlay."""
    files = {"image": ("leaf.jpg", sample_leaf_bytes, "image/jpeg")}
    response = client.post("/predict", files=files, data={"gradcam": "true"})
    assert response.status_code == 200
    data = response.json()

    assert data["gradcam_image"] is not None
    assert data["gradcam_image"].startswith("data:image/png;base64,")


def test_gradcam_endpoint(client, sample_leaf_bytes):
    """Verifies /gradcam returns raw image/png bytes with valid PNG magic header."""
    files = {"image": ("leaf.jpg", sample_leaf_bytes, "image/jpeg")}
    response = client.post("/gradcam", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    # Check PNG magic bytes: \x89PNG\r\n\x1a\n
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_reject_corrupt_image(client):
    """Verifies uploading corrupt non-image bytes returns 400 Bad Request."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_RANDOM_GARBAGE"
    files = {"image": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Invalid or corrupt" in response.json().get("detail", "") or "not a valid" in response.json().get("detail", "")


def test_reject_empty_file(client):
    """Verifies uploading empty 0-byte file returns 400 Bad Request."""
    files = {"image": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json().get("detail", "").lower()


def test_reject_unsupported_format(client):
    """Verifies uploading non-supported image format (e.g. BMP) returns 400."""
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    bmp_bytes = buf.getvalue()

    files = {"image": ("leaf.bmp", bmp_bytes, "image/bmp")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Unsupported image format" in response.json().get("detail", "")


def test_reject_oversized_file(client):
    """Verifies uploading a file over 5 MB returns 413 Payload Too Large."""
    # 5 MB + 1024 bytes of dummy payload
    oversized_bytes = b"0" * (5 * 1024 * 1024 + 1024)
    files = {"image": ("huge.jpg", oversized_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413
    assert "exceeds maximum limit" in response.json().get("detail", "").lower()


def test_status_threshold_logic(synthetic_checkpoints_dir, monkeypatch):
    """Unit test verifying operating-point threshold rules & env overrides using PlantDiseasePredictor."""
    monkeypatch.setenv("APP_CHECKPOINTS", synthetic_checkpoints_dir["resnet_path"])
    monkeypatch.setenv("APP_CALIBRATION", str(B_ENSEMBLE_CALIB))
    monkeypatch.setenv("APP_DEVICE", "cpu")

    predictor = PlantDiseasePredictor()
    assert predictor.mode == "ensemble"
    assert predictor.temperature == pytest.approx(0.90779107, rel=1e-5)
    assert predictor.confident_threshold == pytest.approx(0.67819631, rel=1e-5)
    assert predictor.uncertain_threshold == pytest.approx(0.42896834, rel=1e-5)

    assert predictor.classify_confidence(0.90) == "confident"
    assert predictor.classify_confidence(predictor.confident_threshold) == "confident"
    assert predictor.classify_confidence(0.678) == "uncertain"
    assert predictor.classify_confidence(0.55) == "uncertain"
    assert predictor.classify_confidence(predictor.uncertain_threshold) == "uncertain"
    assert predictor.classify_confidence(0.428) == "not_recognised"
    assert predictor.classify_confidence(0.10) == "not_recognised"

    # Configurable overrides via environment variables
    monkeypatch.setenv("APP_TEMPERATURE", "1.5")
    monkeypatch.setenv("APP_CONFIDENT_THRESHOLD", "0.75")
    monkeypatch.setenv("APP_UNCERTAIN_THRESHOLD", "0.35")
    custom_predictor = PlantDiseasePredictor()
    assert custom_predictor.temperature == pytest.approx(1.5)
    assert custom_predictor.confident_threshold == pytest.approx(0.75)
    assert custom_predictor.uncertain_threshold == pytest.approx(0.35)
    assert custom_predictor.classify_confidence(0.75) == "confident"
    assert custom_predictor.classify_confidence(0.50) == "uncertain"
    assert custom_predictor.classify_confidence(0.34) == "not_recognised"


def test_app_mode_light_endpoint(synthetic_checkpoints_dir, tmp_path, monkeypatch, sample_leaf_bytes):
    """Verifies APP_MODE=light selects only EfficientNet checkpoint and b_effnet.json thresholds."""
    light_cfg = tmp_path / "light_config.yaml"
    light_cfg.write_text(
        yaml.safe_dump({
            "mode": "ensemble",
            "checkpoints": [
                synthetic_checkpoints_dir["resnet_path"],
                synthetic_checkpoints_dir["effnet_path"],
            ],
            "light_checkpoints": [synthetic_checkpoints_dir["effnet_path"]],
            "calibration_path": str(B_ENSEMBLE_CALIB),
            "light_calibration_path": str(B_EFFNET_CALIB),
            "db_path": str(tmp_path / "light_app.db"),
            "device": "cpu",
        }),
        encoding="utf-8",
    )

    monkeypatch.delenv("APP_CHECKPOINTS", raising=False)
    monkeypatch.delenv("APP_CALIBRATION", raising=False)
    monkeypatch.setenv("APP_MODE", "light")
    monkeypatch.setenv("APP_DB", str(tmp_path / "light_app.db"))

    # Load predictor directly with config override and also test via FastAPI TestClient
    light_predictor = PlantDiseasePredictor(config_path=light_cfg)
    assert light_predictor.mode == "light"
    assert len(light_predictor.models) == 1
    assert light_predictor.architectures == ["efficientnet_v2_s"]
    assert light_predictor.temperature == pytest.approx(1.06752441, rel=1e-5)
    assert light_predictor.confident_threshold == pytest.approx(0.80423635, rel=1e-5)
    assert light_predictor.uncertain_threshold == pytest.approx(0.50241256, rel=1e-5)

    monkeypatch.setenv("APP_CHECKPOINTS", synthetic_checkpoints_dir["effnet_path"])
    monkeypatch.setenv("APP_DEVICE", "cpu")
    with TestClient(app) as light_client:
        h = light_client.get("/health")
        assert h.status_code == 200
        hdata = h.json()
        assert hdata["mode"] == "light"
        assert hdata["models_loaded"] == 1
        assert hdata["model_names"] == ["efficientnet_v2_s"]
        assert hdata["temperature"] == pytest.approx(1.06752441, rel=1e-5)
        assert hdata["confident_threshold"] == pytest.approx(0.80423635, rel=1e-5)
        assert hdata["uncertain_threshold"] == pytest.approx(0.50241256, rel=1e-5)

        pred = light_client.post("/predict", files={"image": ("leaf.jpg", sample_leaf_bytes, "image/jpeg")})
        assert pred.status_code == 200


def test_default_config_yaml_b_models_and_calibration():
    """Verifies app/config.yaml specifies the 3 B checkpoints by default and light EfficientNet config."""
    cfg_path = REPO_ROOT / "app" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["mode"] == "ensemble"
    assert len(cfg["checkpoints"]) == 3
    assert any("b_mixed_resnet50" in p for p in cfg["checkpoints"])
    assert any("b_mixed_efficientnet_v2_s" in p for p in cfg["checkpoints"])
    assert any("b_mixed_vit_b_16" in p for p in cfg["checkpoints"])
    assert cfg["calibration_path"] == "results/stage2/calibration/b_ensemble.json"

    assert len(cfg["light_checkpoints"]) == 1
    assert "b_mixed_efficientnet_v2_s" in cfg["light_checkpoints"][0]
    assert cfg["light_calibration_path"] == "results/stage2/calibration/b_effnet.json"


def test_ensemble_calibration_math_matches_calibrate_py(synthetic_checkpoints_dir, monkeypatch):
    """Verifies predictor averages member probabilities first, then applies log temperature scaling."""
    monkeypatch.setenv("APP_CHECKPOINTS", synthetic_checkpoints_dir["resnet_path"])
    monkeypatch.setenv("APP_CALIBRATION", str(B_ENSEMBLE_CALIB))
    monkeypatch.setenv("APP_DEVICE", "cpu")

    class FixedLogitModel(nn.Module):
        def __init__(self, logits_vec):
            super().__init__()
            self.register_buffer("logits_vec", torch.tensor(logits_vec, dtype=torch.float32))

        def forward(self, x):
            return self.logits_vec.unsqueeze(0).expand(x.shape[0], -1)

    predictor = PlantDiseasePredictor()
    z1 = np.zeros(38, dtype=np.float32)
    z1[0] = 3.0
    z2 = np.zeros(38, dtype=np.float32)
    z2[1] = 2.5
    predictor.models = [FixedLogitModel(z1), FixedLogitModel(z2)]
    predictor.architectures = ["m1", "m2"]

    img = Image.new("RGB", (64, 64), color=(0, 128, 0))
    res = predictor.predict(img)

    # Exact calibrate.py pipeline: average softmax probs -> log -> divide by T -> softmax
    p1 = np.exp(z1) / np.sum(np.exp(z1))
    p2 = np.exp(z2) / np.sum(np.exp(z2))
    p_avg = 0.5 * (p1 + p2)
    log_p = np.log(np.clip(p_avg, 1e-12, 1.0))
    scaled = log_p / predictor.temperature
    cal_probs = np.exp(scaled - np.max(scaled))
    cal_probs /= np.sum(cal_probs)

    assert res["confidence"] == pytest.approx(float(np.max(cal_probs)), abs=1e-4)


def test_history_flow(client, sample_leaf_bytes):
    """Verifies predictions are recorded to SQLite and cleared via DELETE /history."""
    # 1. Clear initial history
    del_res = client.delete("/history")
    assert del_res.status_code == 200

    # Verify history is empty
    hist_res = client.get("/history")
    assert hist_res.status_code == 200
    assert len(hist_res.json()) == 0

    # 2. Make prediction
    files = {"image": ("leaf.jpg", sample_leaf_bytes, "image/jpeg")}
    pred_res = client.post("/predict", files=files)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    # 3. Check history has 1 entry
    hist_res = client.get("/history")
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) == 1
    item = history[0]
    assert item["predicted_class"] == pred_data["predicted_class"]
    assert item["crop"] == pred_data["crop"]
    assert item["condition"] == pred_data["condition"]
    assert item["confidence"] == pred_data["confidence"]
    assert item["status"] == pred_data["status"]
    assert "resnet50" in item["model_name"]
    # Image storage disabled by default
    assert item["image_path"] is None

    # 4. Clear history
    del_res = client.delete("/history")
    assert del_res.status_code == 200
    assert del_res.json()["count"] == 1

    # Verify history empty again
    hist_res = client.get("/history")
    assert len(hist_res.json()) == 0


def test_fusion_model_returns_501_on_gradcam(synthetic_checkpoints_dir, tmp_path, monkeypatch, sample_leaf_bytes):
    """Verifies that an ensemble containing a FusionModel returns HTTP 501 on /gradcam."""
    test_db = str(tmp_path / "fusion_app.db")
    monkeypatch.setenv("APP_CHECKPOINTS", synthetic_checkpoints_dir["fusion_path"])
    monkeypatch.setenv("APP_DB", test_db)
    monkeypatch.setenv("APP_DEVICE", "cpu")

    with TestClient(app) as fusion_client:
        files = {"image": ("leaf.jpg", sample_leaf_bytes, "image/jpeg")}
        response = fusion_client.post("/gradcam", files=files)
        assert response.status_code == 501
        detail = response.json().get("detail", "")
        assert "dual backbones" in detail.lower() or "fusion" in detail.lower()

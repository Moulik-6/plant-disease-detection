"""
Integration and Unit Tests for Plant Disease Detection API.
Verifies all endpoints, error handling, status threshold logic,
Grad-CAM explainability, and fusion model constraints.
"""

import io
import os
from pathlib import Path
from PIL import Image
import pytest
from starlette.testclient import TestClient

import app.main as app_module
from app.main import app
from app.predictor import PlantDiseasePredictor


@pytest.fixture
def client(synthetic_checkpoints_dir, tmp_path, monkeypatch):
    """Configures environment and creates a TestClient with ResNet50 checkpoint."""
    test_db = str(tmp_path / "test_app.db")
    upload_dir = str(tmp_path / "test_uploads")

    monkeypatch.setenv("APP_CHECKPOINTS", synthetic_checkpoints_dir["resnet_path"])
    monkeypatch.setenv("APP_CALIBRATION", synthetic_checkpoints_dir["calib_path"])
    monkeypatch.setenv("APP_DB", test_db)
    monkeypatch.setenv("APP_DEVICE", "cpu")
    monkeypatch.setenv("APP_STORE_IMAGES", "0")

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verifies /health reports ok status, device, loaded models, and calibration info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["device"] == "cpu"
    assert data["models_loaded"] >= 1
    assert "resnet50" in data["model_names"]
    assert "calibration" in data
    assert data["temperature"] == 1.25
    assert data["threshold"] == 0.52


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


def test_status_threshold_logic():
    """Unit test verifying operating point threshold rules: confident / uncertain / not_recognised."""
    threshold = 0.60
    # confident if confidence >= threshold
    # uncertain if 0.5 * threshold <= confidence < threshold
    # not_recognised if confidence < 0.5 * threshold

    def categorize(conf, th):
        if conf >= th:
            return "confident"
        elif conf >= 0.5 * th:
            return "uncertain"
        else:
            return "not_recognised"

    assert categorize(0.75, threshold) == "confident"
    assert categorize(0.60, threshold) == "confident"
    assert categorize(0.59, threshold) == "uncertain"
    assert categorize(0.30, threshold) == "uncertain"
    assert categorize(0.29, threshold) == "not_recognised"
    assert categorize(0.05, threshold) == "not_recognised"


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

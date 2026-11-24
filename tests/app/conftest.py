"""
Pytest fixtures and synthetic model checkpoints for app test suite.
All tests execute on CPU with lightweight random weights without requiring real training runs.
"""

import json
from pathlib import Path
import pytest
from PIL import Image
import torch

from ml.src.models import build_model
from ml.src.fusion import FusionModel
from ml.src.datasets import load_unified_class_to_idx


@pytest.fixture(scope="session")
def class_to_idx():
    return load_unified_class_to_idx()


@pytest.fixture(scope="session")
def synthetic_checkpoints_dir(tmp_path_factory, class_to_idx):
    """Creates synthetic CPU checkpoints for ResNet50 and Fusion models."""
    d = tmp_path_factory.mktemp("synthetic_models")

    # 1. ResNet50 Checkpoint
    resnet = build_model({"model": {"architecture": "resnet50"}}, num_classes=38, pretrained=False)
    resnet_path = d / "resnet50_best.pth"
    torch.save({
        "model_state": resnet.state_dict(),
        "state_dict": resnet.state_dict(),
        "architecture": "resnet50",
        "class_to_idx": class_to_idx,
        "config": {"model": {"architecture": "resnet50"}},
    }, str(resnet_path))

    # 2. Fusion Checkpoint
    fusion = FusionModel(num_classes=38, mode="concat", pretrained=False)
    fusion_path = d / "fusion_best.pth"
    torch.save({
        "model_state": fusion.state_dict(),
        "state_dict": fusion.state_dict(),
        "architecture": "fusion_concat",
        "class_to_idx": class_to_idx,
        "config": {"model": {"architecture": "fusion_concat", "mode": "concat"}},
    }, str(fusion_path))

    # 3. Calibration JSON
    calib_path = d / "calibration.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump({
            "temperature": 1.25,
            "threshold": 0.52,
            "operating_points": {
                "youden_j": {"threshold": 0.52}
            }
        }, f)

    return {
        "dir": d,
        "resnet_path": str(resnet_path),
        "fusion_path": str(fusion_path),
        "calib_path": str(calib_path),
    }


@pytest.fixture
def sample_leaf_bytes():
    """Generates an in-memory JPEG image bytes of a green leaf for testing."""
    import io
    img = Image.new("RGB", (128, 128), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes():
    """Generates an in-memory PNG image bytes for testing."""
    import io
    img = Image.new("RGB", (64, 64), color=(60, 179, 113))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

import pytest
import torch
import torch.nn as nn
from ml.src.model_zoo import build_any_model, load_checkpoint_model

@pytest.mark.parametrize("arch", ["resnet50", "efficientnet_v2_s", "vit_b_16"])
def test_architecture_forward_pass(arch):
    """Verify that each core backbone can be instantiated (pretrained=False) and execute a forward pass."""
    cfg = {
        "model": {
            "architecture": arch,
            "pretrained": False,
            "dropout": 0.2,
        },
        "dataset": {"img_size": 224}
    }
    model = build_any_model(config=cfg, num_classes=38, pretrained=False)
    model.eval()

    # Synthetic batch of size 2
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 38), f"Expected output shape (2, 38) for {arch}, got {out.shape}"
    assert not torch.isnan(out).any(), f"Forward pass produced NaNs for {arch}"


def test_checkpoint_roundtrip(tmp_path):
    """Verify saving a model state dict and loading it back through load_checkpoint_model."""
    cfg = {
        "model": {"architecture": "resnet50", "pretrained": False, "dropout": 0.1},
        "dataset": {"img_size": 224}
    }
    model = build_any_model(config=cfg, num_classes=38, pretrained=False)

    ckpt_path = tmp_path / "test_model.pth"
    dummy_class_to_idx = {f"class_{i}": i for i in range(38)}
    payload = {
        "state_dict": model.state_dict(),
        "architecture": "resnet50",
        "class_to_idx": dummy_class_to_idx,
        "config": cfg,
    }
    torch.save(payload, ckpt_path)

    loaded_model, loaded_cfg, loaded_arch, loaded_c2i, loaded_i2c = load_checkpoint_model(
        checkpoint_path=ckpt_path,
        device="cpu"
    )

    assert loaded_model is not None
    assert loaded_arch == "resnet50"
    assert len(loaded_c2i) == 38
    assert len(loaded_i2c) == 38

    # Verify parameter match
    p_orig = next(model.parameters())
    p_loaded = next(loaded_model.parameters())
    assert torch.equal(p_orig, p_loaded), "Model weights did not match after checkpoint round-trip"


def test_fusion_model_stage_freezing_and_eval_mode():
    """Verify FusionModel stage freezing, layer graduation, and train/eval mode behaviour."""
    try:
        from ml.src.fusion import FusionModel
    except ImportError:
        pytest.skip("FusionModel fixes not merged yet; skipping fusion tests.")

    model = FusionModel(num_classes=38, mode="concat", pretrained=False)

    # 1. In Stage 1: backbones must be frozen
    model.set_stage(1)
    assert model.stage == 1
    for p in model.efficientnet.parameters():
        assert not p.requires_grad, "EfficientNet params should be frozen in stage 1"
    for p in model.vit.parameters():
        assert not p.requires_grad, "ViT params should be frozen in stage 1"
    for p in model.head.parameters():
        assert p.requires_grad, "Head params should be trainable in stage 1"

    # In stage 1, train() must keep backbones in eval() mode
    model.train(True)
    assert not model.efficientnet.training, "EfficientNet should remain in eval mode during stage 1"
    assert not model.vit.training, "ViT should remain in eval mode during stage 1"
    assert model.head.training, "Head should follow train mode"

    # 2. In Stage 2: backbones partially unfrozen
    model.set_stage(2)
    assert model.stage == 2

    # In stage 2, train() keeps BatchNorm in eval mode
    model.train(True)
    for m in model.efficientnet.modules():
        if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            assert not m.training, "EfficientNet BatchNorm layers must stay in eval mode during stage 2"

    # Forward pass sanity check
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 38)

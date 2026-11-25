"""
Tests for ml/analysis/benchmark_latency.py supporting --checkpoint,
new run layouts (best.pth), fusion models, sequential ensembles, and legacy fallback.
"""

import json
from pathlib import Path
import sys

import pytest
import torch

from ml.analysis import benchmark_latency
from ml.src.fusion import FusionModel
from ml.src.models import build_model


def _save_synthetic_checkpoint(path: Path, arch: str, num_classes: int = 38, key: str = "model_state") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arch in ("fusion_concat", "fusion_gated"):
        mode = "concat" if arch == "fusion_concat" else "gated"
        model = FusionModel(num_classes=num_classes, mode=mode, pretrained=False)
        cfg = {"model": {"architecture": arch, "mode": mode}}
    else:
        model = build_model({"model": {"architecture": arch}}, num_classes=num_classes, pretrained=False)
        cfg = {"model": {"architecture": arch}}

    class_to_idx = {f"Class_{i}": i for i in range(num_classes)}
    payload = {
        key: model.state_dict(),
        "architecture": arch,
        "class_to_idx": class_to_idx,
        "config": cfg,
    }
    torch.save(payload, str(path))


def test_single_checkpoint_latency_benchmark(tmp_path, monkeypatch):
    """Benchmarking one --checkpoint reports only that chosen model without an ensemble entry."""
    ckpt = tmp_path / "runs" / "b_mixed_resnet50" / "20261121_131527" / "best.pth"
    _save_synthetic_checkpoint(ckpt, "resnet50")
    out_json = tmp_path / "out" / "single.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_latency.py",
            "--checkpoint", str(ckpt),
            "--device", "cpu",
            "--threads", "2",
            "--warmup", "1",
            "--rounds", "1",
            "--runs-per-round", "2",
            "--output", str(out_json),
        ],
    )
    benchmark_latency.main()

    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert list(data["models"].keys()) == ["resnet50"]
    assert "ensemble" not in data["models"]
    assert data["models"]["resnet50"]["sample_count"] == 2
    assert data["models"]["resnet50"]["median_ms"] > 0
    assert data["benchmark_config"]["checkpoints"]["resnet50"] == str(ckpt)


def test_multi_checkpoint_ensemble_and_fusion(tmp_path, monkeypatch):
    """Benchmarking multiple --checkpoint paths measures each model plus sequential ensemble."""
    ckpt_res = tmp_path / "resnet.pth"
    ckpt_fus = tmp_path / "fusion.pth"
    _save_synthetic_checkpoint(ckpt_res, "resnet50")
    _save_synthetic_checkpoint(ckpt_fus, "fusion_concat")
    out_json = tmp_path / "multi.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_latency.py",
            "--checkpoint", str(ckpt_res), str(ckpt_fus),
            "--device", "cpu",
            "--warmup", "1",
            "--rounds", "1",
            "--runs-per-round", "2",
            "--output", str(out_json),
        ],
    )
    benchmark_latency.main()

    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(data["models"].keys()) == {"resnet50", "fusion_concat", "ensemble"}
    assert data["models"]["ensemble"]["sample_count"] == 2
    assert data["models"]["fusion_concat"]["median_ms"] > 0


def test_checkpoint_helpers_and_duplicate_arch_naming(tmp_path):
    """Verifies argument flattening, run-directory resolution, and duplicate architecture disambiguation."""
    flat = benchmark_latency._flatten_checkpoint_args([["a.pth,b.pth"], ["c.pth"]])
    assert flat == ["a.pth", "b.pth", "c.pth"]

    run_dir = tmp_path / "runs" / "b_mixed_resnet50" / "20261121_131527"
    run_dir.mkdir(parents=True)
    best_pth = run_dir / "best.pth"
    best_pth.write_bytes(b"stub")
    assert benchmark_latency._resolve_checkpoint_file(str(run_dir)) == str(best_pth)

    used = set()
    n1 = benchmark_latency._derive_model_name(
        tmp_path / "runs" / "a_pv_only_resnet50" / "20261121_100000" / "best.pth",
        "resnet50",
        used,
    )
    used.add(n1)
    n2 = benchmark_latency._derive_model_name(best_pth, "resnet50", used)
    assert n1 == "resnet50"
    assert n2 == "resnet50_b_mixed_resnet50"


def test_legacy_checkpoints_dir_fallback(tmp_path, monkeypatch):
    """When --checkpoint is omitted, legacy checkpoints-dir names are used and ensemble is measured."""
    for name, arch in [("resnet50", "resnet50"), ("efficientnet_v2", "efficientnet_v2_s"), ("vit", "vit_b_16")]:
        ckpt_p = tmp_path / "ckpts" / name / "best_model.pth"
        _save_synthetic_checkpoint(ckpt_p, arch, key="state_dict")

    # Stub load_model_and_classes with tiny identity modules to keep test sub-second on CPU
    class TinyModel(torch.nn.Module):
        def forward(self, x):
            return x[:, :1, 0, 0]

    def fake_load(cfg_path, cpath, device):
        return TinyModel(), {}, "resnet50", {"c": 0}, {0: "c"}

    monkeypatch.setattr(benchmark_latency, "load_model_and_classes", fake_load)
    out_json = tmp_path / "legacy.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_latency.py",
            "--checkpoints-dir", str(tmp_path / "ckpts"),
            "--device", "cpu",
            "--warmup", "1",
            "--rounds", "1",
            "--runs-per-round", "2",
            "--output", str(out_json),
        ],
    )
    benchmark_latency.main()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(data["models"].keys()) == {"resnet50", "efficientnet_v2", "vit", "ensemble"}

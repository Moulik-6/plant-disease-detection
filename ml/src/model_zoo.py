import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ml.src.models import build_model
from ml.src.fusion import FusionModel

logger = logging.getLogger("model_zoo")


def build_any_model(
    config: dict,
    num_classes: int,
    pretrained: Optional[bool] = None,
) -> nn.Module:
    """
    Unified model factory:
      - Returns FusionModel when model.architecture is "fusion_concat" or "fusion_gated"
      - Otherwise delegates to build_model from ml/src/models.py unchanged.
    """
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    architecture = str(model_cfg.get("architecture", config.get("architecture", "resnet50") if isinstance(config, dict) else "resnet50")).lower().strip()

    if pretrained is None:
        pretrained = bool(model_cfg.get("pretrained", True))
    else:
        pretrained = bool(pretrained)

    if architecture in ("fusion_concat", "fusion_gated"):
        mode = "concat" if architecture == "fusion_concat" else "gated"
        dropout = float(model_cfg.get("dropout", config.get("dropout", 0.3) if isinstance(config, dict) else 0.3))

        fusion_cfg = config.get("fusion", {}) if isinstance(config, dict) else {}
        eff_init = fusion_cfg.get("efficientnet_init") or model_cfg.get("efficientnet_init")
        vit_init = fusion_cfg.get("vit_init") or model_cfg.get("vit_init")

        model = FusionModel(
            num_classes=num_classes,
            mode=mode,
            dropout=dropout,
            pretrained=pretrained,
            efficientnet_init=eff_init,
            vit_init=vit_init,
        )
        return model

    return build_model(config=config, num_classes=num_classes, pretrained=pretrained)


def load_checkpoint_model(
    checkpoint_path: Union[str, Path],
    config: Optional[dict] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, dict, str, dict, dict]:
    """
    Loads a checkpoint for fusion models or delegates.
    Returns: (model, config, architecture, class_to_idx, idx_to_class)
    """
    ckpt_p = Path(checkpoint_path).resolve()
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    ckpt = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)

    # 1. Resolve architecture
    arch = (
        ckpt.get("architecture")
        or ckpt.get("config", {}).get("model", {}).get("architecture")
        or (config.get("model", {}).get("architecture") if config else None)
        or "resnet50"
    )
    arch = str(arch).lower().strip()

    # 2. Resolve class_to_idx & idx_to_class
    class_to_idx = ckpt.get("class_to_idx")
    if class_to_idx is None:
        from ml.src.datasets import load_unified_class_to_idx
        class_to_idx = load_unified_class_to_idx()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 3. Resolve config
    raw_cfg = ckpt.get("config", {}) or (config if config else {})
    ckpt_cfg = copy.deepcopy(raw_cfg) if isinstance(raw_cfg, dict) else {}
    if "model" not in ckpt_cfg or not isinstance(ckpt_cfg["model"], dict):
        ckpt_cfg["model"] = {}
    ckpt_cfg["model"]["architecture"] = arch
    if "mode" in ckpt:
        ckpt_cfg["model"]["mode"] = ckpt["mode"]
    elif arch == "fusion_concat":
        ckpt_cfg["model"]["mode"] = "concat"
    elif arch == "fusion_gated":
        ckpt_cfg["model"]["mode"] = "gated"

    # Remove fusion.efficientnet_init, fusion.vit_init, model.efficientnet_init and model.vit_init
    if "fusion" in ckpt_cfg and isinstance(ckpt_cfg["fusion"], dict):
        ckpt_cfg["fusion"].pop("efficientnet_init", None)
        ckpt_cfg["fusion"].pop("vit_init", None)
    if "model" in ckpt_cfg and isinstance(ckpt_cfg["model"], dict):
        ckpt_cfg["model"].pop("efficientnet_init", None)
        ckpt_cfg["model"].pop("vit_init", None)

    num_classes = len(class_to_idx)
    model = build_any_model(ckpt_cfg, num_classes=num_classes, pretrained=False)

    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    return model, ckpt_cfg, arch, class_to_idx, idx_to_class

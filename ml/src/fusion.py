import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("fusion")


class FusionModel(nn.Module):
    """
    Two-branch late fusion architecture combining:
      1. EfficientNetV2-S (1280-d globally pooled features)
      2. ViT-B/16 (768-d class-token features)

    Modes:
      - "concat":
          Concatenates features (2048-d) -> LayerNorm(2048) -> Dropout(p)
          -> Linear(2048, 512) -> GELU -> Dropout(p) -> Linear(512, num_classes)
      - "gated":
          Projects each branch to 512-d (proj_eff, proj_vit). Computes a learned
          per-sample sigmoid gate g in (0, 1) from the concatenated raw features (2048-d).
          Blends: fused = g * a + (1 - g) * b (512-d).
          Head: LayerNorm(512) -> Dropout(p) -> Linear(512, num_classes).
          Exposes mean gate value via property `mean_gate` and `last_gate_mean`.

    Stages:
      - set_stage(1): Freezes both backbones, trains only classification head.
      - set_stage(2): Unfreezes last 2 blocks of EfficientNet (features[-2:])
                      and last 2 encoder layers of ViT plus final LayerNorm.
    """

    def __init__(
        self,
        num_classes: int,
        mode: str = "concat",
        dropout: float = 0.3,
        pretrained: bool = True,
        efficientnet_init: Optional[Union[str, Path]] = None,
        vit_init: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        mode = str(mode).lower().strip()
        if mode not in ("concat", "gated"):
            raise ValueError(f"Unknown fusion mode: '{mode}'. Allowed: 'concat', 'gated'")

        self.num_classes = num_classes
        self.mode = mode
        self.dropout_rate = dropout
        self.stage = 1
        self._last_gate_mean: Optional[Union[torch.Tensor, float]] = None

        # 1. EfficientNetV2-S branch (1280-d globally pooled features)
        weights_eff = "DEFAULT" if pretrained else None
        self.efficientnet = models.efficientnet_v2_s(weights=weights_eff)
        # Drop classifier head; forward now produces 1280-d pooled features
        self.efficientnet.classifier = nn.Identity()

        # 2. ViT-B/16 branch (768-d class-token features)
        weights_vit = "DEFAULT" if pretrained else None
        self.vit = models.vit_b_16(weights=weights_vit)
        # Drop heads; forward now produces 768-d class-token features
        self.vit.heads = nn.Identity()

        # 3. Fusion Head
        if self.mode == "concat":
            # LayerNorm(2048) -> Dropout -> Linear(2048,512) -> GELU -> Dropout -> Linear(512,num_classes)
            self.head = nn.Sequential(
                nn.LayerNorm(2048),
                nn.Dropout(p=dropout),
                nn.Linear(2048, 512),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, num_classes),
            )
        elif self.mode == "gated":
            # Project each branch to 512-d
            self.proj_eff = nn.Linear(1280, 512)
            self.proj_vit = nn.Linear(768, 512)
            # Learned per-sample sigmoid gate g computed from the concatenated features (2048)
            self.gate = nn.Linear(2048, 1)
            # LayerNorm -> Dropout -> Linear(512, num_classes)
            self.classifier = nn.Sequential(
                nn.LayerNorm(512),
                nn.Dropout(p=dropout),
                nn.Linear(512, num_classes),
            )

        # Default stage 1: freeze backbones, train only head
        self.set_stage(1)

        # Optional loading of backbone checkpoints
        if efficientnet_init or vit_init:
            self.load_backbone_weights(
                efficientnet_init=efficientnet_init,
                vit_init=vit_init,
            )

    def train(self, mode: bool = True):
        """
        Overrides train mode:
          - Stage 1: keeps both backbones in eval() regardless of mode
                     (BatchNorm statistics frozen, no stochastic depth).
          - Stage 2: keeps every BatchNorm in the EfficientNet branch in eval()
                     and lets everything else follow mode.
        """
        super().train(mode)
        if self.stage == 1:
            self.efficientnet.eval()
            self.vit.eval()
        elif self.stage == 2:
            if mode:
                for m in self.efficientnet.modules():
                    if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.eval()
        return self

    @property
    def last_gate_mean(self) -> Optional[float]:
        """Exposes the mean gate value computed in the most recent forward pass (lazily evaluated)."""
        if self._last_gate_mean is None:
            return None
        if isinstance(self._last_gate_mean, torch.Tensor):
            return float(self._last_gate_mean.item())
        return float(self._last_gate_mean)

    @last_gate_mean.setter
    def last_gate_mean(self, value: Any) -> None:
        self._last_gate_mean = value

    @property
    def mean_gate(self) -> Optional[float]:
        """Exposes the mean gate value computed in the most recent forward pass."""
        return self.last_gate_mean

    def get_mean_gate(self) -> Optional[float]:
        """Accessor for the mean gate value."""
        return self.last_gate_mean

    def set_stage(self, stage: int) -> None:
        """
        Switches between training stages:
          - Stage 1: Freezes both backbones; trains only the head.
          - Stage 2: Unfreezes the last 2 blocks of EfficientNet (features[-2:])
                     and the last 2 encoder layers of ViT plus its final norm.
        """
        if stage == 1:
            for p in self.efficientnet.parameters():
                p.requires_grad = False
            for p in self.vit.parameters():
                p.requires_grad = False

            if self.mode == "concat":
                for p in self.head.parameters():
                    p.requires_grad = True
            elif self.mode == "gated":
                for p in self.proj_eff.parameters():
                    p.requires_grad = True
                for p in self.proj_vit.parameters():
                    p.requires_grad = True
                for p in self.gate.parameters():
                    p.requires_grad = True
                for p in self.classifier.parameters():
                    p.requires_grad = True
            self.stage = 1
            self.train(self.training)

        elif stage == 2:
            # First freeze all backbone parameters
            for p in self.efficientnet.parameters():
                p.requires_grad = False
            for p in self.vit.parameters():
                p.requires_grad = False

            # Unfreeze the last 2 blocks of EfficientNet (features[-2:])
            for block in list(self.efficientnet.features)[-2:]:
                for p in block.parameters():
                    p.requires_grad = True

            # Unfreeze the last 2 encoder layers of ViT plus its final norm
            for layer in list(self.vit.encoder.layers)[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(self.vit.encoder, "ln"):
                for p in self.vit.encoder.ln.parameters():
                    p.requires_grad = True

            # Keep head parameters trainable
            if self.mode == "concat":
                for p in self.head.parameters():
                    p.requires_grad = True
            elif self.mode == "gated":
                for p in self.proj_eff.parameters():
                    p.requires_grad = True
                for p in self.proj_vit.parameters():
                    p.requires_grad = True
                for p in self.gate.parameters():
                    p.requires_grad = True
                for p in self.classifier.parameters():
                    p.requires_grad = True
            self.stage = 2
            self.train(self.training)
        else:
            raise ValueError(f"Unknown stage: {stage}. Allowed: 1, 2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_eff = self.efficientnet(x)   # (B, 1280)
        feat_vit = self.vit(x)           # (B, 768)

        # Concatenate backbone features
        cat_features = torch.cat([feat_eff, feat_vit], dim=-1)  # (B, 2048)

        if self.mode == "concat":
            return self.head(cat_features)
        elif self.mode == "gated":
            a = self.proj_eff(feat_eff)  # (B, 512)
            b = self.proj_vit(feat_vit)  # (B, 512)
            g = torch.sigmoid(self.gate(cat_features))  # (B, 1)
            self._last_gate_mean = g.detach().mean()
            fused = g * a + (1.0 - g) * b  # (B, 512)
            return self.classifier(fused)
        else:
            raise RuntimeError(f"Unsupported fusion mode: {self.mode}")

    def load_backbone_weights(
        self,
        efficientnet_init: Optional[Union[str, Path]] = None,
        vit_init: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Loads backbone weights from two checkpoints:
          - efficientnet_init: checkpoint path for EfficientNetV2-S
          - vit_init: checkpoint path for ViT-B/16
        Takes the state_dict and keeps only matching backbone keys, dropping old
        classifier heads, and logs how many keys were loaded and missing.
        """
        if efficientnet_init:
            path_eff = Path(efficientnet_init).resolve()
            if not path_eff.exists():
                logger.warning("EfficientNet init checkpoint not found: %s", path_eff)
            else:
                ckpt = torch.load(path_eff, map_location="cpu", weights_only=False)
                sd = ckpt.get("model_state")
                if sd is None:
                    sd = ckpt.get("state_dict", ckpt)

                target_sd = self.efficientnet.state_dict()
                matching_sd = {}

                for k, v in sd.items():
                    # Drop old classifier heads
                    if k.startswith("classifier") or k.startswith("heads") or k.startswith("fc"):
                        continue
                    clean_k = k
                    for prefix in ("efficientnet.", "model.", "module.", "backbone."):
                        if clean_k.startswith(prefix):
                            clean_k = clean_k[len(prefix):]
                    if clean_k in target_sd and target_sd[clean_k].shape == v.shape:
                        matching_sd[clean_k] = v

                missing_keys, _ = self.efficientnet.load_state_dict(matching_sd, strict=False)
                missing_backbone_keys = [k for k in missing_keys if not k.startswith("classifier")]
                logger.info(
                    "EfficientNet backbone init from %s: loaded %d keys, missing %d keys",
                    path_eff,
                    len(matching_sd),
                    len(missing_backbone_keys),
                )

        if vit_init:
            path_vit = Path(vit_init).resolve()
            if not path_vit.exists():
                logger.warning("ViT init checkpoint not found: %s", path_vit)
            else:
                ckpt = torch.load(path_vit, map_location="cpu", weights_only=False)
                sd = ckpt.get("model_state")
                if sd is None:
                    sd = ckpt.get("state_dict", ckpt)

                target_sd = self.vit.state_dict()
                matching_sd = {}

                for k, v in sd.items():
                    # Drop old classifier heads
                    if k.startswith("heads") or k.startswith("classifier") or k.startswith("head."):
                        continue
                    clean_k = k
                    for prefix in ("vit.", "model.", "module.", "backbone."):
                        if clean_k.startswith(prefix):
                            clean_k = clean_k[len(prefix):]
                    if clean_k in target_sd and target_sd[clean_k].shape == v.shape:
                        matching_sd[clean_k] = v

                missing_keys, _ = self.vit.load_state_dict(matching_sd, strict=False)
                missing_backbone_keys = [k for k in missing_keys if not (k.startswith("heads") or k.startswith("head"))]
                logger.info(
                    "ViT backbone init from %s: loaded %d keys, missing %d keys",
                    path_vit,
                    len(matching_sd),
                    len(missing_backbone_keys),
                )

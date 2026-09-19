from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


def build_model(
    config: dict,
    num_classes: int,
    pretrained: Optional[bool] = None
) -> nn.Module:
    """
    Model factory to instantiate classification backbones:
    - resnet50 (torchvision)
    - efficientnet_v2_s / efficientnet_v2_m / efficientnet_v2_l (torchvision)
    - vit_b_16 (torchvision)
    - Any model supported by timm if timm is installed.

    Allows replacing the final classification head and freezing backbones.
    """
    model_cfg = config.get("model", {})
    architecture = model_cfg.get("architecture", "resnet50").lower()
    
    if pretrained is None:
        pretrained = model_cfg.get("pretrained", True)
        
    dropout = model_cfg.get("dropout", 0.2)
    freeze_backbone = model_cfg.get("freeze_backbone", False)

    # 1. ResNet family (e.g. resnet50, resnet18, resnet34, resnet101)
    if architecture.startswith("resnet"):
        weights = "DEFAULT" if pretrained else None
        if hasattr(models, architecture):
            model = getattr(models, architecture)(weights=weights)
        else:
            raise ValueError(f"Unknown torchvision ResNet variant: {architecture}")
        
        in_features = model.fc.in_features
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

        return model

    # 2. EfficientNetV2 family (e.g. efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l)
    elif architecture.startswith("efficientnet"):
        weights = "DEFAULT" if pretrained else None
        if hasattr(models, architecture):
            model = getattr(models, architecture)(weights=weights)
        else:
            raise ValueError(f"Unknown torchvision EfficientNet variant: {architecture}")
        
        # In torchvision efficientnet_v2, classifier is Sequential(Dropout, Linear)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

        return model

    # 3. Vision Transformer family (e.g. vit_b_16, vit_b_32, vit_l_16)
    elif architecture.startswith("vit"):
        weights = "DEFAULT" if pretrained else None
        if hasattr(models, architecture):
            model = getattr(models, architecture)(weights=weights)
            in_features = model.heads.head.in_features
            if dropout > 0:
                model.heads = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features, num_classes)
                )
            else:
                model.heads.head = nn.Linear(in_features, num_classes)

            if freeze_backbone:
                for name, param in model.named_parameters():
                    if not name.startswith("heads"):
                        param.requires_grad = False

            return model
        # If not directly on torchvision, fallback to timm check below

    # 4. Fallback: timm (PyTorch Image Models)
    try:
        import timm
        model = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        if freeze_backbone:
            # freeze all except classifier head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.get_classifier().parameters():
                param.requires_grad = True
        return model
    except ImportError:
        pass
    except Exception as e:
        raise ValueError(
            f"Failed to create model '{architecture}' via timm: {e}"
        )

    raise ValueError(
        f"Unsupported architecture '{architecture}'. "
        f"Supported standard architectures: resnet50, efficientnet_v2_s, vit_b_16, or install timm."
    )

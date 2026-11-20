#!/usr/bin/env python3
"""
Modern Experiment Training Pipeline for Plant Disease Detection.

Features:
  - CLI: python ml/train_exp.py --exp ml/configs/experiment/<name>.yaml [--set a.b=c ...] [--resume runs/<dir>] [--smoke]
  - Config inheritance via a `base:` key using `deep_update` from ml/src/utils.py (resolved config saved)
  - Hardware Acceleration:
      * bf16 autocast on CUDA with no GradScaler
      * channels_last memory format
      * TF32 enabled (matmul & cuDNN)
  - Optimizer & Scheduler:
      * AdamW
      * Cosine schedule with linear warmup
      * Label smoothing from config
      * Gradient clipping 1.0
  - DataLoader:
      * persistent_workers and prefetch_factor 4
      * default num_workers 6 (0 on Windows)
  - Transforms:
      * config preprocess: squash | crop
      * augmentation presets:
          - baseline: identical to get_transforms in ml/src/dataset.py
          - field: RandomResizedCrop(224, scale=(0.4,1)), flips, stronger ColorJitter,
                   GaussianBlur, TrivialAugmentWide, RandomErasing
          - ImageNet normalization
  - Evaluation & Checkpointing:
      * Every epoch logs train loss and (accuracy, macro-F1) for all configured validation splits
      * Selects best epoch by config selection.metric (default: field_val_macro_f1) with early stopping
      * Output artifacts in runs/<exp_name>/<timestamp>/:
          - config.yaml (resolved)
          - git_commit.txt (try/except)
          - log.csv
          - best.pth (state_dict, architecture and class_to_idx only)
          - last.pth (full state including optimizer, scheduler, epoch, best metric, RNG states for --resume)
          - metrics.json
      * --smoke: runs 20 batches per epoch for 2 epochs
      * Models instantiated from build_model in ml/src/models.py
"""

import argparse
import csv
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torchvision import transforms
import yaml

# Ensure project root and ml/ directory are in sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ml_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_repo_root, _ml_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from ml.src.datasets import ManifestDataset, load_unified_class_to_idx
    from ml.src.models import build_model
    from ml.src.sampling import build_weighted_sampler
    from ml.src.utils import deep_update, set_seed
except ImportError:
    from src.datasets import ManifestDataset, load_unified_class_to_idx
    from src.models import build_model
    from src.sampling import build_weighted_sampler
    from src.utils import deep_update, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_exp")


def parse_override_value(val_str: str) -> Any:
    """Parses scalar string override into native Python bool, None, int, float, or string."""
    v_lower = val_str.lower().strip()
    if v_lower == "true":
        return True
    if v_lower == "false":
        return False
    if v_lower in ("null", "none"):
        return None
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def apply_dot_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Applies dot-separated overrides (e.g. ['model.architecture=vit_b_16']) to config."""
    for item in overrides:
        if "=" not in item:
            logger.warning("Ignoring invalid override format (expected key=value): %s", item)
            continue
        key_path, raw_val = item.split("=", 1)
        parsed_val = parse_override_value(raw_val)

        keys = key_path.strip().split(".")
        current = config
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = parsed_val
        logger.info("Override applied: %s = %r", key_path.strip(), parsed_val)

    return config


def load_inherited_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads YAML config and recursively resolves inheritance via the `base:` key using `deep_update`.
    """
    p = Path(config_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_ref = cfg.get("base") or cfg.get("_base_")
    if base_ref:
        base_path = (p.parent / base_ref).resolve()
        if not base_path.exists():
            # Check relative to ml/configs/experiment or project root
            cand = Path("ml/configs/experiment") / base_ref
            if cand.exists():
                base_path = cand.resolve()
            else:
                raise FileNotFoundError(f"Base config not found: {base_ref} (resolved: {base_path})")

        base_cfg = load_inherited_config(base_path)
        cfg = deep_update(base_cfg, cfg)

    return cfg


def resolve_channels_last(cfg_val: Any, architecture: str) -> bool:
    """
    Resolves channels_last setting from config (true, false, or "auto").
    "auto" resolves to:
      - False when architecture starts with "efficientnet" (bf16 + channels_last fails in cuDNN on this torch build)
      - True otherwise.
    """
    if isinstance(cfg_val, str):
        v = cfg_val.lower().strip()
        if v == "auto":
            return not architecture.startswith("efficientnet")
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
    elif isinstance(cfg_val, bool):
        return cfg_val
    return not architecture.startswith("efficientnet")


def build_transforms(config: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Builds training and evaluation transforms based on:
      - config preprocess: squash | crop
      - config augmentation preset: baseline | field
    """
    img_size = int(config.get("img_size") or config.get("dataset", {}).get("img_size", 224))
    preprocess = str(config.get("preprocess", "squash")).lower().strip()
    aug_preset = str(config.get("augmentation", "baseline")).lower().strip()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Preprocessing resize / crop operation
    if preprocess == "crop":
        base_resize = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size)
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # squash (default)
        base_resize = transforms.Resize((img_size, img_size))
        eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # Augmentation presets
    if aug_preset == "field":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.4, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value="random"
            )
        ])
    else:
        # Baseline preset: identical to get_transforms in ml/src/dataset.py
        aug_cfg = config.get("augmentation_cfg", {})
        rot = aug_cfg.get("rotation_degrees", 30)
        hflip = aug_cfg.get("hflip_prob", 0.5)
        vflip = aug_cfg.get("vflip_prob", 0.2)
        cj = aug_cfg.get("color_jitter", {})
        b = cj.get("brightness", 0.2)
        c = cj.get("contrast", 0.2)
        s = cj.get("saturation", 0.2)
        h = cj.get("hue", 0.05)

        train_transform = transforms.Compose([
            base_resize,
            transforms.RandomRotation(degrees=rot),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.RandomVerticalFlip(p=vflip),
            transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return train_transform, eval_transform


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates a linear warmup followed by cosine annealing learning rate scheduler."""
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min_lr / base_lr
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_git_commit() -> str:
    """Attempts to retrieve current git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return out
    except Exception:
        return "unknown"


def evaluate_split(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_cuda: bool,
    channels_last: bool = False,
    smoke: bool = False
) -> Tuple[float, float, float, int]:
    """
    Evaluates model on a DataLoader.
    Computes macro-F1 only over classes that appear in that split's true labels.
    Returns: (avg_loss, accuracy, macro_f1, classes_covered_count)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(dataloader):
            if smoke and batch_idx >= 20:
                break

            if images.dim() == 4 and use_cuda and channels_last:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            else:
                images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda" if use_cuda else "cpu",
                dtype=torch.bfloat16,
                enabled=use_cuda
            ):
                outputs = model(images)
                loss = criterion(outputs, targets)

            batch_sz = targets.size(0)
            total_loss += loss.item() * batch_sz
            total_samples += batch_sz

            preds = outputs.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0

    # Compute macro-F1 ONLY over classes that appear in that split's true labels
    if all_targets:
        labels_in_split = sorted(list(set(all_targets)))
        classes_covered = len(labels_in_split)
        macro_f1 = f1_score(all_targets, all_preds, labels=labels_in_split, average="macro", zero_division=0)
    else:
        classes_covered = 0
        macro_f1 = 0.0

    return avg_loss, float(acc), float(macro_f1), classes_covered


def run_self_check() -> int:
    """
    Self-check test: builds model for each architecture (resnet50, efficientnet_v2_s, vit_b_16),
    saves a tiny best.pth to a temp folder with the required config schema, and loads it back
    with load_model_and_classes from ml/gradcam.py, printing PASS or FAIL for each.
    Also executes a one-batch training step for each architecture on CUDA when available
    (bf16 autocast with the resolved channels_last setting, forward, backward, optimizer step).
    """
    import tempfile

    try:
        from ml.gradcam import load_model_and_classes
    except ImportError:
        from gradcam import load_model_and_classes

    architectures = ["resnet50", "efficientnet_v2_s", "vit_b_16"]
    class_to_idx = load_unified_class_to_idx()
    num_classes = len(class_to_idx)
    cuda_available = torch.cuda.is_available()

    all_passed = True
    print("=" * 60)
    print("RUNNING SELF-CHECK: Model Building & Grad-CAM Checkpoint Compatibility")
    print("=" * 60)

    for arch in architectures:
        try:
            model_cfg = {
                "architecture": arch,
                "pretrained": False,
                "dropout": 0.2,
                "freeze_backbone": False
            }
            test_config = {
                "model": model_cfg,
                "dataset": {"img_size": 224}
            }

            # 1. Build model
            model = build_model(test_config, num_classes=num_classes, pretrained=False)

            with tempfile.TemporaryDirectory() as tmp_dir:
                best_pth_path = Path(tmp_dir) / f"best_{arch}.pth"
                best_payload = {
                    "state_dict": model.state_dict(),
                    "architecture": arch,
                    "class_to_idx": class_to_idx,
                    "config": {
                        "model": {
                            "architecture": arch,
                            "pretrained": False,
                            "dropout": 0.2,
                            "freeze_backbone": False
                        },
                        "dataset": {"img_size": 224}
                    }
                }
                torch.save(best_payload, best_pth_path)

                # 2. Load back with load_model_and_classes from ml/gradcam.py
                loaded_model, loaded_cfg, loaded_arch, loaded_c2i, loaded_i2c = load_model_and_classes(
                    config_path="ml/configs/config.yaml",
                    checkpoint_path=str(best_pth_path),
                    device=torch.device("cpu")
                )

                assert loaded_model is not None, "Model failed to instantiate"
                assert loaded_arch.lower() == arch.lower(), f"Arch mismatch: expected {arch}, got {loaded_arch}"
                assert len(loaded_c2i) == num_classes, f"Class count mismatch: expected {num_classes}, got {len(loaded_c2i)}"

            # 3. One-batch training step on CUDA when available
            if cuda_available:
                use_cl = resolve_channels_last("auto", arch)
                device = torch.device("cuda")
                if use_cl:
                    model = model.to(memory_format=torch.channels_last)
                model = model.to(device)
                model.train()

                dummy_images = torch.randn(2, 3, 224, 224, device=device)
                if use_cl:
                    dummy_images = dummy_images.to(memory_format=torch.channels_last)
                dummy_targets = torch.tensor([0, 1 % num_classes], device=device, dtype=torch.long)

                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                crit = nn.CrossEntropyLoss()
                opt.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(dummy_images)
                    loss = crit(out, dummy_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                del dummy_images, dummy_targets, opt, crit
                model = model.to("cpu")
                torch.cuda.empty_cache()

            print(f"[{arch}] Self-check: PASS")
        except Exception as e:
            all_passed = False
            print(f"[{arch}] Self-check: FAIL ({e})")

    print("=" * 60)
    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Unified Experiment Training Pipeline for Plant Disease Detection."
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Path to experiment YAML config (e.g. ml/configs/experiment/a_pv_only.yaml)"
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Config overrides using dot notation (e.g. --set model.architecture=vit_b_16 training.learning_rate=3e-5)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file (last.pth) or run directory to resume training from"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test flag: runs 20 batches per epoch for 2 epochs"
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run self-check test building models, saving best.pth, and loading via ml/gradcam.py"
    )

    args = parser.parse_args()

    if args.self_check:
        sys.exit(run_self_check())

    # 1. Determine configuration source
    resume_path: Optional[Path] = None
    if args.resume:
        res = Path(args.resume).resolve()
        if res.is_dir():
            res_file = res / "last.pth"
        else:
            res_file = res
        if not res_file.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {res_file}")
        resume_path = res_file

    raw_child_cfg: Dict[str, Any] = {}
    if args.exp:
        with open(Path(args.exp), "r", encoding="utf-8") as f:
            raw_child_cfg = yaml.safe_load(f) or {}
        config = load_inherited_config(args.exp)
    elif resume_path:
        # Load config from resume checkpoint or sibling config.yaml
        parent_cfg = resume_path.parent / "config.yaml"
        if parent_cfg.exists():
            config = load_inherited_config(parent_cfg)
        else:
            ckpt_data = torch.load(resume_path, map_location="cpu", weights_only=False)
            config = ckpt_data.get("full_config") or ckpt_data.get("config", {})
    else:
        raise ValueError("Must provide either --exp <config.yaml>, --resume <run_dir/last.pth>, or --self-check")

    # 2. Apply dot overrides
    if args.set:
        config = apply_dot_overrides(config, args.set)

    # 3. Model Architecture & Learning Rate Rules
    model_cfg = config.get("model", {})
    architecture = str(model_cfg.get("architecture", "resnet50")).lower().strip()
    training_cfg = config.get("training", {})

    raw_channels_last = training_cfg.get("channels_last", "auto")
    use_channels_last = resolve_channels_last(raw_channels_last, architecture)
    logger.info("Resolved channels_last: %s (config: %s, architecture: %s)",
                use_channels_last, raw_channels_last, architecture)

    # Detect whether learning_rate is explicit: only if in child config or in --set
    has_explicit_lr_in_child = (
        "training" in raw_child_cfg and "learning_rate" in raw_child_cfg.get("training", {})
    )
    has_explicit_lr_in_cli = any(item.startswith("training.learning_rate=") for item in args.set)
    is_explicit_lr = has_explicit_lr_in_child or has_explicit_lr_in_cli

    is_base_default_lr = bool(training_cfg.get("_is_base_learning_rate", False))

    # For vit_b_16, an unmodified base value of 0.0001 is replaced by 3e-5
    if architecture.startswith("vit"):
        if not is_explicit_lr and (is_base_default_lr or math.isclose(float(training_cfg.get("learning_rate", 1e-4)), 1e-4)):
            training_cfg["learning_rate"] = 3e-5
            logger.info("ViT architecture detected with unmodified base learning rate: replaced with 3e-5")
        else:
            logger.info("Explicit learning_rate (%s) specified in child config/CLI for %s; overrides ViT default 3e-5",
                        training_cfg.get("learning_rate"), architecture)

    # Remove internal base-config marker from resolved config
    training_cfg.pop("_is_base_learning_rate", None)

    base_lr = float(training_cfg.get("learning_rate", 1e-4))
    print(f"Resolved Learning Rate: {base_lr}")
    logger.info("Resolved Learning Rate: %s", base_lr)

    # 4. Global Seed
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # 5. Device Setup & Precision flags
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    if use_cuda:
        # TF32 enabled for performance on Ampere+ architectures
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA detected: TF32 enabled, bf16 autocast ready.")
    else:
        logger.info("Running on CPU.")

    # 6. Unified Classes & Datasets
    unified_classes_path = config.get("unified_classes", "ml/configs/labels/unified_classes.yaml")
    class_to_idx = load_unified_class_to_idx(unified_classes_path)
    num_classes = len(class_to_idx)
    logger.info("Loaded unified class mapping with %d classes.", num_classes)

    train_transform, eval_transform = build_transforms(config)

    dataset_cfg = config.get("dataset", {})
    train_splits = dataset_cfg.get("train_splits") or config.get("train_splits", ["pv_train"])
    val_splits = dataset_cfg.get("val_splits") or config.get("val_splits", ["field_val", "pv_val"])
    split_column = dataset_cfg.get("split_column", "split")
    source_filter = dataset_cfg.get("source_filter", None)

    # DataLoader worker configuration: default 6 on Linux, 0 on Windows
    is_windows = (sys.platform == "win32") or (os.name == "nt")
    configured_workers = config.get("dataloader", {}).get("num_workers")
    if configured_workers is None:
        num_workers = 0 if is_windows else 6
    else:
        num_workers = int(configured_workers)

    persistent_workers = (num_workers > 0)
    prefetch_factor = 4 if persistent_workers else None
    batch_size = int(training_cfg.get("batch_size", 32))
    pin_memory = use_cuda

    logger.info("DataLoader config: num_workers=%d, persistent_workers=%s, prefetch_factor=%s, batch_size=%d",
                num_workers, persistent_workers, prefetch_factor, batch_size)

    # Instantiate Datasets
    manifest_csv = config.get("manifest_path", "data/manifest.csv")
    source_roots = config.get("sources", {})
    use_cache = config.get("use_cache", True)

    train_dataset = ManifestDataset(
        manifest_path=manifest_csv,
        splits=train_splits,
        transform=train_transform,
        class_to_idx=class_to_idx,
        source_roots=source_roots,
        use_cache=use_cache,
        filter_unmapped=True,
        split_column=split_column,
        source_filter=source_filter,
    )

    # WeightedRandomSampler for field-fraction and class-balancing
    train_sampler = build_weighted_sampler(train_dataset, config=config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    val_loaders: Dict[str, torch.utils.data.DataLoader] = {}
    for v_split in val_splits:
        v_ds = ManifestDataset(
            manifest_path=manifest_csv,
            splits=[v_split],
            transform=eval_transform,
            class_to_idx=class_to_idx,
            source_roots=source_roots,
            use_cache=use_cache,
            filter_unmapped=True,
            split_column=split_column,
            source_filter=source_filter,
        )
        val_loaders[v_split] = torch.utils.data.DataLoader(
            v_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )

    # 7. Model Construction
    model = build_model(config, num_classes=num_classes)

    # Initial checkpoint loading (e.g. for fine-tuning)
    # Resolves architecture folder: resnet50 -> resnet50, efficientnet_v2_s -> efficientnet_v2, vit_b_16 -> vit
    init_ckpt = config.get("init_checkpoint") or model_cfg.get("init_checkpoint")
    if init_ckpt:
        ARCH_TO_CKPT_FOLDER = {
            "resnet50": "resnet50",
            "efficientnet_v2_s": "efficientnet_v2",
            "efficientnet_v2_m": "efficientnet_v2",
            "efficientnet_v2_l": "efficientnet_v2",
            "efficientnet_v2": "efficientnet_v2",
            "vit_b_16": "vit",
            "vit_b_32": "vit",
            "vit_l_16": "vit",
            "vit": "vit",
        }
        ckpt_folder = ARCH_TO_CKPT_FOLDER.get(architecture, architecture)
        init_ckpt_str = str(init_ckpt).replace("<name>", ckpt_folder).replace("{name}", ckpt_folder).replace("<arch>", ckpt_folder)
        init_path = Path(init_ckpt_str).resolve()
        if not init_path.exists():
            # Fallback to direct architecture string if folder differed
            cand_alt = Path(str(init_ckpt).replace("<name>", architecture).replace("{name}", architecture)).resolve()
            if cand_alt.exists():
                init_path = cand_alt

        if init_path.exists():
            logger.info("Initializing backbone weights from checkpoint: %s", init_path)
            ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
            model.load_state_dict(sd, strict=False)
        else:
            logger.warning("Requested init_checkpoint not found on disk: %s", init_path)

    # channels_last memory format
    if use_cuda and use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    # 8. Optimization, Loss & Learning Rate Schedule
    base_lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    total_epochs = 2 if args.smoke else int(training_cfg.get("epochs", 25))
    warmup_epochs = 1 if args.smoke else int(training_cfg.get("warmup_epochs", 2))
    min_lr = float(training_cfg.get("min_lr", 1e-6))
    grad_clip = float(training_cfg.get("gradient_clip", 1.0))

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = build_cosine_warmup_scheduler(
        optimizer=optimizer,
        epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        base_lr=base_lr,
        min_lr=min_lr
    )

    # 9. Output Directory Setup
    if resume_path:
        run_dir = resume_path.parent
        exp_name = config.get("exp_name") or run_dir.parent.name
        logger.info("Resuming run in existing directory: %s", run_dir.resolve())
    else:
        exp_name = config.get("exp_name") or Path(args.exp or "resumed_exp").stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / exp_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run directory initialized: %s", run_dir.resolve())

    # Save resolved config.yaml
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Save git_commit.txt
    git_hash = get_git_commit()
    with open(run_dir / "git_commit.txt", "w", encoding="utf-8") as f:
        f.write(git_hash + "\n")

    # CSV Logger
    log_csv_path = run_dir / "log.csv"
    write_header = not log_csv_path.exists()
    csv_headers = ["epoch", "lr", "train_loss", "train_acc", "distinct_field_images"]
    for v_split in val_splits:
        csv_headers.extend([f"{v_split}_loss", f"{v_split}_acc", f"{v_split}_macro_f1"])
    csv_headers.append("epoch_time_sec")

    if write_header:
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    # 10. Resume State Restoration
    start_epoch = 0
    resumed_from_epoch: Optional[int] = None
    best_metric = -float("inf")
    best_epoch = 0
    best_snapshot: Dict[str, Any] = {}

    if resume_path:
        logger.info("Resuming full training state from: %s", resume_path)
        ckpt_full = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_full["model_state_dict"])
        optimizer.load_state_dict(ckpt_full["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_full["scheduler_state_dict"])
        resumed_from_epoch = int(ckpt_full.get("epoch", 0))
        start_epoch = resumed_from_epoch + 1
        best_metric = ckpt_full.get("best_metric", -float("inf"))
        best_epoch = ckpt_full.get("best_epoch", 0)

        # Restore previous best_snapshot from metrics.json if available
        if (run_dir / "metrics.json").exists():
            try:
                with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
                    prev_metrics = json.load(f)
                    if "best_epoch_metrics" in prev_metrics:
                        best_snapshot = prev_metrics["best_epoch_metrics"]
            except Exception:
                pass

        # RNG restore
        if "torch_rng" in ckpt_full and ckpt_full["torch_rng"] is not None:
            try:
                torch_rng = ckpt_full["torch_rng"]
                if isinstance(torch_rng, torch.Tensor):
                    torch.set_rng_state(torch_rng.cpu().to(torch.uint8))
                else:
                    torch.set_rng_state(torch_rng)
            except Exception as e:
                logger.warning("Failed to restore torch RNG state: %s", e)

        if use_cuda and "cuda_rng" in ckpt_full and ckpt_full["cuda_rng"] is not None:
            try:
                cuda_rng = ckpt_full["cuda_rng"]
                if isinstance(cuda_rng, (list, tuple)) and len(cuda_rng) > 0:
                    torch.cuda.set_rng_state(cuda_rng[0].cpu().to(torch.uint8))
                elif isinstance(cuda_rng, torch.Tensor):
                    torch.cuda.set_rng_state(cuda_rng.cpu().to(torch.uint8))
                else:
                    torch.cuda.set_rng_state(cuda_rng)
            except Exception as e:
                logger.warning("Failed to restore CUDA RNG state: %s", e)

        if "numpy_rng" in ckpt_full and ckpt_full["numpy_rng"] is not None:
            np.random.set_state(ckpt_full["numpy_rng"])

        if "python_rng" in ckpt_full and ckpt_full["python_rng"] is not None:
            random.setstate(ckpt_full["python_rng"])

    # 11. Early Stopping and Selection Configuration
    selection_metric = config.get("selection", {}).get("metric", "field_val_macro_f1")
    early_stop_cfg = config.get("early_stopping", {})
    early_stop_enabled = early_stop_cfg.get("enabled", True)
    patience = int(early_stop_cfg.get("patience", 7))
    patience_counter = 0

    logger.info("Selection metric: '%s' | Early stopping enabled: %s (patience: %d)",
                selection_metric, early_stop_enabled, patience)

    run_start_time = time.time()
    last_trained_epoch = start_epoch

    # 12. Main Training Loop
    for epoch in range(start_epoch, total_epochs):
        last_trained_epoch = epoch + 1
        epoch_t0 = time.time()
        model.train()

        train_loss_sum = 0.0
        train_samples = 0
        train_correct = 0
        distinct_field_indices: Set[int] = set()

        current_lr = scheduler.get_last_lr()[0]

        for batch_idx, (images, targets, row_idxs) in enumerate(train_loader):
            if args.smoke and batch_idx >= 20:
                break

            # Track distinct field images drawn this epoch
            for r_idx in row_idxs.cpu().tolist():
                if r_idx < len(train_dataset.is_field) and train_dataset.is_field[r_idx]:
                    distinct_field_indices.add(r_idx)

            if images.dim() == 4 and use_cuda and use_channels_last:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            else:
                images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # bf16 autocast on CUDA with no GradScaler
            with torch.autocast(
                device_type="cuda" if use_cuda else "cpu",
                dtype=torch.bfloat16,
                enabled=use_cuda
            ):
                outputs = model(images)
                loss = criterion(outputs, targets)

            loss.backward()

            # Gradient clipping 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            batch_sz = targets.size(0)
            train_loss_sum += loss.item() * batch_sz
            train_samples += batch_sz
            preds = outputs.argmax(dim=-1)
            train_correct += (preds == targets).sum().item()

        scheduler.step()

        train_loss = train_loss_sum / train_samples if train_samples > 0 else 0.0
        train_acc = train_correct / train_samples if train_samples > 0 else 0.0
        num_distinct_field = len(distinct_field_indices)

        logger.info("Epoch [%d/%d] Distinct field images drawn: %d",
                    epoch + 1, total_epochs, num_distinct_field)

        # Evaluate validation splits
        epoch_metrics: Dict[str, float] = {}
        for v_split, v_loader in val_loaders.items():
            v_loss, v_acc, v_f1, v_covered = evaluate_split(
                model=model,
                dataloader=v_loader,
                criterion=criterion,
                device=device,
                use_cuda=use_cuda,
                channels_last=use_channels_last,
                smoke=args.smoke
            )
            epoch_metrics[f"{v_split}_loss"] = v_loss
            epoch_metrics[f"{v_split}_acc"] = v_acc
            epoch_metrics[f"{v_split}_macro_f1"] = v_f1
            logger.info(
                "Epoch [%d/%d] Split '%s': Loss=%.4f, Acc=%.4f, Macro-F1=%.4f (covered %d classes)",
                epoch + 1, total_epochs, v_split, v_loss, v_acc, v_f1, v_covered
            )

        epoch_elapsed = round(time.time() - epoch_t0, 2)

        # Logging to console
        val_summary_str = " | ".join(
            f"{k}={v:.4f}" for k, v in epoch_metrics.items() if "acc" in k or "macro_f1" in k
        )
        logger.info(
            "Epoch [%d/%d] LR: %.2e | Train Loss: %.4f Acc: %.4f | Field Drawn: %d | %s | Time: %.1fs",
            epoch + 1, total_epochs, current_lr, train_loss, train_acc, num_distinct_field, val_summary_str, epoch_elapsed
        )

        # Append to log.csv
        csv_row = [epoch + 1, f"{current_lr:.6e}", f"{train_loss:.4f}", f"{train_acc:.4f}", num_distinct_field]
        for v_split in val_splits:
            csv_row.extend([
                f"{epoch_metrics.get(f'{v_split}_loss', 0.0):.4f}",
                f"{epoch_metrics.get(f'{v_split}_acc', 0.0):.4f}",
                f"{epoch_metrics.get(f'{v_split}_macro_f1', 0.0):.4f}"
            ])
        csv_row.append(epoch_elapsed)

        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)

        # Checkpoint config schema compatible with load_model_and_classes
        img_size_val = int(config.get("img_size") or config.get("dataset", {}).get("img_size", 224))
        ckpt_config = {
            "model": {
                "architecture": architecture,
                "pretrained": False,
                "dropout": float(model_cfg.get("dropout", 0.2)),
                "freeze_backbone": bool(model_cfg.get("freeze_backbone", False)),
            },
            "dataset": {
                "img_size": img_size_val,
            },
        }

        # Save last.pth (full state for --resume)
        last_ckpt = {
            "model_state_dict": model.state_dict(),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "architecture": architecture,
            "class_to_idx": class_to_idx,
            "config": ckpt_config,
            "full_config": config,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if use_cuda else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        torch.save(last_ckpt, run_dir / "last.pth")

        # Metric Selection & Best Model Checkpoint
        current_selection_metric = epoch_metrics.get(selection_metric)
        if current_selection_metric is None and selection_metric == "pv_val_macro_f1":
            # For legacy split configurations where split name is 'val' instead of 'pv_val'
            current_selection_metric = epoch_metrics.get("val_macro_f1")

        if current_selection_metric is None:
            # Fallback to first available macro-f1 or accuracy if exact name not present
            candidates = [v for k, v in epoch_metrics.items() if "macro_f1" in k] or list(epoch_metrics.values())
            current_selection_metric = candidates[0] if candidates else train_acc

        if current_selection_metric > best_metric:
            best_metric = current_selection_metric
            best_epoch = epoch + 1
            best_snapshot = dict(epoch_metrics)
            best_snapshot["train_loss"] = train_loss
            best_snapshot["train_acc"] = train_acc
            patience_counter = 0

            # Save best.pth: state_dict, architecture, class_to_idx, and config
            best_payload = {
                "state_dict": model.state_dict(),
                "architecture": architecture,
                "class_to_idx": class_to_idx,
                "config": ckpt_config,
            }
            torch.save(best_payload, run_dir / "best.pth")
            logger.info(">>> Best model saved at epoch %d with %s = %.4f",
                        best_epoch, selection_metric, best_metric)
        else:
            patience_counter += 1
            if early_stop_enabled and patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    total_training_time = round(time.time() - run_start_time, 2)

    # Save metrics.json
    final_metrics = {
        "experiment_name": exp_name,
        "architecture": architecture,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_metric_value": best_metric,
        "best_epoch_metrics": best_snapshot,
        "total_epochs_trained": last_trained_epoch,
        "total_training_time_sec": total_training_time,
        "smoke": args.smoke,
        "git_commit": git_hash,
    }
    if resumed_from_epoch is not None:
        final_metrics["resumed_from_epoch"] = resumed_from_epoch

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training finished in %.1fs. Artifacts saved in %s", total_training_time, run_dir)


if __name__ == "__main__":
    main()

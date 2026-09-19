import os
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    Supports inheriting from base config via `_base_: <filename>`.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    if "_base_" in config:
        base_rel_path = config.pop("_base_")
        base_path = path.parent / base_rel_path
        base_config = load_config(str(base_path))
        config = deep_update(base_config, config)

    return config


class MetricTracker:
    """Computes and tracks running loss and accuracy."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0

    def update(self, loss: float, preds: torch.Tensor, targets: torch.Tensor):
        batch_size = targets.size(0)
        self.loss_sum += loss * batch_size
        self.correct += (preds == targets).sum().item()
        self.total += batch_size

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / self.total if self.total > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    is_best: bool = False,
    filename: str = "last_model.pth"
):
    """Save model checkpoint and optionally mark as best."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_filepath)


def save_classes(class_to_idx: Dict[str, int], checkpoint_dir: str):
    """Save class_to_idx and idx_to_class mappings as classes.json."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    data = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "classes": sorted(list(class_to_idx.keys()), key=lambda c: class_to_idx[c])
    }
    with open(os.path.join(checkpoint_dir, "classes.json"), "w") as f:
        json.dump(data, f, indent=2)

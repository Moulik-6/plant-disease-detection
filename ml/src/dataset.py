import os
import random
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def get_transforms(config: dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build training and validation/test transforms according to config.
    Augmentation includes:
    - Random rotation
    - Horizontal and vertical flips
    - Brightness and contrast adjustments (ColorJitter)
    - Resizing and ImageNet normalization
    """
    img_size = config.get("dataset", {}).get("img_size", 224)
    aug_cfg = config.get("augmentation", {})
    
    rotation_degrees = aug_cfg.get("rotation_degrees", 30)
    hflip_prob = aug_cfg.get("hflip_prob", 0.5)
    vflip_prob = aug_cfg.get("vflip_prob", 0.2)
    
    color_jitter_cfg = aug_cfg.get("color_jitter", {})
    brightness = color_jitter_cfg.get("brightness", 0.2)
    contrast = color_jitter_cfg.get("contrast", 0.2)
    saturation = color_jitter_cfg.get("saturation", 0.2)
    hue = color_jitter_cfg.get("hue", 0.05)
    
    mean = aug_cfg.get("mean", [0.485, 0.456, 0.406])
    std = aug_cfg.get("std", [0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.RandomVerticalFlip(p=vflip_prob),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


class TransformedSubset(Dataset):
    """
    Subset wrapper that applies a specific transformation to the underlying dataset.
    This allows splitting an ImageFolder dataset into subsets with distinct transforms
    (e.g., training with augmentations, val/test without augmentations).
    """
    def __init__(self, dataset: datasets.ImageFolder, indices: List[int], transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx: int):
        image, label = self.dataset.samples[self.indices[idx]]
        # Load image via ImageFolder's loader (PIL Image)
        img = self.dataset.loader(image)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.indices)


def stratified_split_indices(
    dataset: datasets.ImageFolder,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Performs stratified train/val/test split across all classes.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    rng = random.Random(seed)
    class_indices: Dict[int, List[int]] = {}
    
    for idx, (_, class_idx) in enumerate(dataset.samples):
        if class_idx not in class_indices:
            class_indices[class_idx] = []
        class_indices[class_idx].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for _, indices in sorted(class_indices.items()):
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def get_dataloaders(
    config: dict,
    data_dir_override: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Creates train, val, and test DataLoaders along with the class_to_idx mapping.
    
    Args:
        config: Dictionary containing dataset and augmentation settings.
        data_dir_override: Optional directory path override for dataset.
    
    Returns:
        (train_loader, val_loader, test_loader, class_to_idx)
    """
    ds_cfg = config.get("dataset", {})
    raw_data_dir = data_dir_override or ds_cfg.get("data_dir", "data/plantvillage_dataset/color")
    
    # Try resolving relative to current working directory, then relative to ml/, then project root
    if os.path.exists(raw_data_dir):
        data_dir = raw_data_dir
    elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), raw_data_dir)):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), raw_data_dir)
    elif raw_data_dir.startswith("../") and os.path.exists(raw_data_dir[3:]):
        data_dir = raw_data_dir[3:]
    elif os.path.exists(os.path.join("data", raw_data_dir)):
        data_dir = os.path.join("data", raw_data_dir)
    else:
        raise FileNotFoundError(
            f"Dataset directory '{raw_data_dir}' does not exist. "
            f"Please ensure PlantVillage dataset is placed at '{raw_data_dir}' or specify 'data_dir' in config."
        )

    # Base dataset without transforms to inspect samples and classes
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_to_idx = base_dataset.class_to_idx

    train_ratio = ds_cfg.get("train_split", 0.70)
    val_ratio = ds_cfg.get("val_split", 0.15)
    test_ratio = ds_cfg.get("test_split", 0.15)
    seed = ds_cfg.get("seed", 42)
    batch_size = ds_cfg.get("batch_size", 32)
    num_workers = ds_cfg.get("num_workers", 4)

    train_idx, val_idx, test_idx = stratified_split_indices(
        base_dataset, train_ratio, val_ratio, test_ratio, seed=seed
    )

    train_transform, eval_transform = get_transforms(config)

    train_dataset = TransformedSubset(base_dataset, train_idx, transform=train_transform)
    val_dataset = TransformedSubset(base_dataset, val_idx, transform=eval_transform)
    test_dataset = TransformedSubset(base_dataset, test_idx, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_to_idx

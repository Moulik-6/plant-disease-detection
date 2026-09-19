import argparse
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Ensure ml/ directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders
from src.models import build_model
from src.utils import (
    load_config,
    set_seed,
    MetricTracker,
    save_checkpoint,
    save_classes
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease classification model")
    parser.add_argument("--config", type=str, default="ml/configs/config.yaml", help="Path to config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory override")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--model", type=str, default=None, help="Model architecture override (resnet50, efficientnet_v2_s, vit_b_16)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Output checkpoint directory")
    parser.add_argument("--max-batches", type=int, default=None, help="Maximum batches per epoch (for quick dry runs/testing)")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp: bool = False,
    max_batches: int = None
):
    model.train()
    tracker = MetricTracker()
    
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    pbar = tqdm(loader, desc="Train", total=total_batches, leave=False)
    for batch_idx, (images, targets) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and device.type == "cuda":
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        preds = torch.argmax(outputs, dim=1)
        tracker.update(loss.item(), preds, targets)
        pbar.set_postfix({"loss": f"{tracker.avg_loss:.4f}", "acc": f"{tracker.accuracy:.4f}"})
        
    return tracker.avg_loss, tracker.accuracy


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device, max_batches: int = None):
    model.eval()
    tracker = MetricTracker()
    
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    pbar = tqdm(loader, desc="Val", total=total_batches, leave=False)
    for batch_idx, (images, targets) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        preds = torch.argmax(outputs, dim=1)
        tracker.update(loss.item(), preds, targets)
        pbar.set_postfix({"loss": f"{tracker.avg_loss:.4f}", "acc": f"{tracker.accuracy:.4f}"})
        
    return tracker.avg_loss, tracker.accuracy


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.model is not None:
        config["model"]["architecture"] = args.model
    if args.checkpoint_dir is not None:
        config["output"]["checkpoint_dir"] = args.checkpoint_dir
        
    # Seeding
    seed = config.get("dataset", {}).get("seed", 42)
    set_seed(seed)
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    # Data loaders
    print("[*] Loading dataset splits and transforms...")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        config, data_dir_override=args.data_dir
    )
    num_classes = len(class_to_idx)
    print(f"[*] Found {num_classes} classes. Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Checkpoint dir
    checkpoint_dir = config.get("output", {}).get("checkpoint_dir", "ml/checkpoints")
    save_classes(class_to_idx, checkpoint_dir)
    
    # Build Model
    arch = config.get("model", {}).get("architecture", "resnet50")
    print(f"[*] Initializing model: {arch} with {num_classes} output classes...")
    model = build_model(config, num_classes=num_classes)
    model = model.to(device)
    
    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=train_cfg.get("momentum", 0.9), weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
        
    epochs = train_cfg.get("epochs", 25)
    
    # LR Scheduler
    sched_name = train_cfg.get("lr_scheduler", "cosine").lower()
    if sched_name == "cosine":
        min_lr = float(train_cfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get("step_size", 7),
            gamma=train_cfg.get("gamma", 0.1)
        )
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    else:
        scheduler = None

    # Mixed precision scaler
    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    
    # Early stopping & best model tracking
    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", True)
    es_patience = es_cfg.get("patience", 7)
    es_monitor = es_cfg.get("monitor", "val_loss")
    es_mode = es_cfg.get("mode", "min")
    
    best_metric = float("inf") if es_mode == "min" else -float("inf")
    patience_counter = 0
    
    print(f"[*] Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        
        max_batches = args.max_batches or config.get("training", {}).get("max_batches", None)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, max_batches=max_batches
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, max_batches=max_batches)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )
        
        # Checkpoint state
        state = {
            "epoch": epoch,
            "architecture": arch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": config,
            "class_to_idx": class_to_idx
        }
        
        # Assess best metric
        current_metric = val_loss if es_monitor == "val_loss" else val_acc
        is_best = (current_metric < best_metric) if es_mode == "min" else (current_metric > best_metric)
        
        if is_best:
            best_metric = current_metric
            patience_counter = 0
            save_checkpoint(state, checkpoint_dir, is_best=True)
            print(f"  --> Saved new best checkpoint ({es_monitor}: {current_metric:.4f})")
        else:
            patience_counter += 1
            if not config.get("output", {}).get("save_best_only", False):
                save_checkpoint(state, checkpoint_dir, is_best=False)
                
        if es_enabled and patience_counter >= es_patience:
            print(f"[*] Early stopping triggered after {epoch} epochs without improvement.")
            break
            
    total_time = time.time() - start_time
    print(f"[*] Training finished in {total_time / 60:.2f} minutes.")
    print(f"[*] Best {es_monitor}: {best_metric:.4f}. Model checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()

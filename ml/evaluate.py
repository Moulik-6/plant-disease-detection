import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders
from src.models import build_model
from src.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate plant disease classification model")
    parser.add_argument("--config", type=str, default="ml/configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="ml/checkpoints/best_model.pth", help="Checkpoint file path")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory override")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file '{args.checkpoint}' not found.")
        
    print(f"[*] Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    config = checkpoint.get("config", load_config(args.config))
    class_to_idx = checkpoint.get("class_to_idx")
    
    # Load test split
    _, _, test_loader, discovered_classes = get_dataloaders(config, data_dir_override=args.data_dir)
    
    if class_to_idx is None:
        class_to_idx = discovered_classes
        
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Evaluating on device: {device}")
    
    # Build model & load state dict
    model = build_model(config, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print(f"[*] Running inference on {len(test_loader.dataset)} test samples...")
    for images, targets in tqdm(test_loader, desc="Test"):
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        
    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\n================ Test Results ================")
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print(report)
    
    # Save report to evaluation directory
    eval_dir = os.path.dirname(args.checkpoint)
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    with open(os.path.join(eval_dir, "test_evaluation_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[*] Evaluation report saved to {os.path.join(eval_dir, 'test_evaluation_report.json')}")


if __name__ == "__main__":
    main()

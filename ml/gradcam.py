import argparse
import os
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

# Add ml/ to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import get_dataloaders, get_transforms
from src.models import build_model
from src.utils import load_config


def build_parser():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for Plant Disease Detection")
    parser.add_argument("--config", type=str, default="ml/configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="ml/checkpoints/resnet50/best_model.pth", help="Checkpoint file path")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--output", type=str, default=None, help="Output path for single image overlay or output directory for batch")
    parser.add_argument("--opacity", type=float, default=0.6, help="Heatmap overlay opacity between 0.0 and 1.0 (default: 0.6)")
    parser.add_argument("--batch", type=int, default=None, help="Number of random test images to process in batch mode")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory override")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    parser.add_argument("--target-class", type=int, default=None, help="Optional target class index for Grad-CAM (defaults to predicted class)")
    return parser


def parse_args():
    return build_parser().parse_args()


def get_target_layers_and_transform(model: torch.nn.Module, architecture: str):
    """
    Returns the target layers and optional reshape transform for Grad-CAM
    based on the model architecture.
    """
    arch = architecture.lower()

    if arch.startswith("resnet"):
        # ResNet final convolutional layer is layer4[-1]
        target_layers = [model.layer4[-1]]
        reshape_transform = None

    elif arch.startswith("efficientnet"):
        # EfficientNet final feature extraction block is features[-1]
        target_layers = [model.features[-1]]
        reshape_transform = None

    elif arch.startswith("vit"):
        # Vision Transformer (torchvision vit_b_16)
        # Target layer is layer norm in the final encoder block
        target_layers = [model.encoder.layers[-1].ln_1]

        def vit_reshape_transform(tensor, height=14, width=14):
            # Shape: (batch_size, num_patches + 1, hidden_dim)
            # Remove class token (index 0) and reshape patch tokens into 2D grid
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
            # Permute dimensions to match standard (batch, channels, height, width)
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        reshape_transform = vit_reshape_transform

    else:
        # Fallback for models or timm architectures
        if hasattr(model, "conv_head"):
            target_layers = [model.conv_head]
        elif hasattr(model, "features"):
            target_layers = [model.features[-1]]
        elif hasattr(model, "layer4"):
            target_layers = [model.layer4[-1]]
        else:
            raise ValueError(f"Cannot automatically determine target layer for architecture '{architecture}'.")
        reshape_transform = None

    return target_layers, reshape_transform


def load_model_and_classes(config_path: str, checkpoint_path: str, device: torch.device):
    """
    Loads checkpoint, config, model, and class mappings.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", load_config(config_path))
    class_to_idx = checkpoint.get("class_to_idx")

    if class_to_idx is None:
        # Look for classes.json in checkpoint directory
        ckpt_dir = os.path.dirname(checkpoint_path)
        classes_file = os.path.join(ckpt_dir, "classes.json")
        if os.path.exists(classes_file):
            import json
            with open(classes_file, "r") as f:
                data = json.load(f)
                class_to_idx = data.get("class_to_idx")

    arch = checkpoint.get("architecture") or config.get("model", {}).get("architecture", "resnet50")
    num_classes = len(class_to_idx) if class_to_idx else config.get("model", {}).get("num_classes", 38)

    model = build_model(config, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else {i: str(i) for i in range(num_classes)}
    return model, config, arch, class_to_idx, idx_to_class


def preprocess_image(image_path: str, img_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Loads image from disk, creates preprocessed normalized tensor, and returns
    RGB float32 image normalized in [0, 1] for Grad-CAM overlay.
    """
    pil_img = Image.open(image_path).convert("RGB")
    pil_resized = pil_img.resize((img_size, img_size))

    # Float RGB image in [0, 1] for visualization
    rgb_img = np.float32(pil_resized) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (rgb_img - mean) / std

    # Convert to (1, 3, H, W) tensor
    tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, rgb_img


def run_gradcam_on_image(
    model: torch.nn.Module,
    cam: GradCAM,
    image_path: str,
    output_path: str,
    idx_to_class: dict,
    device: torch.device,
    opacity: float = 0.6,
    img_size: int = 224,
    target_class: Optional[int] = None
) -> Tuple[str, float, int]:
    """
    Runs inference, generates Grad-CAM heatmap, overlays it on the image,
    saves the visualization, and returns (pred_class_name, confidence, pred_idx).
    """
    input_tensor, rgb_img = preprocess_image(image_path, img_size=img_size)
    input_tensor = input_tensor.to(device)

    # 1. Forward pass for prediction & confidence
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    pred_name = idx_to_class.get(pred_idx, str(pred_idx))

    # 2. Target for Grad-CAM
    cam_target_class = target_class if target_class is not None else pred_idx
    targets = [ClassifierOutputTarget(cam_target_class)]

    # 3. Generate Grad-CAM grayscale heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # 4. Overlay heatmap onto RGB image
    # Note: image_weight in show_cam_on_image is 1 - opacity
    image_weight = max(0.0, min(1.0, 1.0 - opacity))
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=image_weight)

    # 5. Save visualization
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pil_out = Image.fromarray(visualization)
    pil_out.save(output_path)

    return pred_name, confidence, pred_idx


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Load model and class dictionary
    print(f"[*] Loading checkpoint: {args.checkpoint}")
    model, config, architecture, class_to_idx, idx_to_class = load_model_and_classes(
        args.config, args.checkpoint, device
    )
    print(f"[*] Model loaded: {architecture} ({len(idx_to_class)} classes)")

    # Target layers and transform
    target_layers, reshape_transform = get_target_layers_and_transform(model, architecture)
    print(f"[*] Grad-CAM target layers: {target_layers}")

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    img_size = config.get("dataset", {}).get("img_size", 224)

    # Mode 1: Single image inference
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Input image '{args.image}' not found.")

        out_path = args.output or "ml/visualizations/gradcam_output.jpg"
        pred_name, conf, pred_idx = run_gradcam_on_image(
            model=model,
            cam=cam,
            image_path=args.image,
            output_path=out_path,
            idx_to_class=idx_to_class,
            device=device,
            opacity=args.opacity,
            img_size=img_size,
            target_class=args.target_class
        )

        file_size_kb = os.path.getsize(out_path) / 1024.0
        print("\n================ Grad-CAM Result ================")
        print(f"Image:         {args.image}")
        print(f"Predicted:     {pred_name} (Class ID: {pred_idx})")
        print(f"Confidence:    {conf * 100:.2f}%")
        print(f"Saved Overlay: {out_path} ({file_size_kb:.1f} KB)")
        print("=================================================\n")

    # Mode 2: Batch inference on test split
    elif args.batch is not None and args.batch > 0:
        print(f"[*] Loading test set for batch Grad-CAM generation (N={args.batch})...")
        _, _, test_loader, discovered_classes = get_dataloaders(config, data_dir_override=args.data_dir)
        test_dataset = test_loader.dataset  # TransformedSubset

        # Group indices by class to ensure diversity (one per class if possible)
        class_to_indices = {}
        for sub_idx in range(len(test_dataset)):
            dataset_idx = test_dataset.indices[sub_idx]
            _, class_idx = test_dataset.dataset.samples[dataset_idx]
            if class_idx not in class_to_indices:
                class_to_indices[class_idx] = []
            class_to_indices[class_idx].append(sub_idx)

        selected_sub_indices = []
        classes_pool = list(class_to_indices.keys())
        random.seed(42)
        random.shuffle(classes_pool)

        # Pick one per class first, then fill remainder if needed
        for c in classes_pool:
            if len(selected_sub_indices) >= args.batch:
                break
            selected_sub_indices.append(random.choice(class_to_indices[c]))

        if len(selected_sub_indices) < args.batch:
            remaining_needed = args.batch - len(selected_sub_indices)
            all_indices = set(range(len(test_dataset))) - set(selected_sub_indices)
            selected_sub_indices.extend(random.sample(list(all_indices), min(remaining_needed, len(all_indices))))

        out_dir = args.output or f"ml/visualizations/gradcam_batch_{architecture}"
        os.makedirs(out_dir, exist_ok=True)
        print(f"[*] Generating visualizations for {len(selected_sub_indices)} images in {out_dir}...")

        summary_results = []
        for i, sub_idx in enumerate(selected_sub_indices):
            dataset_idx = test_dataset.indices[sub_idx]
            img_path, true_label_idx = test_dataset.dataset.samples[dataset_idx]
            true_label_name = idx_to_class.get(true_label_idx, str(true_label_idx))

            base_name = Path(img_path).stem
            out_filename = f"{i+1:02d}_{true_label_name}_{base_name}_cam.jpg"
            out_path = os.path.join(out_dir, out_filename)

            pred_name, conf, pred_idx = run_gradcam_on_image(
                model=model,
                cam=cam,
                image_path=img_path,
                output_path=out_path,
                idx_to_class=idx_to_class,
                device=device,
                opacity=args.opacity,
                img_size=img_size
            )

            file_size_kb = os.path.getsize(out_path) / 1024.0
            status = "MATCH" if pred_idx == true_label_idx else "MISMATCH"
            summary_results.append({
                "file": out_filename,
                "ground_truth": true_label_name,
                "predicted": pred_name,
                "confidence": conf,
                "status": status,
                "size_kb": file_size_kb
            })
            print(f"[{i+1}/{len(selected_sub_indices)}] {status} | GT: {true_label_name} | Pred: {pred_name} ({conf*100:.2f}%) -> {out_filename} ({file_size_kb:.1f} KB)")

        print(f"\n[*] All {len(selected_sub_indices)} Grad-CAM visualizations saved to: {out_dir}")

    else:
        print("[!] Error: Either specify --image <path> for single image or --batch <N> for batch mode.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

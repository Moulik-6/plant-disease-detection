import os
import sys
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Ensure repository root is in sys.path
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from ml.analysis.gradcam import (
    load_model_and_classes,
    get_target_layers_and_transform,
    preprocess_image,
    run_gradcam_on_image
)
from pytorch_grad_cam import GradCAM

# ---------------------------------------------------------------------------
# Setup definitions
# ---------------------------------------------------------------------------
MODELS = [
    {
        "id": "resnet50",
        "name": "ResNet50",
        "config": "ml/legacy/config.yaml",
        "checkpoint": "ml/checkpoints/resnet50/best_model.pth",
        "out_dir": "ml/visualizations/resnet50",
    },
    {
        "id": "efficientnet_v2",
        "name": "EfficientNetV2-S",
        "config": "ml/legacy/efficientnet_v2.yaml",
        "checkpoint": "ml/checkpoints/efficientnet_v2/best_model.pth",
        "out_dir": "ml/visualizations/efficientnet_v2",
    },
    {
        "id": "vit_b_16",
        "name": "ViT-B/16",
        "config": "ml/legacy/vit.yaml",
        "checkpoint": "ml/checkpoints/vit/best_model.pth",
        "out_dir": "ml/visualizations/vit_b_16",
    },
]

IMAGES = [
    {
        "short_name": "Orange HLB",
        "filename": "orange_hlb.jpg",
        "comparison_filename": "orange_hlb_comparison.jpg",
        "ground_truth": "Orange___Haunglongbing_(Citrus_greening)",
        "path": "data/plantvillage_dataset/color/Orange___Haunglongbing_(Citrus_greening)/68ea5572-d729-481b-93dd-6e0a9898bc91___UF.Citrus_HLB_Lab 9704.JPG",
    },
    {
        "short_name": "Strawberry Scorch",
        "filename": "strawberry_leaf_scorch.jpg",
        "comparison_filename": "strawberry_leaf_scorch_comparison.jpg",
        "ground_truth": "Strawberry___Leaf_scorch",
        "path": "data/plantvillage_dataset/color/Strawberry___Leaf_scorch/804af05b-899f-44c2-8606-1b87bbe78c69___RS_L.Scorch 9961.JPG",
    },
    {
        "short_name": "Corn Common Rust",
        "filename": "corn_common_rust.jpg",
        "comparison_filename": "corn_common_rust_comparison.jpg",
        "ground_truth": "Corn_(maize)___Common_rust_",
        "path": "data/plantvillage_dataset/color/Corn_(maize)___Common_rust_/RS_Rust 1754.JPG",
    },
]

COMPARISON_DIR = "ml/visualizations/comparison"


def create_panel_image(orig_path, resnet_path, eff_path, vit_path, output_path, title=""):
    """
    Creates a 4-panel comparison image:
    | Original | ResNet50 | EfficientNetV2-S | ViT-B/16 |
    with label text above each panel.
    """
    panels = [
        ("Original", Image.open(orig_path).convert("RGB").resize((224, 224))),
        ("ResNet50", Image.open(resnet_path).convert("RGB").resize((224, 224))),
        ("EfficientNetV2-S", Image.open(eff_path).convert("RGB").resize((224, 224))),
        ("ViT-B/16", Image.open(vit_path).convert("RGB").resize((224, 224))),
    ]

    panel_w, panel_h = 224, 224
    label_h = 36
    padding = 10
    total_w = 4 * panel_w + 5 * padding
    total_h = panel_h + label_h + 2 * padding

    comp_img = Image.new("RGB", (total_w, total_h), color=(250, 250, 250))
    draw = ImageDraw.Draw(comp_img)

    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    for i, (label, img) in enumerate(panels):
        x = padding + i * (panel_w + padding)
        y = padding

        # Draw label centered
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x + (panel_w - text_w) // 2
        draw.text((text_x, y + 8), label, fill=(30, 30, 30), font=font)

        # Paste panel image
        comp_img.paste(img, (x, y + label_h))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comp_img.save(output_path, quality=95)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing Grad-CAM experiment on device: {device}\n")

    # Validate image paths
    for img_info in IMAGES:
        if not os.path.exists(img_info["path"]):
            raise FileNotFoundError(f"Input image not found: {img_info['path']}")

    # Create directories
    for m in MODELS:
        os.makedirs(m["out_dir"], exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    # Store results: results_by_image[img_filename][model_id] = {pred, conf, class_id, correct, out_path}
    results_by_image = {img["filename"]: {} for img in IMAGES}

    # Iterate models
    for m in MODELS:
        print(f"============================================================")
        print(f"Running Model: {m['name']} ({m['id']})")
        print(f"Checkpoint:   {m['checkpoint']}")
        print(f"============================================================")

        model, config, arch, class_to_idx, idx_to_class = load_model_and_classes(
            m["config"], m["checkpoint"], device
        )

        assert len(idx_to_class) == 38, f"Expected 38 classes, got {len(idx_to_class)}"

        target_layers, reshape_transform = get_target_layers_and_transform(model, arch)

        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        )

        img_size = config.get("dataset", {}).get("img_size", 224)

        for img_info in IMAGES:
            out_file = os.path.join(m["out_dir"], img_info["filename"])
            pred_name, conf, pred_idx = run_gradcam_on_image(
                model=model,
                cam=cam,
                image_path=img_info["path"],
                output_path=out_file,
                idx_to_class=idx_to_class,
                device=device,
                opacity=0.6,
                img_size=img_size
            )

            is_correct = (pred_name == img_info["ground_truth"])
            result_str = "MATCH" if is_correct else "MISMATCH"

            # Print prediction info formatted exactly as requested
            print(f"Model: {m['name']}")
            print(f"Input: {img_info['filename']}")
            print(f"Ground Truth: {img_info['ground_truth']}")
            print(f"Predicted: {pred_name}")
            print(f"Class ID: {pred_idx}")
            print(f"Confidence: {conf * 100:.2f}%")
            print(f"Result: {result_str}")
            print(f"Output: {out_file}\n")

            results_by_image[img_info["filename"]][m["id"]] = {
                "pred": pred_name,
                "conf": conf,
                "class_id": pred_idx,
                "is_correct": is_correct,
                "out_path": out_file
            }

    # ---------------------------------------------------------------------------
    # Generate 4-panel comparison images
    # ---------------------------------------------------------------------------
    print("============================================================")
    print("Generating Cross-Model Comparison Images")
    print("============================================================")
    for img_info in IMAGES:
        fname = img_info["filename"]
        comp_out = os.path.join(COMPARISON_DIR, img_info["comparison_filename"])
        create_panel_image(
            orig_path=img_info["path"],
            resnet_path=results_by_image[fname]["resnet50"]["out_path"],
            eff_path=results_by_image[fname]["efficientnet_v2"]["out_path"],
            vit_path=results_by_image[fname]["vit_b_16"]["out_path"],
            output_path=comp_out,
            title=img_info["short_name"]
        )
        file_sz = os.path.getsize(comp_out) / 1024.0
        print(f"Created comparison figure: {comp_out} ({file_sz:.1f} KB)")

    # ---------------------------------------------------------------------------
    # Generate results.csv
    # ---------------------------------------------------------------------------
    csv_path = os.path.join(COMPARISON_DIR, "results.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "ground_truth",
            "resnet50_prediction",
            "resnet50_confidence",
            "efficientnet_prediction",
            "efficientnet_confidence",
            "vit_prediction",
            "vit_confidence"
        ])
        for img_info in IMAGES:
            fname = img_info["filename"]
            r_res = results_by_image[fname]["resnet50"]
            r_eff = results_by_image[fname]["efficientnet_v2"]
            r_vit = results_by_image[fname]["vit_b_16"]
            writer.writerow([
                img_info["filename"],
                img_info["ground_truth"],
                r_res["pred"],
                f"{r_res['conf']*100:.2f}%",
                r_eff["pred"],
                f"{r_eff['conf']*100:.2f}%",
                r_vit["pred"],
                f"{r_vit['conf']*100:.2f}%",
            ])
    print(f"Saved results CSV: {csv_path}\n")

    # ---------------------------------------------------------------------------
    # Final terminal summary
    # ---------------------------------------------------------------------------
    print("=" * 84)
    print(f"{'Image':<20} {'ResNet50':<20} {'EfficientNetV2-S':<22} {'ViT-B/16':<20}")
    print("-" * 84)
    for img_info in IMAGES:
        fname = img_info["filename"]
        r_res = results_by_image[fname]["resnet50"]
        r_eff = results_by_image[fname]["efficientnet_v2"]
        r_vit = results_by_image[fname]["vit_b_16"]

        cell_res = f"{r_res['conf']*100:.1f}%"
        cell_eff = f"{r_eff['conf']*100:.1f}%"
        cell_vit = f"{r_vit['conf']*100:.1f}%"

        # Show concise label + conf
        lbl_res = f"{r_res['pred'][:12]}.. ({cell_res})" if len(r_res['pred']) > 14 else f"{r_res['pred']} ({cell_res})"
        lbl_eff = f"{r_eff['pred'][:12]}.. ({cell_eff})" if len(r_eff['pred']) > 14 else f"{r_eff['pred']} ({cell_eff})"
        lbl_vit = f"{r_vit['pred'][:12]}.. ({cell_vit})" if len(r_vit['pred']) > 14 else f"{r_vit['pred']} ({cell_vit})"

        print(f"{img_info['short_name']:<20} {lbl_res:<20} {lbl_eff:<22} {lbl_vit:<20}")
    print("=" * 84)
    print()
    print("Total images: 3")
    print("Models: 3")
    print("Grad-CAM outputs: 9")
    print("Comparison images: 3")


if __name__ == "__main__":
    main()

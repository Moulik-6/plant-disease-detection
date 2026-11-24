import sys
import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

# Ensure repository root is in sys.path
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

BASE = Path("ml/visualizations")

images = {
    "orange_hlb": {
        "title": "Orange HLB",
        "original": Path(
            "data/plantvillage_dataset/color/"
            "Orange___Haunglongbing_(Citrus_greening)/"
            "68ea5572-d729-481b-93dd-6e0a9898bc91___UF.Citrus_HLB_Lab 9704.JPG"
        ),
    },
    "strawberry_leaf_scorch": {
        "title": "Strawberry Leaf Scorch",
        "original": Path(
            "data/plantvillage_dataset/color/"
            "Strawberry___Leaf_scorch/"
            "804af05b-899f-44c2-8606-1b87bbe78c69___RS_L.Scorch 9961.JPG"
        ),
    },
    "corn_common_rust": {
        "title": "Corn Common Rust",
        "original": Path(
            "data/plantvillage_dataset/color/"
            "Corn_(maize)___Common_rust_/"
            "RS_Rust 1754.JPG"
        ),
    },
}

models = [
    ("Original", None),
    ("ResNet50", BASE / "resnet50"),
    ("EfficientNetV2-S", BASE / "efficientnet_v2"),
    ("ViT-B/16", BASE / "vit_b_16"),
]

OUTPUT = BASE / "comparison"

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
except OSError:
    font = ImageFont.load_default()


def add_label(img, text):
    label_height = 60
    canvas = Image.new("RGB", (img.width, img.height + label_height), "white")
    canvas.paste(img, (0, label_height))

    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]

    draw.text(
        ((img.width - text_width) // 2, 15),
        text,
        fill="black",
        font=font,
    )

    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Create side-by-side Grad-CAM comparison figures")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for key, info in images.items():
        panels = []

        original = Image.open(info["original"]).convert("RGB")

        # Use the same dimensions for every panel.
        width = 500
        ratio = width / original.width
        height = int(original.height * ratio)

        original = original.resize((width, height), Image.Resampling.LANCZOS)

        panels.append(add_label(original, "Original"))

        for model_name, model_dir in models[1:]:
            path = model_dir / f"{key}.jpg"

            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")

            img = Image.open(path).convert("RGB")
            img = img.resize((width, height), Image.Resampling.LANCZOS)

            panels.append(add_label(img, model_name))

        # Combine horizontally.
        total_width = sum(img.width for img in panels)
        total_height = max(img.height for img in panels)

        comparison = Image.new(
            "RGB",
            (total_width, total_height),
            "white",
        )

        x = 0
        for panel in panels:
            comparison.paste(panel, (x, 0))
            x += panel.width

        output = OUTPUT / f"{key}_comparison.jpg"
        comparison.save(output, quality=95)

        print(f"Saved: {output}")

    print("\nAll comparison images created successfully.")


if __name__ == "__main__":
    main()

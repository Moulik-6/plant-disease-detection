# Plant Disease Classification - ML Pipeline

PyTorch training pipeline for plant disease detection using the PlantVillage dataset. Supports easy backbone swapping (ResNet50, EfficientNetV2, Vision Transformer) without rewriting code.

## Directory Structure

```
ml/
├── configs/
│   ├── config.yaml          # Default configuration (ResNet50)
│   ├── efficientnet_v2.yaml # Preset for EfficientNetV2-S
│   └── vit.yaml             # Preset for Vision Transformer (ViT-B/16)
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Stratified train/val/test splits & augmentations
│   ├── models.py            # Model factory for ResNet, EfficientNetV2, ViT, timm
│   └── utils.py             # Config parser, metrics tracker, checkpointing
├── train.py                 # Configurable training script
├── evaluate.py              # Test split evaluation & classification report
└── requirements.txt         # Dependencies
```

## Dataset Format

The pipeline expects images organized by category in `data/PlantVillage` (or configured via `data_dir`):

```
data/PlantVillage/
├── Tomato___Bacterial_spot/
│   ├── image1.jpg
│   └── ...
├── Tomato___Early_blight/
│   └── ...
└── Tomato___healthy/
    └── ...
```

## Installation

```bash
pip install -r ml/requirements.txt
```

## Training

### Train default ResNet50:
```bash
python ml/train.py --config ml/configs/config.yaml
```

### Override parameters via CLI:
```bash
python ml/train.py --config ml/configs/config.yaml --epochs 30 --lr 0.0003 --batch-size 64
```

### Swap Architecture to EfficientNetV2:
```bash
python ml/train.py --config ml/configs/efficientnet_v2.yaml
# Or via CLI flag:
python ml/train.py --model efficientnet_v2_s
```

### Swap Architecture to Vision Transformer (ViT):
```bash
python ml/train.py --config ml/configs/vit.yaml
# Or via CLI flag:
python ml/train.py --model vit_b_16 --batch-size 16 --lr 0.00003
```

## Evaluation

Evaluate the saved `best_model.pth` on the test split:

```bash
python ml/evaluate.py --config ml/configs/config.yaml --checkpoint ml/checkpoints/best_model.pth
```

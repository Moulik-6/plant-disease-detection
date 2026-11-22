# Plant Disease Classification - Legacy ML Pipeline (Stage I)

PyTorch training pipeline for plant disease detection using the PlantVillage dataset. Archived Stage-I implementation preserved for reproducibility and backward compatibility.

## Directory Structure

```
ml/legacy/
├── config.yaml          # Default configuration (ResNet50)
├── efficientnet_v2.yaml # Preset for EfficientNetV2-S
├── vit.yaml             # Preset for Vision Transformer (ViT-B/16)
├── train.py             # Legacy training script
├── evaluate.py          # Legacy test split evaluation & classification report
├── full_eval.py         # Multi-model benchmarking & ensemble pipeline
├── run_experiment.py    # Multi-model 4-panel Grad-CAM comparison runner
└── create_gradcam_comparisons.py # Grad-CAM panel stitcher
```

## Installation

```bash
pip install -r ml/requirements.txt
```

## Training (Stage I)

### Train default ResNet50:
```bash
python ml/legacy/train.py --config ml/legacy/config.yaml
```

### Override parameters via CLI:
```bash
python ml/legacy/train.py --config ml/legacy/config.yaml --epochs 30 --lr 0.0003 --batch-size 64
```

### Swap Architecture to EfficientNetV2:
```bash
python ml/legacy/train.py --config ml/legacy/config.yaml --config-override ml/legacy/efficientnet_v2.yaml
# Or via CLI flag:
python ml/legacy/train.py --model efficientnet_v2_s
```

### Swap Architecture to Vision Transformer (ViT):
```bash
python ml/legacy/train.py --config ml/legacy/config.yaml --config-override ml/legacy/vit.yaml
# Or via CLI flag:
python ml/legacy/train.py --model vit_b_16 --batch-size 16 --lr 0.00003
```

## Evaluation (Stage I)

Evaluate the saved `best_model.pth` on the test split:

```bash
python ml/legacy/evaluate.py --config ml/legacy/config.yaml --checkpoint ml/checkpoints/resnet50/best_model.pth
```

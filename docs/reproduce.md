# Reproduction Guide: Stage-II Experiments

This guide provides the exact terminal commands required to reproduce all Stage-II data generation, model training, and evaluation benchmarks.

---

## 1. Environment Preparation

```bash
# Clone the repository
git clone https://github.com/[Your-Username]/plant-disease-detection.git
cd plant-disease-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r ml/requirements.txt
pip install -r requirements-dev.txt
```

---

## 2. Dataset Setup & Manifest Generation

Ensure raw datasets are placed under `data/`:
```
data/
├── plantvillage_dataset/color/
├── plantdoc/PlantDoc-Dataset/
└── plantwild/plantwild/
```

Generate the unified manifest with perceptual-hash duplicate clustering:
```bash
python ml/data/build_manifest.py \
    --pv-dir data/plantvillage_dataset/color \
    --plantdoc-dir data/plantdoc/PlantDoc-Dataset \
    --plantwild-dir data/plantwild/plantwild \
    --output-manifest data/manifest.csv \
    --seed 42 \
    --threshold 4
```

*(Optional)* Pre-cache resized $224 \times 224$ images to accelerate training throughput:
```bash
python ml/data/make_cache.py \
    --manifest data/manifest.csv \
    --out data/cache_224 \
    --img-size 224
```

---

## 3. Training Commands

### Automated Queue Execution (All 9 Core Experiments)
To run all 9 baseline and mixed experiments sequentially with automatic S3 syncing and timeout management:
```bash
export S3_BUCKET="<your-bucket-name>"
chmod +x scripts/run_queue.sh
./scripts/run_queue.sh scripts/queues/stage2_main.txt
```

### Manual Experiment Commands

#### Regime A: PlantVillage-Only Baselines
```bash
# ResNet-50
python ml/train.py --exp ml/configs/experiment/a_pv_only.yaml --set model.architecture=resnet50

# EfficientNetV2-S
python ml/train.py --exp ml/configs/experiment/a_pv_only.yaml --set model.architecture=efficientnet_v2_s

# Vision Transformer ViT-B/16
python ml/train.py --exp ml/configs/experiment/a_pv_only.yaml --set model.architecture=vit_b_16
```

### Regime B: Mixed Domain Training (PV + Field)
```bash
# ResNet-50
python ml/train.py --exp ml/configs/experiment/b_mixed.yaml --set model.architecture=resnet50

# EfficientNetV2-S
python ml/train.py --exp ml/configs/experiment/b_mixed.yaml --set model.architecture=efficientnet_v2_s

# Vision Transformer ViT-B/16
python ml/train.py --exp ml/configs/experiment/b_mixed.yaml --set model.architecture=vit_b_16
```

### Regime C: Sequential Fine-Tuning
```bash
# ResNet-50
python ml/train.py --exp ml/configs/experiment/c_finetune.yaml --set model.architecture=resnet50

# EfficientNetV2-S
python ml/train.py --exp ml/configs/experiment/c_finetune.yaml --set model.architecture=efficientnet_v2_s

# Vision Transformer ViT-B/16
python ml/train.py --exp ml/configs/experiment/c_finetune.yaml --set model.architecture=vit_b_16
```

### Multi-Backbone Late Fusion Training
```bash
# Concatenation Fusion
python ml/train.py --exp ml/configs/experiment/d_fusion_concat.yaml

# Dynamic Gated Fusion
python ml/train.py --exp ml/configs/experiment/d_fusion_gated.yaml
```

---

## 4. Evaluation Commands

### Evaluating Single Runs
```bash
python ml/evaluate.py \
    --run runs/b_mixed_resnet50/<timestamp> \
    --sets pv_clean_test,field_test
```

### Evaluating Baseline Checkpoints
```bash
python ml/evaluate.py \
    --checkpoint ml/checkpoints/resnet50/best_model.pth \
    --name resnet50_baseline \
    --sets pv_clean_test,field_test
```

### Full Multi-Model Comparison & Ensembling
```bash
python ml/evaluate.py \
    --run runs/b_mixed_resnet50/<timestamp> \
          runs/b_mixed_efficientnet_v2_s/<timestamp> \
          runs/b_mixed_vit_b_16/<timestamp> \
    --sets pv_clean_test,field_test \
    --ensemble \
    --out results/mixed_ensemble_report/
```

# Plant Disease Detection & Classification

Deep learning pipeline for automated crop leaf disease detection and classification across 38 crop-disease pairs. The repository benchmarks convolutional and transformer architectures (ResNet-50, EfficientNetV2-S, Vision Transformer ViT-B/16, and multi-backbone fusion models) and ensembles, paired with explainability (Grad-CAM), perceptual hash data leakage audits, multi-dataset generalization manifests (PlantVillage, PlantDoc, PlantWild), and CPU inference latency profiling.

---

## Dataset

- **Source**: [Kaggle abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Local Location**: `data/plantvillage_dataset/color`
- **Volume & Classes**: 54,305 color leaf images across 38 distinct crop-disease categories (apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, tomato).
- **Split Configuration**: Stratified 70% train / 15% validation / 15% test split with fixed seed `42`, implemented in `ml/src/dataset.py` (and integrated into multi-source manifests via `ml/data/build_manifest.py`):
  - **Train**: 37,997 images
  - **Validation**: 8,129 images
  - **Test**: 8,179 images

---

## Checkpoints

Model checkpoint weights are organized under:
```
ml/checkpoints/
├── resnet50/best_model.pth
├── efficientnet_v2/best_model.pth
└── vit/best_model.pth
```
> [!NOTE]
> Checkpoint binaries (`*.pth`) are not tracked in Git (`.gitignore`); they are archived and versioned in Amazon S3.

---

## Repository Layout

```
plant-disease-detection/
├── data/                              # Raw datasets (PlantVillage, PlantDoc, PlantWild)
├── ml/
│   ├── configs/                       # Configurations
│   │   ├── experiment/                # Stage-II experiment configs (base, a_pv_only, b_mixed, c_finetune)
│   │   ├── labels/                    # Label maps (pv_38.yaml)
│   │   └── mappings/                  # Dataset cross-alignment (plantdoc_map, plantwild_map)
│   ├── src/                           # Core library modules
│   │   ├── datasets.py                # Manifest-based dataset loader & transforms
│   │   ├── models.py                  # Standard backbones (ResNet, EfficientNetV2, ViT)
│   │   ├── fusion.py                  # Multi-backbone fusion models (concat, gated)
│   │   ├── model_zoo.py               # Unified model factory and checkpoint loader
│   │   ├── sampling.py                # Source-balanced batch sampling
│   │   ├── utils.py                   # YAML configs, metric tracking, checkpoint IO
│   │   └── dataset.py                 # Legacy stratified split (imported by build_manifest.py)
│   ├── data/                          # Dataset preprocessing & manifest tooling
│   │   ├── build_manifest.py          # Unified multi-source manifest builder
│   │   ├── make_cache.py              # Pre-resized image caching tool
│   │   └── sources/                   # Dataset ingestion adapters (plantwild.py)
│   ├── analysis/                      # Diagnostic and explainability tools
│   │   ├── gradcam.py                 # Grad-CAM attention visualizer
│   │   ├── leakage_check.py           # Perceptual hash leakage audit
│   │   ├── analyze_plantdoc.py        # PlantDoc split & label diagnostics
│   │   ├── eval_plantdoc.py           # PlantDoc standalone evaluation
│   │   └── benchmark_latency.py       # Single-image CPU inference latency profiling
│   ├── legacy/                        # Stage-I scripts and configurations
│   │   ├── train.py                   # Stage-I training script
│   │   ├── evaluate.py                # Stage-I evaluation script
│   │   ├── full_eval.py               # Stage-I multi-model evaluation & ensemble
│   │   ├── run_experiment.py          # Stage-I 4-panel Grad-CAM comparison runner
│   │   ├── create_gradcam_comparisons.py # Grad-CAM panel stitcher
│   │   ├── config.yaml                # Stage-I ResNet-50 config
│   │   ├── efficientnet_v2.yaml       # Stage-I EfficientNetV2 config
│   │   ├── vit.yaml                   # Stage-I ViT config
│   │   └── README.md                  # Stage-I documentation
│   ├── checkpoints/                   # Checkpoints per architecture (stored in S3)
│   ├── results/                       # Evaluation reports, probs, metrics JSON, leakage
│   ├── train.py                       # Stage-II unified training pipeline
│   ├── evaluate.py                    # Stage-II multi-run evaluation & benchmarking suite
│   └── requirements.txt               # Dependencies & platform instructions
├── scripts/                           # Queue runner & automation
│   ├── run_queue.sh                   # Sequential 9-experiment queue runner with S3 sync
│   └── aws_setup.md                   # AWS GPU training instance setup guide
├── docs/                              # Project documentation & reorganization records
│   └── REORG_PLAN.md
├── tests/                             # Unit & integration tests
├── .gitignore
└── README.md
```

---

## Script Guide

### 1. Unified Training (`ml/train.py`)
Trains models using configuration presets with support for mixed precision (`auto`, `fp16`, `bf16`, `fp32`), multi-stage training (e.g. fusion backbones), GradScaler, source-balanced sampling, and progress tracking.

```bash
# Run experiment with overrides:
python ml/train.py --exp ml/configs/experiment/a_pv_only.yaml --set model.architecture=resnet50

# Self-check sanity test:
python ml/train.py --self-check
```

---

### 2. Multi-Run & Checkpoint Evaluation (`ml/evaluate.py`)
Evaluates completed runs and baseline checkpoints across manifest splits (`pv_clean_test`, `field_test`, etc.) with leakage guards and probability-averaging ensembling.

```bash
# Evaluate baseline checkpoint:
python ml/evaluate.py --checkpoint ml/checkpoints/resnet50/best_model.pth --name resnet50_baseline

# Evaluate specific runs:
python ml/evaluate.py --runs runs/a_pv_only_resnet50/* runs/b_mixed_resnet50/*
```

---

### 3. Dataset Manifest Generation (`ml/data/build_manifest.py`)
Builds unified dataset manifests linking PlantVillage, PlantDoc, and PlantWild with category harmonization.

```bash
python ml/data/build_manifest.py
```

---

### 4. Data Leakage & Near-Duplicate Audit (`ml/analysis/leakage_check.py`)
Computes 64-bit perceptual hashes (`imagehash.phash`) across train and test partitions to identify and visualize data leakage and near-duplicate pairs.

```bash
python ml/analysis/leakage_check.py
```

---

### 5. Latency Benchmarking (`ml/analysis/benchmark_latency.py`)
Profiles single-image CPU inference latency across models with warm-up cycles and interleaved timing rounds.

```bash
python ml/analysis/benchmark_latency.py --threads 4
```

---

### 6. Grad-CAM Explainability (`ml/analysis/gradcam.py`)
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps across CNN and Vision Transformer architectures.

```bash
python ml/analysis/gradcam.py --image path/to/leaf.jpg --checkpoint ml/checkpoints/resnet50/best_model.pth
```

---

### 7. Legacy Pipeline (`ml/legacy/`)
Stage-I training, evaluation, and comparative experiment scripts are preserved under `ml/legacy/`. See [ml/legacy/README.md](file:///c:/Major%20Project/plant-disease-detection/ml/legacy/README.md) for details.

---

## Results

Evaluation on the 8,179-image PlantVillage stratified test split:

| Model | Accuracy (%) | Macro-F1 | Test Set Size |
| :--- | :---: | :---: | :---: |
| **ResNet50** | **99.80%** | **0.9969** | 8,179-image PlantVillage test split |
| **EfficientNetV2-S** | **99.88%** | **0.9983** | 8,179-image PlantVillage test split |
| **ViT-B/16** | **99.87%** | **0.9983** | 8,179-image PlantVillage test split |
| **Probability-Averaging Ensemble** | **99.93%** | **0.9988** | 8,179-image PlantVillage test split |

> [!WARNING]
> These are in-domain results and field-image generalisation is evaluated via the Stage-II evaluation suite (`ml/evaluate.py`).

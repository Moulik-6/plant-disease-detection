# Plant Disease Detection & Classification

Deep learning pipeline for automated crop leaf disease detection and classification across 38 crop-disease pairs. The repository benchmarks convolutional and transformer architectures (ResNet-50, EfficientNetV2-S, Vision Transformer ViT-B/16) and a probability-averaging ensemble, paired with explainability (Grad-CAM), perceptual hash data leakage audits, and CPU inference latency profiling.

---

## Dataset

- **Source**: [Kaggle abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Local Location**: `data/plantvillage_dataset/color`
- **Volume & Classes**: 54,305 color leaf images across 38 distinct crop-disease categories (apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, tomato).
- **Split Configuration**: Stratified 70% train / 15% validation / 15% test split with fixed seed `42`, implemented in `ml/src/dataset.py`:
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
├── data/
│   └── plantvillage_dataset/
│       └── color/                     # 38 class subdirectories (54,305 images)
├── ml/
│   ├── configs/
│   │   ├── config.yaml                # ResNet-50 default configuration
│   │   ├── efficientnet_v2.yaml       # EfficientNetV2-S preset
│   │   └── vit.yaml                   # Vision Transformer (ViT-B/16) preset
│   ├── src/
│   │   ├── dataset.py                 # Stratified 70/15/15 dataset loader & augmentations
│   │   ├── models.py                  # Architecture factory (ResNet50, EfficientNetV2, ViT)
│   │   └── utils.py                   # YAML config loader, metric tracking, checkpoint IO
│   ├── checkpoints/                   # Checkpoints per architecture (stored in S3)
│   ├── results/                       # Evaluation reports, probs, metrics JSON, leakage
│   │   └── leakage/                   # Visualizations of near-duplicate test/train pairs
│   ├── visualizations/                # Grad-CAM overlays and comparative panels
│   ├── train.py                       # Supervised training loop
│   ├── evaluate.py                    # Standalone checkpoint evaluation
│   ├── full_eval.py                   # Multi-model benchmarking & ensemble pipeline
│   ├── leakage_check.py               # 64-bit pHash train/test duplicate detection
│   ├── benchmark_latency.py           # Single-image latency profiling across CPU threads
│   ├── gradcam.py                     # Grad-CAM explainability generator
│   ├── run_experiment.py              # Multi-model 4-panel Grad-CAM comparison runner
│   └── requirements.txt               # Dependencies & platform instructions
├── .gitignore
└── README.md
```

---

## Script Guide

### 1. Training (`train.py`)
Trains an individual backbone architecture on the stratified training set with cross-entropy loss, learning rate scheduling, and early stopping.

```bash
# Train default ResNet-50:
python ml/train.py --config ml/configs/config.yaml

# Train EfficientNetV2-S:
python ml/train.py --config ml/configs/efficientnet_v2.yaml

# Train ViT-B/16:
python ml/train.py --config ml/configs/vit.yaml
```
- **Outputs**: Best checkpoint (`best_model.pth`), class index mapping (`classes.json`), and validation metrics logs saved in `ml/checkpoints/<architecture>/`.

---

### 2. Standalone Evaluation (`evaluate.py`)
Evaluates a single model checkpoint against the 8,179-image test partition.

```bash
python ml/evaluate.py --checkpoint ml/checkpoints/resnet50/best_model.pth
```
- **Outputs**: Test classification report (precision, recall, F1, support per class) and accuracy printed to console; summary JSON saved to `<checkpoint_dir>/test_evaluation_report.json`.

---

### 3. Full Multi-Model Evaluation & Ensembling (`full_eval.py`)
Loads all three checkpoints (`resnet50`, `efficientnet_v2`, `vit`), verifies identical `class_to_idx` mappings with the data loader, exports raw softmax probabilities, builds a probability-averaging ensemble, calculates accuracy and macro F1 metrics, records model weight footprint (MB), measures median single-image latency, and prints a comparative summary table.

```bash
# Full evaluation:
python ml/full_eval.py

# Fast smoke-test (limits test loader to N samples):
python ml/full_eval.py --limit 5

# Skip single-image latency benchmarking:
python ml/full_eval.py --skip-latency
```
- **Outputs**:
  - `ml/results/test_files.csv`: List of test image paths and ground-truth labels.
  - `ml/results/probs_<model>.npy`: Softmax probability distributions for each model and ensemble.
  - `ml/results/per_class_<model>.csv` & `ml/results/per_class_metrics.csv`: Per-class precision, recall, and F1.
  - `ml/results/confusion_matrix_<model>.png` & `ml/results/confusion_matrix.png`: Normalized confusion matrix heatmaps.
  - `ml/results/metrics.json`: Full evaluation results, metadata, and hardware environment specs.

---

### 4. Data Leakage & Near-Duplicate Audit (`leakage_check.py`)
Rebuilds the dataset split via `get_dataloaders(num_workers=0)` and computes 64-bit perceptual hashes (`imagehash.phash`) for all 37,997 train and 8,179 test images. Evaluates minimum Hamming distances from every test image to any train image using vectorized matrix operations.

```bash
python ml/leakage_check.py
```
- **Outputs**:
  - `ml/results/hashes.npy`: Cached train and test perceptual hashes for near-instant reruns.
  - `ml/results/leakage/`: High-resolution side-by-side PNGs (`pair_01_dist_0.png` through `pair_30_dist_4.png`) illustrating the 30 closest train/test pairs.
  - `ml/results/leakage_metrics.json`: Leakage counts, worst classes, and model accuracy on near-duplicates ($d_H \le 4$) versus clean samples ($d_H > 4$).

---

### 5. Single-Image Latency Benchmarking (`benchmark_latency.py`)
Measures single-image forward-pass execution latency (batch size 1) across individual models and the sequential ensemble inside `torch.inference_mode()`. Employs 20 warm-up runs and 10 interleaved rounds (20 runs/round = 200 timed runs per model) to eliminate thermal drift and scheduling bias.

```bash
# Benchmark with default 4 PyTorch CPU threads:
python ml/benchmark_latency.py --threads 4
```
- **Outputs**:
  - Console report displaying hardware environment (PyTorch version, thread count, CPU model, Windows active power plan from `powercfg /getactivescheme`).
  - Latency statistics (median, p10, p90, and $p_{90}/p_{10}$ stability ratio) with noise alerts if $p_{90}/p_{10} > 1.5$.
  - `ml/results/latency.json`: Structured benchmark metrics.

---

### 6. Grad-CAM Explainability (`gradcam.py`)
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) overlays showing spatial attention regions supporting model predictions. Adapts automatically to CNN final conv layers (`layer4[-1]`, `features[-1]`) and Vision Transformers (using token reshaping to reproject 1D patch embeddings into a 2D spatial grid).

```bash
# Single image visualization:
python ml/gradcam.py --image path/to/leaf.jpg --checkpoint ml/checkpoints/resnet50/best_model.pth

# Batch generation across test split:
python ml/gradcam.py --batch 10 --checkpoint ml/checkpoints/efficientnet_v2/best_model.pth
```
- **Outputs**: Visualizations saved to `ml/visualizations/`.

---

### 7. Multi-Model Grad-CAM Experiment Comparison (`run_experiment.py`)
Runs comparative Grad-CAM across ResNet-50, EfficientNetV2-S, and ViT-B/16 on representative disease samples (e.g., Orange Huanglongbing, Strawberry Leaf Scorch, Corn Common Rust).

```bash
python ml/run_experiment.py
```
- **Outputs**: 4-panel comparison images (`Original | ResNet50 | EfficientNetV2-S | ViT-B/16`) saved in `ml/visualizations/comparison/` and summary CSV records.

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
> These are in-domain results and field-image generalisation is still to be evaluated.

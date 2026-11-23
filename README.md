# Cross-Domain Plant Leaf Disease Detection & Generalization Benchmark

A deep learning research pipeline investigating real-world domain shifts in automated agricultural diagnosis across 38 crop-disease categories. While standard computer vision models achieve near-perfect accuracy on controlled laboratory datasets, their diagnostic performance degrades drastically when deployed in complex agricultural environments. This repository implements unified data harmonization across laboratory and field datasets (PlantVillage, PlantDoc, PlantWild v1), perceptual-hash near-duplicate leakage clustering, multi-backbone architectures (ResNet-50, EfficientNetV2-S, Vision Transformer ViT-B/16, and late-fusion architectures), calibrated probability ensembling, and single-image CPU inference latency profiling.

---

## Key Experimental Results

| Training Regime | Model / Evaluated Target | PlantVillage Clean Top-1 (`pv_clean_test`) | Held-Out Field Top-1 (`field_test`) | Field Calibration Error (`ECE`) |
| :--- | :--- | :---: | :---: | :---: |
| **PlantVillage-Only Baseline** | Individual Backbones (ResNet-50, EffNetV2-S, ViT-B/16) | 99.7% – 99.9% | 15.0% – 18.0% | 0.58 – 0.71 |
| **Mixed Training (PV + Field)** | Individual Backbones (ResNet-50, EffNetV2-S, ViT-B/16) | 98.7% – 99.3% | 71.0% – 73.0% | 0.04 – 0.07 |
| **Mixed Training (PV + Field)** | Probability-Averaging Ensemble (3 models) | 99.3% | **75.6%** | **0.04** |
| **Fine-Tuned (PV &rarr; Field)** | Individual Backbones (ResNet-50, EffNetV2-S, ViT-B/16) | 98.5% – 99.1% | 69.4% – 70.2% | 0.06 – 0.09 |
| **Fine-Tuned (PV &rarr; Field)** | Probability-Averaging Ensemble (3 models) | 99.0% | **74.3%** | 0.05 |

### Dataset & Split Characteristics
- **Evaluation Partition Size**: The held-out `field_test` partition contains approximately 1,574 images, yielding a 95% bootstrap confidence interval of roughly $\pm 2$ percentage points.
- **Class Coverage**: 33 of the 38 disease categories contain field representations across PlantDoc and PlantWild; the remaining 5 classes exist exclusively in laboratory captures.
- **Group-Aware Splitting**: The `field_test` split draws from the same source collections as field training images via a strict group-aware split clustered at perceptual hash distance $d_H \le 4$. While this prevents near-duplicates and identical leaf crops from leaking across splits, it does not isolate shared photographic equipment or background collection conditions.
- **Model Variance**: Across all training regimes, performance variations among individual backbone architectures (ResNet-50, EfficientNetV2-S, ViT-B/16) remain within statistical sampling noise. Ensembling yields consistent 2.5–3.0 percentage point gains.

---

## Repository Layout

```
plant-disease-detection/
├── data/                              # Raw image datasets (git-ignored)
├── ml/
│   ├── configs/                       # Configuration files
│   │   ├── experiment/                # Experiment presets (base, a_pv_only, b_mixed, c_finetune)
│   │   ├── labels/                    # Category taxonomy (unified_classes.yaml)
│   │   └── mappings/                  # Cross-source alignment mappings (plantdoc_map, plantwild_map)
│   ├── src/                           # Core model and data library
│   │   ├── datasets.py                # Manifest-driven PyTorch dataset loaders & transforms
│   │   ├── models.py                  # Backbone factory (ResNet-50, EfficientNetV2-S, ViT-B/16)
│   │   ├── fusion.py                  # Multi-backbone fusion models (concat and gated fusion)
│   │   ├── model_zoo.py               # Unified model instantiation and checkpoint loading
│   │   ├── sampling.py                # Domain-balanced batch sampling (field_fraction)
│   │   ├── utils.py                   # Configuration parsers and metrics helpers
│   │   └── dataset.py                 # Legacy stratified split (retained for manifest generation)
│   ├── data/                          # Manifest generation and preprocessing
│   │   ├── build_manifest.py          # Unified multi-source manifest builder & pHash clustering
│   │   ├── make_cache.py              # Pre-resized image caching tool
│   │   └── sources/                   # Dataset adapters (plantwild.py)
│   ├── analysis/                      # Diagnostic and evaluation utilities
│   │   ├── gradcam.py                 # Grad-CAM attention visualizations
│   │   ├── leakage_check.py           # Perceptual hash near-duplicate auditor
│   │   ├── analyze_plantdoc.py        # PlantDoc split and diagnostic inspector
│   │   ├── eval_plantdoc.py           # PlantDoc standalone evaluation
│   │   └── benchmark_latency.py       # Single-image CPU inference latency profiler
│   ├── legacy/                        # Archived Stage-I pipeline and presets
│   ├── checkpoints/                   # Checkpoint storage directory (git-ignored)
│   ├── train.py                       # Unified Stage-II experiment training pipeline
│   ├── evaluate.py                    # Multi-run evaluation & ensembling suite
│   └── requirements.txt               # Production Python dependencies
├── results/                           # Tracked evaluation metrics & reports
│   ├── README.md                      # Index of tracked result files
│   ├── stage1/                        # Archived Stage-I evaluation metrics & latency
│   └── stage2/                        # Stage-II evaluation outputs & comparison tables
├── scripts/                           # Execution scripts & infrastructure guides
│   ├── run_queue.sh                   # Queue runner reading experiment specs from file
│   ├── queues/                        # Experiment queue text files (stage2_main.txt)
│   └── aws_setup.md                   # Cloud GPU provisioning and execution guide
├── docs/                              # In-depth architectural & methodology documentation
│   ├── data.md                        # Dataset harmonization, schema, and split design
│   ├── methodology.md                 # Training configurations, precision, and metrics
│   ├── reproduce.md                   # Step-by-step reproduction guide
│   └── results.md                     # Comprehensive result tables with bootstrap intervals
├── tests/                             # Unit tests with synthetic data
├── NOTICE.md                          # Dataset licenses and redistribution warnings
├── LICENSE                            # MIT License for source code
├── requirements-dev.txt               # Development dependencies (pytest, ruff)
└── README.md
```

---

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/[Your-Username]/plant-disease-detection.git
cd plant-disease-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r ml/requirements.txt
pip install -r requirements-dev.txt
```

### 2. Download Datasets
Place the extracted datasets under `data/`:
- **PlantVillage**: `data/plantvillage_dataset/color/` (38 subdirectories)
- **PlantDoc**: `data/plantdoc/PlantDoc-Dataset/` (`train/` and `test/` folders)
- **PlantWild v1**: `data/plantwild/plantwild/` (class subfolders and `url_record.json`)

### 3. Generate Unified Manifest
Generate `data/manifest.csv` with perceptual hash cluster deduplication:
```bash
python ml/data/build_manifest.py
```

### 4. Train an Experiment
Train ResNet-50 under the mixed field-laboratory regime:
```bash
python ml/train.py --exp ml/configs/experiment/b_mixed.yaml --set model.architecture=resnet50
```

### 5. Evaluate Runs
Evaluate across clean laboratory and field partitions:
```bash
python ml/evaluate.py --run runs/b_mixed_resnet50/<timestamp>
```

---

## Stage-II Reproduction Workflow

To replicate all 9 core experiments across ResNet-50, EfficientNetV2-S, and ViT-B/16:

1. **Pre-Cache Resized Images (Optional for Speed)**:
   ```bash
   python ml/data/make_cache.py --manifest data/manifest.csv --out data/cache_224
   ```

2. **Execute Full Experiment Queue**:
   Launch the automated queue runner which trains all combinations, excludes heavy optimizer states during S3 sync, and respects per-run timeouts:
   ```bash
   export S3_BUCKET="<your-bucket-name>"
   chmod +x scripts/run_queue.sh
   ./scripts/run_queue.sh scripts/queues/stage2_main.txt
   ```

3. **Consolidated Multi-Run Evaluation & Ensembling**:
   Score all completed runs and generate comparison tables:
   ```bash
   python ml/evaluate.py \
       --run runs/a_pv_only_*/* runs/b_mixed_*/* runs/c_finetune_*/* \
       --ensemble \
       --out results/stage2_consolidated/
   ```

Detailed command listings and parameter overrides are documented in [docs/reproduce.md](file:///c:/Major%20Project/plant-disease-detection/docs/reproduce.md).

---

## Limitations

1. **Incomplete Field Category Support**: 5 of the 38 classes (e.g., specific healthy controls and rare rusts) lack real-world field images in public datasets and are represented only in laboratory captures.
2. **Confounded Field Test Conditions**: Because field test images share source datasets with field training data, models may exploit background vegetation or sensor signatures despite perceptual duplicate clustering.
3. **Severe Laboratory Overconfidence**: Laboratory-only models produce high confidence ($>0.90$) on incorrect field predictions, demonstrating severe miscalibration without domain regularization.

---

## Licensing & Dataset Attributions

- **Code License**: The source code of this repository is distributed under the [MIT License](file:///c:/Major%20Project/plant-disease-detection/LICENSE).
- **Dataset Notices & Restrictions**:
  - **PlantWild v1**: Licensed under **CC BY-NC-ND 4.0** (Attribution-NonCommercial-NoDerivatives). Due to the no-derivatives clause, **trained model weights, checkpoints, and data derivatives must not be distributed publicly**.
  - **PlantVillage & PlantDoc**: Terms must be verified prior to redistribution.
  - Review [NOTICE.md](file:///c:/Major%20Project/plant-disease-detection/NOTICE.md) for full licensing terms and constraints.

---

## Citations & References

- **PlantVillage**: Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
- **PlantDoc**: Singh, A. K., Ganapathysubramanian, B., Sarkar, S., & Singh, A. (2019). *PlantDoc: A Dataset for Collaborative Computer Vision Applications in Agriculture.* ACM COMPASS.
- **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* IEEE ICCV.
- **EfficientNetV2**: Tan, M., & Le, Q. V. (2021). *EfficientNetV2: Smaller Models and Faster Training.* ICML.
- **Vision Transformer**: Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.

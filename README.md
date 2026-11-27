# Cross-Domain Plant Leaf Disease Detection & Generalization Benchmark

A deep learning research pipeline investigating real-world domain shifts in automated agricultural diagnosis across 38 crop-disease categories. While standard computer vision models achieve near-perfect accuracy on controlled laboratory datasets, their diagnostic performance degrades drastically when deployed in complex agricultural environments. This repository implements unified data harmonization across laboratory and field datasets (PlantVillage, PlantDoc, PlantWild v1), perceptual-hash near-duplicate leakage clustering, multi-backbone architectures (ResNet-50, EfficientNetV2-S, Vision Transformer ViT-B/16, and late-fusion architectures), calibrated probability ensembling, and single-image CPU inference latency profiling.

---

## Key Experimental Results

The table below summarizes all 22 evaluated runs from [results/stage2/summary_table.csv](results/stage2/summary_table.csv) (`95% CI` shown where present).

| Training Regime | Model / Run | `pv_clean` Top-1 (1,342 imgs) | `field_test` Top-1 (1,574 imgs) [95% CI] | Macro-F1 | ECE | Unknown &ge; 0.9 | PlantDoc Top-1 (362 imgs) | PlantWild Top-1 (1,212 imgs) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Originals (Stage-I PV-Only)** | ResNet-50 | 99.9% | 14.9% [13.0, 16.9] | 0.146 | 0.657 | 43.7% | 18.8% | 13.8% |
| | EfficientNetV2-S | 99.9% | 14.9% [13.0, 16.9] | 0.146 | 0.713 | 58.4% | 17.4% | 14.1% |
| | ViT-B/16 | 99.7% | 15.4% [13.5, 17.3] | 0.154 | 0.582 | 30.5% | 17.7% | 14.7% |
| | **3-Model Ensemble** | 99.9% | **16.2%** [14.3, 18.3] | 0.162 | 0.480 | 15.5% | 20.2% | 15.0% |
| **A. PV-Only (New Pipeline)** | ResNet-50 | 100.0% | 16.7% [14.8, 18.7] | 0.165 | 0.677 | 55.1% | 21.5% | 15.3% |
| | EfficientNetV2-S | 99.6% | 15.9% [13.9, 17.8] | 0.165 | 0.685 | 58.1% | 18.8% | 15.0% |
| | ViT-B/16 | 99.9% | 17.2% [15.3, 19.2] | 0.173 | 0.610 | 42.5% | 21.8% | 15.8% |
| | **3-Model Ensemble** | 100.0% | **18.6%** [16.4, 20.6] | 0.195 | 0.391 | 9.1% | 23.8% | 17.0% |
| **A + Field Augmentation** | ResNet-50 | 100.0% | 19.1% | 0.177 | 0.418 | 12.8% | 24.6% | 17.4% |
| **B. Mixed (PV + Field)** | ResNet-50 | 99.3% | 71.9% [69.4, 74.3] | 0.719 | 0.041 | 10.7% | 79.3% | 69.6% |
| | EfficientNetV2-S | 99.3% | 73.0% [70.7, 75.4] | 0.727 | 0.073 | 10.1% | 77.6% | 71.6% |
| | ViT-B/16 | 98.7% | 71.0% [68.5, 73.5] | 0.701 | 0.049 | 7.6% | 77.1% | 69.1% |
| | EffNetV2-S + ViT-B/16 Ensemble | 99.7% | 74.8% | 0.742 | 0.045 | 4.3% | 80.4% | 73.1% |
| | **3-Model Ensemble** | 99.8% | **75.6%** [73.2, 78.0] | **0.751** | 0.049 | 3.9% | 82.9% | 73.4% |
| **C. Fine-Tune (PV &rarr; Field)** | ResNet-50 | 99.6% | 69.6% [67.1, 72.0] | 0.691 | 0.040 | 7.1% | 79.0% | 66.7% |
| | EfficientNetV2-S | 99.7% | 69.4% [67.0, 71.8] | 0.686 | 0.046 | 5.3% | 75.1% | 67.7% |
| | ViT-B/16 | 99.6% | 70.2% [67.8, 72.8] | 0.699 | 0.041 | 6.5% | 78.2% | 67.8% |
| | **3-Model Ensemble** | 100.0% | **74.3%** [72.0, 76.8] | 0.740 | 0.083 | 2.1% | 82.9% | 71.8% |
| **D. Late Fusion** | Fusion Concat | 99.5% | 74.3% | 0.741 | 0.064 | 17.0% | 79.0% | 72.9% |
| | Fusion Gated | 99.6% | 73.1% | 0.729 | 0.076 | 12.4% | 78.2% | 71.6% |
| **E. Leave-One-Source-Out (EffNet)** | Held-Out PlantWild (PV + PlantDoc) | 99.6% | 53.6% | 0.485 | 0.266 | 12.5% | 75.7% | 47.0% |
| | Held-Out PlantDoc (PV + PlantWild) | 99.6% | 70.3% | 0.707 | 0.086 | 10.0% | 70.4% | 70.3% |

### Summary Highlights & Evaluation Characteristics
- **Partition Size & Intervals**: Held-out `field_test` contains **1,574 images** across **1,396** perceptual-hash `group_id` clusters and **33 of 38** classes; `pv_clean_test` has **1,342** images. Reported confidence intervals are 95% group-cluster bootstrap intervals (`seed = 42`). Mixed training (**B**) is 0.3-1.0 points lower than the originals and **C** on `pv_clean_test` (4-13 of 1,342 images), about the size of the &plusmn;0.5 point sampling margin of that test set.
- **Paired Differences (first-named minus second-named model, `field_test`)**: Field augmentation alone adds **+2.4 pp [+0.6, +4.0]**, whereas mixed training with field data jumps **+52.8 pp [+49.7, +55.8]** over field-augmented PV-only training. On `B` vs `C`, mixed training outperforms fine-tuning on EfficientNetV2-S (**+3.6 pp [+1.3, +6.0]**) and ResNet-50 (**+2.3 pp [+0.4, +4.2]**) and is comparable on ViT-B/16 (**+0.8 pp [-1.1, +2.7]**). The 3-model ensemble outperforms EfficientNetV2-S by **+2.6 pp [+1.2, +4.0]**; late fusion does not beat averaging the same two backbones (concat -0.5 pp [-1.7, +0.7]; gated -1.7 pp [-3.0, -0.3]); concat beats EfficientNetV2-S alone by +1.3 pp [+0.4, +2.2].
- **Source Held-Out (`EfficientNetV2-S`)**: Without PlantWild (**1,843** PlantDoc field training images), accuracy is **75.7%** on PlantDoc and **47.0%** on held-out PlantWild; without PlantDoc (about **5,256** PlantWild field training images), held-out PlantDoc (**362** images, 95% interval about &plusmn;4.7 points) is **70.4%** and PlantWild is **70.3%**. Because the two runs differ in training size and source difficulty (every model scores higher on PlantDoc), source and amount cannot be separated.
- **Calibration & Unknown Rejection**: Temperature scaling fitted on `field_val` lowers ECE from **0.073 &rarr; 0.039** ($T = 1.0675$, AUROC **0.687**) for EfficientNetV2-S and **0.049 &rarr; 0.027** ($T = 0.9078$, AUROC **0.762**) for the 3-model ensemble. Ensemble operating points on `field_test` / `unknown_test`: threshold **0.678** &rarr; **69.2%** coverage / **87.6%** acc / **70.7%** unknown rejected; **0.429** &rarr; **90.7%** / **80.4%** / **35.9%**; **0.825** &rarr; **53.2%** / **91.5%** / **83.7%**.
- Detailed tables, calibration breakdowns, CPU latency status, and source provenance are in [docs/results.md](docs/results.md).

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

1. **Shared Field Sources (`field_test` vs. Field Train)**: Held-out `field_test` images are drawn from the same web field collections (`plantdoc`, `plantwild`) as the field training split. While `group_id` perceptual-hash clustering prevents near-duplicates from leaking across splits, shared photographic and source collection biases remain.
2. **No Non-Plant Imagery in Unknown Set**: The open-set `other_pool` evaluation partition contains other crops and plant diseases outside the 38-class taxonomy, with **no non-plant images** (e.g., soil, hands, or indoor objects); rejection rates reflect out-of-taxonomy plant filtering rather than general non-plant rejection.
3. **Incomplete Field Class Coverage (33 of 38 Classes)**: Only 33 of the 38 unified disease categories have real-world field images across PlantDoc and PlantWild; the remaining 5 classes exist exclusively in PlantVillage laboratory captures.
4. **Web Image Label Noise**: Field samples scraped from online sources contain annotation noise, visual ambiguity between related disorders, and multi-leaf backgrounds.
5. **PlantWild v1 Licensing**: PlantWild v1 is licensed CC BY-NC-ND 4.0; it is unclear whether trained weights count as derivatives; the weights are kept private unless the dataset authors or the project guide permit otherwise.
6. **Single Training Run per Configuration**: Every configuration was trained once, so confidence intervals cover test-set sampling rather than training randomness.
7. **`A + field_aug` Scope**: `A + field augmentation` was run only for ResNet-50.
8. **Post-Hoc Configuration Selection**: Hyper-parameters and epochs were chosen on validation sets only, but the deployed configuration was chosen after seeing test numbers (a mild selection effect).

---

## Licensing & Dataset Attributions

- **Code License**: The source code of this repository is distributed under the [MIT License](LICENSE).
- **Dataset Notices & Restrictions**:
  - **PlantWild v1**: Licensed under **CC BY-NC-ND 4.0** (Attribution-NonCommercial-NoDerivatives); it is unclear whether trained weights count as derivatives, so weights are kept private unless the dataset authors or the project guide permit otherwise.
  - **PlantVillage & PlantDoc**: Terms must be verified prior to redistribution.
  - Review [NOTICE.md](NOTICE.md) for full licensing terms and constraints.

---

## Citations & References

- **PlantVillage**: Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
- **PlantDoc**: Singh, A. K., Ganapathysubramanian, B., Sarkar, S., & Singh, A. (2019). *PlantDoc: A Dataset for Collaborative Computer Vision Applications in Agriculture.* ACM COMPASS.
- **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* IEEE ICCV.
- **EfficientNetV2**: Tan, M., & Le, Q. V. (2021). *EfficientNetV2: Smaller Models and Faster Training.* ICML.
- **Vision Transformer**: Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.

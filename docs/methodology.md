# Modeling, Training & Evaluation Methodology

This document outlines the architectural configurations, optimization choices, precision handling, evaluation protocol, and modular code architecture governing the Stage-II plant disease classification experiments.

---

## 1. Model Architectures

The benchmark evaluates three complementary backbone families alongside late-fusion multi-branch networks:

| Architecture | Model Family | Parameters | Pretrained Source | Input Resolution | Key Design Feature |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **ResNet-50** | Residual CNN | ~23.5M | ImageNet-1K (`DEFAULT`) | 224 &times; 224 | Classic residual skip connections |
| **EfficientNetV2-S** | Scaled ConvNet | ~20.2M | ImageNet-1K (`DEFAULT`) | 224 &times; 224 | Fused-MBConv layers & progressive learning |
| **ViT-B/16** | Vision Transformer | ~85.8M | ImageNet-1K (`DEFAULT`) | 224 &times; 224 | 16&times;16 patch embeddings & self-attention |
| **Fusion (Concat)** | Multi-Backbone Late Fusion | ~107.0M | Backbone Checkpoints | 224 &times; 224 | Concatenates 1280-d EffNet + 768-d ViT tokens |
| **Fusion (Gated)** | Dynamic Gated Late Fusion | ~108.1M | Backbone Checkpoints | 224 &times; 224 | Per-sample learned gating: $g \cdot a + (1-g) \cdot b$ |

---

## 2. Training Regimes & Sampling Strategies

### A. In-Domain Laboratory Baseline (`a_pv_only.yaml`)
- **Dataset**: `pv_train` and `pv_val` only.
- **Purpose**: Establishes upper-bound laboratory accuracy and documents failure modes when evaluated on real-world field images.

### B. Mixed Domain Training (`b_mixed.yaml`)
- **Dataset**: Joint training on `pv_train` and `field_train`.
- **Sampling**: `FieldFractionWeightedSampler` draws batches with fixed field probability mass ($field\_fraction = 0.50$).
- **Augmentations**: Stronger field augmentations (RandomResizedCrop, TrivialAugmentWide, GaussianBlur, ColorJitter, RandomErasing).

### C. Sequential Fine-Tuning (`c_finetune.yaml`)
- **Dataset**: Initialized from completed `a_pv_only` checkpoints; trained on `field_train` with optional laboratory replay ($field\_fraction = 0.80$).

### D. Multi-Stage Fusion Optimization
- **Stage 1 (Head-Only)**: Backbones frozen; head trained with AdamW ($\text{lr} = 10^{-3}$, cosine annealing).
- **Stage 2 (Joint Fine-Tuning)**: Final 2 blocks of EfficientNet and final 2 encoder layers of ViT unfrozen ($\text{lr} = 10^{-5}$); head tuned at $\text{lr} = 10^{-4}$.

---

## 3. Mixed Precision & Memory Optimization

The training pipeline supports four precision configurations (`training.precision`):
- **`auto`** *(Default)*: Automatically resolves to `fp16` on CUDA-enabled GPUs and `fp32` on CPU.
- **`fp16`**: Utilizes `torch.autocast(device_type="cuda", dtype=torch.float16)` coupled with `torch.amp.GradScaler("cuda")`. Gradient clipping (`max_norm = 1.0`) is performed after `scaler.unscale_()`.
- **`bf16`**: Employs Bfloat16 autocasting without requiring a gradient scaler (suitable for NVIDIA Ampere and newer).
- **`fp32`**: Standard single-precision training without autocasting.

---

## 4. Evaluation Protocol & Metrics

Evaluation is performed across partitions using [ml/evaluate.py](file:///c:/Major%20Project/plant-disease-detection/ml/evaluate.py):

1. **Top-1 Accuracy**:
   - $\text{Top-1 (All 38)}$: Standard accuracy computed over all 38 output classes.
   - $\text{Top-1 (Restricted)}$: Accuracy when predictions are restricted strictly to classes present in the test partition.
2. **Macro-F1 (Over Present Classes)**:
   - Unweighted average of per-class F1-scores, evaluated strictly over categories with non-zero ground-truth support in that partition.
3. **Expected Calibration Error (ECE)**:
   - 15 equally spaced probability bins in $[0, 1]$.
   - Measures reliability and probability overconfidence:
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
4. **Cluster Bootstrap 95% Confidence Intervals**:
   - 1,000 resamples clustered by `group_id` so perceptual near-duplicates are not treated as independent observations.
5. **Leakage Guard**:
   - Evaluator verifies that checkpoints initialized on legacy splits do not evaluate against `pv_test` or `pv_legacy_test`, automatically substituting `pv_clean_test`.

---

## 5. How the Code is Organised

The repository separates data ingestion, sampling, model instantiation, training execution, and multi-run evaluation into distinct, single-responsibility components:

### 1. Manifest Builder (`ml/data/build_manifest.py`)
- **Reads**:
  - Raw image trees (`data/plantvillage_dataset/color`, `data/plantdoc/PlantDoc-Dataset`, `data/plantwild/plantwild`).
  - Taxonomic alignment YAML mappings (`ml/configs/mappings/plantdoc_map.yaml`, `ml/configs/mappings/plantwild_map.yaml`).
  - Optional precomputed perceptual hashes from disk.
- **Writes**:
  - `data/manifest.csv`: The canonical dataset table storing relative file paths, unified labels, integer class indices, perceptual hash cluster IDs (`group_id`), cross-domain duplicate flags (`pv_dup`), and partition assignments (`split`, `legacy_split`).
  - `data/manifest_report.json`: Cluster size distribution, class imbalance statistics, and partition integrity diagnostics.

### 2. Domain-Balanced Sampler (`ml/src/sampling.py`)
- **Reads**:
  - Sample metadata (`labels`, `is_field` flags) from manifest-backed PyTorch datasets (`ml/src/datasets.py`).
  - Sampling hyperparameters (`sampling.field_fraction`, `sampling.class_balanced`).
- **Produces**:
  - Normalized sampling weight tensors (`torch.DoubleTensor`) for PyTorch's `WeightedRandomSampler`, enforcing a fixed field image proportion per batch (e.g. 50% field / 50% laboratory) and optional inverse-square-root class frequency balancing without mutating underlying datasets.

### 3. Model Zoo (`ml/src/model_zoo.py`)
- **Reads**:
  - Architecture strings, pretrained weights specifications, dropout rates, and backbone checkpoint paths (`fusion.efficientnet_init`, `fusion.vit_init`) from experiment configurations or checkpoint dictionaries.
- **Produces / Writes**:
  - Instantiates individual torchvision backbones (`resnet50`, `efficientnet_v2_s`, `vit_b_16`) via `ml/src/models.py` or late-fusion architectures (`fusion_concat`, `fusion_gated`) via `ml/src/fusion.py`.
  - Safely deserializes checkpoints via `load_checkpoint_model()`, automatically purging initialization references to prevent circular reload dependencies.

### 4. Unified Trainer (`ml/train.py`)
- **Reads**:
  - Experiment configuration hierarchy (`ml/configs/experiment/*.yaml` inheriting from `base.yaml`).
  - Dataset manifest (`data/manifest.csv`) and optional pre-resized image cache (`data/cache_224/`).
  - Optional resume/fine-tune checkpoints (`--resume` or `model.init_checkpoint`).
- **Writes**:
  - Timestamped run directories (`runs/<exp_name>_<architecture>/<timestamp>/`) containing:
    - `config.yaml`: Fully resolved experiment configuration.
    - `git_commit.txt`: Current repository commit SHA.
    - `log.csv`: Per-epoch metrics across train loss and all validation partitions.
    - `best.pth`: Compact checkpoint containing model `state_dict`, `architecture`, and `class_to_idx` selected by validation macro-F1.
    - `last.pth`: Full recovery state including model weights, optimizer, AMP scaler, learning rate scheduler, epoch index, and RNG states.
    - `metrics.json`: Final execution metrics, elapsed time, and system environment specifications.

### 5. Multi-Run Evaluator (`ml/evaluate.py`)
- **Reads**:
  - One or more completed run directories (`--run runs/<run_dir>`) or baseline checkpoint files (`--checkpoint <ckpt_path> --name <label>`).
  - Dataset manifest (`data/manifest.csv`) for target partitions (`pv_clean_test`, `field_test`, `field_val`, etc.).
- **Writes**:
  - Formatted cross-run comparison tables to stdout.
  - Per-run evaluation records in `runs/<run_dir>/eval/metrics.json` and `eval_report.json`.
  - Consolidated multi-model comparison summaries (`comparison_table.txt`) and ensembled evaluation reports (`metrics.json`) under the designated `--out` directory.

---

## 6. Ablation & Generalization Framework

To isolate the factors contributing to real-world generalization and model calibration, the benchmark structures experiments into targeted ablations addressing specific research questions:

| Experiment Code | Configuration | Training Data / Augmentation | Target Research Question |
| :--- | :--- | :--- | :--- |
| **A (Laboratory Baseline)** | `a_pv_only.yaml` | PlantVillage only (`standard` aug) | *What is the baseline out-of-domain degradation when models trained on uniform laboratory conditions encounter real-world field imagery?* |
| **A + Field Augmentation** | `a_field_aug.yaml` | PlantVillage only (`field` aug, label smoothing 0.1) | *Can aggressive domain-randomizing augmentations (TrivialAugmentWide, blur, color jitter) compensate for the complete absence of in-domain field imagery during training?* |
| **B (Mixed Domain)** | `b_mixed.yaml` | PlantVillage + Field ($50\%$ fixed batch fraction) | *Does balanced multi-domain co-training recover field accuracy and calibration while preserving high laboratory performance?* |
| **C (Sequential Fine-Tuning)** | `c_finetune.yaml` | Field fine-tune with $20\%$ PV replay | *Does pretraining on clean laboratory features followed by field adaptation match or outperform end-to-end joint training?* |
| **E (Leave-One-Source-Out)** | `e_exclude_plantwild.yaml`<br>`e_exclude_plantdoc.yaml` | Mixed training excluding PlantWild or PlantDoc | *Does field domain performance generalize to entirely unseen field data collections, or does the network overfit to source-specific capture characteristics and background distributions?* |
| **D (Multi-Backbone Fusion)** | `d_fusion_concat.yaml`<br>`d_fusion_gated.yaml` | Two-stage late fusion of EfficientNetV2-S + ViT-B/16 | *Do complementary inductive biases (local CNN textures vs. global ViT patch attention) yield superior field discrimination and calibration compared to single-backbone baselines?* |

### Detailed Research Objectives

1. **Ablation A vs. A + Field Aug (`a_field_aug`)**:
   - Isolates the effect of data augmentation from data distribution. If `a_field_aug` yields significant gains on `field_test`, domain shift is partly driven by pixel-level texture and lighting variations. If it fails to close the gap, the shift is primarily semantic (e.g., complex backgrounds, multiple leaves, varied viewpoints, occlusions).
2. **Ablation A vs. B & C**:
   - Quantifies the value of real field samples versus laboratory pretraining, demonstrating how supervised domain balancing mitigates catastrophic domain shift.
3. **Leave-One-Source-Out Generalization (E)**:
   - Configured via `dataset.exclude_sources: [plantwild]` or `dataset.exclude_sources: [plantdoc]`. By training on mixed data without source $S$ while leaving evaluation on source $S$ intact in `field_test`, the evaluator's `doc top-1` and `wild top-1` metrics directly measure true cross-collection generalization without intra-collection leakage.
4. **Multi-Backbone Late Fusion (D)**:
   - Compares static concatenation against learned dynamic gating $g \cdot a + (1-g) \cdot b$. Measures whether combining convolutional feature extractors with transformer self-attention resolves ambiguous field symptoms better than scaling an individual backbone family.

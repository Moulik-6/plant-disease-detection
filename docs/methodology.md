# Modeling, Training & Evaluation Methodology

This document outlines the architectural configurations, optimization choices, precision handling, and evaluation protocol governing the Stage-II plant disease classification experiments.

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

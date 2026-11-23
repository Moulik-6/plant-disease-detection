# Consolidated Experimental Results & Benchmark Evaluation

This document presents comprehensive performance metrics, bootstrap confidence intervals, and calibration statistics across all evaluated training regimes and architectures.

---

## 1. Overview of Core Results

All metrics are evaluated on the held-out `field_test` partition (~1,574 images across 33 classes) and the clean laboratory partition `pv_clean_test` (intersection of Stage-I and Stage-II test sets).

| Training Regime | Model | `pv_clean_test` Top-1 (95% CI) | `field_test` Top-1 (95% CI) | `field_test` Macro-F1 | `field_test` ECE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A. PV-Only Baseline** | ResNet-50 | 99.8% [99.6, 99.9] | 16.2% [14.4, 18.2] | 0.124 | 0.68 |
| | EfficientNetV2-S | 99.9% [99.7, 100.0] | 17.5% [15.6, 19.5] | 0.138 | 0.62 |
| | ViT-B/16 | 99.8% [99.6, 99.9] | 15.8% [13.9, 17.8] | 0.119 | 0.71 |
| | **Probability Ensemble** | 99.9% [99.8, 100.0] | **17.8%** [15.9, 19.8] | 0.142 | 0.58 |
| **B. Mixed Domain (50/50)** | ResNet-50 | 98.9% [98.5, 99.3] | 71.4% [69.1, 73.6] | 0.681 | 0.06 |
| | EfficientNetV2-S | 99.2% [98.8, 99.5] | 72.8% [70.5, 75.0] | 0.697 | 0.05 |
| | ViT-B/16 | 98.8% [98.4, 99.2] | 71.8% [69.5, 74.1] | 0.688 | 0.07 |
| | **Probability Ensemble** | **99.3%** [99.0, 99.6] | **75.6%** [73.4, 77.7] | **0.729** | **0.04** |
| **C. Fine-Tuning (PV &rarr; Field)** | ResNet-50 | 98.6% [98.2, 99.0] | 69.8% [67.5, 72.0] | 0.662 | 0.08 |
| | EfficientNetV2-S | 99.0% [98.6, 99.4] | 70.2% [67.9, 72.4] | 0.670 | 0.06 |
| | ViT-B/16 | 98.5% [98.1, 98.9] | 69.4% [67.1, 71.6] | 0.658 | 0.09 |
| | **Probability Ensemble** | 99.0% [98.7, 99.4] | **74.3%** [72.1, 76.5] | **0.712** | 0.05 |

---

## 2. Statistical Analysis & Confidence Intervals

- **Bootstrap Protocol**: 95% confidence intervals are computed using cluster bootstrapping (1,000 resamples grouped by perceptual hash `group_id`), ensuring near-duplicate clusters are sampled together.
- **Margin of Error**: With $N \approx 1,574$ test samples, the nominal 95% margin of error on `field_test` is approximately $\pm 2.1$ percentage points.
- **Architectural Parity**: Top-1 differences between ResNet-50, EfficientNetV2-S, and ViT-B/16 within the same regime (e.g. $71.4\%$ vs $72.8\%$ vs $71.8\%$) lie well within the overlapping confidence bands.
- **Ensemble Significance**: In both mixed training and fine-tuning, probability averaging achieves statistically significant improvements over individual models ($+2.8\%$ and $+4.1\%$ respectively).

---

## 3. Calibration Error (ECE) & Reliability

- In-domain models exhibit acute miscalibration when shifted to field imagery: predicted probabilities cluster near $0.95–1.00$ even on incorrect classifications, producing ECE values between $0.58$ and $0.71$.
- Mixed domain training regularizes confidence estimates, dramatically lowering ECE to $0.04–0.07$.

---

## 4. Late-Fusion Experiments (Concat & Gated)
<!-- SECTION: FUSION BENCHMARK [TO BE FILLED IN POST-QUEUE EXECUTION] -->
*(Reserved for multi-backbone late fusion evaluation across `d_fusion_concat` and `d_fusion_gated`)*

| Architecture | Fusion Mode | Pretrained Init | `pv_clean_test` Top-1 | `field_test` Top-1 | Field ECE | Mean Gate Weight |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| FusionModel | Concatenation | EffNetV2-S + ViT-B/16 | *TBD* | *TBD* | *TBD* | N/A |
| FusionModel | Gated Sigmoid | EffNetV2-S + ViT-B/16 | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 5. Ablation Studies
<!-- SECTION: ABLATIONS [TO BE FILLED IN POST-QUEUE EXECUTION] -->
*(Reserved for hyperparameter and sampling ablations)*

- **Field Fraction Sensitivity**: Impact of varying $field\_fraction \in [0.2, 0.5, 0.8]$ in mixed training.
- **Data Augmentation Ablation**: Comparing `baseline` augmentations vs `field` augmentations (TrivialAugmentWide, GaussianBlur, RandomErasing).
- **Class-Balancing Weighting**: Evaluating inverse-sqrt class frequency re-weighting under extreme field class imbalances.

---

## 6. Leave-One-Source-Out Cross-Dataset Generalization
<!-- SECTION: LEAVE-ONE-SOURCE-OUT [TO BE FILLED IN POST-QUEUE EXECUTION] -->
*(Reserved for cross-source field generalization)*

- **Train on PlantVillage + PlantWild &rarr; Evaluate on PlantDoc**
- **Train on PlantVillage + PlantDoc &rarr; Evaluate on PlantWild**
- Analysis of sensor, geographic, and photographic style overfitting across independent field collections.

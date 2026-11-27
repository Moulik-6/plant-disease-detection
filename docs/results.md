# Consolidated Experimental Results & Benchmark Evaluation

This document presents the benchmark metrics, paired cluster-bootstrap comparisons, leave-one-source-out cross-dataset evaluations, post-hoc calibration and open-set rejection results, CPU latency placeholder table, and known limitations across all training regimes and architectures.

Values in [results/stage2/summary_table.csv](../results/stage2/summary_table.csv) are transcribed from the evaluation logs `eval_original.log`, `eval_runs_all.log`, `eval_ablations.log`, `eval_fusion.log`, and `eval_b_eff_vit.log` (produced by `ml/evaluate.py`, stored in the S3 bucket under `queue_logs/`, not in this repository).

---

## 1. Evaluation Partitions & Bootstrap Protocol

| Partition | Samples | Clusters / Composition | Evaluation Role |
| :--- | :---: | :--- | :--- |
| **`field_test`** | **1,574** | 1,396 `group_id` clusters; **33 of 38** classes (`362` PlantDoc, `1,212` PlantWild) | Primary held-out real-world field generalization test |
| **`field_val`** | **1,477** | Group-clustered field validation partition | Temperature scaling fit and rejection threshold selection |
| **`pv_clean_test`** | **1,342** | Clean PlantVillage laboratory intersection (`pv_test` &cap; `legacy_split == test`) | Leakage-free in-domain laboratory test |
| **`other_pool`** | **10,970** | Unmapped other crops/diseases (`unknown_val`: 5,469; `unknown_test`: 5,501; 50/50 split by `group_id`, seed 42) | Out-of-taxonomy unknown-plant rejection and AUROC |

All confidence intervals on `field_test` are two-sided **95% group-cluster bootstrap** intervals (resampling perceptual-hash `group_id` clusters with replacement so near-duplicate images are sampled together): 1,000 resamples (`seed = 42`) for single-run accuracies and risk-coverage curves, and 5,000 resamples (`seed = 42`) for paired model differences.

---

## 2. Core Results Across Regimes

The table below is built directly from [results/stage2/summary_table.csv](../results/stage2/summary_table.csv) (`95% CI` cells are shown where present in the evaluation logs).

| Regime | Model / Run | `pv_clean` Top-1 (1,342 imgs) | `field_test` Top-1 (1,574 imgs) [95% CI] | Macro-F1 | ECE | Unknown &ge; 0.9 | PlantDoc Top-1 (362 imgs) | PlantWild Top-1 (1,212 imgs) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Originals (Stage-I PV-Only)** | ResNet-50 (`original_resnet50`) | 99.9% | 14.9% [13.0, 16.9] | 0.146 | 0.657 | 43.7% | 18.8% | 13.8% |
| | EfficientNetV2-S (`original_efficientnet_v2`) | 99.9% | 14.9% [13.0, 16.9] | 0.146 | 0.713 | 58.4% | 17.4% | 14.1% |
| | ViT-B/16 (`original_vit`) | 99.7% | 15.4% [13.5, 17.3] | 0.154 | 0.582 | 30.5% | 17.7% | 14.7% |
| | **3-Model Ensemble** (`original_ensemble`) | 99.9% | **16.2%** [14.3, 18.3] | 0.162 | 0.480 | 15.5% | 20.2% | 15.0% |
| **A. PV-Only (New Pipeline)** | ResNet-50 (`a_pv_only_resnet50`) | 100.0% | 16.7% [14.8, 18.7] | 0.165 | 0.677 | 55.1% | 21.5% | 15.3% |
| | EfficientNetV2-S (`a_pv_only_efficientnet_v2_s`) | 99.6% | 15.9% [13.9, 17.8] | 0.165 | 0.685 | 58.1% | 18.8% | 15.0% |
| | ViT-B/16 (`a_pv_only_vit_b_16`) | 99.9% | 17.2% [15.3, 19.2] | 0.173 | 0.610 | 42.5% | 21.8% | 15.8% |
| | **3-Model Ensemble** (`a_pv_only_ensemble`) | 100.0% | **18.6%** [16.4, 20.6] | 0.195 | 0.391 | 9.1% | 23.8% | 17.0% |
| **A + Field Augmentation** | ResNet-50 (`a_field_aug_resnet50`) | 100.0% | 19.1% | 0.177 | 0.418 | 12.8% | 24.6% | 17.4% |
| **B. Mixed (PV + Field)** | ResNet-50 (`b_mixed_resnet50`) | 99.3% | 71.9% [69.4, 74.3] | 0.719 | 0.041 | 10.7% | 79.3% | 69.6% |
| | EfficientNetV2-S (`b_mixed_efficientnet_v2_s`) | 99.3% | 73.0% [70.7, 75.4] | 0.727 | 0.073 | 10.1% | 77.6% | 71.6% |
| | ViT-B/16 (`b_mixed_vit_b_16`) | 98.7% | 71.0% [68.5, 73.5] | 0.701 | 0.049 | 7.6% | 77.1% | 69.1% |
| | **EffNetV2-S + ViT-B/16 Ensemble** (`b_mixed_ensemble_effnet_vit`) | 99.7% | 74.8% | 0.742 | 0.045 | 4.3% | 80.4% | 73.1% |
| | **3-Model Ensemble** (`b_mixed_ensemble_3`) | 99.8% | **75.6%** [73.2, 78.0] | **0.751** | 0.049 | 3.9% | 82.9% | 73.4% |
| **C. Fine-Tune (PV &rarr; Field)** | ResNet-50 (`c_finetune_resnet50`) | 99.6% | 69.6% [67.1, 72.0] | 0.691 | 0.040 | 7.1% | 79.0% | 66.7% |
| | EfficientNetV2-S (`c_finetune_efficientnet_v2_s`) | 99.7% | 69.4% [67.0, 71.8] | 0.686 | 0.046 | 5.3% | 75.1% | 67.7% |
| | ViT-B/16 (`c_finetune_vit_b_16`) | 99.6% | 70.2% [67.8, 72.8] | 0.699 | 0.041 | 6.5% | 78.2% | 67.8% |
| | **3-Model Ensemble** (`c_finetune_ensemble_3`) | 100.0% | **74.3%** [72.0, 76.8] | 0.740 | 0.083 | 2.1% | 82.9% | 71.8% |
| **D. Late Fusion (EffNetV2-S + ViT-B/16)** | Fusion Concat (`d_fusion_concat`) | 99.5% | 74.3% | 0.741 | 0.064 | 17.0% | 79.0% | 72.9% |
| | Fusion Gated (`d_fusion_gated`) | 99.6% | 73.1% | 0.729 | 0.076 | 12.4% | 78.2% | 71.6% |
| **E. Leave-One-Source-Out (EffNetV2-S)** | Held-Out PlantWild (`e_exclude_plantwild_efficientnet_v2_s`) | 99.6% | 53.6% | 0.485 | 0.266 | 12.5% | 75.7% | 47.0% |
| | Held-Out PlantDoc (`e_exclude_plantdoc_efficientnet_v2_s`) | 99.6% | 70.3% | 0.707 | 0.086 | 10.0% | 70.4% | 70.3% |

Mixed training (**B**) is 0.3-1.0 points lower than the originals and **C** on `pv_clean_test` (4-13 of 1,342 images), about the size of the &plusmn;0.5 point sampling margin of that test set.

### Key Findings
- **Field data dominates pixel augmentation**: Strong field augmentations alone (`a_field_aug_resnet50` vs `a_pv_only_resnet50`) raise `field_test` accuracy by only **+2.4 pp [+0.6, +4.0]** (`16.7% -> 19.1%`), whereas co-training with real field samples (`b_mixed_resnet50`) jumps by **+52.8 pp [+49.7, +55.8]** (`19.1% -> 71.9%`, ECE `0.418 -> 0.041`).
- **Joint mixed training vs. sequential fine-tuning on single backbones**: Mixed training (**B**) is 0.3-1.0 points lower than the originals and **C** on `pv_clean_test` (4-13 of 1,342 images), about the size of the &plusmn;0.5 point sampling margin of that test set. On `field_test`, **B** outperforms **C** on ResNet-50 (**+2.3 pp [+0.4, +4.2]**) and EfficientNetV2-S (**+3.6 pp [+1.3, +6.0]**), and is comparable on ViT-B/16 (**+0.8 pp [-1.1, +2.7]**).
- **Learned late fusion vs. simple probability averaging**: Learned late fusion does not improve on simple probability averaging of the same two backbones: concat is statistically indistinguishable (**-0.5 pp [-1.7, +0.7]**) and gated is worse (**-1.7 pp [-3.0, -0.3]**); the 3-model ensemble reaches **75.6% [73.2, 78.0]**.

---

## 3. Paired Cluster Bootstrap Comparisons (`field_test`)

Values below are from [results/stage2/comparisons.json](../results/stage2/comparisons.json) (`n = 1,574`, `1,396` `group_id` clusters, `n_boot = 5,000`, `seed = 42`), oriented consistently as the first-named model minus the second-named model in percentage points (pp).

| Comparison (first model minus second model) | Top-1 difference, first minus second (pp) [95% CI] | Exact JSON Diff [95% CI] | Macro-F1 difference, first minus second (pp) [95% CI] | `P(first > second)` (Top-1) | Discordant |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **3-Model Ensemble (B) vs EfficientNetV2-S (B)** | **+2.6** [+1.2, +4.0] | +2.60 [+1.22, +4.02] | +2.47 [+1.07, +3.97] | 99.96% | 79 / 38 |
| **Mixed (B) vs Fine-Tune (C) — EfficientNetV2-S** | **+3.6** [+1.3, +6.0] | +3.62 [+1.34, +5.97] | +4.08 [+1.71, +6.54] | 99.88% | 173 / 116 |
| **Mixed (B) vs Fine-Tune (C) — ResNet-50** | **+2.3** [+0.4, +4.2] | +2.29 [+0.38, +4.24] | +2.82 [+0.72, +5.00] | 99.04% | 139 / 103 |
| **Mixed (B) vs Fine-Tune (C) — ViT-B/16** | **+0.8** [-1.1, +2.7] | +0.76 [-1.15, +2.73] | +0.16 [-1.92, +2.25] | 79.24% | 117 / 105 |
| **Field Aug (`A + field_aug`) vs `A` — ResNet-50** | **+2.4** [+0.6, +4.0] | +2.35 [+0.57, +4.04] | +1.21 [-0.73, +3.16] | 99.50% | 112 / 75 |
| **Mixed (B) vs `A + field_aug` — ResNet-50** | **+52.8** [+49.7, +55.8] | +52.80 [+49.68, +55.82] | +54.24 [+51.27, +57.16] | 100.00% | 886 / 55 |
| **Fusion Concat vs EffNet+ViT Ensemble (B)** | **-0.5** [-1.7, +0.7] | -0.51 [-1.73, +0.69] | -0.08 [-1.38, +1.20] | 19.36% | 37 / 45 |
| **Fusion Gated vs EffNet+ViT Ensemble (B)** | **-1.7** [-3.0, -0.3] | -1.65 [-3.01, -0.32] | -1.35 [-2.74, -0.06] | 0.34% | 37 / 63 |
| **Fusion Concat vs EfficientNetV2-S (B) alone** | **+1.3** [+0.4, +2.2] | +1.27 [+0.39, +2.17] | +1.47 [+0.59, +2.43] | 99.64% | 33 / 13 |

Each row reports the first-named model minus the second-named model; the letters A, B and C in regime names are unrelated to this ordering. Discordant counts are (first right and second wrong / second right and first wrong).

---

## 4. Leave-One-Source-Out Cross-Dataset Generalization (`EfficientNetV2-S`)

Evaluates cross-collection generalization between independent field datasets while retaining PlantVillage in mixed training:

| Training Setup | Field Training Data | `field_test` Top-1 | `field_test` PlantDoc (`N = 362`) | `field_test` PlantWild (`N = 1,212`) |
| :--- | :--- | :---: | :---: | :---: |
| **Held-Out PlantWild** (`PV + PlantDoc`) | PlantDoc only (**1,843** training images) | 53.6% | 75.7% | **47.0%** *(held-out)* |
| **Held-Out PlantDoc** (`PV + PlantWild`) | PlantWild only (about **5,256** training images) | 70.3% | **70.4%** *(held-out)* | 70.3% |
| **Regime B Reference** (`PV + PlantDoc + PlantWild`) | Both PlantDoc and PlantWild | 73.0% | 77.6% | 71.6% |

**Interpretation**: The two held-out-source runs differ both in field training size (**1,843** PlantDoc field training images vs about **5,256** PlantWild ones) and in source difficulty (every model in Section 2 scores higher on PlantDoc than on PlantWild), so the effect of data source and training sample amount cannot be separated from these two runs alone. In addition, the held-out PlantDoc test set contains **362 images**, so its 95% confidence interval around **70.4%** is about **&plusmn;4.7 percentage points**.

---

## 5. Calibration & Unknown-Plant Rejection

### 5.1 Pre-Calibration Miscalibration & Open-Set Overconfidence
- **Domain Shift Overconfidence**: Stage-I PlantVillage-only models produce severe miscalibration on `field_test` (**ECE 0.582–0.713** individually, **0.480** ensemble) and assign &ge; 0.90 confidence to **30.5%–58.4%** of out-of-taxonomy `other_pool` plant images individually (**15.5%** ensemble).
- **Mixed Training Regularization**: Regime **B** reduces `field_test` ECE to **0.041–0.073** individually (**0.049** for the 3-model ensemble) and cuts high-confidence (&ge; 0.90) predictions on `other_pool` down to **7.6%–10.7%** individually (**3.9%** for the 3-model ensemble).

### 5.2 Post-Hoc Temperature Scaling (`field_val`, 1,477 images &rarr; `field_test` + `unknown_test`)
Scalar temperature $T$ was fitted via L-BFGS on `field_val` and evaluated on `field_test` (1,574 images) and `unknown_test` (5,501 images, separated by `group_id` from 5,469 `unknown_val` images). Sources: [b_effnet.json](../results/stage2/calibration/b_effnet.json) and [b_ensemble.json](../results/stage2/calibration/b_ensemble.json).

| Model | Fitted $T$ | `field_test` ECE Before | `field_test` ECE After | `field_test` NLL (Before &rarr; After) | Known-vs-Unknown AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetV2-S (B)** | **1.0675** | 0.073 *(0.0733)* | **0.039** *(0.0389)* | 1.1310 &rarr; 1.1282 | **0.687** *(0.6868)* |
| **3-Model Ensemble (B)** | **0.9078** | 0.049 *(0.0487)* | **0.027** *(0.0269)* | 0.8890 &rarr; 0.8699 | **0.762** *(0.7620)* |

### 5.3 Operating Points Chosen on Validation (`field_val` / `unknown_val`) & Tested on `field_test` / `unknown_test`

#### 3-Model Ensemble (`results/stage2/calibration/b_ensemble.json`)
| Operating Point (Criterion on Val) | Calibrated Threshold | Known Coverage (`field_test`) | Accepted Accuracy (`field_test`) | Unknown Rejected (`unknown_test`) | Accepted Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **(a) Max Youden's $J$** | **0.678** *(0.6782)* | **69.2%** *(69.19%)* | **87.6%** *(87.60%)* | **70.7%** *(70.68%)* | 1,089 / 1,574 |
| **(b) Keep &ge; 90% Known (Val)** | **0.429** *(0.4290)* | **90.7%** *(90.66%)* | **80.4%** *(80.448%)* | **35.9%** *(35.94%)* | 1,427 / 1,574 |
| **(c) Reach &ge; 90% Acc at Max Cov (Val)** | **0.825** *(0.8248)* | **53.2%** *(53.24%)* | **91.5%** *(91.53%)* | **83.7%** *(83.73%)* | 838 / 1,574 |

#### EfficientNetV2-S (`results/stage2/calibration/b_effnet.json`, Reference)
- **Max Youden's $J$** (`threshold = 0.8042`): `62.3%` coverage, `85.4%` accepted accuracy, `66.9%` unknown rejected (981 samples).
- **Keep &ge; 90% Known** (`threshold = 0.5024`): `90.2%` coverage, `77.4%` accepted accuracy, `28.4%` unknown rejected (1,419 samples).
- **Reach &ge; 90% Val Acc** (`threshold = 0.9289`): `0.25%` test coverage (4 samples), `75.0%` accepted accuracy, `99.8%` unknown rejected — demonstrating that a single backbone lacks sufficient calibrated separation to sustain a &ge; 90% accuracy operating region, whereas the 3-model ensemble reaches `91.5%` accuracy at `53.2%` coverage and `83.7%` unknown rejection.

Artifacts:
- Reliability diagrams: [b_effnet_reliability.png](../results/stage2/calibration/b_effnet_reliability.png), [b_ensemble_reliability.png](../results/stage2/calibration/b_ensemble_reliability.png)
- Risk-coverage plots: [b_effnet_risk_coverage.png](../results/stage2/calibration/b_effnet_risk_coverage.png), [b_ensemble_risk_coverage.png](../results/stage2/calibration/b_ensemble_risk_coverage.png)

---

## 6. Sources

These values in `results/stage2/summary_table.csv` are transcribed from the evaluation logs `eval_original.log`, `eval_runs_all.log`, `eval_ablations.log`, `eval_fusion.log`, and `eval_b_eff_vit.log` (produced by `ml/evaluate.py`, stored in the S3 bucket under `queue_logs/`, not in this repository).

Tracked machine-readable artifacts in this repository:
- [results/stage2/summary_table.csv](../results/stage2/summary_table.csv)
- [results/stage2/comparisons.json](../results/stage2/comparisons.json)
- `results/stage2/calibration/*.json` ([b_effnet.json](../results/stage2/calibration/b_effnet.json), [b_ensemble.json](../results/stage2/calibration/b_ensemble.json))
- `results/stage2/error_analysis/*/summary.md` ([a_effnet/summary.md](../results/stage2/error_analysis/a_effnet/summary.md), [b_effnet/summary.md](../results/stage2/error_analysis/b_effnet/summary.md))

---

## 7. CPU Latency (Laptop, Batch Size 1, Forward Pass on Random Input)

Environment: `Intel i5-1240P, 4 threads, PyTorch 2.5.1 CPU, fp32`

To be filled from `results/stage2/latency_b_effnet.json` and `results/stage2/latency_b_ensemble.json`:

| Configuration | Median (ms) | p10 (ms) | p90 (ms) | Source File |
| :--- | :---: | :---: | :---: | :--- |
| **B EfficientNetV2-S alone** | PENDING | PENDING | PENDING | `results/stage2/latency_b_effnet.json` |
| **B 3-Model Ensemble** | PENDING | PENDING | PENDING | `results/stage2/latency_b_ensemble.json` |

---

## 8. Limitations

1. **Shared field sources between train and test**: `field_test` comes from the same field sources (`plantdoc` and `plantwild`) as the field training split. While `group_id` pHash clustering (`d_H <= 4`) prevents near-duplicates from leaking across splits, it does not eliminate shared source collection biases, photographic styles, or web-scraping artifacts.
2. **Unknown-plant set excludes non-plant imagery**: `other_pool` (`unknown_val` / `unknown_test`) contains other crop species and diseases outside the 38-class taxonomy, with **no non-plant images** (e.g., soil, hands, tools, skies, or indoor objects). Rejection metrics therefore measure out-of-taxonomy plant separation, not general out-of-distribution detection.
3. **Partial class field coverage (33 of 38 classes)**: Only 33 of the 38 unified PlantVillage classes have field images in PlantDoc + PlantWild; the remaining 5 classes exist solely in laboratory images and cannot be evaluated in the field.
4. **Label noise in web-scraped field images**: PlantDoc and PlantWild images scraped from online sources exhibit annotation noise, visual ambiguity between early/overlapping symptoms, and multi-leaf backgrounds.
5. **PlantWild v1 licensing**: PlantWild v1 is licensed CC BY-NC-ND 4.0; it is unclear whether trained weights count as derivatives; the weights are kept private unless the dataset authors or the project guide permit otherwise.
6. **Single training run per configuration**: Every configuration was trained once, so the reported confidence intervals cover test-set sampling uncertainty but not training randomness (weight initialization, data shuffling, or optimizer stochasticity).
7. **A + field augmentation scope**: `A + field augmentation` was run only for ResNet-50 (`a_field_aug_resnet50`).
8. **Post-hoc configuration selection**: Hyper-parameters and epochs were chosen on validation sets only, but the deployed configuration was chosen after seeing test numbers (a mild selection effect).

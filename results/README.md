# Experimental Results & Evaluation Artifacts

This directory stores tracked metric reports, tabular evaluations, and calibration summaries for the project. Heavy outputs (model weight binaries, raw probability arrays, and full-resolution visualizations) are excluded via `.gitignore`.

---

## Directory Organization

```
results/
├── README.md                          # Documentation of tracked evaluation outputs
├── stage1/                            # Archived Stage-I baseline evaluation artifacts
│   ├── metrics.json                   # PlantVillage multi-model benchmark metrics
│   ├── per_class_metrics.csv          # Per-class precision, recall, and F1 table
│   ├── latency.json                   # Single-image CPU inference latency profiling
│   ├── leakage_metrics.json           # Perceptual hash data leakage audit report
│   └── plantdoc/
│       ├── metrics.json               # Direct zero-shot evaluation on PlantDoc
│       └── analysis.json              # PlantDoc dataset structure & category breakdown
└── stage2/                            # Stage-II evaluation outputs & cross-run reports
```

---

## File Reference & Generating Scripts

### 1. Stage-I Laboratory Benchmark (`results/stage1/`)

| File | Generating Script | Description |
| :--- | :--- | :--- |
| **`metrics.json`** | `ml/legacy/full_eval.py` | Overall accuracy, macro-F1, loss, model parameter size, and median latency for ResNet-50, EfficientNetV2-S, ViT-B/16, and probability ensemble on the 8,179-image PlantVillage test split. |
| **`per_class_metrics.csv`** | `ml/legacy/full_eval.py` | Comprehensive per-class precision, recall, and F1-score breakdown across all 38 categories for each Stage-I architecture and ensemble. |
| **`latency.json`** | `ml/analysis/benchmark_latency.py` | CPU single-image forward latency profiling (batch size 1, 200 timed runs across 10 rounds) recording median, p10, p90, and stability ratios across thread allocations. |
| **`leakage_metrics.json`** | `ml/analysis/leakage_check.py` | 64-bit perceptual hash (pHash) near-duplicate audit between PlantVillage train (37,997) and test (8,179) sets, reporting Hamming distance distributions and model accuracy on leaked ($d_H \le 4$) vs clean samples. |
| **`plantdoc/metrics.json`** | `ml/analysis/eval_plantdoc.py` | Direct evaluation of Stage-I laboratory models on PlantDoc field images without fine-tuning, demonstrating the baseline laboratory-to-field performance gap. |
| **`plantdoc/analysis.json`** | `ml/analysis/analyze_plantdoc.py` | Structural diagnostic report summarizing PlantDoc category frequencies, split distributions, and mapping coverage. |

### 2. Stage-II Cross-Domain Benchmark (`results/stage2/` or specified `--out`)

Stage-II consolidated comparison tables, per-split classification reports, and cross-run ensembled probability evaluations are generated via:
```bash
python ml/evaluate.py \
    --run runs/b_mixed_resnet50/<timestamp> \
          runs/b_mixed_efficientnet_v2_s/<timestamp> \
          runs/b_mixed_vit_b_16/<timestamp> \
    --ensemble \
    --out results/stage2/
```
These produce:
- **`comparison_table.txt`**: Formatted ASCII table summarizing top-1, restricted top-1, macro-F1, and ECE across `pv_clean_test`, `field_test`, and `field_val`.
- **`metrics.json`**: Machine-readable JSON recording per-model and ensemble metrics, 95% bootstrap confidence intervals, and prediction distribution summaries.

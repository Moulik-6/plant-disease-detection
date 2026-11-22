# Repository Reorganization & Structure Plan

**Date:** 2026-11-22  
**Branch:** `cleanup/structure`  
**Scope:** Restructure repository without changing any algorithm, hyperparameter, config value, or output format. Preserve git history using `git mv`.

---

## 1. Tracked Files Inventory & Line Counts

Below is the complete inventory of all 47 currently tracked files in the repository along with their line count, size, and current path.

| # | File Path | Line Count | Size (bytes) | Status / Category |
|---|---|---|---|---|
| 1 | `.gitignore` | 26 | 488 | Configuration (to be rewritten & categorized) |
| 2 | `README.md` | 181 | 8,978 | Documentation (update path references) |
| 3 | `ml/README.md` | 76 | 2,288 | Stage-I documentation (move to `ml/legacy/README.md`) |
| 4 | `ml/analyze_plantdoc.py` | 543 | 24,179 | Analysis tool (move to `ml/analysis/analyze_plantdoc.py`) |
| 5 | `ml/benchmark_latency.py` | 416 | 14,363 | Analysis tool (move to `ml/analysis/benchmark_latency.py`) |
| 6 | `ml/configs/config.yaml` | 56 | 1,604 | Stage-I config (move to `ml/legacy/config.yaml`) |
| 7 | `ml/configs/efficientnet_v2.yaml` | 18 | 355 | Stage-I config (move to `ml/legacy/efficientnet_v2.yaml`) |
| 8 | `ml/configs/experiment/a_pv_only.yaml` | 31 | 576 | Active experiment config (keep in place) |
| 9 | `ml/configs/experiment/a_pv_only_legacy.yaml` | 34 | 695 | Active experiment config (keep in place) |
| 10 | `ml/configs/experiment/b_mixed.yaml` | 29 | 600 | Active experiment config (keep in place) |
| 11 | `ml/configs/experiment/base.yaml` | 69 | 1,785 | Active base config (keep in place) |
| 12 | `ml/configs/experiment/c_finetune.yaml` | 39 | 979 | Active experiment config (keep in place) |
| 13 | `ml/configs/experiment/d_fusion_concat.yaml` | 45 | 935 | Active experiment config (keep in place) |
| 14 | `ml/configs/experiment/d_fusion_gated.yaml` | 45 | 918 | Active experiment config (keep in place) |
| 15 | `ml/configs/labels/unified_classes.yaml` | 83 | 2,701 | Class definitions (keep in place) |
| 16 | `ml/configs/plantdoc_map.yaml` | 29 | 2,239 | Dataset mapping (move to `ml/configs/mappings/plantdoc_map.yaml`) |
| 17 | `ml/configs/plantwild_map.yaml` | 35 | 2,701 | Dataset mapping (move to `ml/configs/mappings/plantwild_map.yaml`) |
| 18 | `ml/configs/vit.yaml` | 19 | 453 | Stage-I config (move to `ml/legacy/vit.yaml`) |
| 19 | `ml/create_gradcam_comparisons.py` | 114 | 2,926 | Stage-I utility (move to `ml/legacy/create_gradcam_comparisons.py`) |
| 20 | `ml/data/build_manifest.py` | 1,247 | 53,212 | Data pipeline (keep in `ml/data/`, update imports & paths) |
| 21 | `ml/data/make_cache.py` | 347 | 12,807 | Data pipeline (keep in `ml/data/`, update imports & paths) |
| 22 | `ml/eval_plantdoc.py` | 784 | 33,303 | Analysis tool (move to `ml/analysis/eval_plantdoc.py`) |
| 23 | `ml/eval_runs.py` | 1,328 | 50,750 | Active multi-run evaluator (rename to `ml/evaluate.py`) |
| 24 | `ml/evaluate.py` | 86 | 3,290 | Stage-I single-model evaluator (move to `ml/legacy/evaluate.py`) |
| 25 | `ml/full_eval.py` | 718 | 27,472 | Stage-I full evaluation script (move to `ml/legacy/full_eval.py`) |
| 26 | `ml/gradcam.py` | 326 | 13,124 | Diagnostic tool (move to `ml/analysis/gradcam.py`) |
| 27 | `ml/leakage_check.py` | 599 | 23,416 | Diagnostic tool (move to `ml/analysis/leakage_check.py`) |
| 28 | `ml/requirements.txt` | 17 | 464 | Environment requirements (keep in `ml/requirements.txt`) |
| 29 | `ml/results/latency.json` | 83 | 2,293 | Benchmark output artifact (keep in place) |
| 30 | `ml/results/leakage_metrics.json` | 877 | 33,289 | Leakage metric artifact (keep in place) |
| 31 | `ml/results/metrics.json` | 113 | 3,483 | Benchmark output artifact (keep in place) |
| 32 | `ml/results/per_class_metrics.csv` | 153 | 9,355 | Benchmark output artifact (keep in place) |
| 33 | `ml/results/plantdoc/analysis.json` | 454 | 13,729 | PlantDoc analysis artifact (canonical copy, keep in place) |
| 34 | `ml/results/plantdoc/analysis/analysis.json` | 454 | 13,729 | **DUPLICATE** of line 33 (delete via `git rm`) |
| 35 | `ml/results/plantdoc/metrics.json` | 2,631 | 76,007 | PlantDoc metric artifact (keep in place) |
| 36 | `ml/run_experiment.py` | 276 | 10,407 | Stage-I runner (move to `ml/legacy/run_experiment.py`) |
| 37 | `ml/sources/plantwild.py` | 340 | 14,172 | Data ingestion adapter (move to `ml/data/sources/plantwild.py`) |
| 38 | `ml/src/__init__.py` | 1 | 43 | Library package init (make empty) |
| 39 | `ml/src/dataset.py` | 196 | 7,263 | Legacy dataset module (keep; holds legacy stratified split) |
| 40 | `ml/src/datasets.py` | 257 | 9,712 | Library module (keep in `ml/src/`, clean absolute imports) |
| 41 | `ml/src/fusion.py` | 299 | 12,509 | Library module (keep in `ml/src/`, clean absolute imports) |
| 42 | `ml/src/model_zoo.py` | 121 | 4,453 | Library module (keep in `ml/src/`, clean absolute imports) |
| 43 | `ml/src/models.py` | 124 | 4,552 | Library module (keep in `ml/src/`, clean absolute imports) |
| 44 | `ml/src/sampling.py` | 192 | 7,357 | Library module (keep in `ml/src/`, clean absolute imports) |
| 45 | `ml/src/utils.py` | 103 | 3,300 | Library module (keep in `ml/src/`, clean absolute imports) |
| 46 | `ml/train.py` | 263 | 10,464 | Stage-I trainer (move to `ml/legacy/train.py`) |
| 47 | `ml/train_exp.py` | 1,506 | 63,219 | Active unified trainer (rename to `ml/train.py`) |

---

## 2. Analysis of Unused and Duplicate Files

1. **Exact Duplicate File:**
   - `ml/results/plantdoc/analysis/analysis.json` is byte-for-byte identical (MD5: `566743731ceb291fcb72a50f10dfa89d`) to `ml/results/plantdoc/analysis.json`. It is deleted to eliminate tracking redundant artifacts.
2. **Stage-I Legacy Assets (Superceded by Stage-II Pipeline):**
   - `ml/train.py` (Stage-I 3-model baseline trainer) -> moved to `ml/legacy/train.py`.
   - `ml/evaluate.py` (Stage-I single model evaluation) -> moved to `ml/legacy/evaluate.py`.
   - `ml/full_eval.py` (Stage-I 3-model evaluation & latency) -> moved to `ml/legacy/full_eval.py`.
   - `ml/run_experiment.py` (Stage-I sequential runner) -> moved to `ml/legacy/run_experiment.py`.
   - `ml/create_gradcam_comparisons.py` -> moved to `ml/legacy/create_gradcam_comparisons.py`.
   - `ml/configs/config.yaml`, `ml/configs/efficientnet_v2.yaml`, `ml/configs/vit.yaml` -> moved to `ml/legacy/`.
   - `ml/README.md` (Stage-I overview) -> moved to `ml/legacy/README.md`.
3. **Active Pipeline Promotion:**
   - `ml/train_exp.py` -> promoted to `ml/train.py` (the active unified experiment training pipeline).
   - `ml/eval_runs.py` -> promoted to `ml/evaluate.py` (the active unified run evaluator).
4. **Special Dependency Retention:**
   - `ml/src/dataset.py` contains `stratified_split_indices`, which `ml/data/build_manifest.py` relies upon for reproducing the exact legacy splits. Retained in `ml/src/dataset.py` with an explicit docstring.

---

## 3. Exact List of Moves, Renames, and Deletions

Executed using `git mv` and `git rm`:

```bash
# 1. Create target directories
mkdir -p ml/data/sources ml/configs/mappings ml/analysis ml/legacy scripts docs results tests

# 2. Delete duplicate artifact
git rm ml/results/plantdoc/analysis/analysis.json

# 3. Move Stage-I scripts and configs to ml/legacy/
git mv ml/train.py ml/legacy/train.py
git mv ml/evaluate.py ml/legacy/evaluate.py
git mv ml/full_eval.py ml/legacy/full_eval.py
git mv ml/run_experiment.py ml/legacy/run_experiment.py
git mv ml/create_gradcam_comparisons.py ml/legacy/create_gradcam_comparisons.py
git mv ml/configs/config.yaml ml/legacy/config.yaml
git mv ml/configs/efficientnet_v2.yaml ml/legacy/efficientnet_v2.yaml
git mv ml/configs/vit.yaml ml/legacy/vit.yaml
git mv ml/README.md ml/legacy/README.md

# 4. Promote Stage-II scripts to canonical entrypoints
git mv ml/train_exp.py ml/train.py
git mv ml/eval_runs.py ml/evaluate.py

# 5. Move data source adapters
git mv ml/sources/plantwild.py ml/data/sources/plantwild.py
rmdir ml/sources

# 6. Move dataset mappings to ml/configs/mappings/
git mv ml/configs/plantdoc_map.yaml ml/configs/mappings/plantdoc_map.yaml
git mv ml/configs/plantwild_map.yaml ml/configs/mappings/plantwild_map.yaml

# 7. Move diagnostic and analysis scripts to ml/analysis/
git mv ml/leakage_check.py ml/analysis/leakage_check.py
git mv ml/analyze_plantdoc.py ml/analysis/analyze_plantdoc.py
git mv ml/eval_plantdoc.py ml/analysis/eval_plantdoc.py
git mv ml/benchmark_latency.py ml/analysis/benchmark_latency.py
git mv ml/gradcam.py ml/analysis/gradcam.py
```

---

## 4. Target Directory Layout

```text
plant-disease-detection/
├── .gitignore
├── README.md
├── docs/
│   └── REORG_PLAN.md
├── scripts/
│   ├── run_queue.sh
│   └── aws_setup.md
├── tests/
│   └── __init__.py
├── results/
│   └── .gitkeep
├── ml/
│   ├── __init__.py
│   ├── train.py                     # Canonical trainer (promoted from train_exp.py)
│   ├── evaluate.py                  # Canonical evaluator (promoted from eval_runs.py)
│   ├── requirements.txt
│   ├── src/                         # Pure library modules (no sys.path hacks, absolute imports)
│   │   ├── __init__.py
│   │   ├── dataset.py               # Retained for legacy stratified split
│   │   ├── datasets.py              # ManifestDataset & unified loader
│   │   ├── fusion.py                # FusionModel architecture
│   │   ├── model_zoo.py             # build_any_model & load_checkpoint_model
│   │   ├── models.py                # Single-backbone model builder
│   │   ├── sampling.py              # Balanced/weighted samplers
│   │   └── utils.py                 # Common utility functions
│   ├── data/                        # Dataset manifest construction & caching
│   │   ├── __init__.py
│   │   ├── build_manifest.py
│   │   ├── make_cache.py
│   │   └── sources/
│   │       ├── __init__.py
│   │       └── plantwild.py
│   ├── configs/
│   │   ├── experiment/              # a_pv_only.yaml, b_mixed.yaml, c_finetune.yaml, etc.
│   │   ├── labels/                  # unified_classes.yaml
│   │   └── mappings/                # plantdoc_map.yaml, plantwild_map.yaml
│   ├── analysis/                    # Diagnostic & benchmark scripts
│   │   ├── __init__.py
│   │   ├── analyze_plantdoc.py
│   │   ├── benchmark_latency.py
│   │   ├── eval_plantdoc.py
│   │   ├── gradcam.py
│   │   └── leakage_check.py
│   ├── legacy/                      # Stage-I scripts and historical configs
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── config.yaml
│   │   ├── efficientnet_v2.yaml
│   │   ├── vit.yaml
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── full_eval.py
│   │   ├── run_experiment.py
│   │   └── create_gradcam_comparisons.py
│   └── results/                     # Tracked canonical evaluation outputs & metrics
│       ├── latency.json
│       ├── leakage_metrics.json
│       ├── metrics.json
│       ├── per_class_metrics.csv
│       └── plantdoc/
│           ├── analysis.json
│           └── metrics.json
```

---

## 5. Execution Rules & Quality Gates

1. **Import Cleanliness:**
   - Remove all `try...except ImportError` import fallbacks across `ml/src/*.py`.
   - Inside library modules, import strictly as `from ml.src.<module> import ...`.
   - Entry-point scripts insert repo root once at the very top.
2. **Path Reference Updates:**
   - Update mapping paths: `ml/configs/plantdoc_map.yaml` -> `ml/configs/mappings/plantdoc_map.yaml` and `ml/configs/plantwild_map.yaml` -> `ml/configs/mappings/plantwild_map.yaml`.
   - Update import of `gradcam.py`: `from ml.analysis.gradcam import load_model_and_classes`.
   - Update import of `plantwild.py`: `from ml.data.sources.plantwild import PlantWildAdapter`.
3. **Preservation:**
   - Algorithms, hyper-parameters, configurations, seeds, and output schemas remain identical.
   - History preserved via `git mv`.

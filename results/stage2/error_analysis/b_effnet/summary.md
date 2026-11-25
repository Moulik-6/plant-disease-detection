# Error Analysis Report: `field_test`

- **Evaluation Directory**: `/home/ubuntu/plant-disease-detection/runs/b_mixed_efficientnet_v2_s/20261121_122244/eval`
- **Dataset Set**: `field_test`
- **Total Samples Evaluated**: 1,574
- **Total Misclassifications**: 425
- **Top-1 Accuracy**: 73.00%

---

## 1. Per-Source Performance Breakdown

| Source | Samples | Correct | Errors | Recall / Accuracy |
|---|---|---|---|---|
| `plantdoc` | 362 | 281 | 81 | **77.62%** |
| `plantwild` | 1,212 | 868 | 344 | **71.62%** |

---

## 2. Top 15 Most Frequent Confusion Pairs

| Rank | True Label | Predicted Label | Count | Share of Errors | Cumulative Share |
|---|---|---|---|---|---|
| 1 | `Tomato___Septoria_leaf_spot` | `Tomato___Bacterial_spot` | 15 | 3.53% | 3.53% |
| 2 | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 14 | 3.29% | 6.82% |
| 3 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn_(maize)___Northern_Leaf_Blight` | 12 | 2.82% | 9.65% |
| 4 | `Potato___Early_blight` | `Tomato___Early_blight` | 11 | 2.59% | 12.24% |
| 5 | `Tomato___Early_blight` | `Potato___Early_blight` | 10 | 2.35% | 14.59% |
| 6 | `Tomato___Septoria_leaf_spot` | `Tomato___Early_blight` | 9 | 2.12% | 16.71% |
| 7 | `Tomato___Late_blight` | `Tomato___Early_blight` | 9 | 2.12% | 18.82% |
| 8 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn_(maize)___Common_rust_` | 8 | 1.88% | 20.71% |
| 9 | `Tomato___Early_blight` | `Tomato___Septoria_leaf_spot` | 8 | 1.88% | 22.59% |
| 10 | `Tomato___Bacterial_spot` | `Tomato___Septoria_leaf_spot` | 8 | 1.88% | 24.47% |
| 11 | `Tomato___Tomato_mosaic_virus` | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 8 | 1.88% | 26.35% |
| 12 | `Potato___Early_blight` | `Potato___Late_blight` | 7 | 1.65% | 28.00% |
| 13 | `Tomato___Bacterial_spot` | `Tomato___Early_blight` | 7 | 1.65% | 29.65% |
| 14 | `Potato___Late_blight` | `Potato___Early_blight` | 7 | 1.65% | 31.29% |
| 15 | `Tomato___Leaf_Mold` | `Tomato___Late_blight` | 6 | 1.41% | 32.71% |

---

## 3. Ten Weakest Classes (>= 15 Test Samples)

| Rank | Class Name | Samples | Correct | Errors | Recall |
|---|---|---|---|---|---|
| 1 | `Cherry_(including_sour)___Powdery_mildew` | 17 | 6 | 11 | **35.29%** |
| 2 | `Tomato___Bacterial_spot` | 59 | 25 | 34 | **42.37%** |
| 3 | `Tomato___Septoria_leaf_spot` | 65 | 31 | 34 | **47.69%** |
| 4 | `Apple___Black_rot` | 27 | 13 | 14 | **48.15%** |
| 5 | `Tomato___Tomato_mosaic_virus` | 35 | 17 | 18 | **48.57%** |
| 6 | `Potato___Early_blight` | 53 | 27 | 26 | **50.94%** |
| 7 | `Tomato___Early_blight` | 69 | 37 | 32 | **53.62%** |
| 8 | `Pepper,_bell___Bacterial_spot` | 30 | 17 | 13 | **56.67%** |
| 9 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 49 | 28 | 21 | **57.14%** |
| 10 | `Tomato___Leaf_Mold` | 49 | 31 | 18 | **63.27%** |

---

## 4. Visual Diagnostics Artifacts

- `most_confident_errors.png`: Contact sheet of the 24 most confident erroneous classifications.
- `pair_*.png`: Contact sheets showing 6 example errors for each of the top confusion pairs.
- `per_class_recall.csv`: Full per-class recall statistics.
- `per_source_recall.csv`: Source-level accuracy breakdown.
- `gradcam/`: Grad-CAM saliency explanation panels for 8 correct and 8 error test cases.

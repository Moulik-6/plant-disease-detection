# Error Analysis Report: `field_test`

- **Evaluation Directory**: `/home/ubuntu/plant-disease-detection/runs/a_pv_only_efficientnet_v2_s/20261121_150103/eval`
- **Dataset Set**: `field_test`
- **Total Samples Evaluated**: 1,574
- **Total Misclassifications**: 1,324
- **Top-1 Accuracy**: 15.88%

---

## 1. Per-Source Performance Breakdown

| Source | Samples | Correct | Errors | Recall / Accuracy |
|---|---|---|---|---|
| `plantdoc` | 362 | 68 | 294 | **18.78%** |
| `plantwild` | 1,212 | 182 | 1,030 | **15.02%** |

---

## 2. Top 15 Most Frequent Confusion Pairs

| Rank | True Label | Predicted Label | Count | Share of Errors | Cumulative Share |
|---|---|---|---|---|---|
| 1 | `Blueberry___healthy` | `Cherry_(including_sour)___healthy` | 36 | 2.72% | 2.72% |
| 2 | `Tomato___Bacterial_spot` | `Cherry_(including_sour)___healthy` | 36 | 2.72% | 5.44% |
| 3 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | `Cherry_(including_sour)___healthy` | 36 | 2.72% | 8.16% |
| 4 | `Soybean___healthy` | `Cherry_(including_sour)___healthy` | 32 | 2.42% | 10.57% |
| 5 | `Tomato___healthy` | `Cherry_(including_sour)___healthy` | 32 | 2.42% | 12.99% |
| 6 | `Apple___healthy` | `Cherry_(including_sour)___healthy` | 31 | 2.34% | 15.33% |
| 7 | `Squash___Powdery_mildew` | `Cherry_(including_sour)___healthy` | 30 | 2.27% | 17.60% |
| 8 | `Tomato___Septoria_leaf_spot` | `Cherry_(including_sour)___healthy` | 29 | 2.19% | 19.79% |
| 9 | `Corn_(maize)___Northern_Leaf_Blight` | `Cherry_(including_sour)___healthy` | 27 | 2.04% | 21.83% |
| 10 | `Tomato___Late_blight` | `Cherry_(including_sour)___healthy` | 27 | 2.04% | 23.87% |
| 11 | `Tomato___Early_blight` | `Cherry_(including_sour)___healthy` | 25 | 1.89% | 25.76% |
| 12 | `Grape___Black_rot` | `Cherry_(including_sour)___healthy` | 25 | 1.89% | 27.64% |
| 13 | `Apple___Cedar_apple_rust` | `Cherry_(including_sour)___healthy` | 24 | 1.81% | 29.46% |
| 14 | `Orange___Haunglongbing_(Citrus_greening)` | `Cherry_(including_sour)___healthy` | 24 | 1.81% | 31.27% |
| 15 | `Pepper,_bell___healthy` | `Cherry_(including_sour)___healthy` | 23 | 1.74% | 33.01% |

---

## 3. Ten Weakest Classes (>= 15 Test Samples)

| Rank | Class Name | Samples | Correct | Errors | Recall |
|---|---|---|---|---|---|
| 1 | `Tomato___Bacterial_spot` | 59 | 0 | 59 | **0.00%** |
| 2 | `Soybean___healthy` | 46 | 0 | 46 | **0.00%** |
| 3 | `Peach___healthy` | 40 | 0 | 40 | **0.00%** |
| 4 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 38 | 0 | 38 | **0.00%** |
| 5 | `Apple___Cedar_apple_rust` | 59 | 1 | 58 | **1.69%** |
| 6 | `Potato___healthy` | 39 | 1 | 38 | **2.56%** |
| 7 | `Orange___Haunglongbing_(Citrus_greening)` | 35 | 1 | 34 | **2.86%** |
| 8 | `Tomato___Tomato_mosaic_virus` | 35 | 1 | 34 | **2.86%** |
| 9 | `Potato___Late_blight` | 56 | 2 | 54 | **3.57%** |
| 10 | `Tomato___Leaf_Mold` | 49 | 2 | 47 | **4.08%** |

---

## 4. Visual Diagnostics Artifacts

- `most_confident_errors.png`: Contact sheet of the 24 most confident erroneous classifications.
- `pair_*.png`: Contact sheets showing 6 example errors for each of the top confusion pairs.
- `per_class_recall.csv`: Full per-class recall statistics.
- `per_source_recall.csv`: Source-level accuracy breakdown.
- `gradcam/`: Grad-CAM saliency explanation panels for 8 correct and 8 error test cases.

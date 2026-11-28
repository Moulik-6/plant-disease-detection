# Plant Disease Detection Web Application Documentation

This document describes the design, deployment, calibrated decision thresholds, operation, and verification of the AgriScan plant disease detection web service.

---

## 1. Safety, Ethics & Dataset Licensing Notice

### Calibration, Operating Points & Measured Field Accuracy
Field accuracy (`75.6%` top-1 for the 3-model Stage-II Experiment B ensemble; `73.0%` for EfficientNetV2-S) is lower than controlled laboratory benchmark accuracy (`98.7–99.85%`). To communicate uncertainty and filter out-of-taxonomy plant imagery, `PlantDiseasePredictor` averages member softmax probabilities, converts the ensemble distribution to log-probabilities, applies post-hoc temperature scaling ($T$) fitted on `field_val` via [ml/analysis/calibrate.py](file:///c:/Major%20Project/plant-disease-detection/ml/analysis/calibrate.py), and maps the top calibrated probability $p_{\max}$ against two operating-point thresholds fitted strictly on `field_val` / `unknown_val` and measured on held-out `field_test` ($N = 1,574$) and `unknown_test` ($N = 5,501$):

#### Default Ensemble Mode (`APP_MODE=ensemble`, 3-Model B Ensemble, $T = 0.9078$, `AUROC = 0.7620`)
Source: [results/stage2/calibration/b_ensemble.json](file:///c:/Major%20Project/plant-disease-detection/results/stage2/calibration/b_ensemble.json) (`ECE`: `0.0487` &rarr; `0.0269`; overall `field_test` accuracy `75.60%` = `1,190 / 1,574`)

| Status | Calibrated Probability Range | Basis / Operating Point | `field_test` Share (Samples) | Measured Accuracy in Status Band | Cumulative Accepted Accuracy | Unknown Rejected (`unknown_test`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **`confident`** | $p_{\max} \ge 0.6782$ | Maximum Youden's $J$ (`youden_j`, `0.678196`) | **69.19%** *(1,089 / 1,574)* | **87.60%** *(954 / 1,089)* | **87.60%** *($\ge 0.6782$)* | **70.68%** *(at 0.6782)* |
| **`uncertain`** | $0.4290 \le p_{\max} < 0.6782$ | Between `keep_90_known` (`0.428968`) and `youden_j` | **21.47%** *(338 / 1,574)* | **57.40%** *(194 / 338)* | **80.45%** *($\ge 0.4290$, 1,427 imgs / 90.66% cov)* | **35.94%** *(at 0.4290)* |
| **`not_recognised`** | $p_{\max} < 0.4290$ | Below `keep_90_known` threshold (`0.428968`) | **9.34%** *(147 / 1,574)* | **28.57%** *(42 / 147)* | — *(abstained)* | — |

*(Note: A third high-precision operating point `acc_90_max_cov` at threshold `0.8248` is also recorded in `b_ensemble.json`: `53.24%` coverage (`838 / 1,574`), `91.53%` accepted accuracy, and `83.73%` unknown rejection.)*

#### Light Mode (`APP_MODE=light`, Single EfficientNetV2-S, $T = 1.0675$, `AUROC = 0.6868`)
Source: [results/stage2/calibration/b_effnet.json](file:///c:/Major%20Project/plant-disease-detection/results/stage2/calibration/b_effnet.json) (`ECE`: `0.0733` &rarr; `0.0389`; overall `field_test` accuracy `73.00%` = `1,149 / 1,574`)

| Status | Calibrated Probability Range | Basis / Operating Point | `field_test` Share (Samples) | Measured Accuracy in Status Band | Cumulative Accepted Accuracy | Unknown Rejected (`unknown_test`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **`confident`** | $p_{\max} \ge 0.8042$ | Maximum Youden's $J$ (`youden_j`, `0.804236`) | **62.33%** *(981 / 1,574)* | **85.42%** *(838 / 981)* | **85.42%** *($\ge 0.8042$)* | **66.93%** *(at 0.8042)* |
| **`uncertain`** | $0.5024 \le p_{\max} < 0.8042$ | Between `keep_90_known` (`0.502413`) and `youden_j` | **27.83%** *(438 / 1,574)* | **59.59%** *(261 / 438)* | **77.45%** *($\ge 0.5024$, 1,419 imgs / 90.15% cov)* | **28.41%** *(at 0.5024)* |
| **`not_recognised`** | $p_{\max} < 0.5024$ | Below `keep_90_known` threshold (`0.502413`) | **9.85%** *(155 / 1,574)* | **32.26%** *(50 / 155)* | — *(abstained)* | — |

- **Agricultural Decision Warning**: Never apply restricted or expensive fungicides/pesticides based solely on automated screening. Always inspect physical symptoms and consult certified agricultural extension officers before treatment.

### PlantWild Non-Commercial Licensing Clause
The **PlantWild** dataset is released under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** licence.
- All model checkpoints fine-tuned or trained on PlantWild (including Stage-II mixed runs) inherit these non-commercial restrictions.
- You **MUST NOT** use or deploy these model checkpoints in commercial applications or SaaS products, nor publish derived weights, without prior written permission from the original dataset copyright holders.

---

## 2. Configuration & Environment Variables

| Environment Variable | Default (`ensemble` mode) | `APP_MODE=light` Default | Description |
| :--- | :--- | :--- | :--- |
| `APP_MODE` | `"ensemble"` | `"light"` | Selects 3-model B ensemble vs. single EfficientNetV2-S mode |
| `APP_CHECKPOINTS` | 3 `b_mixed` `best.pth` runs (`resnet50`, `efficientnet_v2_s`, `vit_b_16`) | `runs/b_mixed_efficientnet_v2_s/20261121_122244/best.pth` | Comma-separated `.pth` checkpoint paths |
| `APP_CALIBRATION` | `results/stage2/calibration/b_ensemble.json` | `results/stage2/calibration/b_effnet.json` | Calibration JSON loading `temperature` and `operating_points` |
| `APP_TEMPERATURE` | `0.9078` *(from JSON)* | `1.0675` *(from JSON)* | Explicit temperature scalar override |
| `APP_CONFIDENT_THRESHOLD` | `0.6782` *(Youden $J$)* | `0.8042` *(Youden $J$)* | Cutoff at or above which status is `"confident"` (`APP_THRESHOLD` alias) |
| `APP_UNCERTAIN_THRESHOLD` | `0.4290` *(keep 90% known)* | `0.5024` *(keep 90% known)* | Lower cutoff for `"uncertain"`; below this yields `"not_recognised"` |
| `APP_DB` | `"app.db"` | `"app.db"` | SQLite diagnosis history path |
| `APP_DEVICE` | `"auto"` | `"auto"` | `"auto"`, `"cpu"`, or `"cuda"` |
| `APP_STORE_IMAGES` | `"0"` | `"0"` | Store uploaded images on disk (`1` = enable, `0` = disable) |

---

## 3. API Endpoints Reference

### `GET /health`
- **Purpose**: Readiness and liveness check exposing active mode, loaded checkpoints, fitted temperature, and active status thresholds.
- **Response**:
```json
{
  "status": "ok",
  "mode": "ensemble",
  "device": "cpu",
  "models_loaded": 3,
  "model_names": ["resnet50", "efficientnet_v2_s", "vit_b_16"],
  "loaded_models": [
    {
      "architecture": "resnet50",
      "path": "runs/b_mixed_resnet50/20261121_131527/best.pth"
    },
    {
      "architecture": "efficientnet_v2_s",
      "path": "runs/b_mixed_efficientnet_v2_s/20261121_122244/best.pth"
    },
    {
      "architecture": "vit_b_16",
      "path": "runs/b_mixed_vit_b_16/20261121_134843/best.pth"
    }
  ],
  "calibration": {
    "temperature": 0.9077910747054088,
    "threshold": 0.6781963109970093,
    "confident_threshold": 0.6781963109970093,
    "uncertain_threshold": 0.4289683401584625,
    "thresholds": {
      "confident": 0.6781963109970093,
      "uncertain": 0.4289683401584625
    },
    "operating_points": {
      "youden_j": { "threshold": 0.6781963109970093 },
      "keep_90_known": { "threshold": 0.4289683401584625 },
      "acc_90_max_cov": { "threshold": 0.8248436450958252 }
    },
    "source": "results/stage2/calibration/b_ensemble.json"
  },
  "temperature": 0.9077910747054088,
  "threshold": 0.6781963109970093,
  "confident_threshold": 0.6781963109970093,
  "uncertain_threshold": 0.4289683401584625,
  "thresholds": {
    "confident": 0.6781963109970093,
    "uncertain": 0.4289683401584625
  }
}
```

### `GET /classes`
- **Purpose**: Returns the full catalog of 38 disease categories recognized by the unified model.
- **Response**:
```json
[
  {
    "class_name": "Tomato___Late_blight",
    "crop": "Tomato",
    "condition": "Late blight",
    "healthy": false
  },
  {
    "class_name": "Apple___healthy",
    "crop": "Apple",
    "condition": "Healthy",
    "healthy": true
  }
]
```

### `POST /predict`
- **Parameters**:
  - `image` (Multipart File): Leaf photograph (`image/jpeg`, `image/png`, `image/webp`). Max size: 5 MB.
  - `gradcam` (Form/Query Bool): Whether to compute and return Grad-CAM attention heatmap overlay.
- **Response**:
```json
{
  "predicted_class": "Tomato___Late_blight",
  "crop": "Tomato",
  "condition": "Late blight",
  "healthy": false,
  "confidence": 0.8921,
  "status": "confident",
  "top3": [
    {
      "class_name": "Tomato___Late_blight",
      "crop": "Tomato",
      "condition": "Late blight",
      "healthy": false,
      "confidence": 0.8921,
      "probability": 0.8921
    },
    {
      "class_name": "Tomato___Early_blight",
      "crop": "Tomato",
      "condition": "Early blight",
      "healthy": false,
      "confidence": 0.0842,
      "probability": 0.0842
    },
    {
      "class_name": "Potato___Late_blight",
      "crop": "Potato",
      "condition": "Late blight",
      "healthy": false,
      "confidence": 0.0153,
      "probability": 0.0153
    }
  ],
  "latency_ms": 297.0,
  "inference_latency_ms": 297.0,
  "model_names": ["resnet50", "efficientnet_v2_s", "vit_b_16"],
  "caution": "This diagnosis is a preliminary screening aid and not a certified expert diagnosis. For severe crop symptoms, consult a certified local agricultural extension officer or agronomist.",
  "gradcam_image": "data:image/png;base64,iVBORw0KGgoAAA..."
}
```

### `POST /gradcam`
- **Parameters**: `image` (Multipart File).
- **Response**: Direct image binary with `Content-Type: image/png`.
- **Error**: Returns `501 Not Implemented` if the loaded ensemble includes a dual-backbone fusion model (`fusion_concat` or `fusion_gated`).

### `GET /history`
- **Parameters**: `limit` (Query Int, 1–500, default 50).
- **Response**: List of saved predictions from SQLite database.

### `DELETE /history`
- **Purpose**: Clears all records from the SQLite database and removes any stored uploads.
- **Response**: `{"status": "deleted", "count": 1}`.

---

## 4. Deployment Guide

### Local Development
```bash
# Default 3-model calibrated ensemble (T = 0.9078)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Light mode (single EfficientNetV2-S, T = 1.0675)
APP_MODE=light uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build lightweight CPU container without weights
docker build -t agriscan-app:latest .

# Run container with mounted checkpoints and results volumes
docker run -d \
  --name agriscan \
  -p 8000:8000 \
  -v /path/to/runs:/workspace/runs:ro \
  -v /path/to/results:/workspace/results:ro \
  -v /path/to/data:/workspace/data:rw \
  -e APP_DEVICE="cpu" \
  agriscan-app:latest
```

---

## 5. Multi-Language Frontend Architecture
Externalized strings reside in `app/static/strings.json`. The frontend dynamically updates UI elements on language change without page reload:
- English (`en`)
- Telugu (`te`)
- Hindi (`hi`)
To add a new language, simply add a new key block in `strings.json` and a corresponding `<option>` in `index.html`.

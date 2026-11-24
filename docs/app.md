# Plant Disease Detection Web Application Documentation

This document describes the design, deployment, operation, and verification of the AgriScan plant disease detection web service.

---

## 1. Safety, Ethics & Dataset Licensing Notice

### Caution for In-Field Diagnostic Usage
Field accuracy (71–73% top-1, 75.6% ensemble) is lower than clean laboratory benchmark accuracy (99.8%). In real-world conditions:
- Out-of-distribution symptoms, lighting variations, and overlapping infections can cause misclassifications.
- The system includes confidence categorization:
  - **Confident**: $\text{probability} \ge \tau$ (default 0.5)
  - **Uncertain**: $0.5 \tau \le \text{probability} < \tau$
  - **Not Recognised / Out-of-Distribution**: $\text{probability} < 0.5 \tau$
- **Agricultural Decision Warning**: Never apply restricted or expensive fungicides/pesticides based solely on automated screening. Users should always inspect physical symptoms and consult local agricultural extension officers before intervention.

### PlantWild Non-Commercial Licensing Clause
The **PlantWild** dataset is released under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** licence.
- All model checkpoints fine-tuned or trained on PlantWild (including Stage-II mixed runs) inherit these non-commercial restrictions.
- You **MUST NOT** use or deploy these model checkpoints in commercial applications or SaaS products, nor publish derived weights, without prior written permission from the original dataset copyright holders.

---

## 2. API Endpoints Reference

### `GET /health`
- **Purpose**: Readiness and liveness check for monitoring and container orchestration.
- **Response**:
```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": 1,
  "model_names": ["resnet50"],
  "loaded_models": [
    {
      "architecture": "resnet50",
      "path": "runs/resnet50/best.pth"
    }
  ],
  "calibration": {
    "temperature": 1.15,
    "threshold": 0.48,
    "source": "results/calibration.json"
  },
  "temperature": 1.15,
  "threshold": 0.48
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
  "latency_ms": 38.4,
  "inference_latency_ms": 38.4,
  "model_names": ["resnet50"],
  "caution": "Field accuracy is lower than lab accuracy; inspect symptoms in person before treating.",
  "gradcam_image": "data:image/png;base64,iVBORw0KGgoAAA..."
}
```

### `POST /gradcam`
- **Parameters**: `image` (Multipart File).
- **Response**: Direct image binary with `Content-Type: image/png`.
- **Error**: Returns `501 Not Implemented` if the loaded ensemble includes a dual-backbone fusion model (`fusion_concat` or `fusion_gated`).

### `GET /history`
- **Parameters**: `limit` (Query Int, 1–500, default 50).
- **Response**: List of saved predictions from SQLite database:
```json
[
  {
    "id": 1,
    "timestamp": "2026-11-24T12:00:00+00:00",
    "created_at": "2026-11-24T12:00:00+00:00",
    "predicted_class": "Tomato___Late_blight",
    "crop": "Tomato",
    "condition": "Late blight",
    "healthy": false,
    "confidence": 0.8921,
    "status": "confident",
    "model_name": "resnet50",
    "image_path": null
  }
]
```

### `DELETE /history`
- **Purpose**: Clears all records from the SQLite database and removes any stored uploads.
- **Response**: `{"status": "deleted", "count": 1}`.

---

## 3. Deployment Guide

### Local Development
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment
```bash
# Build lightweight CPU container without weights
docker build -t agriscan-app:latest .

# Run container with mounted checkpoints volume
docker run -d \
  --name agriscan \
  -p 8000:8000 \
  -v /path/to/runs:/workspace/runs:ro \
  -v /path/to/data:/workspace/data:rw \
  -e APP_CHECKPOINTS="/workspace/runs/resnet50/best.pth" \
  -e APP_DEVICE="cpu" \
  agriscan-app:latest
```

---

## 4. Multi-Language Frontend Architecture
Externalized strings reside in `app/static/strings.json`. The frontend dynamically updates UI elements on language change without page reload:
- English (`en`)
- Telugu (`te`)
- Hindi (`hi`)
To add a new language, simply add a new key block in `strings.json` and a corresponding `<option>` in `index.html`.

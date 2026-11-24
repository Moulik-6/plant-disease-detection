"""
FastAPI Application for Plant Disease Detection.
Serves prediction APIs, Grad-CAM visualizations, diagnosis history,
and the responsive mobile-first static interface.
"""

from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.predictor import PlantDiseasePredictor
from app.labels import parse_label, get_all_classes
from app.db import init_db, save_prediction, get_history, clear_history
from app.schemas import (
    HealthResponse,
    LoadedModelInfo,
    CalibrationInfo,
    ClassItem,
    Top3Item,
    PredictionResponse,
    HistoryItem,
    HistoryResponse,
    DeleteHistoryResponse,
    ErrorResponse,
)

logger = logging.getLogger("app.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global predictor and config references
predictor: Optional[PlantDiseasePredictor] = None
app_config: Dict[str, Any] = {}
db_path: str = "app.db"
store_images: bool = False
upload_dir: Path = Path("app/uploads")
max_upload_bytes: int = 5 * 1024 * 1024  # 5 MB
max_image_pixels: int = 25_000_000      # 25 MP


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    global predictor, app_config, db_path, store_images, upload_dir, max_upload_bytes, max_image_pixels

    # Initialize predictor
    predictor = PlantDiseasePredictor()
    app_config = predictor.raw_config

    # Initialize DB settings
    db_path = os.environ.get("APP_DB", app_config.get("db_path", "app.db"))
    init_db(db_path)

    # Image storage settings
    env_store = os.environ.get("APP_STORE_IMAGES", str(app_config.get("store_images", 0))).strip().lower()
    store_images = env_store in ("1", "true", "yes")

    cfg_upload_dir = app_config.get("upload_dir", "app/uploads")
    upload_dir = Path(cfg_upload_dir)
    if store_images:
        upload_dir.mkdir(parents=True, exist_ok=True)

    max_upload_bytes = int(app_config.get("max_upload_bytes", 5242880))
    max_image_pixels = int(app_config.get("max_image_pixels", 25000000))

    logger.info("Plant Disease Detection service ready. Device=%s, Models=%d, StoreImages=%s",
                predictor.device, len(predictor.models), store_images)
    yield


app = FastAPI(
    title="Plant Disease Detection API",
    description="High-precision plant disease screening and differential diagnosis API with Grad-CAM explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
origins = os.environ.get("APP_CORS_ORIGINS", "")
allowed_origins = [o.strip() for o in origins.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler to avoid leaking internal tracebacks
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error processing %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An internal error occurred during processing."},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail if isinstance(exc.detail, str) else "Error", "detail": exc.detail},
    )


async def validate_and_load_image(file: UploadFile) -> Image.Image:
    """
    Validates uploaded file size, checks MIME/magic bytes using Pillow,
    and returns verified PIL Image.
    """
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum limit of {max_upload_bytes // (1024 * 1024)} MB.",
        )

    try:
        # First verify format integrity
        test_img = Image.open(io.BytesIO(contents))
        img_format = test_img.format
        test_img.verify()

        # Re-open verified stream for raster processing
        pil_img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid or readable image file.",
        )

    allowed_formats = ("JPEG", "PNG", "WEBP")
    if (img_format or "").upper() not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {img_format}. Allowed formats: {', '.join(allowed_formats)}.",
        )

    width, height = pil_img.size
    if width * height > max_image_pixels:
        raise HTTPException(
            status_code=400,
            detail=f"Image resolution ({width}x{height}) exceeds maximum limit of {max_image_pixels} pixels.",
        )

    # Attach raw bytes as attribute for optional saving
    pil_img._raw_bytes = contents
    return pil_img


@app.get("/health", response_model=HealthResponse)
async def get_health():
    """Returns system status, loaded model architectures, device, and calibration parameters."""
    loaded_info = [
        LoadedModelInfo(architecture=arch, path=p)
        for arch, p in zip(predictor.architectures, predictor.loaded_model_paths)
    ]
    calib_info = CalibrationInfo(
        temperature=predictor.temperature,
        threshold=predictor.threshold,
        source=str(predictor.calibration_path) if predictor.calibration_path else None,
    )
    return HealthResponse(
        status="ok",
        device=str(predictor.device),
        models_loaded=len(predictor.models),
        model_names=predictor.architectures,
        loaded_models=loaded_info,
        calibration=calib_info,
        temperature=predictor.temperature,
        threshold=predictor.threshold,
    )


@app.get("/classes", response_model=List[ClassItem])
async def get_classes():
    """Returns list of all 38 unified classes with crop, condition, and healthy indicator."""
    classes = get_all_classes()
    return [ClassItem(**c) for c in classes]


@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    image: UploadFile = File(...),
    gradcam: bool = Form(False),
):
    """
    Classifies a crop leaf image, returning predicted disease, confidence,
    status classification, top-3 differential diagnoses, and optional Grad-CAM overlay.
    """
    pil_img = await validate_and_load_image(image)

    try:
        # Run inference in worker thread
        result = await anyio.to_thread.run_sync(predictor.predict, pil_img, gradcam)
    except NotImplementedError as e:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # Optional image storage
    saved_img_path = None
    if store_images and hasattr(pil_img, "_raw_bytes"):
        filename = f"{uuid.uuid4().hex}.jpg"
        target_path = upload_dir / filename
        try:
            with open(target_path, "wb") as f:
                f.write(pil_img._raw_bytes)
            saved_img_path = str(target_path)
        except Exception as e:
            logger.warning("Failed to store image to %s: %s", target_path, e)

    # Persist prediction in SQLite history
    model_repr = "+".join(predictor.architectures) if predictor.architectures else "unknown"
    save_prediction(
        db_path=db_path,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        status=result["status"],
        model_name=model_repr,
        image_path=saved_img_path,
    )

    req_id = str(uuid.uuid4())
    result["request_id"] = req_id
    return PredictionResponse(**result)


@app.post("/gradcam")
async def generate_gradcam_overlay(image: UploadFile = File(...)):
    """
    Generates and returns Grad-CAM heatmap overlay as image/png bytes directly.
    Returns HTTP 501 Not Implemented if the ensemble contains a fusion model.
    """
    pil_img = await validate_and_load_image(image)

    try:
        cam_bytes = await anyio.to_thread.run_sync(predictor.generate_gradcam, pil_img)
    except NotImplementedError as e:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return Response(content=cam_bytes, media_type="image/png")


@app.get("/history", response_model=List[HistoryItem])
async def list_history(limit: int = Query(50, ge=1, le=500)):
    """Retrieves recent diagnosis predictions from the SQLite database."""
    raw_items = get_history(db_path, limit=limit)
    history_items = []
    for item in raw_items:
        crop, cond, healthy = parse_label(item["predicted_class"])
        ts = item.get("created_at", "")
        history_items.append(
            HistoryItem(
                id=item["id"],
                timestamp=ts,
                created_at=ts,
                predicted_class=item["predicted_class"],
                crop=crop,
                condition=cond,
                healthy=healthy,
                confidence=float(item["confidence"]),
                status=item["status"],
                model_name=item["model_name"],
                image_path=item.get("image_path"),
            )
        )
    return history_items


@app.delete("/history", response_model=DeleteHistoryResponse)
async def delete_history():
    """Clears all records from the diagnosis history database."""
    deleted_count = clear_history(db_path, upload_dir=str(upload_dir) if store_images else None)
    return DeleteHistoryResponse(status="deleted", count=deleted_count)


# Mount Static UI at root '/'
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

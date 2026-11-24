"""
Pydantic data schemas for the Plant Disease Detection API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LoadedModelInfo(BaseModel):
    architecture: str = Field(..., examples=["resnet50"])
    path: str = Field(..., examples=["ml/checkpoints/resnet50/best_model.pth"])


class CalibrationInfo(BaseModel):
    temperature: float = Field(1.0, examples=[1.0])
    threshold: float = Field(0.5, examples=[0.5])
    source: Optional[str] = Field(None, examples=["results/calibration.json"])


class HealthResponse(BaseModel):
    status: str = Field("ok", examples=["ok"])
    device: str = Field(..., examples=["cpu"])
    models_loaded: int = Field(..., examples=[1])
    model_names: List[str] = Field(default_factory=list, examples=[["resnet50"]])
    loaded_models: List[LoadedModelInfo] = Field(default_factory=list)
    calibration: CalibrationInfo
    temperature: float = Field(..., examples=[1.0])
    threshold: float = Field(..., examples=[0.5])


class ClassItem(BaseModel):
    class_name: str = Field(..., examples=["Tomato___Late_blight"])
    crop: str = Field(..., examples=["Tomato"])
    condition: str = Field(..., examples=["Late blight"])
    healthy: bool = Field(..., examples=[False])


class Top3Item(BaseModel):
    class_name: str = Field(..., examples=["Tomato___Late_blight"])
    crop: str = Field(..., examples=["Tomato"])
    condition: str = Field(..., examples=["Late blight"])
    healthy: bool = Field(..., examples=[False])
    confidence: float = Field(..., examples=[0.9234])
    probability: float = Field(..., examples=[0.9234])


class PredictionResponse(BaseModel):
    request_id: Optional[str] = Field(default=None, examples=["f81d4fae-7dec-11d0-a765-00a0c91e6bf6"])
    predicted_class: str = Field(..., examples=["Tomato___Late_blight"])
    crop: str = Field(..., examples=["Tomato"])
    condition: str = Field(..., examples=["Late blight"])
    healthy: bool = Field(..., examples=[False])
    confidence: float = Field(..., examples=[0.9234])
    status: str = Field(..., examples=["confident"], description="'confident', 'uncertain', or 'not_recognised'")
    top3: List[Top3Item]
    top_3: Optional[List[Top3Item]] = None
    latency_ms: float = Field(..., examples=[42.5])
    inference_latency_ms: float = Field(..., examples=[42.5])
    model_names: List[str] = Field(default_factory=list)
    caution: str = Field(
        ...,
        examples=["This diagnosis is a preliminary screening aid and not a certified expert diagnosis. For severe crop symptoms, consult a certified local agricultural extension officer or agronomist."],
    )
    gradcam_image: Optional[str] = Field(
        default=None,
        description="data:image/png;base64 string if gradcam was requested",
    )


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    created_at: str
    predicted_class: str
    crop: str
    condition: str
    healthy: bool
    confidence: float
    status: str
    model_name: str
    image_path: Optional[str] = None


class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int


class DeleteHistoryResponse(BaseModel):
    status: str = Field("deleted", examples=["deleted"])
    count: int = Field(..., examples=[5])


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

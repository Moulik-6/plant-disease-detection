"""
Model Predictor and Inference Engine for Plant Disease Detection.
Handles checkpoint loading, EXIF-aware preprocessing, calibrated ensembling,
and Grad-CAM explainability.
"""

import base64
import copy
import io
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import yaml

from ml.src.model_zoo import load_checkpoint_model
from ml.src.fusion import FusionModel
from app.labels import parse_label, get_all_classes

logger = logging.getLogger("app.predictor")


class PlantDiseasePredictor:
    """
    Loads one or more trained model checkpoints, performs temperature-scaled
    ensemble inference, and generates Grad-CAM visualizations.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_paths: Optional[List[str]] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        device_str: Optional[str] = None,
        threshold: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        # 1. Load config defaults
        self.config_path = Path(config_path) if config_path else Path(__file__).resolve().parent / "config.yaml"
        self.raw_config: Dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.raw_config = yaml.safe_load(f) or {}

        # 2. Resolve PyTorch Device
        env_device = os.environ.get("APP_DEVICE", "").strip().lower()
        cfg_device = str(device_str or env_device or self.raw_config.get("device", "auto")).lower()
        if cfg_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif cfg_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # 3. Resolve Checkpoint Paths
        env_ckpts = os.environ.get("APP_CHECKPOINTS", "").strip()
        if checkpoint_paths:
            self.ckpt_paths = [str(p).strip() for p in checkpoint_paths if str(p).strip()]
        elif env_ckpts:
            self.ckpt_paths = [p.strip() for p in env_ckpts.split(",") if p.strip()]
        else:
            cfg_ckpts = self.raw_config.get("checkpoints", [])
            if isinstance(cfg_ckpts, str):
                self.ckpt_paths = [p.strip() for p in cfg_ckpts.split(",") if p.strip()]
            else:
                self.ckpt_paths = [str(p).strip() for p in cfg_ckpts if str(p).strip()]

        # 4. Resolve Calibration & Thresholds
        env_calib = os.environ.get("APP_CALIBRATION", "").strip()
        calib_file = Path(calibration_path or env_calib or self.raw_config.get("calibration_path") or "")
        self.calibration_path: Optional[Path] = calib_file if calib_file.is_file() else None

        self.temperature: float = 1.0
        self.threshold: float = float(self.raw_config.get("default_threshold", 0.5))

        if self.calibration_path and self.calibration_path.exists():
            self._load_calibration(self.calibration_path)

        if temperature is not None:
            self.temperature = float(temperature)
        if threshold is not None:
            self.threshold = float(threshold)

        # 5. Load Models
        self.models: List[torch.nn.Module] = []
        self.model_configs: List[Dict[str, Any]] = []
        self.architectures: List[str] = []
        self.loaded_model_paths: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        self._load_all_models()

        # ImageNet constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_calibration(self, calib_path: Path) -> None:
        """Parses temperature and threshold from calibration JSON."""
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Resolve temperature
            if "temperature" in data:
                self.temperature = float(data["temperature"])
            elif "T" in data:
                self.temperature = float(data["T"])
            elif "optimal_temperature" in data:
                self.temperature = float(data["optimal_temperature"])
            elif "scaling" in data and isinstance(data["scaling"], dict):
                self.temperature = float(data["scaling"].get("temperature", 1.0))

            # Resolve threshold
            if "threshold" in data:
                self.threshold = float(data["threshold"])
            elif "operating_points" in data and isinstance(data["operating_points"], dict):
                op = data["operating_points"]
                if "youden_j" in op and isinstance(op["youden_j"], dict):
                    self.threshold = float(op["youden_j"].get("threshold", 0.5))
                elif "keep_90_known" in op and isinstance(op["keep_90_known"], dict):
                    self.threshold = float(op["keep_90_known"].get("threshold", 0.5))
            elif "thresholds" in data and isinstance(data["thresholds"], dict):
                th = data["thresholds"]
                if "youden_j" in th:
                    val = th["youden_j"]
                    self.threshold = float(val.get("threshold", val) if isinstance(val, dict) else val)
                elif "keep_90_known" in th:
                    val = th["keep_90_known"]
                    self.threshold = float(val.get("threshold", val) if isinstance(val, dict) else val)
            logger.info("Loaded calibration from %s: T=%.4f, threshold=%.4f", calib_path, self.temperature, self.threshold)
        except Exception as e:
            logger.warning("Failed to parse calibration JSON at %s: %s. Using defaults.", calib_path, e)

    def _load_all_models(self) -> None:
        """Loads all specified checkpoints into memory."""
        for ckpt_str in self.ckpt_paths:
            ckpt_p = Path(ckpt_str)
            if not ckpt_p.is_file():
                logger.warning("Checkpoint not found, skipping: %s", ckpt_p)
                continue
            try:
                model, m_cfg, arch, c_to_i, i_to_c = load_checkpoint_model(ckpt_p, device=self.device)
                self.models.append(model)
                self.model_configs.append(m_cfg)
                self.architectures.append(arch)
                self.loaded_model_paths.append(str(ckpt_p))

                if not self.class_to_idx:
                    self.class_to_idx = c_to_i
                    self.idx_to_class = i_to_c
                logger.info("Loaded model '%s' from %s", arch, ckpt_p)
            except Exception as e:
                logger.error("Error loading checkpoint %s: %s", ckpt_p, e, exc_info=True)

        # Fallback if no checkpoints could be loaded
        if not self.class_to_idx:
            from ml.src.datasets import load_unified_class_to_idx
            self.class_to_idx = load_unified_class_to_idx()
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def preprocess_image(self, pil_image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocesses a PIL Image exactly matching ml/evaluate.py:
        1. Applies EXIF orientation.
        2. Converts to RGB.
        3. Squash resizes to 224x224.
        4. Scales to float32 in [0, 1].
        5. Normalizes with ImageNet mean and std.
        6. Returns (1, 3, 224, 224) tensor and (224, 224, 3) float32 RGB array.
        """
        # 1. EXIF orientation correction
        img = ImageOps.exif_transpose(pil_image)

        # 2. Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 3. Squash resize to 224x224
        img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)

        # 4. Float RGB in [0, 1]
        rgb_float = np.float32(img_resized) / 255.0

        # 5. ImageNet normalization
        norm_img = (rgb_float - self.mean) / self.std

        # 6. Permute to (1, 3, H, W)
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device), rgb_float

    def predict(self, pil_image: Image.Image, include_gradcam: bool = False) -> Dict[str, Any]:
        """
        Runs model inference, computes calibrated ensemble predictions,
        and determines diagnostic confidence status.
        """
        t0 = time.perf_counter()
        tensor, rgb_float = self.preprocess_image(pil_image)

        if not self.models:
            # If no model is loaded (e.g., bare initialization), raise error
            raise RuntimeError("No model checkpoints loaded in PlantDiseasePredictor.")

        # Ensemble calibrated inference
        all_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(tensor)
                scaled_logits = logits / max(1e-5, self.temperature)
                probs = F.softmax(scaled_logits, dim=-1)
                all_probs.append(probs)

            # Average probabilities across ensemble members
            avg_probs = torch.mean(torch.stack(all_probs), dim=0)[0].cpu().numpy()

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        pred_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])
        pred_class = self.idx_to_class.get(pred_idx, f"Class_{pred_idx}")
        crop, condition, healthy = parse_label(pred_class)

        # Operating point status categorization:
        # - confident: max probability >= threshold
        # - uncertain: 0.5 * threshold <= max probability < threshold
        # - not_recognised: max probability < 0.5 * threshold
        if confidence >= self.threshold:
            status = "confident"
        elif confidence >= 0.5 * self.threshold:
            status = "uncertain"
        else:
            status = "not_recognised"

        # Differential Diagnoses (Top 3)
        top3_indices = np.argsort(avg_probs)[::-1][:3]
        top3_list = []
        for idx in top3_indices:
            cls_name = self.idx_to_class.get(int(idx), f"Class_{idx}")
            t_crop, t_cond, t_healthy = parse_label(cls_name)
            prob_val = float(round(float(avg_probs[idx]), 4))
            top3_list.append({
                "class_name": cls_name,
                "crop": t_crop,
                "condition": t_cond,
                "healthy": t_healthy,
                "confidence": prob_val,
                "probability": prob_val,
            })

        # Grad-CAM if requested
        gradcam_b64 = None
        if include_gradcam:
            cam_bytes = self.generate_gradcam(pil_image, target_class=pred_idx)
            gradcam_b64 = f"data:image/png;base64,{base64.b64encode(cam_bytes).decode('utf-8')}"

        caution = str(self.raw_config.get(
            "caution_text",
            "This diagnosis is a preliminary screening aid and not a certified expert diagnosis. For severe crop symptoms, consult a certified local agricultural extension officer or agronomist.",
        ))

        return {
            "predicted_class": pred_class,
            "crop": crop,
            "condition": condition,
            "healthy": healthy,
            "confidence": float(round(confidence, 4)),
            "status": status,
            "top3": top3_list,
            "top_3": top3_list,
            "latency_ms": latency_ms,
            "inference_latency_ms": latency_ms,
            "model_names": self.architectures,
            "caution": caution,
            "gradcam_image": gradcam_b64,
        }

    def generate_gradcam(self, pil_image: Image.Image, target_class: Optional[int] = None) -> bytes:
        """
        Generates Grad-CAM heatmap overlay for the input image.
        Returns PNG bytes directly.
        Raises NotImplementedError if any model is a dual-backbone fusion model.
        """
        if not self.models:
            raise RuntimeError("No models loaded for Grad-CAM.")

        # Check for fusion models
        for model, arch in zip(self.models, self.architectures):
            if isinstance(model, FusionModel) or arch.lower().startswith("fusion"):
                raise NotImplementedError("Grad-CAM across dual backbones is not supported.")

        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from ml.analysis.gradcam import get_target_layers_and_transform

        # Use primary model for Grad-CAM
        primary_model = self.models[0]
        arch = self.architectures[0]

        tensor, rgb_float = self.preprocess_image(pil_image)

        if target_class is None:
            with torch.no_grad():
                out = primary_model(tensor)
                target_class = int(torch.argmax(out, dim=1).item())

        target_layers, reshape_transform = get_target_layers_and_transform(primary_model, arch)

        # Run GradCAM with gradients enabled
        primary_model.zero_grad()
        with GradCAM(model=primary_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
            targets = [ClassifierOutputTarget(int(target_class))]
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

        # Overlay heatmap (opacity = 0.6 => image_weight = 0.4)
        visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, image_weight=0.4)

        buf = io.BytesIO()
        pil_vis = Image.fromarray(visualization)
        pil_vis.save(buf, format="PNG")
        return buf.getvalue()

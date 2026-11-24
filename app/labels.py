"""
Label parser and taxonomic metadata for Plant Disease Detection.
Converts unified dataset labels (e.g., 'Tomato___Late_blight') into human-readable
crop names, condition descriptions, and binary health indicators.
"""

from typing import Any, Dict, List, Optional, Tuple
import yaml
from pathlib import Path

DEFAULT_UNIFIED_YAML = Path(__file__).resolve().parents[1] / "ml" / "configs" / "labels" / "unified_classes.yaml"


def parse_label(class_name: str) -> Tuple[str, str, bool]:
    """
    Parses a unified dataset label into (crop, condition, is_healthy).

    Examples:
      'Tomato___Late_blight' -> ('Tomato', 'Late blight', False)
      'Apple___healthy' -> ('Apple', 'Healthy', True)
      'Corn_(maize)___Common_rust_' -> ('Corn (maize)', 'Common rust', False)
      'Pepper,_bell___Bacterial_spot' -> ('Pepper, bell', 'Bacterial spot', False)
    """
    if "___" in class_name:
        raw_crop, raw_cond = class_name.split("___", 1)
    else:
        raw_crop, raw_cond = class_name, ""

    # Clean crop name
    crop = raw_crop.replace("_", " ").strip()
    crop = crop.replace(" ,", ",")

    # Clean condition name
    cond = raw_cond.replace("_", " ").strip()
    is_healthy = (cond.lower() == "healthy")

    if is_healthy:
        cond = "Healthy"
    elif cond:
        # Normalize leading lowercase if present
        if cond[0].islower():
            cond = cond[0].upper() + cond[1:]
    else:
        cond = "Unknown Condition"

    return crop, cond, is_healthy


def get_all_classes(yaml_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads all unified classes from yaml and parses their taxonomic metadata."""
    target_yaml = yaml_path or DEFAULT_UNIFIED_YAML
    if not target_yaml.exists():
        return []

    with open(target_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    classes_list = data.get("classes", [])
    results = []
    for cls_name in classes_list:
        crop, cond, healthy = parse_label(cls_name)
        results.append({
            "class_name": cls_name,
            "crop": crop,
            "condition": cond,
            "healthy": healthy,
        })
    return results

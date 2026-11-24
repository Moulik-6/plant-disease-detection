"""
Label parser and taxonomic metadata for Plant Disease Detection.
Converts unified dataset labels (e.g., 'Tomato___Late_blight') into human-readable
crop names, condition descriptions, and binary health indicators.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

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
    raw = str(class_name).strip()
    if "___" in raw:
        raw_crop, raw_cond = raw.split("___", 1)
    else:
        raw_crop, raw_cond = raw, ""

    # Clean crop name
    crop = raw_crop.replace("_", " ").strip()
    crop = crop.replace(" ,", ",")
    while "  " in crop:
        crop = crop.replace("  ", " ")
    if crop and crop[0].islower():
        crop = crop[0].upper() + crop[1:]

    # Clean condition name
    cond = raw_cond.replace("_", " ").strip()
    while "  " in cond:
        cond = cond.replace("  ", " ")

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


def get_class_info(class_name: str) -> Dict[str, Any]:
    """Returns a dictionary representation of parsed class labels."""
    crop, condition, healthy = parse_label(class_name)
    return {
        "class_name": class_name,
        "crop": crop,
        "condition": condition,
        "healthy": healthy,
    }


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
        results.append(get_class_info(cls_name))
    return results

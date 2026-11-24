"""
Label parsing and taxonomic formatting for unified PlantVillage disease classes.
"""

from typing import Any, Dict, List, Tuple


def parse_label(class_name: str) -> Tuple[str, str, bool]:
    """
    Parses a unified class identifier (e.g. 'Tomato___Late_blight') into:
      crop: 'Tomato'
      condition: 'Late blight'
      healthy: False
    """
    raw = str(class_name).strip()
    if "___" in raw:
        crop_part, cond_part = raw.split("___", 1)
    else:
        crop_part = raw
        cond_part = "unknown"

    # 1. Clean crop name
    crop = crop_part.replace("_", " ").strip()
    # Normalize common patterns
    if crop.lower().startswith("pepper, bell") or crop.lower().startswith("pepper,_bell"):
        crop = "Bell pepper"
    elif crop.lower().startswith("cherry (including sour)"):
        crop = "Cherry (sour/sweet)"
    elif crop.lower().startswith("corn (maize)"):
        crop = "Corn (maize)"

    crop = crop[0].upper() + crop[1:] if len(crop) > 1 else crop.upper()

    # 2. Clean condition name
    cond = cond_part.replace("_", " ").strip()
    while "  " in cond:
        cond = cond.replace("  ", " ")

    is_healthy = cond.lower() == "healthy"
    if is_healthy:
        condition = "Healthy"
    else:
        condition = cond[0].upper() + cond[1:] if len(cond) > 1 else cond.upper()

    return crop, condition, is_healthy


def get_class_info(class_name: str) -> Dict[str, Any]:
    """Returns a dictionary representation of parsed class labels."""
    crop, condition, healthy = parse_label(class_name)
    return {
        "class_name": class_name,
        "crop": crop,
        "condition": condition,
        "healthy": healthy,
    }

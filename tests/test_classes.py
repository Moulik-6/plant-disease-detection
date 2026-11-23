import yaml
from pathlib import Path

def test_unified_classes_structure():
    """Verify that unified_classes.yaml defines exactly 38 unique classes in a fixed order."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "ml" / "configs" / "labels" / "unified_classes.yaml"
    assert cfg_path.exists(), f"Missing config file: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "classes" in data, "unified_classes.yaml must have a 'classes' key"
    classes = data["classes"]
    assert len(classes) == 38, f"Expected exactly 38 classes, got {len(classes)}"
    assert len(set(classes)) == 38, "Class names in 'classes' must all be unique"

    if "class_to_idx" in data:
        c2i = data["class_to_idx"]
        assert len(c2i) == 38, f"Expected 38 class_to_idx mappings, got {len(c2i)}"
        for idx, name in enumerate(classes):
            assert c2i[name] == idx, f"Mapping mismatch: class '{name}' has index {c2i[name]}, expected {idx}"

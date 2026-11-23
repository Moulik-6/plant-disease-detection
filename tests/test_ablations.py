import csv
from pathlib import Path
import pytest

from ml.src.datasets import ManifestDataset
from ml.train import load_inherited_config

def test_exclude_sources_synthetic(tmp_path):
    """
    Verify that exclude_sources removes matching sources from training and validation splits,
    does not filter test splits, and that an empty list or None changes nothing.
    """
    manifest_file = tmp_path / "manifest.csv"
    dummy_c2i = {"Apple___healthy": 0}

    # Create synthetic manifest
    fieldnames = [
        "path", "source", "unified_label", "label_idx", "split",
        "legacy_split", "cache_path", "group_id", "confidence_level", "pv_dup"
    ]
    rows = [
        {"path": "p1.jpg", "source": "plantvillage", "unified_label": "Apple___healthy", "split": "pv_train"},
        {"path": "p2.jpg", "source": "plantdoc", "unified_label": "Apple___healthy", "split": "field_train"},
        {"path": "p3.jpg", "source": "plantwild", "unified_label": "Apple___healthy", "split": "field_train"},
        {"path": "p4.jpg", "source": "plantwild", "unified_label": "Apple___healthy", "split": "field_val"},
        {"path": "p5.jpg", "source": "plantdoc", "unified_label": "Apple___healthy", "split": "field_val"},
        {"path": "p6.jpg", "source": "plantwild", "unified_label": "Apple___healthy", "split": "field_test"},
        {"path": "p7.jpg", "source": "plantdoc", "unified_label": "Apple___healthy", "split": "field_test"},
    ]

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # 1. Training split: exclude_sources=['plantwild']
    ds_train = ManifestDataset(
        manifest_path=manifest_file,
        splits=["field_train"],
        class_to_idx=dummy_c2i,
        use_cache=False,
        exclude_sources=["plantwild"],
    )
    assert len(ds_train) == 1, f"Expected 1 sample in field_train, got {len(ds_train)}"
    assert ds_train.samples[0]["source"] == "plantdoc"
    assert ds_train.num_excluded == 1
    assert ds_train.excluded_by_split.get("field_train") == 1

    # 2. Validation split: exclude_sources=['plantwild']
    ds_val = ManifestDataset(
        manifest_path=manifest_file,
        splits=["field_val"],
        class_to_idx=dummy_c2i,
        use_cache=False,
        exclude_sources=["plantwild"],
    )
    assert len(ds_val) == 1, f"Expected 1 sample in field_val, got {len(ds_val)}"
    assert ds_val.samples[0]["source"] == "plantdoc"
    assert ds_val.num_excluded == 1

    # 3. Test split: not filtered (no exclude_sources passed, as in evaluate.py)
    ds_test = ManifestDataset(
        manifest_path=manifest_file,
        splits=["field_test"],
        class_to_idx=dummy_c2i,
        use_cache=False,
    )
    assert len(ds_test) == 2, f"Expected both samples in field_test, got {len(ds_test)}"
    test_sources = {s["source"] for s in ds_test.samples}
    assert test_sources == {"plantwild", "plantdoc"}
    assert ds_test.num_excluded == 0

    # 4. Empty list exclude_sources=[] changes nothing
    ds_empty = ManifestDataset(
        manifest_path=manifest_file,
        splits=["field_train"],
        class_to_idx=dummy_c2i,
        use_cache=False,
        exclude_sources=[],
    )
    assert len(ds_empty) == 2
    assert ds_empty.num_excluded == 0

    # 5. None exclude_sources changes nothing
    ds_none = ManifestDataset(
        manifest_path=manifest_file,
        splits=["field_train"],
        class_to_idx=dummy_c2i,
        use_cache=False,
        exclude_sources=None,
    )
    assert len(ds_none) == 2
    assert ds_none.num_excluded == 0


def test_all_experiment_configs_load():
    """Verify that every config in ml/configs/experiment/ parses and resolves inheritance cleanly."""
    repo_root = Path(__file__).resolve().parents[1]
    exp_dir = repo_root / "ml" / "configs" / "experiment"
    assert exp_dir.exists(), f"Experiment directory not found: {exp_dir}"

    yaml_files = list(exp_dir.glob("*.yaml"))
    assert len(yaml_files) >= 7, f"Expected at least 7 experiment configs, found {len(yaml_files)}"

    for yf in sorted(yaml_files):
        cfg = load_inherited_config(yf)
        assert isinstance(cfg, dict), f"Config {yf.name} did not resolve to a dict"
        assert "exp_name" in cfg, f"Config {yf.name} missing 'exp_name'"
        assert "dataset" in cfg, f"Config {yf.name} missing 'dataset'"
        assert "training" in cfg, f"Config {yf.name} missing 'training'"
        assert "model" in cfg, f"Config {yf.name} missing 'model'"

        # Verify exclude_sources is list when present
        ex_sources = cfg.get("dataset", {}).get("exclude_sources")
        if ex_sources is not None:
            assert isinstance(ex_sources, list), f"exclude_sources in {yf.name} must be a list"

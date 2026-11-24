import ast
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch

from ml.analysis.gradcam import load_model_and_classes
from ml.src.models import build_model


def test_load_model_and_classes_embedded_config_success(tmp_path):
    """
    (a) Save a tiny random-weight resnet50 checkpoint that contains a 'config',
    then load it through load_model_and_classes with a config_path that does not exist,
    and expect success without FileNotFoundError.
    """
    config = {
        "model": {
            "architecture": "resnet50",
            "pretrained": False,
            "num_classes": 3,
        }
    }
    model = build_model(config, num_classes=3, pretrained=False)
    class_to_idx = {"apple_scab": 0, "black_rot": 1, "cedar_rust": 2}

    ckpt_path = tmp_path / "tiny_resnet50.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "class_to_idx": class_to_idx,
            "architecture": "resnet50",
        },
        ckpt_path,
    )

    non_existent_config = str(tmp_path / "non_existent_config.yaml")
    assert not os.path.exists(non_existent_config)

    loaded_model, loaded_cfg, arch, c2i, i2c = load_model_and_classes(
        config_path=non_existent_config,
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
    )

    assert loaded_model is not None
    assert arch == "resnet50"
    assert c2i == class_to_idx
    assert len(i2c) == 3
    assert loaded_cfg == config


def test_all_entrypoints_help_exit_code():
    """
    (b) For every entry-point script in ml/ (train.py, evaluate.py,
    data/build_manifest.py, data/make_cache.py, analysis/*.py, legacy/*.py)
    run 'python <script> --help' in a subprocess and require exit code 0.
    """
    repo_root = Path(__file__).resolve().parents[1]

    entrypoints = [
        repo_root / "ml" / "train.py",
        repo_root / "ml" / "evaluate.py",
        repo_root / "ml" / "data" / "build_manifest.py",
        repo_root / "ml" / "data" / "make_cache.py",
    ]

    for p in sorted((repo_root / "ml" / "analysis").glob("*.py")):
        if p.name != "__init__.py":
            entrypoints.append(p)

    for p in sorted((repo_root / "ml" / "legacy").glob("*.py")):
        if p.name != "__init__.py":
            entrypoints.append(p)

    assert len(entrypoints) >= 14, f"Expected at least 14 entrypoint scripts, found {len(entrypoints)}"

    import concurrent.futures

    def check_help(script: Path):
        assert script.exists(), f"Entrypoint script not found: {script}"
        res = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=40,
        )
        return script, res.returncode, res.stdout, res.stderr

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(check_help, entrypoints))

    for script, code, stdout, stderr in results:
        assert code == 0, (
            f"Script '{script.relative_to(repo_root)}' exited with code {code}.\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


def test_yaml_path_string_literals_exist():
    """
    (c) Scan every .py file under ml/ and tests/ for string literals that look
    like repository-relative yaml paths starting with ml/configs/ or ml/legacy/
    (skip strings containing <, {, * or placeholders) and assert each exists on disk.
    """
    repo_root = Path(__file__).resolve().parents[1]
    matched_paths = []

    for base_dir_name in ["ml", "tests"]:
        base_dir = repo_root / base_dir_name
        for root, _, files in os.walk(base_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                py_path = Path(root) / file
                try:
                    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
                except Exception as exc:
                    pytest.fail(f"Could not parse Python AST in {py_path}: {exc}")

                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        val = node.value.strip().replace("\\", "/")
                        # Match strings starting with ml/configs/ or ml/legacy/
                        if val.startswith("ml/configs/") or val.startswith("ml/legacy/"):
                            # Skip if containing placeholders or template tokens
                            if any(token in val for token in ["<", "{", "*", "$", "placeholder"]):
                                continue
                            # Focus on yaml files or directory targets
                            matched_paths.append((py_path.relative_to(repo_root), val))

    assert len(matched_paths) > 0, "Expected to find repository-relative config/legacy paths in codebase"

    missing = []
    for source_file, rel_path in matched_paths:
        target_path = repo_root / rel_path
        if not target_path.exists():
            missing.append(f"{source_file}: '{rel_path}' does not exist on disk")

    assert not missing, "Found references to non-existent config/legacy paths:\n" + "\n".join(missing)

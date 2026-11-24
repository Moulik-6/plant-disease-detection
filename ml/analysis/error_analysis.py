"""
Comprehensive Error Analysis for Plant Disease Detection Models.
Analyzes evaluation predictions, identifies confusion clusters, computes per-class/source recalls,
generates thumbnail contact sheets, and produces Grad-CAM diagnostic saliency panels.

Usage:
    python ml/analysis/error_analysis.py --eval-dir <eval folder> --set field_test --out results/error_analysis/<name> [--gradcam --run <run dir>]
"""

import argparse
import csv
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw

# Ensure repo root is in sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

logger = logging.getLogger("error_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Error analysis and visual diagnostics on test predictions.")
    parser.add_argument("--eval-dir", type=str, required=True, help="Path to evaluation folder containing predictions_<set>.csv")
    parser.add_argument("--set", type=str, default="field_test", help="Evaluation set name (default: field_test)")
    parser.add_argument("--out", type=str, required=True, help="Output directory for error analysis artifacts")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Path to manifest CSV (default: data/manifest.csv)")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visual explanation panels")
    parser.add_argument("--run", type=str, default=None, help="Run directory containing best.pth (required if --gradcam is set)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling (default: 42)")
    parser.add_argument("--device", type=str, default=None, help="Device for Grad-CAM (cuda/cpu, default: auto)")
    return parser


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads manifest.csv into a dictionary indexed by original path."""
    manifest_map: Dict[str, Dict[str, Any]] = {}
    if not manifest_path.is_file():
        logger.warning("Manifest not found at %s. Will attempt direct image path resolution.", manifest_path)
        return manifest_map

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("path", "").strip()
            if p:
                manifest_map[p] = row
    logger.info("Loaded %d manifest records from %s", len(manifest_map), manifest_path)
    return manifest_map


def load_predictions(eval_dir: Path, set_name: str) -> List[Dict[str, Any]]:
    """Loads predictions_<set>.csv from the evaluation directory."""
    csv_file = eval_dir / f"predictions_{set_name}.csv"
    if not csv_file.is_file():
        # Fallback check inside eval/ subdirectory if eval-dir was pointing to run root
        cand = eval_dir / "eval" / f"predictions_{set_name}.csv"
        if cand.is_file():
            csv_file = cand
        else:
            raise FileNotFoundError(f"Predictions file not found at {csv_file}")

    records: List[Dict[str, Any]] = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse fields safely
            conf_str = row.get("confidence", "0.0")
            try:
                conf = float(conf_str)
            except ValueError:
                conf = 0.0

            true_idx_str = row.get("true_idx", "-1")
            try:
                true_idx = int(true_idx_str)
            except ValueError:
                true_idx = -1

            pred_idx_str = row.get("pred_idx", "-1")
            try:
                pred_idx = int(pred_idx_str)
            except ValueError:
                pred_idx = -1

            corr_str = str(row.get("correct", "")).strip()
            if corr_str in ("1", "True", "true"):
                correct = 1
            elif corr_str in ("0", "False", "false"):
                correct = 0
            else:
                correct = 1 if (true_idx >= 0 and true_idx == pred_idx) else 0

            records.append({
                "path": row.get("path", "").strip(),
                "source": row.get("source", "").strip(),
                "true_label": row.get("true_label", "").strip(),
                "true_idx": true_idx,
                "pred_label": row.get("pred_label", "").strip(),
                "pred_idx": pred_idx,
                "confidence": conf,
                "correct": correct,
                "group_id": row.get("group_id", "").strip(),
            })

    logger.info("Loaded %d prediction records from %s", len(records), csv_file)
    return records


def resolve_image_path(
    pred_path: str,
    source: str,
    manifest_map: Dict[str, Dict[str, Any]],
    project_root: Path = _repo_root,
) -> Optional[Path]:
    """
    Resolves image file on disk according to specification:
    1. Check cache_path in manifest when present and exists.
    2. Check original path under source root or data directory.
    Never reformats file names.
    """
    m_row = manifest_map.get(pred_path)
    if m_row:
        cache_p = m_row.get("cache_path", "").strip()
        if cache_p:
            cand = Path(cache_p)
            if not cand.is_absolute():
                cand = project_root / cand
            if cand.is_file():
                return cand

        orig_p = m_row.get("path", "").strip()
        if orig_p:
            cand = Path(orig_p)
            if not cand.is_absolute():
                cand = project_root / cand
            if cand.is_file():
                return cand

    # Direct relative or absolute path check
    direct = Path(pred_path)
    if direct.is_file():
        return direct
    if (project_root / direct).is_file():
        return project_root / direct

    # Check under data/sources/<source>/ and data/<source>/
    if source:
        candidates = [
            project_root / "data" / "sources" / source / pred_path,
            project_root / "data" / source / pred_path,
            project_root / "data" / pred_path,
        ]
        for c in candidates:
            if c.is_file():
                return c

    return None


def get_thumbnail(img_path: Optional[Path], size: int = 224) -> np.ndarray:
    """Loads image, applies EXIF orientation, squashes to thumbnail size, and returns RGB array."""
    if img_path and img_path.is_file():
        try:
            with Image.open(img_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((size, size), Image.Resampling.BILINEAR)
                return np.array(img)
        except Exception as e:
            logger.warning("Could not read image at %s: %s", img_path, e)

    # Fallback blank placeholder
    placeholder = Image.new("RGB", (size, size), color=(230, 230, 230))
    draw = ImageDraw.Draw(placeholder)
    draw.text((size // 4, size // 2 - 10), "Image Missing", fill=(100, 100, 100))
    return np.array(placeholder)


# ---------------------------------------------------------------------------
# 1. Confusion Pairs Table
# ---------------------------------------------------------------------------

def compute_confusion_pairs(
    records: List[Dict[str, Any]],
    top_k: int = 15,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Computes confusion pairs from misclassified predictions, sorted by frequency.
    Returns: (top_pairs, total_errors)
    """
    errors = [r for r in records if r["correct"] == 0]
    total_errors = len(errors)

    if total_errors == 0:
        return [], 0

    pair_counts: Dict[Tuple[str, str], int] = {}
    for r in errors:
        pair = (r["true_label"], r["pred_label"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    results: List[Dict[str, Any]] = []
    cum_share = 0.0
    for rank, ((t_label, p_label), count) in enumerate(sorted_pairs[:top_k], start=1):
        share = count / total_errors
        cum_share += share
        results.append({
            "rank": rank,
            "true_label": t_label,
            "pred_label": p_label,
            "count": count,
            "share": share,
            "share_pct": share * 100.0,
            "cum_share_pct": cum_share * 100.0,
        })

    return results, total_errors


# ---------------------------------------------------------------------------
# 2. Contact Sheets
# ---------------------------------------------------------------------------

def render_contact_sheet(
    items: List[Dict[str, Any]],
    out_path: Path,
    n_rows: int,
    n_cols: int,
    sheet_title: str = "",
    thumb_size: int = 224,
) -> None:
    """Renders a grid contact sheet of thumbnail images with true/pred/confidence captions."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.6), squeeze=False)
    if sheet_title:
        fig.suptitle(sheet_title, fontsize=13, fontweight="bold", y=0.99)

    for i in range(n_rows * n_cols):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]

        if i < len(items):
            item = items[i]
            thumb = get_thumbnail(item.get("resolved_path"), size=thumb_size)
            ax.imshow(thumb)

            true_txt = item.get("true_label", "")
            pred_txt = item.get("pred_label", "")
            conf_val = item.get("confidence", 0.0)

            # Clean formatting for readability
            caption = f"True: {true_txt}\nPred: {pred_txt}\nConf: {conf_val:.2%}"
            ax.set_title(caption, fontsize=8, pad=4)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_pair_contact_sheets(
    records: List[Dict[str, Any]],
    top_pairs: List[Dict[str, Any]],
    manifest_map: Dict[str, Dict[str, Any]],
    out_dir: Path,
    seed: int = 42,
    top_n_pairs: int = 6,
    samples_per_pair: int = 6,
) -> List[Path]:
    """Generates contact sheets for each of the top N confusion pairs."""
    rng = random.Random(seed)
    saved_paths: List[Path] = []

    for pair_info in top_pairs[:top_n_pairs]:
        rank = pair_info["rank"]
        t_label = pair_info["true_label"]
        p_label = pair_info["pred_label"]

        pair_records = [
            r for r in records
            if r["correct"] == 0 and r["true_label"] == t_label and r["pred_label"] == p_label
        ]

        if not pair_records:
            continue

        # Deterministic sample if more than required
        if len(pair_records) > samples_per_pair:
            # Sort deterministically by confidence, then sample or take top
            selected = rng.sample(pair_records, samples_per_pair)
        else:
            selected = list(pair_records)

        # Resolve paths
        for s in selected:
            s["resolved_path"] = resolve_image_path(s["path"], s.get("source", ""), manifest_map)

        safe_pair_name = f"pair_{rank:02d}_{t_label[:15]}_vs_{p_label[:15]}".replace(" ", "_").replace("/", "_")
        sheet_file = out_dir / f"{safe_pair_name}.png"

        title = f"Pair #{rank}: True: {t_label} -> Pred: {p_label} (n={len(pair_records)}, {pair_info['share_pct']:.1f}% of errors)"
        render_contact_sheet(
            items=selected,
            out_path=sheet_file,
            n_rows=2,
            n_cols=3,
            sheet_title=title,
        )
        saved_paths.append(sheet_file)

    return saved_paths


def generate_confident_errors_contact_sheet(
    records: List[Dict[str, Any]],
    manifest_map: Dict[str, Dict[str, Any]],
    out_dir: Path,
    max_images: int = 24,
) -> Path:
    """Generates contact sheet of the most confident wrong predictions."""
    errors = [r for r in records if r["correct"] == 0]
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    top_errors = errors[:max_images]

    for item in top_errors:
        item["resolved_path"] = resolve_image_path(item["path"], item.get("source", ""), manifest_map)

    out_file = out_dir / "most_confident_errors.png"
    title = f"Top {len(top_errors)} Most Confident Errors"
    render_contact_sheet(
        items=top_errors,
        out_path=out_file,
        n_rows=4,
        n_cols=6,
        sheet_title=title,
    )
    return out_file


# ---------------------------------------------------------------------------
# 3. Recall Analytics
# ---------------------------------------------------------------------------

def compute_recalls(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Computes per-class recall and per-source recall."""
    # Per-class
    class_stats: Dict[str, Dict[str, int]] = {}
    for r in records:
        lbl = r["true_label"]
        if not lbl:
            continue
        if lbl not in class_stats:
            class_stats[lbl] = {"total": 0, "correct": 0}
        class_stats[lbl]["total"] += 1
        if r["correct"] == 1:
            class_stats[lbl]["correct"] += 1

    per_class: List[Dict[str, Any]] = []
    for lbl, stats in sorted(class_stats.items(), key=lambda x: x[0]):
        tot = stats["total"]
        cor = stats["correct"]
        rec = cor / tot if tot > 0 else 0.0
        err = tot - cor
        per_class.append({
            "class_name": lbl,
            "total": tot,
            "correct": cor,
            "errors": err,
            "recall": rec,
            "recall_pct": rec * 100.0,
        })

    # Per-source
    source_stats: Dict[str, Dict[str, int]] = {}
    for r in records:
        src = r.get("source") or "unknown"
        if src not in source_stats:
            source_stats[src] = {"total": 0, "correct": 0}
        source_stats[src]["total"] += 1
        if r["correct"] == 1:
            source_stats[src]["correct"] += 1

    per_source: List[Dict[str, Any]] = []
    for src, stats in sorted(source_stats.items(), key=lambda x: x[0]):
        tot = stats["total"]
        cor = stats["correct"]
        rec = cor / tot if tot > 0 else 0.0
        err = tot - cor
        per_source.append({
            "source": src,
            "total": tot,
            "correct": cor,
            "errors": err,
            "recall": rec,
            "recall_pct": rec * 100.0,
        })

    return per_class, per_source


def get_weakest_classes(
    per_class_recall: List[Dict[str, Any]],
    min_samples: int = 15,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Filters classes with at least min_samples test images and returns top_k weakest by recall."""
    filtered = [c for c in per_class_recall if c["total"] >= min_samples]
    # Sort ascending by recall, then descending by total
    sorted_weakest = sorted(filtered, key=lambda x: (x["recall"], -x["total"], x["class_name"]))
    return sorted_weakest[:top_k]


def save_recalls_csv(
    per_class: List[Dict[str, Any]],
    per_source: List[Dict[str, Any]],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Exports per-class and per-source recall tables to CSV files."""
    class_csv = out_dir / "per_class_recall.csv"
    with open(class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "total", "correct", "errors", "recall", "recall_pct"])
        writer.writeheader()
        for r in per_class:
            writer.writerow({
                "class_name": r["class_name"],
                "total": r["total"],
                "correct": r["correct"],
                "errors": r["errors"],
                "recall": f"{r['recall']:.6f}",
                "recall_pct": f"{r['recall_pct']:.2f}",
            })

    source_csv = out_dir / "per_source_recall.csv"
    with open(source_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "total", "correct", "errors", "recall", "recall_pct"])
        writer.writeheader()
        for r in per_source:
            writer.writerow({
                "source": r["source"],
                "total": r["total"],
                "correct": r["correct"],
                "errors": r["errors"],
                "recall": f"{r['recall']:.6f}",
                "recall_pct": f"{r['recall_pct']:.2f}",
            })

    return class_csv, source_csv


# ---------------------------------------------------------------------------
# 4. Grad-CAM Diagnostics
# ---------------------------------------------------------------------------

def run_gradcam_diagnostics(
    run_dir: Path,
    records: List[Dict[str, Any]],
    manifest_map: Dict[str, Dict[str, Any]],
    out_dir: Path,
    seed: int = 42,
    device_str: Optional[str] = None,
) -> Optional[Path]:
    """
    Loads checkpoint from run_dir, checks architecture, and creates panels:
    [Original | Grad-CAM Predicted | Grad-CAM True] for 8 correct and 8 wrong images.
    Skips fusion models with an informative message.
    """
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from ml.analysis.gradcam import get_target_layers_and_transform, preprocess_image
    from ml.src.model_zoo import load_checkpoint_model

    # Locate checkpoint
    ckpt_path = run_dir / "best.pth"
    if not ckpt_path.is_file():
        cand = run_dir / "checkpoints" / "best_model.pth"
        if cand.is_file():
            ckpt_path = cand
        else:
            cand2 = run_dir / "best_model.pth"
            if cand2.is_file():
                ckpt_path = cand2
            else:
                logger.warning("Checkpoint best.pth not found in %s. Skipping Grad-CAM.", run_dir)
                return None

    # Resolve device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, ckpt_cfg, arch, c_to_i, i_to_c = load_checkpoint_model(ckpt_path, device=device)
    except Exception as e:
        logger.error("Failed to load checkpoint %s: %s. Skipping Grad-CAM.", ckpt_path, e)
        return None

    arch_lower = str(arch).lower()
    if "fusion" in arch_lower:
        msg = f"[Grad-CAM] Skipping Grad-CAM: architecture '{arch}' is a fusion model (dual-backbone Grad-CAM is not supported)."
        print(msg)
        logger.info(msg)
        return None

    logger.info("Running Grad-CAM explainability with architecture: %s on device: %s", arch, device)
    try:
        target_layers, reshape_transform = get_target_layers_and_transform(model, arch)
    except Exception as e:
        logger.warning("Could not determine Grad-CAM target layers for %s: %s", arch, e)
        return None

    rng = random.Random(seed)
    correct_all = [r for r in records if r["correct"] == 1]
    wrong_all = [r for r in records if r["correct"] == 0]

    correct_sample = rng.sample(correct_all, min(8, len(correct_all)))
    wrong_sample = rng.sample(wrong_all, min(8, len(wrong_all)))

    gradcam_dir = out_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    def process_and_save_panel(sample_item: Dict[str, Any], prefix: str, idx: int) -> None:
        p_resolved = resolve_image_path(sample_item["path"], sample_item.get("source", ""), manifest_map)
        if not p_resolved or not p_resolved.is_file():
            return

        try:
            input_tensor, rgb_img = preprocess_image(str(p_resolved), img_size=224)
            input_tensor = input_tensor.to(device)

            pred_c = sample_item["pred_idx"]
            true_c = sample_item["true_idx"]

            # Compute Grad-CAM for Predicted
            model.zero_grad()
            with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
                cam_pred = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(int(pred_c))])[0, :]
            overlay_pred = show_cam_on_image(rgb_img, cam_pred, use_rgb=True, image_weight=0.4)

            # Compute Grad-CAM for True
            model.zero_grad()
            with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
                cam_true = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(int(true_c))])[0, :]
            overlay_true = show_cam_on_image(rgb_img, cam_true, use_rgb=True, image_weight=0.4)

            # Render 3-column panel: Original | Grad-CAM (Pred) | Grad-CAM (True)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.8))
            ax1.imshow((rgb_img * 255).astype(np.uint8))
            ax1.set_title("Original Photo", fontsize=9)
            ax1.axis("off")

            ax2.imshow(overlay_pred)
            ax2.set_title(f"CAM Pred: {sample_item['pred_label']}\nConf: {sample_item['confidence']:.2%}", fontsize=8)
            ax2.axis("off")

            ax3.imshow(overlay_true)
            ax3.set_title(f"CAM True: {sample_item['true_label']}", fontsize=8)
            ax3.axis("off")

            status_str = "CORRECT" if sample_item["correct"] == 1 else "WRONG"
            fig.suptitle(f"{status_str} Prediction — Group: {sample_item.get('group_id', '')}", fontsize=10, fontweight="bold")
            plt.tight_layout()

            out_panel = gradcam_dir / f"{prefix}_{idx:02d}.png"
            fig.savefig(out_panel, dpi=140, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning("Error generating Grad-CAM for %s: %s", sample_item["path"], e)

    for i, s in enumerate(correct_sample, start=1):
        process_and_save_panel(s, "correct", i)

    for i, s in enumerate(wrong_sample, start=1):
        process_and_save_panel(s, "error", i)

    logger.info("Saved Grad-CAM diagnostic panels to %s", gradcam_dir)
    return gradcam_dir


# ---------------------------------------------------------------------------
# 5. Summary Markdown Generation & Console Printing
# ---------------------------------------------------------------------------

def generate_summary_markdown(
    eval_dir: Path,
    set_name: str,
    total_images: int,
    total_errors: int,
    overall_acc: float,
    top_pairs: List[Dict[str, Any]],
    weakest_classes: List[Dict[str, Any]],
    per_source: List[Dict[str, Any]],
    out_dir: Path,
    gradcam_dir: Optional[Path] = None,
) -> Path:
    """Writes executive summary.md to the output directory."""
    lines = [
        f"# Error Analysis Report: `{set_name}`",
        "",
        f"- **Evaluation Directory**: `{eval_dir}`",
        f"- **Dataset Set**: `{set_name}`",
        f"- **Total Samples Evaluated**: {total_images:,}",
        f"- **Total Misclassifications**: {total_errors:,}",
        f"- **Top-1 Accuracy**: {overall_acc:.2%}",
        "",
        "---",
        "",
        "## 1. Per-Source Performance Breakdown",
        "",
        "| Source | Samples | Correct | Errors | Recall / Accuracy |",
        "|---|---|---|---|---|",
    ]

    for s in per_source:
        lines.append(f"| `{s['source']}` | {s['total']:,} | {s['correct']:,} | {s['errors']:,} | **{s['recall_pct']:.2f}%** |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Top 15 Most Frequent Confusion Pairs",
        "",
        "| Rank | True Label | Predicted Label | Count | Share of Errors | Cumulative Share |",
        "|---|---|---|---|---|---|",
    ])

    for p in top_pairs:
        lines.append(f"| {p['rank']} | `{p['true_label']}` | `{p['pred_label']}` | {p['count']} | {p['share_pct']:.2f}% | {p['cum_share_pct']:.2f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Ten Weakest Classes (≥ 15 Test Samples)",
        "",
        "| Rank | Class Name | Samples | Correct | Errors | Recall |",
        "|---|---|---|---|---|---|",
    ])

    for rank, c in enumerate(weakest_classes, start=1):
        lines.append(f"| {rank} | `{c['class_name']}` | {c['total']} | {c['correct']} | {c['errors']} | **{c['recall_pct']:.2f}%** |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Visual Diagnostics Artifacts",
        "",
        "- `most_confident_errors.png`: Contact sheet of the 24 most confident erroneous classifications.",
        "- `pair_*.png`: Contact sheets showing 6 example errors for each of the top confusion pairs.",
        "- `per_class_recall.csv`: Full per-class recall statistics.",
        "- `per_source_recall.csv`: Source-level accuracy breakdown.",
    ])

    if gradcam_dir and gradcam_dir.exists():
        lines.append("- `gradcam/`: Grad-CAM saliency explanation panels for 8 correct and 8 error test cases.")
    else:
        lines.append("- *Grad-CAM skipped or not requested.*")

    summary_file = out_dir / "summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return summary_file


def print_console_tables(
    set_name: str,
    total_images: int,
    total_errors: int,
    overall_acc: float,
    top_pairs: List[Dict[str, Any]],
    weakest_classes: List[Dict[str, Any]],
    per_source: List[Dict[str, Any]],
) -> None:
    """Prints clear, human-readable tables to stdout."""
    print("\n" + "=" * 80)
    print(f" ERROR ANALYSIS SUMMARY: {set_name.upper()} (Total: {total_images:,} | Acc: {overall_acc:.2%})")
    print("=" * 80)

    print("\n[Per-Source Breakdown]")
    print(f"{'Source':<18} {'Samples':>8} {'Correct':>8} {'Errors':>8} {'Accuracy':>10}")
    print("-" * 56)
    for s in per_source:
        print(f"{s['source']:<18} {s['total']:>8} {s['correct']:>8} {s['errors']:>8} {s['recall_pct']:>9.2f}%")

    print("\n[Top 15 Most Frequent Confusion Pairs]")
    print(f"{'Rank':<5} {'True Label':<30} {'Predicted Label':<30} {'Count':>6} {'Share':>8} {'Cum%':>8}")
    print("-" * 92)
    for p in top_pairs:
        t_short = (p['true_label'][:28] + '..') if len(p['true_label']) > 30 else p['true_label']
        p_short = (p['pred_label'][:28] + '..') if len(p['pred_label']) > 30 else p['pred_label']
        print(f"#{p['rank']:<4} {t_short:<30} {p_short:<30} {p['count']:>6} {p['share_pct']:>7.2f}% {p['cum_share_pct']:>7.2f}%")

    print(f"\n[10 Weakest Classes (≥ 15 Test Samples)]")
    print(f"{'Rank':<5} {'Class Name':<42} {'Samples':>8} {'Correct':>8} {'Recall':>9}")
    print("-" * 76)
    for rank, c in enumerate(weakest_classes, start=1):
        c_short = (c['class_name'][:40] + '..') if len(c['class_name']) > 42 else c['class_name']
        print(f"#{rank:<4} {c_short:<42} {c['total']:>8} {c['correct']:>8} {c['recall_pct']:>8.2f}%")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    out_dir = Path(args.out).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions
    records = load_predictions(eval_dir, args.set)
    total_images = len(records)
    total_correct = sum(1 for r in records if r["correct"] == 1)
    total_errors = total_images - total_correct
    overall_acc = total_correct / total_images if total_images > 0 else 0.0

    # 2. Load manifest for image path resolution
    manifest_map = load_manifest(manifest_path)

    # 3. Top 15 confusion pairs
    top_pairs, _ = compute_confusion_pairs(records, top_k=15)

    # 4. Per-class & per-source recalls
    per_class, per_source = compute_recalls(records)
    save_recalls_csv(per_class, per_source, out_dir)
    weakest_classes = get_weakest_classes(per_class, min_samples=15, top_k=10)

    # 5. Visual Contact Sheets
    generate_pair_contact_sheets(
        records=records,
        top_pairs=top_pairs,
        manifest_map=manifest_map,
        out_dir=out_dir,
        seed=args.seed,
        top_n_pairs=6,
        samples_per_pair=6,
    )

    generate_confident_errors_contact_sheet(
        records=records,
        manifest_map=manifest_map,
        out_dir=out_dir,
        max_images=24,
    )

    # 6. Grad-CAM Diagnostics if requested
    gradcam_dir = None
    if args.gradcam:
        if not args.run:
            logger.warning("--gradcam specified but --run <run dir> not provided. Skipping Grad-CAM.")
        else:
            run_p = Path(args.run).resolve()
            gradcam_dir = run_gradcam_diagnostics(
                run_dir=run_p,
                records=records,
                manifest_map=manifest_map,
                out_dir=out_dir,
                seed=args.seed,
                device_str=args.device,
            )

    # 7. Print to console & write summary.md
    print_console_tables(
        set_name=args.set,
        total_images=total_images,
        total_errors=total_errors,
        overall_acc=overall_acc,
        top_pairs=top_pairs,
        weakest_classes=weakest_classes,
        per_source=per_source,
    )

    summary_file = generate_summary_markdown(
        eval_dir=eval_dir,
        set_name=args.set,
        total_images=total_images,
        total_errors=total_errors,
        overall_acc=overall_acc,
        top_pairs=top_pairs,
        weakest_classes=weakest_classes,
        per_source=per_source,
        out_dir=out_dir,
        gradcam_dir=gradcam_dir,
    )
    logger.info("Summary markdown written to %s", summary_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Field Dataset Pre-Caching & Resizing Tool.

Reads `data/manifest.csv` and generates optimized, resized JPEG copies for all
field images (PlantDoc and PlantWild) at 512px on the longest side:
  - Longest side 512px (aspect ratio strictly preserved, never upscaled)
  - Color mode: RGB
  - Encoding: JPEG quality 92
  - Output path: data/cache/<source>/<same relative path with a .jpg extension>
  - Acceleration: PIL Image.draft() for fast JPEG decoder downscaling
  - Safe file handling: pathlib/os.scandir preserving odd/raw file names
  - Multi-threading: ThreadPoolExecutor (workers default 8)
  - Idempotency: Skips files already cached
  - Fault tolerance: Skips and counts corrupt or unreadable files
  - Manifest update: Adds `cache_path` column (empty for PlantVillage, which stays 256x256)
  - Atomic write: Replaces data/manifest.csv atomically via temporary file rename
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("make_cache")


def format_bytes(size_bytes: int) -> str:
    """Formats raw byte counts into human-readable B, KB, MB, or GB strings."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB ({size_bytes:,} bytes)"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB ({size_bytes:,} bytes)"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB ({size_bytes:,} bytes)"


def resolve_source_path(
    row: Dict[str, str],
    pv_dir: Path,
    plantdoc_dir: Path,
    plantwild_dir: Path
) -> Path:
    """
    Resolves the absolute or relative disk path of an image record based on its source.
    Preserves exact raw path characters without string reformatting or decoding.
    """
    source = row.get("source", "").lower().strip()
    rel_path = Path(row["path"])

    # If the path in manifest already points to an existing file, use it directly
    if rel_path.exists():
        return rel_path.resolve()

    if source == "plantdoc":
        return (plantdoc_dir / rel_path).resolve()
    elif source == "plantwild":
        return (plantwild_dir / rel_path).resolve()
    elif source == "plantvillage":
        return (pv_dir / rel_path).resolve()
    else:
        return rel_path.resolve()


def process_single_image(
    src_path: Path,
    dst_path: Path,
    max_size: int = 512,
    quality: int = 92
) -> Tuple[str, int]:
    """
    Processes a single field image:
      1. Checks if already cached on disk (skips if non-empty).
      2. Opens with PIL and uses Image.draft("RGB", (max_size, max_size)) for fast JPEG decode.
      3. Converts to RGB.
      4. Downscales if max(w, h) > max_size (preserves aspect ratio, never upscales).
      5. Writes atomically to destination JPEG with quality=92.

    Returns:
        (status, file_size):
            status in {'cached', 'processed', 'corrupt'}
            file_size: size in bytes on disk
    """
    # 1. Skip if already cached and non-empty
    if dst_path.exists():
        try:
            sz = dst_path.stat().st_size
            if sz > 0:
                return "cached", sz
        except OSError:
            pass

    # Verify source file exists
    if not src_path.exists():
        logger.warning("Source image missing: %s", src_path)
        return "corrupt", 0

    # 2. Decode and resize
    try:
        with Image.open(src_path) as im:
            # Instruct libjpeg decoder to downscale at DCT phase for speed
            im.draft("RGB", (max_size, max_size))
            im = im.convert("RGB")

            w, h = im.size
            # Never upscale: only resize if longest side exceeds max_size
            if max(w, h) > max_size:
                if w >= h:
                    new_w = max_size
                    new_h = max(1, int(round(h * float(max_size) / float(w))))
                else:
                    new_h = max_size
                    new_w = max(1, int(round(w * float(max_size) / float(h))))
                im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

            # Ensure target parent directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write via temporary file
            tmp_dst = dst_path.with_name(f"{dst_path.stem}.tmp_{os.getpid()}_{time.time_ns()}.jpg")
            im.save(tmp_dst, format="JPEG", quality=quality, optimize=True)
            os.replace(tmp_dst, dst_path)

            return "processed", dst_path.stat().st_size
    except Exception as e:
        logger.warning("Corrupt or unreadable image skipped '%s': %s", src_path, e)
        return "corrupt", 0


def make_cache(
    manifest_csv: str = "data/manifest.csv",
    cache_dir: str = "data/cache",
    pv_dir: str = "data/plantvillage_dataset/color",
    plantdoc_dir: str = "data/plantdoc/PlantDoc-Dataset",
    plantwild_dir: str = "data/plantwild/plantwild",
    workers: int = 8,
    max_size: int = 512,
    quality: int = 92
) -> None:
    """Reads manifest.csv, caches resized field images, updates manifest with cache_path atomically."""
    manifest_path = Path(manifest_csv).resolve()
    cache_root = Path(cache_dir).resolve()
    pv_root = Path(pv_dir).resolve()
    pdoc_root = Path(plantdoc_dir).resolve()
    pwild_root = Path(plantwild_dir).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    logger.info("Reading manifest from: %s", manifest_path)
    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    total_records = len(rows)
    logger.info("Loaded %d manifest records.", total_records)

    # Determine tasks for field images
    # PlantVillage stays 256x256 and is used directly; cache_path remains empty
    tasks: List[Tuple[int, Path, Path]] = []
    pv_count = 0
    field_count = 0

    for idx, row in enumerate(rows):
        src_name = row.get("source", "").lower().strip()
        if src_name in {"plantdoc", "plantwild"}:
            field_count += 1
            src_file = resolve_source_path(row, pv_root, pdoc_root, pwild_root)
            # Dest: data/cache/<source>/<same relative path with .jpg extension>
            rel_with_jpg = Path(row["path"]).with_suffix(".jpg")
            dst_file = cache_root / src_name / rel_with_jpg
            tasks.append((idx, src_file, dst_file))
        else:
            pv_count += 1
            # Empty cache_path for PlantVillage
            row["cache_path"] = ""

    logger.info("Cache Target Partitioning:")
    logger.info("  - PlantVillage images (skipped, direct 256x256): %d", pv_count)
    logger.info("  - Field images to cache (PlantDoc + PlantWild):   %d", field_count)

    # Execute parallel processing
    processed_count = 0
    cached_skip_count = 0
    corrupt_count = 0
    total_cache_bytes = 0

    logger.info("Processing %d field images using ThreadPoolExecutor(workers=%d)...", len(tasks), workers)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(process_single_image, src, dst, max_size, quality): (idx, dst)
            for idx, src, dst in tasks
        }

        for future in as_completed(future_to_idx):
            idx, dst_path = future_to_idx[future]
            status, byte_size = future.result()

            if status == "processed":
                processed_count += 1
                total_cache_bytes += byte_size
                # Record path relative to project root or normalized posix path
                rel_cache_str = Path("data/cache") / dst_path.relative_to(cache_root)
                rows[idx]["cache_path"] = rel_cache_str.as_posix()
            elif status == "cached":
                cached_skip_count += 1
                total_cache_bytes += byte_size
                rel_cache_str = Path("data/cache") / dst_path.relative_to(cache_root)
                rows[idx]["cache_path"] = rel_cache_str.as_posix()
            elif status == "corrupt":
                corrupt_count += 1
                rows[idx]["cache_path"] = ""

    elapsed = time.time() - t0
    rate = len(tasks) / elapsed if elapsed > 0 else 0.0
    logger.info("Field caching finished in %.2fs (%.1f img/s).", elapsed, rate)

    # Ensure cache_path column is in fieldnames
    if "cache_path" not in fieldnames:
        fieldnames.append("cache_path")

    # Write data/manifest.csv atomically
    tmp_manifest = manifest_path.with_name(f"{manifest_path.stem}.tmp_{os.getpid()}_{time.time_ns()}.csv")
    logger.info("Writing updated manifest atomically to: %s", manifest_path)
    with open(tmp_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    os.replace(tmp_manifest, manifest_path)
    logger.info("Atomically updated %s with 'cache_path' column.", manifest_path)

    # Compute overall cache directory size on disk
    disk_cache_bytes = 0
    cached_files_on_disk = 0
    if cache_root.exists():
        for root, _, files in os.walk(cache_root):
            for file_name in files:
                if file_name.lower().endswith(".jpg"):
                    p = Path(root) / file_name
                    try:
                        disk_cache_bytes += p.stat().st_size
                        cached_files_on_disk += 1
                    except OSError:
                        pass

    # Print comprehensive diagnostic summary
    print("\n" + "=" * 80)
    print(" FIELD IMAGE CACHE GENERATION SUMMARY")
    print("=" * 80)
    print(f"  Total Manifest Records:           {total_records:,}")
    print(f"  PlantVillage (Direct 256x256):    {pv_count:,} (skipped from cache)")
    print(f"  Field Records Evaluated:          {field_count:,} (PlantDoc + PlantWild)")
    print(f"  Newly Processed & Resized:        {processed_count:,}")
    print(f"  Already Cached (Skipped):         {cached_skip_count:,}")
    print(f"  Corrupt / Unreadable (Skipped):   {corrupt_count:,}")
    print(f"  Total Valid Cached Files on Disk: {cached_files_on_disk:,}")
    print(f"  Total Cache Disk Size:            {format_bytes(disk_cache_bytes)}")
    print(f"  Manifest Output:                  {manifest_path.as_posix()} (atomic update)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-caches and resizes field images (PlantDoc & PlantWild) to 512px longest side."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Path to manifest CSV (default: data/manifest.csv)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Directory to store resized JPEG cache (default: data/cache)"
    )
    parser.add_argument(
        "--pv-dir",
        type=str,
        default="data/plantvillage_dataset/color",
        help="PlantVillage dataset root (default: data/plantvillage_dataset/color)"
    )
    parser.add_argument(
        "--plantdoc-dir",
        type=str,
        default="data/plantdoc/PlantDoc-Dataset",
        help="PlantDoc dataset root (default: data/plantdoc/PlantDoc-Dataset)"
    )
    parser.add_argument(
        "--plantwild-dir",
        type=str,
        default="data/plantwild/plantwild",
        help="PlantWild dataset root (default: data/plantwild/plantwild)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Longest side maximum size in pixels (default: 512)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=92,
        help="JPEG output quality (default: 92)"
    )

    args = parser.parse_args()

    make_cache(
        manifest_csv=args.manifest,
        cache_dir=args.cache_dir,
        pv_dir=args.pv_dir,
        plantdoc_dir=args.plantdoc_dir,
        plantwild_dir=args.plantwild_dir,
        workers=args.workers,
        max_size=args.max_size,
        quality=args.quality
    )


if __name__ == "__main__":
    main()

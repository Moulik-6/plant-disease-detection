"""
PlantWild v1 Dataset Ingestion Adapter.

Parses PlantWild v1 dataset structure:
    Expected layout:
        data/plantwild/plantwild/
            ├── images/
            │   ├── <class name>/
            │   │   ├── <image_1>.jpg
            │   │   └── ...
            ├── url_record.json (source URLs for images)
            ├── train.txt / val.txt / test.txt (official split lists)
            └── ...

Aligns PlantWild categories with the 38 PlantVillage classes using `ml/configs/plantwild_map.yaml`.
Folders not present in the mapping configuration receive `unified_label = "other"` and are
preserved in the manifest for background/out-of-distribution evaluation ('other_pool').

Integrates `url_record.json` to extract raw source URLs and normalized URLs (with query strings
and URL fragments stripped) to force identical-source imagery into the same duplicate group.
"""

from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import yaml

logger = logging.getLogger(__name__)


def normalize_source_url(raw_url: str) -> str:
    """
    Normalizes a source URL:
      - Lowercase scheme and host
      - Drop fragment
      - Strip trailing slash from path
      - Keep query string, dropping only tracking parameters (utm_*, fbclid, gclid, ref)
      - Sort remaining query parameters
    """
    if not raw_url or not isinstance(raw_url, str):
        return ""
    clean = raw_url.strip()
    try:
        parsed = urlparse(clean)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        params = parsed.params

        filtered_params: List[Tuple[str, str]] = []
        if parsed.query:
            for k, v in parse_qsl(parsed.query, keep_blank_values=True):
                k_lower = k.lower()
                if k_lower.startswith("utm_") or k_lower in {"fbclid", "gclid", "ref"}:
                    continue
                filtered_params.append((k, v))
            filtered_params.sort(key=lambda x: (x[0], x[1]))

        query = urlencode(filtered_params, doseq=True) if filtered_params else ""

        normalized = urlunparse((
            scheme,
            netloc,
            path,
            params,
            query,
            ""  # strip fragment
        ))
        return normalized
    except Exception:
        # Fallback manual extraction
        return clean.split("#")[0].strip().rstrip("/")


def parse_url_records(record_file: Path) -> Dict[str, Tuple[str, str]]:
    """
    Parses `url_record.json` into a mapping of image identifiers to (raw_url, normalized_url).
    Supports multiple common JSON schemas:
        1. Dict[image_path_or_filename, url_string]
        2. Dict[url_string, List[image_path_or_filename]]
        3. List[Dict[str, Any]] with keys (e.g. 'image'/'path'/'file_name' and 'url')
    """
    if not record_file.exists():
        logger.info("URL record file '%s' does not exist on disk.", record_file)
        return {}

    try:
        with open(record_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to parse URL record JSON at '%s': %s", record_file, e)
        return {}

    url_lookup: Dict[str, Tuple[str, str]] = {}

    def _register(key_str: str, url_str: str):
        if not key_str or not url_str:
            return
        norm_url = normalize_source_url(url_str)
        cleaned_key = key_str.replace("\\", "/").strip().lstrip("/")
        url_lookup[cleaned_key] = (url_str.strip(), norm_url)

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                _register(k, v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        _register(item, k)
                    elif isinstance(item, dict):
                        p = item.get("path") or item.get("image") or item.get("filename")
                        u = item.get("url") or k
                        if p and u:
                            _register(str(p), str(u))
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                p = entry.get("image") or entry.get("path") or entry.get("file_name") or entry.get("filename")
                u = entry.get("url") or entry.get("source_url") or entry.get("image_url")
                if p and u:
                    _register(str(p), str(u))

    logger.info("Loaded %d URL record lookups from %s", len(url_lookup), record_file)
    return url_lookup


class PlantWildAdapter:
    """Ingestion adapter for PlantWild v1 agricultural leaf disease dataset."""

    VALID_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

    def __init__(
        self,
        data_dir: str = "data/plantwild/plantwild",
        mapping_file: str = "ml/configs/plantwild_map.yaml",
        url_record_file: Optional[str] = None
    ):
        """
        Args:
            data_dir: Root directory of unzipped PlantWild dataset.
            mapping_file: Path to YAML category alignment mapping file.
            url_record_file: Optional explicit path to url_record.json.
        """
        self.data_dir = Path(data_dir).resolve()
        self.mapping_file = Path(mapping_file).resolve()
        self.url_record_file = Path(url_record_file).resolve() if url_record_file else None
        self.mapping: Dict[str, Dict[str, str]] = {}
        self.unmapped_policy: str = "other"

    def load_mapping(self) -> Dict[str, Dict[str, str]]:
        """Loads and normalizes category mapping configuration from YAML."""
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"PlantWild mapping file not found: {self.mapping_file}")

        with open(self.mapping_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.mapping = cfg.get("mapping", {})
        self.unmapped_policy = cfg.get("unmapped_policy", "other")
        logger.info("Loaded %d category mapping rules from %s", len(self.mapping), self.mapping_file)
        return self.mapping

    def resolve_layout(self) -> Tuple[Path, Path]:
        """
        Resolves root directory and images directory to accommodate nested unzipping layouts:
          1. data/plantwild/plantwild/images/
          2. data/plantwild/images/
        """
        if (self.data_dir / "images").exists():
            root_dir = self.data_dir
            images_dir = self.data_dir / "images"
        elif (self.data_dir / "plantwild" / "images").exists():
            root_dir = self.data_dir / "plantwild"
            images_dir = self.data_dir / "plantwild" / "images"
        elif self.data_dir.exists():
            root_dir = self.data_dir
            images_dir = self.data_dir
        else:
            raise FileNotFoundError(f"PlantWild directory does not exist: {self.data_dir}")

        return root_dir, images_dir

    def discover_official_split_files(self, root_dir: Path) -> List[str]:
        """Discovers official split lists and metadata files residing in PlantWild root."""
        split_files: List[str] = []
        if not root_dir.exists():
            return split_files

        candidate_dirs = [root_dir]
        if (root_dir / "splits").exists():
            candidate_dirs.append(root_dir / "splits")

        for d in candidate_dirs:
            for item in sorted(d.iterdir()):
                if item.is_file() and item.suffix.lower() in {".txt", ".csv", ".json"} and item.name != "url_record.json":
                    rel_to_root = item.relative_to(root_dir).as_posix()
                    split_files.append(rel_to_root)

        return split_files

    def _match_category(self, folder_name: str) -> Tuple[str, str]:
        """
        Matches a raw folder name against the mapping table using direct, lowercase,
        and space/underscore normalized variants.
        """
        if folder_name in self.mapping:
            info = self.mapping[folder_name]
            return info["target"], str(info.get("confidence", "high"))

        lower_name = folder_name.lower().strip()
        if lower_name in self.mapping:
            info = self.mapping[lower_name]
            return info["target"], str(info.get("confidence", "high"))

        normalized = lower_name.replace("_", " ")
        if normalized in self.mapping:
            info = self.mapping[normalized]
            return info["target"], str(info.get("confidence", "high"))

        return self.unmapped_policy, ""

    def scan(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Scans PlantWild v1 directory tree, associates URL metadata from `url_record.json`,
        extracts records, and returns metadata.

        Returns:
            records: List of manifest record dictionaries.
            metadata: Diagnostic information including discovered official split files and URL statistics.
        """
        if not self.mapping:
            self.load_mapping()

        root_dir, images_dir = self.resolve_layout()
        logger.info("Scanning PlantWild v1 images at: %s", images_dir)

        # Detect or load url_record.json
        url_file = self.url_record_file or (root_dir / "url_record.json")
        url_lookups = parse_url_records(url_file) if url_file.exists() else {}

        official_split_files = self.discover_official_split_files(root_dir)
        logger.info("Recorded %d official split/list files in %s: %s", len(official_split_files), root_dir, official_split_files)

        records: List[Dict[str, Any]] = []
        unmapped_folders: Set[str] = set()
        mapped_folders: Set[str] = set()
        class_image_counts: Counter = Counter()
        url_match_count = 0

        for class_dir in sorted(images_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            orig_label = class_dir.name
            unified_label, confidence = self._match_category(orig_label)

            if unified_label == "other":
                unmapped_folders.add(orig_label)
            else:
                mapped_folders.add(orig_label)

            for img_file in sorted(class_dir.iterdir()):
                if img_file.is_file() and img_file.suffix in self.VALID_EXTENSIONS:
                    rel_path = img_file.relative_to(root_dir).as_posix()
                    class_image_counts[unified_label] += 1

                    # Look up URL information strictly by "plantwild/" + rel_path
                    # (no bare-file-name registration and no basename fallback)
                    lookup_key = f"plantwild/{rel_path}"
                    raw_url, norm_url = "", ""
                    if lookup_key in url_lookups:
                        raw_url, norm_url = url_lookups[lookup_key]
                        url_match_count += 1

                    records.append({
                        "absolute_path": str(img_file.resolve()),
                        "path": rel_path,
                        "source": "plantwild",
                        "orig_label": orig_label,
                        "unified_label": unified_label,
                        "confidence_level": confidence,
                        "orig_split": "",       # Do not use official split lists yet
                        "legacy_split": "",     # PlantVillage only
                        "raw_url": raw_url,
                        "normalized_url": norm_url
                    })

        # Calculate URL statistics across all discovered records
        images_with_url = sum(1 for r in records if r["normalized_url"])
        norm_urls = [r["normalized_url"] for r in records if r["normalized_url"]]
        distinct_norm_urls = len(set(norm_urls))
        url_counts = Counter(norm_urls)
        top_10_urls = url_counts.most_common(10)

        url_to_orig_labels: Dict[str, Set[str]] = defaultdict(set)
        for r in records:
            if r["normalized_url"]:
                url_to_orig_labels[r["normalized_url"]].add(r["orig_label"])

        urls_spanning_multiple_labels = sum(1 for labels in url_to_orig_labels.values() if len(labels) > 1)

        # Log comprehensive URL audit statistics
        logger.info("PlantWild URL Scan Statistics:")
        logger.info("  - Images with a URL: %d", images_with_url)
        logger.info("  - Distinct normalised URLs: %d", distinct_norm_urls)
        logger.info("  - 10 most frequent normalised URLs with counts:")
        for rank, (u, count) in enumerate(top_10_urls, 1):
            logger.info("      [%2d] count=%d | %s", rank, count, u)
        logger.info("  - Normalised URLs spanning more than one orig_label folder: %d", urls_spanning_multiple_labels)

        metadata = {
            "root_dir": str(root_dir),
            "images_dir": str(images_dir),
            "url_record_file": str(url_file) if url_file.exists() else None,
            "url_matches_count": url_match_count,
            "images_with_url": images_with_url,
            "distinct_normalized_urls": distinct_norm_urls,
            "top_10_normalized_urls": top_10_urls,
            "urls_spanning_multiple_orig_labels": urls_spanning_multiple_labels,
            "official_split_files": official_split_files,
            "total_images": len(records),
            "mapped_categories_count": len(mapped_folders),
            "unmapped_categories_count": len(unmapped_folders),
            "unmapped_categories": sorted(list(unmapped_folders)),
            "other_images_count": class_image_counts.get("other", 0),
            "mapped_images_count": len(records) - class_image_counts.get("other", 0)
        }

        logger.info(
            "Discovered %d PlantWild records (%d mapped, %d assigned to 'other', %d with source URLs).",
            len(records), metadata["mapped_images_count"], metadata["other_images_count"], url_match_count
        )

        return records, metadata

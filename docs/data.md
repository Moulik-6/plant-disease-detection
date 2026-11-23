# Dataset Harmonization, Schema & Split Methodology

This document provides a technical specification of the datasets, taxonomy harmonization, duplicate clustering, split definitions, and manifest structure used in the cross-domain plant leaf disease benchmark.

---

## 1. Primary Datasets

Three image datasets form the foundation of this benchmark:

1. **PlantVillage (Laboratory Domain)**:
   - Volume: 54,305 color leaf images across 38 distinct crop-disease categories.
   - Capture Environment: Detached leaves photographed under controlled laboratory conditions against uniform gray/black paper backgrounds.
   - Purpose: Primary source for in-domain training and baseline laboratory evaluation.

2. **PlantDoc (Field Domain)**:
   - Volume: 2,569 field images across 17 crops and multiple diseases.
   - Capture Environment: In-the-wild Internet captures featuring full plants, complex natural backgrounds, variable illumination, and multiple diseased leaves per frame.
   - Role: Real-world field evaluation and mixed domain training.

3. **PlantWild v1 (Field Domain)**:
   - Volume: Field collections of wild agricultural disease captures.
   - Capture Environment: Outdoor farm captures under natural ambient lighting.
   - Role: Augmented field training and cross-source field evaluation.

---

## 2. Taxonomy Harmonization & Class Mapping

The target taxonomy consists of **38 unified categories** defined in [ml/configs/labels/unified_classes.yaml](file:///c:/Major%20Project/plant-disease-detection/ml/configs/labels/unified_classes.yaml). These match the class definitions of PlantVillage to ensure full compatibility with established benchmarks.

### Cross-Dataset Mapping Files
- **PlantDoc Mapping** ([ml/configs/mappings/plantdoc_map.yaml](file:///c:/Major%20Project/plant-disease-detection/ml/configs/mappings/plantdoc_map.yaml)):
  - Maps raw PlantDoc directory tags to the 38 unified labels.
  - Categories not representing one of the 38 classes are mapped to `"other"` and assigned to `other_pool`.
- **PlantWild Mapping** ([ml/configs/mappings/plantwild_map.yaml](file:///c:/Major%20Project/plant-disease-detection/ml/configs/mappings/plantwild_map.yaml)):
  - Aligns PlantWild class strings with the 38 unified classes.
  - Unmapped or irrelevant disease classes are mapped to `"other"`.

### Coverage
- **Laboratory Coverage**: All 38 classes are present in PlantVillage.
- **Field Coverage**: 33 of the 38 classes contain real-world field images across PlantDoc and PlantWild. The remaining 5 classes exist exclusively in laboratory images.

---

## 3. Perceptual Hash Duplicate Grouping

Field image datasets scraped from online sources frequently contain repeated downloads, re-encodings, or multi-scale crops of the same physical plant. Splitting images naively by filename results in severe test-train leakage.

To address this, [ml/data/build_manifest.py](file:///c:/Major%20Project/plant-disease-detection/ml/data/build_manifest.py) clusters images using **64-bit perceptual hashing (`imagehash.phash`)**:

1. **Hashing**: Each image is converted to a 64-bit binary perceptual hash representing low-frequency DCT coefficients.
2. **Distance Threshold**: Two images with Hamming distance $d_H \le 4$ are designated as near-duplicates.
3. **Graph Union-Find**: A Disjoint Set (Union-Find) algorithm constructs connected components, grouping near-duplicates into discrete `group_id` clusters.
4. **Leakage Protection**: All images belonging to a shared `group_id` are strictly forced into the same partition. No `group_id` is ever divided across train and test sets.
5. **Laboratory-Field Duplicate Flagging**: Images in field datasets that duplicate laboratory PlantVillage samples are flagged with `pv_dup = 1` and restricted to training splits.

---

## 4. Manifest Schema (`data/manifest.csv`)

The central manifest records all sample metadata with the following column specification:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `source` | `str` | Dataset origin (`plantvillage`, `plantdoc`, `plantwild`) |
| `image_path` | `str` | Relative file path from repository root |
| `raw_label` | `str` | Original dataset category identifier |
| `unified_label` | `str` | Normalized target class string (or `"other"`) |
| `label_idx` | `int` | Integer class index in $[0, 37]$ (or $-1$ for `"other"`) |
| `group_id` | `str` | Unique cluster identifier from perceptual hash grouping |
| `pv_dup` | `int` | Binary flag ($1$ if near-duplicate of a PlantVillage image, $0$ otherwise) |
| `split` | `str` | Partition assignment under the group-aware split |
| `legacy_split` | `str` | Historical stratified 70/15/15 assignment (`train`, `val`, `test`) |
| `is_field` | `bool` | True if field image (`plantdoc`, `plantwild`), False if `plantvillage` |

---

## 5. Split Definitions & Partition Rules

### Laboratory Partitions
- **`pv_train`**: 70% of PlantVillage laboratory groups.
- **`pv_val`**: 15% of PlantVillage laboratory groups.
- **`pv_test`**: 15% of PlantVillage laboratory groups.
- **`pv_clean_test`**:
  - **Definition**: The exact intersection of images assigned to `pv_test` in the Stage-II manifest AND assigned to `test` in the historical 70/15/15 stratified split (`legacy_split == "test"`).
  - **Purpose**: Prevents data leakage when testing models initialized from legacy Stage-I checkpoints that were trained on the historical training split.

### Field Partitions
- **`field_train`**: 70% of group-clustered field images across the 33 covered classes, plus all clusters with $pv\_dup = 1$.
- **`field_val`**: 15% of group-clustered field images.
- **`field_test`**: 15% of group-clustered field images (~1,574 images), held out exclusively for cross-domain evaluation.
- **`other_pool`**: All images with `unified_label == "other"`, reserved exclusively for out-of-distribution rejection evaluation.

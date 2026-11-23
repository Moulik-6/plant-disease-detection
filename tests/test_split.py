import pytest
from ml.data.build_manifest import assign_group_splits_field

def test_assign_group_splits_field_integrity():
    """Verify that group-aware splitting never splits a group and isolates 'other'."""
    records = []
    
    # 1. Create 'other' groups
    for g_idx in range(5):
        gid = f"group_other_{g_idx}"
        for img_idx in range(3):
            records.append({
                "group_id": gid,
                "unified_label": "other",
                "pv_dup": 0,
                "image_path": f"other/{gid}_{img_idx}.jpg",
            })

    # 2. Create normal class groups (Class A has 10 groups, Class B has 8 groups)
    for cls in ["Apple___Apple_scab", "Tomato___healthy"]:
        for g_idx in range(10):
            gid = f"group_{cls}_{g_idx}"
            for img_idx in range(2):
                records.append({
                    "group_id": gid,
                    "unified_label": cls,
                    "pv_dup": 0,
                    "image_path": f"{cls}/{gid}_{img_idx}.jpg",
                })

    # 3. Create groups with pv_dup = 1
    for g_idx in range(3):
        gid = f"group_pv_dup_{g_idx}"
        for img_idx in range(2):
            records.append({
                "group_id": gid,
                "unified_label": "Apple___Apple_scab",
                "pv_dup": 1,
                "image_path": f"dup/{gid}_{img_idx}.jpg",
            })

    # Execute splitting
    split_records, tiny_report, forced_stats = assign_group_splits_field(
        records=records,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
    )

    # Invariant 1: No group is EVER split into more than one split
    group_to_splits = {}
    for r in split_records:
        gid = r["group_id"]
        group_to_splits.setdefault(gid, set()).add(r["split"])

    for gid, splits in group_to_splits.items():
        assert len(splits) == 1, f"Group '{gid}' was assigned to multiple splits: {splits}"

    # Invariant 2: 'other' groups are strictly and exclusively assigned to 'other_pool'
    for r in split_records:
        if r["unified_label"] == "other":
            assert r["split"] == "other_pool", (
                f"Record with label 'other' assigned to {r['split']} instead of 'other_pool'"
            )
        else:
            assert r["split"] != "other_pool", (
                f"Classified disease record '{r['unified_label']}' was mistakenly assigned to 'other_pool'"
            )

    # Invariant 3: Groups with pv_dup == 1 are forced into field_train
    for r in split_records:
        if r.get("pv_dup") == 1:
            assert r["split"] == "field_train", (
                f"Group with pv_dup=1 assigned to {r['split']} instead of 'field_train'"
            )

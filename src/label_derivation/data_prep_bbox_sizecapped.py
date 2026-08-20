"""
bbox_size_capped_v1: a size-controlled variant of bbox_mm_v1.

check_bbox_size_shortcut.py showed that bbox_mm_v1's crop size varies
from case to case, and that size alone (with no image content at all)
predicts the label almost as well as the trained model. This suggests
part of bbox_mm_v1's performance may come from a zoom shortcut
rather than genuine learning from the anatomy.

This version keeps bbox_mm_v1's box position and content exactly the
same (same tumour+vessel region, same margin). The only change is
that every box is padded out to a common fixed physical size
(TARGET_SIZE_MM, chosen from the p90 of bbox_mm_v1's own box
sizes; see find_bbox_target_size.py) before resizing to CROP_SIZE_XY.
This removes the zoom cue while still keeping each crop centred
on the case-specific tumour and vessel.

If a case's original box is already larger than TARGET_SIZE_MM on a
given axis, it is left as-is rather than cropped down, since shrinking
could cut off tumour or vessel content that bbox_mm_v1 deliberately
included. These oversized cases are logged separately (n_oversized) so
their remaining size variability can be checked afterward.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

LABEL_DERIVATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

from config import DERIVED_ANGLES_DIR

from label_derivation.data_prep import (
    load_ct, window_and_normalize, get_inplane_axes_and_spacing,
    resize_to_output, CROP_SIZE_XY,
)
from label_derivation.data_prep_bbox import compute_crop_geometry, crop_bbox_from_slice, load_tumor_mask

# Set this from find_bbox_target_size.py's output (p90 of
# max(h_mm, w_mm) across bbox_mm_v1's actual boxes).
TARGET_SIZE_MM = 220.0   # p90 of bbox_mm_v1's box sizes

N_SLICES_2_5D = 5

OUT_DIR = DERIVED_ANGLES_DIR.parent / "classification_inputs" / "bbox_size_capped_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pad_range_to_target(a_range, spacing, target_size_mm, axis_len_vox):
    """
    Given an existing [lo, hi] voxel range and its physical spacing,
    grow the range (centered on its own midpoint) up to target_size_mm
    if it's currently smaller. If already >= target, returns the range
    unchanged and flags it
    """
    lo, hi = a_range
    current_size_vox = hi - lo
    target_size_vox = int(round(target_size_mm / spacing))

    if current_size_vox >= target_size_vox:
        return (lo, hi), True  # oversized, unchanged

    center = (lo + hi) / 2.0
    half = target_size_vox / 2.0
    new_lo = int(round(center - half))
    new_hi = int(round(center + half))

    # clip to volume bounds, shifting the window if it would go
    # out-of-bounds on one side
    if new_lo < 0:
        new_hi += -new_lo
        new_lo = 0
    if new_hi > axis_len_vox - 1:
        shift = new_hi - (axis_len_vox - 1)
        new_lo = max(0, new_lo - shift)
        new_hi = axis_len_vox - 1

    return (new_lo, new_hi), False  # padded, not oversized


def compute_sizecapped_geometry(case_id, driving_vessel):
    """
    Same box centre as bbox_mm_v1 (via compute_crop_geometry,
    unchanged), but a_range/b_range are padded to TARGET_SIZE_MM
    before returning, unless the original box is already largerr
    """
    geo = compute_crop_geometry(case_id, driving_vessel)
    ct_volume = geo["ct_volume"]
    affine = geo["affine"]
    slice_axis = geo["slice_axis"]
    inplane_a, inplane_b = geo["inplane_a"], geo["inplane_b"]

    _, _, spacing_a, spacing_b = get_inplane_axes_and_spacing(affine, slice_axis=slice_axis)

    a_len = ct_volume.shape[inplane_a]
    b_len = ct_volume.shape[inplane_b]

    new_a_range, a_oversized = pad_range_to_target(
        geo["a_range"], spacing_a, TARGET_SIZE_MM, a_len)
    new_b_range, b_oversized = pad_range_to_target(
        geo["b_range"], spacing_b, TARGET_SIZE_MM, b_len)

    geo["a_range"] = new_a_range
    geo["b_range"] = new_b_range
    geo["oversized"] = a_oversized or b_oversized

    return geo


def extract_case(case_id, driving_vessel, mode="2d"):
    geo = compute_sizecapped_geometry(case_id, driving_vessel)
    ct_volume = geo["ct_volume"]
    slice_axis = geo["slice_axis"]
    slice_center = geo["slice_center"]
    n_slices_along_axis = geo["n_slices_along_axis"]
    a_range, b_range = geo["a_range"], geo["b_range"]

    def take_slice(slice_index):
        return np.take(ct_volume, indices=slice_index, axis=slice_axis)

    if mode == "2d":
        ct_slice = take_slice(slice_center)
        ct_slice = window_and_normalize(ct_slice)
        crop = crop_bbox_from_slice(ct_slice, a_range, b_range)
        crop = resize_to_output(crop, out_size=CROP_SIZE_XY)
        return crop

    elif mode == "2.5d":
        half_n = N_SLICES_2_5D // 2
        slices = []
        for d in range(-half_n, half_n + 1):
            idx = int(np.clip(slice_center + d, 0, n_slices_along_axis - 1))
            ct_slice = window_and_normalize(take_slice(idx))
            crop = crop_bbox_from_slice(ct_slice, a_range, b_range)
            crop = resize_to_output(crop, out_size=CROP_SIZE_XY)
            slices.append(crop)
        return np.stack(slices, axis=0)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_extraction(df, mode="2d"):
    out_subdir = OUT_DIR / mode
    out_subdir.mkdir(parents=True, exist_ok=True)

    failures = []
    n_oversized = 0
    log_rows = []

    for i, row in enumerate(df.itertuples(), 1):
        case_id = row.case_id
        driving_vessel = row.driving_vessel
        try:
            geo = compute_sizecapped_geometry(case_id, driving_vessel)
            if geo["oversized"]:
                n_oversized += 1

            crop = extract_case(case_id, driving_vessel, mode=mode)
            np.save(out_subdir / f"{case_id}.npy", crop)

            a_range, b_range = geo["a_range"], geo["b_range"]
            log_rows.append({
                "case_id": case_id,
                "driving_vessel": driving_vessel,
                "oversized": geo["oversized"],
                "box_h_vox": a_range[1] - a_range[0],
                "box_w_vox": b_range[1] - b_range[0],
            })
        except Exception as e:
            failures.append((case_id, str(e)))
        if i % 100 == 0 or i == len(df):
            print(f"[{i}/{len(df)}] extracted... ({len(failures)} failed so far, "
                  f"{n_oversized} oversized/unchanged)")

    log_df = pd.DataFrame(log_rows)
    log_path = out_subdir / "extraction_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Per-case log saved to {log_path}")

    print(f"\n[DONE] {len(df) - len(failures)} succeeded, {len(failures)} failed.")
    print(f"{n_oversized} / {len(df)} cases were already >= TARGET_SIZE_MM "
          f"and kept their original (larger) size.")
    if failures:
        print("Failures (first 10):", failures[:10])
    return failures


if __name__ == "__main__":
    df = pd.read_csv(DERIVED_ANGLES_DIR / "derived_labels.csv")
    print(f"Extracting bbox_size_capped_v1 crops for {len(df)} cases "
          f"(TARGET_SIZE_MM={TARGET_SIZE_MM})...")

    print("\n2D crops:")
    run_extraction(df, mode="2d")

    print("\n2.5D crops:")
    run_extraction(df, mode="2.5d")

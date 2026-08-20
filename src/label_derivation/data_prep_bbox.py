"""
bbox_mm_v1: (tumour + driving-vessel bounding box crop)

This is the second crop strategy attempted in this thesis.

This version draws a box around exactly the tumour and the vessel most responsible for its label
(the "driving vessel", from derived_labels.csv), with a small margin added around both.
The box size therefore varies per case instead of being fixed to show the model less irrelevant
background and more of the anatomy that actually matters for the label.

Slice-axis and in-plane logic is reused directly from data_prep.py (get_slice_axis, get_inplane_axes_and_spacing)
rather than reimplemented, since that logic is already validated.

Vessel masks are loaded from LABEL_DIR, the same raw CT coordinate space as the tumour mask
and not from ISO_RESAMPLING_DIR, which is a separate resampled space used only for centreline and
angle computation and would not line up correctly with the raw CT here.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
# from scipy.ndimage import zoom

LABEL_DERIVATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

from config import IMAGE_DIR, LABEL_DIR, DERIVED_ANGLES_DIR, IMAGE_TE_DIR

# Reused the validated axis-aware helpers from the fov_mm_v4 pipeline.
from label_derivation.data_prep import (
    load_ct, window_and_normalize, get_slice_axis,
    get_inplane_axes_and_spacing, resize_to_output, CROP_SIZE_XY,
)

TUMOUR_MASK_NAME = "pancreatic_lesion.nii.gz"

# Maps driving_vessel values (from derived_labels.csv) to their filename,
# matching VESSEL_CONFIG in the label-derivation main.py exactly.
VESSEL_FILENAMES = {
    "sma":       "superior_mesenteric_artery.nii.gz",
    "ca":        "celiac_artery.nii.gz",
    "aorta":     "aorta.nii.gz",
    "postcava":  "postcava.nii.gz",
    "veins":     "veins.nii.gz",
}

BBOX_MARGIN_MM = 20.0  # physical margin added around the tumour+vessel bbox

N_SLICES_2_5D = 5

OUT_DIR = DERIVED_ANGLES_DIR.parent / "classification_inputs" / "bbox_mm_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_mask(path: Path):
    nii = nib.load(str(path))
    return (nii.get_fdata() > 0.5).astype(np.uint8)


def load_tumor_mask(case_id):
    mask_path = LABEL_DIR / case_id / "segmentations" / TUMOUR_MASK_NAME
    return load_mask(mask_path)


def load_driving_vessel_mask(case_id, driving_vessel):
    filename = VESSEL_FILENAMES.get(driving_vessel)
    if filename is None:
        return None
    vessel_path = LABEL_DIR / case_id / "segmentations" / filename
    if not vessel_path.exists():
        return None
    mask = load_mask(vessel_path)
    return mask if mask.sum() > 0 else None


def bbox_from_mask_union(masks, margin_voxels_per_axis):
    """
    Given a list of 3D binary masks (same shape), return the bounding
    box (min, max) per axis spanning the union of all nonzero voxels
    across all masks, expanded by margin_voxels_per_axis
    Returns None if all masks are empty.
    """
    union = np.zeros_like(masks[0])
    for m in masks:
        if m is not None:
            union = np.logical_or(union, m)

    if not union.any():
        return None

    coords = np.argwhere(union)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    mins = [max(0, mins[i] - margin_voxels_per_axis[i]) for i in range(3)]
    maxs = [min(union.shape[i] - 1, maxs[i] + margin_voxels_per_axis[i]) for i in range(3)]

    return mins, maxs


def crop_bbox_from_slice(slice_2d, bbox_a_range, bbox_b_range):
    """Crop a 2D slice to the given [min, max] voxel ranges along its
    two axes then return it"""
    a0, a1 = bbox_a_range
    b0, b1 = bbox_b_range
    return slice_2d[a0:a1 + 1, b0:b1 + 1]


def compute_crop_geometry(case_id, driving_vessel):
    """
    Returns a dict with everything needed to produce the crop:
    ct_volume, tumor_mask, affine, slice_axis, slice_center, a_range,
    b_range (in inplane_a/inplane_b terms), inplane_a, inplane_b.
    """
    ct_volume, affine = load_ct(case_id)
    tumor_mask = load_tumor_mask(case_id)
    vessel_mask = load_driving_vessel_mask(case_id, driving_vessel)

    if tumor_mask.shape != ct_volume.shape:
        raise ValueError(f"{case_id}: tumour mask shape {tumor_mask.shape} != "
                          f"CT shape {ct_volume.shape}")
    if vessel_mask is not None and vessel_mask.shape != ct_volume.shape:
        raise ValueError(f"{case_id}: vessel mask shape {vessel_mask.shape} != "
                          f"CT shape {ct_volume.shape}")
    if tumor_mask.sum() == 0:
        raise ValueError(f"{case_id}: tumour mask is empty")

    slice_axis = get_slice_axis(affine)
    inplane_a, inplane_b, spacing_a, spacing_b = get_inplane_axes_and_spacing(affine, slice_axis=slice_axis)
    spacing_slice = np.linalg.norm(affine[:3, slice_axis])

    margin_voxels = [0, 0, 0]
    margin_voxels[inplane_a] = max(1, int(round(BBOX_MARGIN_MM / spacing_a)))
    margin_voxels[inplane_b] = max(1, int(round(BBOX_MARGIN_MM / spacing_b)))
    margin_voxels[slice_axis] = max(1, int(round(BBOX_MARGIN_MM / spacing_slice)))

    tumor_coords = np.argwhere(tumor_mask)
    tumor_centroid = tumor_coords.mean(axis=0)
    slice_center = int(round(tumor_centroid[slice_axis]))
    n_slices_along_axis = ct_volume.shape[slice_axis]

    slice_window_half = margin_voxels[slice_axis] * 2
    window_lo = max(0, slice_center - slice_window_half)
    window_hi = min(n_slices_along_axis - 1, slice_center + slice_window_half)

    def restrict_to_window(mask):
        restricted = np.zeros_like(mask)
        sl = [slice(None)] * 3
        sl[slice_axis] = slice(window_lo, window_hi + 1)
        restricted[tuple(sl)] = mask[tuple(sl)]
        return restricted

    tumor_local = restrict_to_window(tumor_mask)
    masks = [tumor_local]
    if vessel_mask is not None:
        vessel_local = restrict_to_window(vessel_mask)
        if vessel_local.any():
            masks.append(vessel_local)

    bbox = bbox_from_mask_union(masks, margin_voxels)
    if bbox is None:
        bbox = bbox_from_mask_union([tumor_mask], margin_voxels)
    mins, maxs = bbox

    a_range = (int(mins[inplane_a]), int(maxs[inplane_a]))
    b_range = (int(mins[inplane_b]), int(maxs[inplane_b]))

    return {
        "ct_volume": ct_volume,
        "tumor_mask": tumor_mask,
        "affine": affine,
        "slice_axis": slice_axis,
        "slice_center": slice_center,
        "n_slices_along_axis": n_slices_along_axis,
        "inplane_a": inplane_a,
        "inplane_b": inplane_b,
        "a_range": a_range,
        "b_range": b_range,
    }


def extract_case(case_id, driving_vessel, mode="2d"):
    """
    Extract a tumour+driving-vessel bbox crop for one case, resized to
    CROP_SIZE_XY x CROP_SIZE_XY. Uses compute_crop_geometry as the
    single source of truth for bbox selection.
    """
    geo = compute_crop_geometry(case_id, driving_vessel)
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
    for i, row in enumerate(df.itertuples(), 1):
        case_id = row.case_id
        driving_vessel = row.driving_vessel
        try:
            crop = extract_case(case_id, driving_vessel, mode=mode)
            np.save(out_subdir / f"{case_id}.npy", crop)
        except Exception as e:
            failures.append((case_id, str(e)))
        if i % 100 == 0 or i == len(df):
            print(f"[{i}/{len(df)}] extracted... ({len(failures)} failed so far)")

    print(f"\n[DONE] {len(df) - len(failures)} succeeded, {len(failures)} failed.")
    if failures:
        print("Failures (first 10):", failures[:10])
    return failures


if __name__ == "__main__":
    df = pd.read_csv(DERIVED_ANGLES_DIR / "derived_labels.csv")
    print(f"Extracting bbox crops for {len(df)} cases...")

    print("\n2D crops:")
    run_extraction(df, mode="2d")

    print("\n2.5D crops:")
    run_extraction(df, mode="2.5d")

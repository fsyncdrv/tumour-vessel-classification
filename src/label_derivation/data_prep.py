"""
fov_mm_v4 Baseline crop (centred on tumour)

Due to the differences in spacing across the dataset (confirmed to vary between 0.7mm and 2.5mm)
a fixed physical size was chosen over a fixed voxel count for a more consistent representation
of anatomical relationships within the data.
A fixed-size crop of the CT scan, centred on the tumour centroid, is extracted in both 2D and 2.5D
format.
Then, each case's normalised numpy array is saved to disk so this extraction is not repeated every
epoch during training

This is the baseline of three final crops strategies compared in this thesis
"""

import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass, zoom

LABEL_DERIVATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

from config import IMAGE_DIR, LABEL_DIR, DERIVED_ANGLES_DIR, IMAGE_TE_DIR


TUMOUR_MASK_NAME = "pancreatic_lesion.nii.gz"

# Output crop is always resized to this crop size, regardless of the
# physical field-of-view used to extract it.
CROP_SIZE_XY = 224          # this number was chosen because it matches typical ResNet input size

# Physical field-of-view extracted around the tumour centroid, in mm.
# Chosen to fit pancreas + adjacent vessels. This replaces the old fixed-voxel-count crop,
# which gave inconsistent real-world FOV across cases with different
# voxel spacing (~0.70mm-2.5mm in this dataset)
CROP_SIZE_MM = 150.0

N_SLICES_2_5D = 5

HU_WINDOW_MIN = -100
HU_WINDOW_MAX = 250

OUT_DIR = DERIVED_ANGLES_DIR.parent / "classification_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def load_ct(case_id):
    for base_dir in (IMAGE_DIR, IMAGE_TE_DIR):
        ct_path = base_dir / case_id / "ct.nii.gz"
        if ct_path.exists():
            nii = nib.load(str(ct_path))
            return nii.get_fdata(), nii.affine
    raise FileNotFoundError(f"CT scan not found for {case_id} in ImageTr or ImageTe")


def load_tumor_mask(case_id):
    mask_path = LABEL_DIR / case_id / "segmentations" / TUMOUR_MASK_NAME
    nii = nib.load(str(mask_path))
    return (nii.get_fdata() > 0.5).astype(np.uint8)


def get_axis_spacings(affine):
    """Return voxel spacing (mm) for each of the 3 array axes"""
    return [np.linalg.norm(affine[:3, i]) for i in range(3)]


def get_slice_axis(affine, dominance_ratio=1.5):
    """
    Return the array axis that is actually the slice-selection
    (through-plane, coarsest-spacing) axis.

    BUG FIX: We cannot assume this is always axis 2. A dataset check found
    159/1033 cases (mostly older/legacy acquisitions, e.g. a 2006 scan
    with only 69 voxels and 5mm spacing on axis 1) where axis 2 is not
    the coarsest axis. For those cases, slicing ct_volume[:, :, z]
    cuts through the wrong plane entirely.

    Also, taking argmax(spacing) is not reliable. In some cases, the three
    spacings differ only by scanner rounding noise, and argmax will
    confidently declare whichever axis is nominally 0.01mm larger as
    the slice axis. This silently mis-slicing a case that was actually fine
    under the plain axis-2 default (confirmed: PanTS_00001876 regressed
    under a pure-argmax version of this function).

    Only override the axis-2 default when the coarsest axis is
    clearly dominant. This catches genuine legacy thick-slice scans
    (5mm vs 1.5mm = 3.2x) without misfiring on near-isotropic ones.
    """
    spacings = get_axis_spacings(affine)
    order = sorted(range(3), key=lambda i: spacings[i], reverse=True)
    coarsest_axis, second_axis = order[0], order[1]
    coarsest_val, second_val = spacings[coarsest_axis], spacings[second_axis]

    if second_val > 0 and (coarsest_val / second_val) >= dominance_ratio:
        return coarsest_axis
    return 2  # default assumption, used whenever spacing doesn't clearly indicate otherwise


def get_inplane_axes_and_spacing(affine, slice_axis=None):
    """
    Return (inplane_axis_a, inplane_axis_b, spacing_a, spacing_b):
    the two array axes that are nott the slice-selection axis, in
    ascending axis-index order, with their respective mm/voxel spacing.

    BUG FIX: this used to recompute the slice axis independently via
    plain argmax(spacings), which disagreed with get_slice_axis's
    dominance-ratio-aware pick for isotropic/near-isotropic spacing
    cases - e.g. spacing [1,1,1] -> get_slice_axis correctly defaults
    to axis 2, but plain argmax picked axis 0, causing this function to
    report inplane axes as (1, 2) - i.e. axis 2 claimed as both the
    slice axis and an in-plane axis simultaneously. Confirmed to crash
    the bbox extraction (division by zero) and almost certainly
    silently corrupted a subset of fov_mm_v3 crops too, since that
    pipeline never surfaced an error for the same inconsistency.

    Updated to take the already-computed slice_axis as a parameter so there
    is exactly one source of truth for which axis is the slice axis everywhere.
    """
    if slice_axis is None:
        slice_axis = get_slice_axis(affine)
    spacings = get_axis_spacings(affine)
    inplane_axes = [ax for ax in range(3) if ax != slice_axis]
    a, b = inplane_axes
    return a, b, spacings[a], spacings[b]


def window_and_normalize(ct_slice, hu_min=HU_WINDOW_MIN, hu_max=HU_WINDOW_MAX):
    """Clip to HU window, then normalize to [0, 1]."""
    clipped = np.clip(ct_slice, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


def crop_around_center(volume_2d, center_yx, crop_size_voxels_yx):
    """Crop a fixed-size (in voxels) rectangular region from a 2D array,
    centred at center_yx. Pads with zeros if the crop would go out of
    bounds (e.g. tumor near scan edge).

    crop_size_voxels_yx: (size_y, size_x) in voxels. This may differ per
    case now, since it's derived from that case's physical spacing.
    """
    size_y, size_x = crop_size_voxels_yx
    half_y, half_x = size_y // 2, size_x // 2
    cy, cx = int(round(center_yx[0])), int(round(center_yx[1]))

    y0, y1 = cy - half_y, cy + half_y
    x0, x1 = cx - half_x, cx + half_x

    padded = np.zeros((size_y, size_x), dtype=volume_2d.dtype)

    src_y0, src_y1 = max(y0, 0), min(y1, volume_2d.shape[0])
    src_x0, src_x1 = max(x0, 0), min(x1, volume_2d.shape[1])
    dst_y0, dst_y1 = src_y0 - y0, src_y1 - y0
    dst_x0, dst_x1 = src_x0 - x0, src_x1 - x0

    padded[dst_y0:dst_y1, dst_x0:dst_x1] = volume_2d[src_y0:src_y1, src_x0:src_x1]
    return padded


def resize_to_output(crop_2d, out_size=CROP_SIZE_XY):
    """Resize a variable-voxel-size crop to the fixed model input size,
    so every case produces a uniform-shaped array regardless of the
    voxel spacing it was extracted at"""
    zoom_y = out_size / crop_2d.shape[0]
    zoom_x = out_size / crop_2d.shape[1]
    resized = zoom(crop_2d, (zoom_y, zoom_x), order=1)
    # zoom() on non-integer ratios can be off by a pixel; force exact shape.
    if resized.shape != (out_size, out_size):
        fixed = np.zeros((out_size, out_size), dtype=resized.dtype)
        y_lim = min(out_size, resized.shape[0])
        x_lim = min(out_size, resized.shape[1])
        fixed[:y_lim, :x_lim] = resized[:y_lim, :x_lim]
        resized = fixed
    return resized


def extract_case(case_id, mode="2d"):
    """
    Extract a model-ready crop for one case.

    Crop is taken at a fixed physical size (CROP_SIZE_MM x CROP_SIZE_MM),
    using this case's own voxel spacing to work out how many voxels that
    corresponds to, then resized to the fixed CROP_SIZE_XY pixel output
    so every case still yields a uniform-shaped array.

    The slice-selection axis (which array axis we slide along for 2.5D,
    and which single index we pick for 2D) is determined per case via
    get_slice_axis, not assumed to always be axis 2 ** see get_slice_axis
    docstring for why (159/1033 cases violate that assumption).
    """
    ct_volume, affine = load_ct(case_id)
    tumor_mask = load_tumor_mask(case_id)

    centroid = center_of_mass(tumor_mask)
    center_vox = [int(round(c)) for c in centroid]

    slice_axis = get_slice_axis(affine)
    inplane_a, inplane_b, spacing_a, spacing_b = get_inplane_axes_and_spacing(affine, slice_axis=slice_axis)

    crop_size_voxels_ab = (
        max(1, int(round(CROP_SIZE_MM / spacing_a))),
        max(1, int(round(CROP_SIZE_MM / spacing_b))),
    )
    center_ab = (center_vox[inplane_a], center_vox[inplane_b])
    slice_center = center_vox[slice_axis]
    n_slices_along_axis = ct_volume.shape[slice_axis]


    def take_slice(slice_index):
        """Extract the 2D in-plane slice at slice_index along slice_axis,
        as an array ordered (inplane_a, inplane_b) regardless of where
        slice_axis physically sits (0, 1, or 2)."""
        return np.take(ct_volume, indices=slice_index, axis=slice_axis)

    if mode == "2d":
        ct_slice = take_slice(slice_center)
        ct_slice = window_and_normalize(ct_slice)
        crop = crop_around_center(ct_slice, center_ab, crop_size_voxels_ab)
        crop = resize_to_output(crop)
        return crop

    elif mode == "2.5d":
        half_n = N_SLICES_2_5D // 2
        slices = []
        for d in range(-half_n, half_n + 1):
            idx = int(np.clip(slice_center + d, 0, n_slices_along_axis - 1))
            ct_slice = window_and_normalize(take_slice(idx))
            crop = crop_around_center(ct_slice, center_ab, crop_size_voxels_ab)
            crop = resize_to_output(crop)
            slices.append(crop)
        return np.stack(slices, axis=0)

    else:
        raise ValueError(f"Unknown mode: {mode}")



def run_extraction(case_ids, mode="2d"):
    out_subdir = OUT_DIR / mode
    out_subdir.mkdir(parents=True, exist_ok=True)

    failures = []
    for i, case_id in enumerate(case_ids, 1):
        try:
            crop = extract_case(case_id, mode=mode)
            np.save(out_subdir / f"{case_id}.npy", crop)
        except Exception as e:
            failures.append((case_id, str(e)))
        if i % 100 == 0 or i == len(case_ids):
            print(f"[{i}/{len(case_ids)}] extracted... ({len(failures)} failed so far)")

    print(f"\n[DONE] {len(case_ids) - len(failures)} succeeded, {len(failures)} failed.")
    if failures:
        print("Failures:", failures[:10])
    return failures


# ct_data, _ = load_ct("PanTS_00000086")
# tumor_mask = load_tumor_mask("PanTS_00000086")
# print(ct_data.shape, tumor_mask.shape)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(DERIVED_ANGLES_DIR / "derived_labels.csv")
    case_ids = df["case_id"].tolist()

    print(f"Extracting 2D crops for {len(case_ids)} cases...")
    run_extraction(case_ids, mode="2d")

    print(f"\nExtracting 2.5D crops for {len(case_ids)} cases...")
    run_extraction(case_ids, mode="2.5d")

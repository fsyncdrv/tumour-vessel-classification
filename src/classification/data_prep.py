import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass

LABEL_DERIVATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

from config import IMAGE_DIR, LABEL_DIR, DERIVED_ANGLES_DIR, IMAGE_TE_DIR


TUMOUR_MASK_NAME = "pancreatic_lesion.nii.gz"
CROP_SIZE_XY = 224          # this number matches typical ResNet input size
N_SLICES_2_5D = 5           # the number of adjacent slices for 2.5D input

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
    """
    Load the ORIGINAL (unresampled) tumor mask. This matches the CT scan's
    native coordinate space (raw CT scans were never isotropically resampled;
    only the masks used for label derivation were). Using the isotropic mask
    here would give a centroid in the wrong coordinate space relative to the
    raw CT, causing incorrect crop locations.
    """
    mask_path = LABEL_DIR / case_id / "segmentations" / TUMOUR_MASK_NAME
    nii = nib.load(str(mask_path))
    return (nii.get_fdata() > 0.5).astype(np.uint8)

def window_and_normalize(ct_slice, hu_min=HU_WINDOW_MIN, hu_max=HU_WINDOW_MAX):
    """Clip to HU window, then normalize to [0, 1]."""
    clipped = np.clip(ct_slice, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)

def crop_around_center(volume_2d, center_yx, crop_size):
    """Crop a fixed-size square region from a 2D array, centred at center_yx.
    Pads with zeros if the crop would go out of bounds (e.g. tumor near scan edge)."""
    half = crop_size // 2
    cy, cx = int(round(center_yx[0])), int(round(center_yx[1]))

    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    padded = np.zeros((crop_size, crop_size), dtype=volume_2d.dtype)

    src_y0, src_y1 = max(y0, 0), min(y1, volume_2d.shape[0])
    src_x0, src_x1 = max(x0, 0), min(x1, volume_2d.shape[1])
    dst_y0, dst_y1 = src_y0 - y0, src_y1 - y0
    dst_x0, dst_x1 = src_x0 - x0, src_x1 - x0

    padded[dst_y0:dst_y1, dst_x0:dst_x1] = volume_2d[src_y0:src_y1, src_x0:src_x1]
    return padded


# ------------------------- main extraction -------------------------

def extract_case(case_id, mode="2d"):
    """
    Extract a model-ready crop for one case.

    mode="2d"   -> returns (CROP_SIZE_XY, CROP_SIZE_XY) single-channel array
    mode="2.5d" -> returns (N_SLICES_2_5D, CROP_SIZE_XY, CROP_SIZE_XY) array
    """
    ct_volume, _ = load_ct(case_id)
    tumor_mask = load_tumor_mask(case_id)

    centroid = center_of_mass(tumor_mask)
    z_center, y_center, x_center = [int(round(c)) for c in centroid]
    # print(tumor_mask.shape)

    if mode == "2d":
        ct_slice = ct_volume[z_center, :, :]
        ct_slice = window_and_normalize(ct_slice)
        crop = crop_around_center(ct_slice, (y_center, x_center), CROP_SIZE_XY)
        return crop

    elif mode == "2.5d":
        half_n = N_SLICES_2_5D // 2
        slices = []
        for dz in range(-half_n, half_n + 1):
            z = np.clip(z_center + dz, 0, ct_volume.shape[0] - 1)
            ct_slice = window_and_normalize(ct_volume[z, :, :])
            crop = crop_around_center(ct_slice, (y_center, x_center), CROP_SIZE_XY)
            slices.append(crop)
        return np.stack(slices, axis=0)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------- batch run -------------------------

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

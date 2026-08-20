"""
Main label-derivation script

It does the following:
For each case,
1. dilate the tumour mask by 2mm to catch near-contact cases then split into components
2. For each vessel type the mask is skeletonised then
   2.1 Centreline extraction is run on (SMA, CA, aorta and postcava)
   2.2 Branch selection is run on veins.
3. Angle is computed per tumour component
4. Take the max angle across componets to get vessel's angle
5. Take the max across all five vessels give the case-level angle. This means only the single worst
vessel contact is captured not how many vessels are involved. This is a simplication compared to
clinical assessment where multiple vessel contacts would be considered.
6. The case is then labelled as either low_vascular_contact or high_vascular_contact using a 180 degree
threshold.

"""

import sys
import csv
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.morphology import skeletonize
from scipy import ndimage as ndi


LABEL_DERIVATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("LABEL_DERIVATION_DIR:", LABEL_DERIVATION_DIR)
assert (PROJECT_ROOT / "config.py").exists(), f"config.py not found at {PROJECT_ROOT}"

from config import ISO_RESAMPLING_DIR, DERIVED_ANGLES_DIR
from angle_quantify import skeleton_to_graph, ordered_centerline, quantify_angles
from branch_selection import get_tumour_components, get_branch_path_coords


TUMOUR_MASK_NAME = "pancreatic_lesion.nii.gz"

VESSEL_CONFIG = {
    "sma":       {"file": "superior_mesenteric_artery.nii.gz", "type": "single_tube"},
    "ca":        {"file": "celiac_artery.nii.gz",               "type": "single_tube"},
    "aorta":     {"file": "aorta.nii.gz",                       "type": "single_tube"},
    "postcava":  {"file": "postcava.nii.gz",                    "type": "single_tube"},
    "veins":     {"file": "veins.nii.gz",                       "type": "branching"},
}

VASCULAR_CONTACT_THRESHOLD_DEG = 180.0
TUMOUR_DILATION_MARGIN_MM = 2.0
ISOTROPIC_SPACING_MM = 0.7

OUT_DIR = DERIVED_ANGLES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_LABELS_CSV = OUT_DIR / "derived_labels.csv"
FAILED_CASES_LOG = OUT_DIR / "failed_cases.log"


def load_bin(path: Path):
    nii = nib.load(str(path))
    data = (nii.get_fdata() > 0.5).astype(np.uint8)
    return data, nii.affine, nii.header

## some segmentation masks have some disconnected noise. this function was
## added to only keep the single largest connected component during skeletonisation
def keep_lcc(mask):
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (lab == np.argmax(counts)).astype(np.uint8)

def path_to_mask(path_xyz, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for p in path_xyz:
        mask[int(p[0]), int(p[1]), int(p[2])] = 1
    return mask


# To ensure both overlap and near vessel contact are taken into consideration
## 2mm is used to match the contact threshold alreadyh used to define the dataset's
# vascular-contact labels
def dilate_mask(mask, margin_mm, voxel_spacing_mm=0.7):
    radius_voxels = int(round(margin_mm / voxel_spacing_mm))
    if radius_voxels <= 0:
        return mask
    struct = ndi.generate_binary_structure(3, 1)
    dilated = ndi.binary_dilation(mask, structure=struct, iterations=radius_voxels)
    return dilated.astype(np.uint8)


def process_vessel_for_case(case_id, seg_dir, vessel_key, vessel_cfg, tumor_components, out_dir):
    """
    Returns the max angle for this vessel across all tumour components,
    plus which component drove it (index), or (None, None) if vessel
    mask is missing/empty/all components fail.
    """
    vessel_path = seg_dir / vessel_cfg["file"]
    if not vessel_path.exists():
        return None, None

    vessel_mask, affine, header = load_bin(vessel_path)
    if vessel_mask.sum() == 0:
        return None, None

    vessel_mask_lcc = keep_lcc(vessel_mask)
    skel = skeletonize(vessel_mask_lcc).astype(np.uint8)
    if skel.sum() == 0:
        return None, None

    vessel_out_dir = out_dir / case_id / vessel_key
    vessel_out_dir.mkdir(parents=True, exist_ok=True)

    # For single-tube vessels, the centreline doesn't depend on the tumour
    # component. It is computed once and reused across components
    shared_centerline_array = None
    if vessel_cfg["type"] == "single_tube":
        try:
            path_xyz = ordered_centerline(skel)
            shared_centerline_array = path_to_mask(path_xyz, vessel_mask_lcc.shape)
        except Exception as e:
            print(f"    [{vessel_key}] centreline extraction failed: {e}")
            return None, None
    else:
        G = skeleton_to_graph(skel)

    component_angles = []
    for comp_idx, tumor_component_mask in enumerate(tumor_components):
        try:
            if vessel_cfg["type"] == "branching":
                path_xyz = get_branch_path_coords(G, tumor_component_mask)
                centerline_array = path_to_mask(path_xyz, vessel_mask_lcc.shape)
            else:
                centerline_array = shared_centerline_array

            max_angle = quantify_angles(
                vessel_array=vessel_mask_lcc,
                tumor_array=tumor_component_mask,
                centerline_array=centerline_array,
                out_dir=vessel_out_dir / f"component_{comp_idx}",
                save_png=False,
            )
            component_angles.append(max_angle)

        except Exception as e:
            print(f"    [{vessel_key}] component {comp_idx} failed: {e}")
            component_angles.append(0.0)

    if not component_angles:
        return None, None

    best_component_idx = int(np.argmax(component_angles))
    return component_angles[best_component_idx], best_component_idx


def process_case(case_id):
    seg_dir = ISO_RESAMPLING_DIR / case_id / "segmentations"
    tumor_path = seg_dir / TUMOUR_MASK_NAME

    if not tumor_path.exists():
        return {"case_id": case_id, "status": "failed", "reason": "no tumour mask"}

    full_tumor_mask, _, _ = load_bin(tumor_path)
    if full_tumor_mask.sum() == 0:
        return {"case_id": case_id, "status": "failed", "reason": "empty tumour mask"}

    # Dilate the tumour mask by a small physical margin to capture near-contact cases
    full_tumor_mask = dilate_mask(full_tumor_mask, TUMOUR_DILATION_MARGIN_MM, ISOTROPIC_SPACING_MM)

    tumor_components = get_tumour_components(full_tumor_mask)
    if not tumor_components:
        return {"case_id": case_id, "status": "failed", "reason": "no tumour components above size threshold"}

    vessel_max_angles = {}
    vessel_driving_component = {}

    for vessel_key, vessel_cfg in VESSEL_CONFIG.items():
        max_angle, comp_idx = process_vessel_for_case(
            case_id, seg_dir, vessel_key, vessel_cfg, tumor_components, OUT_DIR
        )
        vessel_max_angles[vessel_key] = max_angle if max_angle is not None else 0.0
        vessel_driving_component[vessel_key] = comp_idx

    overall_max_angle = max(vessel_max_angles.values())
    driving_vessel = max(vessel_max_angles, key=vessel_max_angles.get)

    label = ("low_vascular_contact" if overall_max_angle <= VASCULAR_CONTACT_THRESHOLD_DEG
              else "high_vascular_contact")

    return {
        "case_id": case_id,
        "status": "ok",
        "n_tumour_components": len(tumor_components),
        "overall_max_angle": round(overall_max_angle, 2),
        "driving_vessel": driving_vessel,
        "driving_component": vessel_driving_component.get(driving_vessel),
        "label": label,
        **{f"angle_{k}": round(v, 2) for k, v in vessel_max_angles.items()},
    }


def run_pipeline(case_ids):
    results = []
    failures = []

    fieldnames = [
        "case_id", "status", "n_tumour_components", "overall_max_angle",
        "driving_vessel", "driving_component", "label",
        "angle_sma", "angle_ca", "angle_aorta", "angle_postcava", "angle_veins",
    ]

    with open(FINAL_LABELS_CSV, "w", newline="") as f, open(FAILED_CASES_LOG, "w") as flog:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, case_id in enumerate(case_ids, 1):
            try:
                result = process_case(case_id)
            except Exception as e:
                result = {"case_id": case_id, "status": "failed", "reason": str(e)}

            if result["status"] == "failed":
                failures.append(result)
                flog.write(f"{case_id}: {result.get('reason', 'unknown error')}\n")
            else:
                results.append(result)
                writer.writerow({k: result.get(k, "") for k in fieldnames})

            if i % 25 == 0 or i == len(case_ids):
                print(f"[{i}/{len(case_ids)}] processed... "
                      f"({len(results)} ok, {len(failures)} failed)")

    print(f"\n[DONE] {len(results)} succeeded, {len(failures)} failed.")
    print(f"Labels saved to: {FINAL_LABELS_CSV}")
    print(f"Failure log: {FAILED_CASES_LOG}")

    return results, failures


if __name__ == "__main__":
    import json
    with open(PROJECT_ROOT / "notebooks" / "outputs" / "vessel_distances.json") as f:
        vessel_distances = json.load(f)

    CONTACT_DISTANCE_THRESHOLD_MM = 2.0
    case_ids = sorted([
        case_id for case_id, distances in vessel_distances.items()
        if distances and min(distances.values()) <= CONTACT_DISTANCE_THRESHOLD_MM
    ])

    print(f"Running full label-derivation pipeline on {len(case_ids)} cases...")

    run_pipeline(case_ids)

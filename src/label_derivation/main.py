import sys
import csv
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.morphology import skeletonize
from scipy import ndimage as ndi


LABEL_DERIVATION_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

from config import ISO_RESAMPLING_DIR
from angle_quantify import skeleton_to_graph, ordered_centerline, quantify_angles
from branch_selection import get_tumor_components, get_branch_path_coords


TUMOUR_MASK_NAME = "pancreatic_lesion.nii.gz"

VESSEL_CONFIG = {
    "sma":       {"file": "superior_mesenteric_artery.nii.gz", "type": "single_tube"},
    "ca":        {"file": "celiac_artery.nii.gz",               "type": "single_tube"},
    "aorta":     {"file": "aorta.nii.gz",                       "type": "single_tube"},
    "postcava":  {"file": "postcava.nii.gz",                    "type": "single_tube"},
    "veins":     {"file": "veins.nii.gz",                       "type": "branching"},
}

RESECTABILITY_THRESHOLD_DEG = 180.0

OUT_DIR = PROJECT_ROOT / "outputs" / "label_derivation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_LABELS_CSV = OUT_DIR / "derived_labels.csv"
FAILED_CASES_LOG = OUT_DIR / "failed_cases.log"


# ------------------------- helpers -------------------------

def load_bin(path: Path):
    nii = nib.load(str(path))
    data = (nii.get_fdata() > 0.5).astype(np.uint8)
    return data, nii.affine, nii.header

def keep_lcc(mask):
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (lab == np.argmax(counts)).astype(np.uint8)

def path_to_mask(path_xyz, shape):
    """Rasterize a coordinate path back into a binary centerline mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    for p in path_xyz:
        mask[int(p[0]), int(p[1]), int(p[2])] = 1
    return mask


# ------------------------- per-vessel processing -------------------------

def process_vessel_for_case(case_id, seg_dir, vessel_key, vessel_cfg,
                             tumor_components, out_dir):
    """
    Returns the max angle for this vessel across all tumor components,
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


# ------------------------- per-case processing -------------------------

def process_case(case_id):
    seg_dir = ISO_RESAMPLING_DIR / case_id / "segmentations"
    tumor_path = seg_dir / TUMOUR_MASK_NAME

    if not tumor_path.exists():
        return {"case_id": case_id, "status": "failed", "reason": "no tumor mask"}

    full_tumor_mask, _, _ = load_bin(tumor_path)
    if full_tumor_mask.sum() == 0:
        return {"case_id": case_id, "status": "failed", "reason": "empty tumor mask"}

    tumor_components = get_tumor_components(full_tumor_mask)
    if not tumor_components:
        return {"case_id": case_id, "status": "failed", "reason": "no tumor components above size threshold"}

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

    label = ("resectable" if overall_max_angle <= RESECTABILITY_THRESHOLD_DEG
              else "borderline_or_locally_advanced")

    return {
        "case_id": case_id,
        "status": "ok",
        "n_tumor_components": len(tumor_components),
        "overall_max_angle": round(overall_max_angle, 2),
        "driving_vessel": driving_vessel,
        "driving_component": vessel_driving_component.get(driving_vessel),
        "label": label,
        **{f"angle_{k}": round(v, 2) for k, v in vessel_max_angles.items()},
    }


# ------------------------- run across positive resampled cases -------------------------

def run_pipeline(case_ids):
    results = []
    failures = []

    fieldnames = [
        "case_id", "status", "n_tumor_components", "overall_max_angle",
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

    print(f"Running the label-derivation pipeline on {len(case_ids)} cases...")
    run_pipeline(case_ids)

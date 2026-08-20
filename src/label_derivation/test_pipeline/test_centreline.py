"""
Visual chexck of centreline extraction for both artery and veins masks

This test does three things:
1. Test the adapted Zhang et al. method on the SMA/CA cases
2. Test the branch selection logic I implemented for the veins
3. then it plots the centrelines and export as png

"""

import sys
from pathlib import Path

LABEL_DERIVATION_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("LABEL_DERIVATION_DIR:", LABEL_DERIVATION_DIR)
assert (PROJECT_ROOT / "config.py").exists(), f"config.py not found at: {PROJECT_ROOT}"
assert (LABEL_DERIVATION_DIR / "angle_quantify.py").exists(), f"angle_quantify.py not found at {LABEL_DERIVATION_DIR}"

import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

from config import ISO_RESAMPLING_DIR
from angle_quantify import skeleton_to_graph, ordered_centerline
from branch_selection import get_branch_path_coords
from skimage.morphology import skeletonize
from scipy import ndimage as ndi


VESSEL_DISTANCES_PATH = PROJECT_ROOT / "notebooks" / "outputs" / "vessel_distances.json"

TUMOUR_MASK_NAME = "pancreatic_lesion.nii.gz"
VEINS_MASK_NAME = "veins.nii.gz"
SMA_MASK_NAME = "superior_mesenteric_artery.nii.gz"
CA_MASK_NAME = "celiac_artery.nii.gz"

N_VEINS_CASES = 6
N_SMA_CA_CASES = 6

OUT_DIR = Path("test_results/batch_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(VESSEL_DISTANCES_PATH) as f:
    vessel_distances = json.load(f)



def sample_cases_for_vessel(vessel_key, n, distance_threshold=2.0):
    candidates = [
        (case_id, d[vessel_key])
        for case_id, d in vessel_distances.items()
        if vessel_key in d and d[vessel_key] <= distance_threshold
    ]
    candidates.sort(key=lambda x: x[1])
    if len(candidates) <= n:
        return [c[0] for c in candidates]
    idxs = np.linspace(0, len(candidates) - 1, n).astype(int)
    return [candidates[i][0] for i in idxs]


veins_test_cases = sample_cases_for_vessel("veins", N_VEINS_CASES)
sma_test_cases = sample_cases_for_vessel("superior_mesenteric_artery", N_SMA_CA_CASES // 2)
ca_test_cases = sample_cases_for_vessel("celiac_artery", N_SMA_CA_CASES // 2)

print(f"Veins test cases ({len(veins_test_cases)}): {veins_test_cases}")
print(f"SMA test cases ({len(sma_test_cases)}): {sma_test_cases}")
print(f"CA test cases ({len(ca_test_cases)}): {ca_test_cases}")


def load_bin(path):
    nii = nib.load(str(path))
    return (nii.get_fdata() > 0.5).astype(np.uint8), nii.affine, nii.header

def keep_lcc(mask):
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (lab == np.argmax(counts)).astype(np.uint8)

def sparse_coords(mask, step=4):
    coords = np.argwhere(mask > 0)
    return coords[::step]

def plot_and_save(case_id, vessel_mask, tumour_mask, path_xyz, out_png, title_suffix=""):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    v_pts = sparse_coords(vessel_mask, step=4)
    t_pts = sparse_coords(tumour_mask, step=2)
    ax.scatter(v_pts[:, 2], v_pts[:, 1], v_pts[:, 0], c='blue', s=1, alpha=0.15, label='vessel')
    ax.scatter(t_pts[:, 2], t_pts[:, 1], t_pts[:, 0], c='red', s=3, alpha=0.3, label='tumour')
    ax.plot(path_xyz[:, 2], path_xyz[:, 1], path_xyz[:, 0], c='lime', linewidth=3, label='selected path')
    ax.legend()
    ax.set_title(f"{case_id} {title_suffix}")
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


# TEST 1: Branch selection on veins
print("Branch ------->")
for case_id in veins_test_cases:
    try:
        seg_dir = ISO_RESAMPLING_DIR / case_id / "segmentations"
        veins_mask, affine, header = load_bin(seg_dir / VEINS_MASK_NAME)
        tumour_mask, _, _ = load_bin(seg_dir / TUMOUR_MASK_NAME)

        veins_mask = keep_lcc(veins_mask)
        skel = skeletonize(veins_mask).astype(np.uint8)

        G = skeleton_to_graph(skel)
        degrees = [d for _, d in G.degree()]
        n_tips = sum(1 for d in degrees if d == 1)
        n_branch_pts = sum(1 for d in degrees if d >= 3)

        path_xyz = get_branch_path_coords(G, tumour_mask)

        print(f"\n{case_id}: graph nodes={G.number_of_nodes()}, tips={n_tips}, "
              f"branch_pts={n_branch_pts}, selected path length={len(path_xyz)}")

        out_png = OUT_DIR / f"{case_id}_veins_branch.png"
        plot_and_save(case_id, veins_mask, tumour_mask, path_xyz, out_png, "veins branch selection")
        print(f"  Saved plot -> {out_png}")

    except Exception as e:
        print(f"  FAILED on {case_id}: {e}")


# TEST 2: centreline on SMA/CA
print("Cenrtreline ------->")
for vessel_name, mask_fname, case_list in [
    ("SMA", SMA_MASK_NAME, sma_test_cases),
    ("CA", CA_MASK_NAME, ca_test_cases),
]:
    for case_id in case_list:
        try:
            seg_dir = ISO_RESAMPLING_DIR / case_id / "segmentations"
            vessel_mask, affine, header = load_bin(seg_dir / mask_fname)
            tumour_mask, _, _ = load_bin(seg_dir / TUMOUR_MASK_NAME)

            vessel_mask_lcc = keep_lcc(vessel_mask)
            skel = skeletonize(vessel_mask_lcc).astype(np.uint8)

            path_xyz = ordered_centerline(skel)

            print(f"\n{case_id} ({vessel_name}): skeleton voxels={skel.sum()}, "
                  f"path length={len(path_xyz)}")

            out_png = OUT_DIR / f"{case_id}_{vessel_name}_centreline.png"
            plot_and_save(case_id, vessel_mask_lcc, tumour_mask, path_xyz, out_png,
                          f"{vessel_name} centreline")
            print(f"  Saved plot -> {out_png}")

        except Exception as e:
            print(f"  FAILED on {case_id} ({vessel_name}): {e}")

print(f"\n[DONE] plots saved to: {OUT_DIR}")

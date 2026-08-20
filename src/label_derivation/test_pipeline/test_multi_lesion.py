"""
Visual check to test branch_selection module correctly assigns branch to
corresponding tumour component
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
from scipy.ndimage import label

LABEL_DERIVATION_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

assert (PROJECT_ROOT / "config.py").exists(), f"config.py not found at {PROJECT_ROOT}"
assert (LABEL_DERIVATION_DIR / "angle_quantify.py").exists(), f"angle_quantify.py not found at {LABEL_DERIVATION_DIR}"

from config import ISO_RESAMPLING_DIR, LABEL_DIR
from angle_quantify import skeleton_to_graph
from branch_selection import get_tumour_components, get_branch_path_coords

VEINS_MASK_NAME = "veins.nii.gz"
TUMOUR_MASK_NAME = "pancreatic_lesion.nii.gz"

OUT_DIR = Path("test_results/multi_lesion_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TEST_CASES = 2


def load_bin(path):
    nii = nib.load(str(path))
    return (nii.get_fdata() > 0.5).astype(np.uint8)

def keep_lcc(mask):
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (lab == np.argmax(counts)).astype(np.uint8)

def sparse_coords(mask, step=4):
    return np.argwhere(mask > 0)[::step]


# Identify multiple lesion cases
# NOTE: updated to test on resampled masks here insteadd of the original LABEL_DIR masks

print("Rebuilding mask-based tumour-positive cohort...")
cases = sorted([c for c in os.listdir(LABEL_DIR) if c.startswith("PanTS_")])
lesion_counts = {}


for case in cases:
    lesion_path = LABEL_DIR / case / "segmentations" / TUMOUR_MASK_NAME
    if lesion_path.exists():
        data = load_bin(lesion_path)
        lesion_counts[case] = int(np.count_nonzero(data))

mask_positive_cases = set(k for k, v in lesion_counts.items() if v > 0)
print(f"Mask-based tumour-positive: {len(mask_positive_cases)}")
assert len(mask_positive_cases) == 1033, f"Expected 1033, got {len(mask_positive_cases)}"


print("Identifying multi-lesion cases (on resampled masks)...")
multi_lesion_cases = []

for case_id in sorted(mask_positive_cases):
    tumour_path = ISO_RESAMPLING_DIR / case_id / "segmentations" / TUMOUR_MASK_NAME
    if not tumour_path.exists():
        continue
    tumour_mask = load_bin(tumour_path)
    _, n_components = label(tumour_mask)
    if n_components > 1:
        multi_lesion_cases.append((case_id, n_components))

print(f"Cases with >1 tumour component: {len(multi_lesion_cases)}")


def load_bin(path):
    nii = nib.load(str(path))
    return (nii.get_fdata() > 0.5).astype(np.uint8)


def keep_lcc(mask):
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (lab == np.argmax(counts)).astype(np.uint8)


def sparse_coords(mask, step=4):
    return np.argwhere(mask > 0)[::step]


test_case_ids = [c[0] for c in multi_lesion_cases[:N_TEST_CASES]]
print(f"Testing multi-lesion handling on: {test_case_ids}")


for case_id in test_case_ids:
    print(f"\n{'='*60}\n{case_id}\n{'='*60}")

    seg_dir = ISO_RESAMPLING_DIR / case_id / "segmentations"
    veins_mask = load_bin(seg_dir / VEINS_MASK_NAME)
    full_tumour_mask = load_bin(seg_dir / TUMOUR_MASK_NAME)

    veins_mask_lcc = keep_lcc(veins_mask)
    skel = skeletonize(veins_mask_lcc).astype(np.uint8)
    G = skeleton_to_graph(skel)

    tumour_components = get_tumour_components(full_tumour_mask)
    print(f"  Found {len(tumour_components)} tumour component(s), "
          f"sizes = {[c.sum() for c in tumour_components]}")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    v_pts = sparse_coords(veins_mask_lcc, step=4)
    ax.scatter(v_pts[:, 2], v_pts[:, 1], v_pts[:, 0], c='blue', s=1, alpha=0.15, label='veins')

    colors = ['red', 'orange', 'purple', 'brown', 'pink']
    path_colors = ['lime', 'cyan', 'yellow', 'magenta', 'black']

    for i, component_mask in enumerate(tumour_components):
        t_pts = sparse_coords(component_mask, step=2)
        color = colors[i % len(colors)]
        ax.scatter(t_pts[:, 2], t_pts[:, 1], t_pts[:, 0], c=color, s=3, alpha=0.3,
                   label=f'tumour component {i+1} ({component_mask.sum()} vox)')

        try:
            path_xyz = get_branch_path_coords(G, component_mask)
            path_color = path_colors[i % len(path_colors)]
            ax.plot(path_xyz[:, 2], path_xyz[:, 1], path_xyz[:, 0],
                    c=path_color, linewidth=3, label=f'branch for component {i+1}')
            print(f"  Component {i+1} ({component_mask.sum()} vox): "
                  f"branch path length = {len(path_xyz)}")
        except Exception as e:
            print(f"  Component {i+1} ({component_mask.sum()} vox): FAILED -> {e}")

    ax.legend(fontsize=7)
    ax.set_title(f"{case_id}: multi-lesion branch selection")

    out_png = OUT_DIR / f"{case_id}_multi_lesion.png"
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot -> {out_png}")

print(f"\n[DONE] Results saved to {OUT_DIR}")

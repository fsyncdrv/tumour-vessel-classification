"""
Determine a fixed target crop size (mm) for bbox_size_capped_v1 based
on the distribution of bbox_mm_v1's actual (variable) box sizes
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent
LABEL_DERIVATION_DIR = EXPERIMENTS_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

from config import DERIVED_ANGLES_DIR
from data_prep import load_ct, get_slice_axis, get_inplane_axes_and_spacing
from data_prep_bbox import compute_crop_geometry

df = pd.read_csv(DERIVED_ANGLES_DIR / "derived_labels.csv")

sizes_mm = []
for row in df.itertuples():
    try:
        geo = compute_crop_geometry(row.case_id, row.driving_vessel)
        _, affine = load_ct(row.case_id)
        slice_axis = geo["slice_axis"]
        _, _, spacing_a, spacing_b = get_inplane_axes_and_spacing(affine, slice_axis=slice_axis)
        a_range, b_range = geo["a_range"], geo["b_range"]
        h_mm = (a_range[1] - a_range[0]) * spacing_a
        w_mm = (b_range[1] - b_range[0]) * spacing_b
        sizes_mm.append(max(h_mm, w_mm))  # larger dimension per case
    except Exception as e:
        print(f"skip {row.case_id}: {e}")

sizes_mm = np.array(sizes_mm)
print(f"n={len(sizes_mm)}")
print(f"median={np.median(sizes_mm):.1f}mm, "
      f"p75={np.percentile(sizes_mm,75):.1f}mm, "
      f"p90={np.percentile(sizes_mm,90):.1f}mm, "
      f"p95={np.percentile(sizes_mm,95):.1f}mm, "
      f"max={sizes_mm.max():.1f}mm")

with open(EXPERIMENTS_DIR / "bbox_target_size_results.txt", "w") as f:
    f.write(f"n={len(sizes_mm)}\n")
    f.write(f"median={np.median(sizes_mm):.1f}mm\n")
    f.write(f"p75={np.percentile(sizes_mm,75):.1f}mm\n")
    f.write(f"p90={np.percentile(sizes_mm,90):.1f}mm\n")
    f.write(f"p95={np.percentile(sizes_mm,95):.1f}mm\n")
    f.write(f"max={sizes_mm.max():.1f}mm\n")

"""
This tests whether the crop's physical size alone, with no image content at all, can predict the label.
A simple logistic regression is trained on just the box dimensions (width, height, area, aspect
ratio). If it scores well above the majority-class baseline, that's evidence the model's real performance
could partly come from crop zoom level rather than genuine anatomical content.


Result: 5-fold CV AUROC of 0.768 +/- 0.152 (majority-class baseline accuracy 0.840), close to bbox_mm_v1's own AUROC (0.78).
It is possible tthen that bbox_mm_v1's performance oculd be due to size rather than the featuresw in the image
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

EXPERIMENTS_DIR = Path(__file__).resolve().parent
LABEL_DERIVATION_DIR = EXPERIMENTS_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

from config import DERIVED_ANGLES_DIR
from data_prep_bbox_sizecapped import compute_sizecapped_geometry


def main():
    df = pd.read_csv(DERIVED_ANGLES_DIR / "derived_labels.csv")

    rows = []
    for row in df.itertuples():
        try:
            geo = compute_sizecapped_geometry(row.case_id, row.driving_vessel)
        except Exception:
            continue
        a0, a1 = geo["a_range"]
        b0, b1 = geo["b_range"]
        width = a1 - a0
        height = b1 - b0
        area = width * height
        aspect = width / height if height > 0 else 0
        rows.append({
            "case_id": row.case_id,
            "width": width, "height": height,
            "area": area, "aspect": aspect,
            "derived_label": 1 if row.derived_label == "high_vascular_contact" else 0,
        })

    feat_df = pd.DataFrame(rows)
    print(f"Computed bbox geometry for {len(feat_df)} / {len(df)} cases")

    X = feat_df[["width", "height", "area", "aspect"]].values
    y = feat_df["derived_label"].values

    majority_baseline = max(y.mean(), 1 - y.mean())
    print(f"Majority-class baseline accuracy: {majority_baseline:.3f}")

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"\n5-fold CV AUROC using only bbox size/shape (no image content): "
          f"{scores.mean():.3f} +/- {scores.std():.3f}")


if __name__ == "__main__":
    main()

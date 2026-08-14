"""
Stratified train/val/test split for the classification cohort.

Splits derived_labels.csv into train/val/test sets while preserving the
class ratio (low_vascular_contact vs high_vascular_contact) in each
subset
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_DERIVATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

from config import DERIVED_ANGLES_DIR

FINAL_LABELS_CSV = DERIVED_ANGLES_DIR / "derived_labels.csv"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

RANDOM_SEED = 42


def main():
    df = pd.read_csv(FINAL_LABELS_CSV)
    print(f"Loaded {len(df)} cases")
    print(df["derived_label"].value_counts())

    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_FRAC + TEST_FRAC),
        stratify=df["derived_label"],
        random_state=RANDOM_SEED,
    )

    relative_test_size = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["derived_label"],
        random_state=RANDOM_SEED,
    )

    print(f"\nTrain: {len(train_df)} cases")
    print(train_df["derived_label"].value_counts())
    print(f"\nVal: {len(val_df)} cases")
    print(val_df["derived_label"].value_counts())
    print(f"\nTest: {len(test_df)} cases")
    print(test_df["derived_label"].value_counts())

    train_ids = set(train_df["case_id"])
    val_ids = set(val_df["case_id"])
    test_ids = set(test_df["case_id"])
    assert not (train_ids & val_ids), "Overlap between train and val!"
    assert not (train_ids & test_ids), "Overlap between train and test!"
    assert not (val_ids & test_ids), "Overlap between val and test!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(df), "Case count mismatch!"
    print("\n[OK] No case leakage between splits, all cases accounted for.")

    train_df.to_csv(DERIVED_ANGLES_DIR / "split_train.csv", index=False)
    val_df.to_csv(DERIVED_ANGLES_DIR / "split_val.csv", index=False)
    test_df.to_csv(DERIVED_ANGLES_DIR / "split_test.csv", index=False)
    print(f"\nSaved split_train.csv, split_val.csv, split_test.csv to {DERIVED_ANGLES_DIR}")


if __name__ == "__main__":
    main()

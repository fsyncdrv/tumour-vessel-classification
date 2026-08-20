"""
Stratified K-fold cross-validation split.

This step was added to address the small positive group (16%).

Keeps a held-out TEST set completely separate (with same 15%, same random_state as in stratifed_split.py)

The remaining 85% (original TRAIN+VAL pool) is split into K stratified folds. For each fold i,
that fold becomes the val set and the other K-1 folds become the train set. It producing split_train_fold{i}.csv
and split_val_fold{i}.csv for i in 0..K-1.

"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

LABEL_DERIVATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))

from config import DERIVED_ANGLES_DIR

FINAL_LABELS_CSV = DERIVED_ANGLES_DIR / "derived_labels.csv"

TEST_FRAC = 0.15
N_FOLDS = 5
RANDOM_SEED = 42


def main():
    df = pd.read_csv(FINAL_LABELS_CSV)
    print(f"Loaded {len(df)} cases")
    print(df["derived_label"].value_counts())

    # Same test split as stratified_split.py (w/ same seed, same fraction)
    trainval_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        stratify=df["derived_label"],
        random_state=RANDOM_SEED,
    )

    print(f"\nHeld-out test set: {len(test_df)}")
    print(test_df["derived_label"].value_counts())

    test_df.to_csv(DERIVED_ANGLES_DIR / "split_test.csv", index=False)

    existing_test_path = DERIVED_ANGLES_DIR / "split_test.csv"
    print(f"[OK] split_test.csv written/confirmed at {existing_test_path}")


    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    trainval_df = trainval_df.reset_index(drop=True)
    labels = trainval_df["derived_label"]

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(trainval_df, labels)):
        fold_train_df = trainval_df.iloc[train_idx]
        fold_val_df = trainval_df.iloc[val_idx]

        train_ids = set(fold_train_df["case_id"])
        val_ids = set(fold_val_df["case_id"])
        test_ids = set(test_df["case_id"])
        assert not (train_ids & val_ids), f"Fold {fold_i}: overlap train/val!"
        assert not (train_ids & test_ids), f"Fold {fold_i}: overlap train/test!"
        assert not (val_ids & test_ids), f"Fold {fold_i}: overlap val/test!"

        fold_train_df.to_csv(DERIVED_ANGLES_DIR / f"split_train_fold{fold_i}.csv", index=False)
        fold_val_df.to_csv(DERIVED_ANGLES_DIR / f"split_val_fold{fold_i}.csv", index=False)

        print(f"\nFold {fold_i}: train={len(fold_train_df)}, val={len(fold_val_df)}")
        print(f"  train label counts: {fold_train_df['derived_label'].value_counts().to_dict()}")
        print(f"  val label counts:   {fold_val_df['derived_label'].value_counts().to_dict()}")

    print(f"\n[DONE] Wrote split_test.csv + {N_FOLDS} fold pairs "
          f"(split_train_fold{{0..{N_FOLDS-1}}}.csv, split_val_fold{{0..{N_FOLDS-1}}}.csv) "
          f"to {DERIVED_ANGLES_DIR}")


if __name__ == "__main__":
    main()

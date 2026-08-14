"""
PyTorch Dataset for the CT vascular-contact classification task.

Loads pre-extracted .npy crops (2D or 2.5D) and their derived labels,
with optional data augmentation applied only during training.

"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"low_vascular_contact": 0, "high_vascular_contact": 1}


class CTVascularContactDataset(Dataset):
    def __init__(self, split_csv, crop_dir, mode="2d", transform=None):
        """
        split_csv: path to split_train.csv / split_val.csv / split_test.csv
        crop_dir:  path to the folder containing .npy crops for this mode
        mode:      "2d" or "2.5d" > only affects expected array shape
        transform: optional callable applied to the crop (numpy array in,
                   numpy array out). Augmentation, only pass this for
                   the TRAINING split, not val/test
        """
        self.df = pd.read_csv(split_csv)
        self.crop_dir = Path(crop_dir)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]
        label_str = row["derived_label"]

        crop = np.load(self.crop_dir / f"{case_id}.npy")

        if self.transform is not None:
            crop = self.transform(crop)

        if self.mode == "2d":
            crop = crop[np.newaxis, :, :]   # (1, H, W)

        label = LABEL_MAP[label_str]

        return torch.from_numpy(crop).float(), torch.tensor(label, dtype=torch.long), case_id


# ------------------------- simple augmentation -------------------------

def train_augment(crop):
    if np.random.rand() < 0.5:
        crop = np.flip(crop, axis=-1).copy()  # horizontal flip
    k = np.random.choice([0, 1, 2, 3])
    if k > 0:
        crop = np.rot90(crop, k=k, axes=(-2, -1)).copy()
    return crop


# ------------------------- local test -------------------------

if __name__ == "__main__":
    LABEL_DERIVATION_DIR = Path(__file__).resolve().parent.parent / "label_derivation"
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(LABEL_DERIVATION_DIR))

    from config import DERIVED_ANGLES_DIR

    CROP_DIR_2D = DERIVED_ANGLES_DIR.parent / "classification_inputs" / "2d"

    train_ds = CTVascularContactDataset(
        split_csv=DERIVED_ANGLES_DIR / "split_train.csv",
        crop_dir=CROP_DIR_2D,
        mode="2d",
        transform=train_augment,
    )

    print(f"Train dataset size: {len(train_ds)}")

    # test loading a single item
    crop, label, case_id = train_ds[0]
    print(f"First item: case_id={case_id}, crop.shape={crop.shape}, "
          f"crop.dtype={crop.dtype}, label={label.item()}")

    # test with a DataLoader (batching)
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    batch_crops, batch_labels, batch_ids = next(iter(loader))
    print(f"\nBatch shapes: crops={batch_crops.shape}, labels={batch_labels.shape}")
    print(f"Batch labels: {batch_labels.tolist()}")
    print(f"Batch case_ids: {batch_ids}")

    # confirm value range looks sane (should be roughly 0-1, given normalization)
    print(f"\nBatch crop value range: min={batch_crops.min():.3f}, max={batch_crops.max():.3f}")

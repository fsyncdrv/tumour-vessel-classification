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
from scipy.ndimage import rotate as scipy_rotate

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

# def train_augment(crop):
#     if np.random.rand() < 0.5:
#         crop = np.flip(crop, axis=-1).copy()  # horizontal flip
#     k = np.random.choice([0, 1, 2, 3])
#     if k > 0:
#         crop = np.rot90(crop, k=k, axes=(-2, -1)).copy()
#     return crop

def train_augment(crop, rotation_max_deg=15, brightness_range=0.15,
                   contrast_range=0.15, noise_std=0.02,
                   cutout_prob=0.3, cutout_max_frac=0.15):
    """
    crop: numpy float32 array, (H, W) for 2D or (C, H, W) for 2.5D,
          values already in [0, 1].
    """
    is_2_5d = crop.ndim == 3

    # ---- flip (existing) ----
    if np.random.rand() < 0.5:
        crop = np.flip(crop, axis=-1).copy()

    # ---- 90-degree rotation (existing) ----
    k = np.random.choice([0, 1, 2, 3])
    if k > 0:
        crop = np.rot90(crop, k=k, axes=(-2, -1)).copy()

    # ---- continuous small-angle rotation ----
    # Fills the corners introduced by rotation with 0 (background HU
    # after windowing/normalization), consistent with the zero-padding
    # already used at crop boundaries elsewhere in the pipeline.
    angle = np.random.uniform(-rotation_max_deg, rotation_max_deg)
    if is_2_5d:
        # rotate each slice identically so anatomy stays aligned across
        # the stack. This is appliesd along the last two axes only.
        crop = scipy_rotate(crop, angle, axes=(-2, -1), reshape=False,
                             order=1, mode="constant", cval=0.0)
    else:
        crop = scipy_rotate(crop, angle, reshape=False, order=1,
                             mode="constant", cval=0.0)

    # ---- brightness/contrast jitter ----
    # Simulates the kind of HU-windowing/scanner-calibration variation
    # already observed across the dataset's different acquisition eras.
    brightness = np.random.uniform(-brightness_range, brightness_range)
    contrast = np.random.uniform(1 - contrast_range, 1 + contrast_range)
    crop = (crop - 0.5) * contrast + 0.5 + brightness

    # ---- light gaussian noise ----
    if noise_std > 0:
        crop = crop + np.random.normal(0, noise_std, size=crop.shape)

    crop = np.clip(crop, 0.0, 1.0).astype(np.float32)

    # ---- coarse dropout / cutout ----
    # Randomly zero a small rectangular patch. It discourages the model
    # from relying too heavily on any single localized region
    if np.random.rand() < cutout_prob:
        h, w = crop.shape[-2], crop.shape[-1]
        cut_h = int(np.random.uniform(0.05, cutout_max_frac) * h)
        cut_w = int(np.random.uniform(0.05, cutout_max_frac) * w)
        y0 = np.random.randint(0, max(1, h - cut_h))
        x0 = np.random.randint(0, max(1, w - cut_w))
        if is_2_5d:
            crop[:, y0:y0 + cut_h, x0:x0 + cut_w] = 0.0
        else:
            crop[y0:y0 + cut_h, x0:x0 + cut_w] = 0.0

    return crop.copy()



if __name__ == "__main__":
    LABEL_DERIVATION_DIR = Path(__file__).resolve().parent.parent / "label_derivation"
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(LABEL_DERIVATION_DIR))

    from config import DERIVED_ANGLES_DIR

    CROP_DIR_2D = DERIVED_ANGLES_DIR.parent / "classification_inputs" / "slice_mm_v1" / "2d"

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

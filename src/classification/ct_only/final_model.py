"""
This script trains the final bbox_size_capped_v1 model on the full train+val pool for the held-out test-set evaluation

"""

import sys
import json
import time
# import random
from pathlib import Path
# import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

CLASSIFICATION_DIR = Path(__file__).resolve().parent
LABEL_DERIVATION_DIR = CLASSIFICATION_DIR.parent / "label_derivation"
PROJECT_ROOT = CLASSIFICATION_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))
sys.path.insert(0, str(CLASSIFICATION_DIR))

from config import DERIVED_ANGLES_DIR
from dataset import CTVascularContactDataset, train_augment
from model import build_resnet50, count_trainable_params, prepare_2d_batch
from train import evaluate, set_seed

## These are the same hyperparameters I used during training bbox_size_capped_v1
SEED = 123
CROP_VERSION = "bbox_size_capped_v1"
FREEZE_UNTIL = "layer3_layer4"
LR = 1e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 16
PATIENCE = 30
MAX_EPOCHS = 100
INTERNAL_VAL_FRAC = 0.10


def build_merged_split(seed):
    train_df = pd.read_csv(DERIVED_ANGLES_DIR / "split_train.csv")
    val_df = pd.read_csv(DERIVED_ANGLES_DIR / "split_val.csv")
    pool_df = pd.concat([train_df, val_df], ignore_index=True)

    final_train_df, internal_val_df = train_test_split(
        pool_df, test_size=INTERNAL_VAL_FRAC, stratify=pool_df["derived_label"],
        random_state=seed,
    )
    return final_train_df, internal_val_df


def run_final_training(mode):
    set_seed(SEED)

    out_dir = PROJECT_ROOT / "outputs" / "classification_runs" / "final_model" / CROP_VERSION / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[{mode}] Using device: {device}")

    crop_dir_name = "2d" if mode == "2d" else "2.5d"
    crop_dir = DERIVED_ANGLES_DIR.parent / "classification_inputs" / CROP_VERSION / crop_dir_name

    final_train_df, internal_val_df = build_merged_split(SEED)

    final_train_csv = out_dir / "final_train_split.csv"
    internal_val_csv = out_dir / "internal_val_split.csv"
    final_train_df.to_csv(final_train_csv, index=False)
    internal_val_df.to_csv(internal_val_csv, index=False)
    print(f"[{mode}] Final train: {len(final_train_df)}, internal val: {len(internal_val_df)}")

    train_ds = CTVascularContactDataset(final_train_csv, crop_dir, mode=mode, transform=train_augment)
    val_ds = CTVascularContactDataset(internal_val_csv, crop_dir, mode=mode, transform=None)

    train_labels_arr = final_train_df["derived_label"].map(
        {"low_vascular_contact": 0, "high_vascular_contact": 1}
    ).values
    n_neg = (train_labels_arr == 0).sum()
    n_pos = (train_labels_arr == 1).sum()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"[{mode}] Class weights (neg, pos): {class_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    input_channels = 3 if mode == "2d" else 5
    model = build_resnet50(input_channels=input_channels, num_classes=2, freeze_until=FREEZE_UNTIL)
    model = model.to(device)

    trainable, total = count_trainable_params(model)
    print(f"[{mode}] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    best_val_auroc = -1.0
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for crops, labels, _ in train_loader:
            crops, labels = crops.to(device), labels.to(device)
            if mode == "2d":
                crops = prepare_2d_batch(crops)

            optimizer.zero_grad()
            outputs = model(crops)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * crops.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, mode, criterion)
        scheduler.step(val_metrics["auroc"])

        epoch_time = time.time() - epoch_start
        print(f"[{mode}] Epoch {epoch}/{MAX_EPOCHS} ({epoch_time:.1f}s) | "
              f"train_loss={train_loss:.4f} | val_auroc={val_metrics['auroc']:.4f} | "
              f"val_auprc={val_metrics['auprc']:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, **{
            f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"
        }})

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "final_model.pt")
            print(f"  -> New best val AUROC ({best_val_auroc:.4f}), checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[{mode}] Early stopping at epoch {epoch}.")
                break

    with open(out_dir / "final_training_history.json", "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_val_auroc": best_val_auroc}, f, indent=2)

    print(f"[{mode}] DONE. Best internal-val AUROC: {best_val_auroc:.4f} (epoch {best_epoch})")
    return out_dir / "final_model.pt"


# if __name__ == "__main__":
#     run_final_training("2.5d")

if __name__ == "__main__":
    for mode in ["2d", "2.5d"]:
        run_final_training(mode)

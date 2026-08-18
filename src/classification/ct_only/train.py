"""
Training script for the CT vascular-contact classifier

"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score


CLASSIFICATION_DIR = Path(__file__).resolve().parent
LABEL_DERIVATION_DIR = CLASSIFICATION_DIR.parent / "label_derivation"
PROJECT_ROOT = CLASSIFICATION_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))
sys.path.insert(0, str(CLASSIFICATION_DIR))

from config import DERIVED_ANGLES_DIR
from dataset import CTVascularContactDataset, train_augment
from model import build_resnet50, count_trainable_params, prepare_2d_batch


def parse_args():
    p = argparse.ArgumentParser(description="Train CT vascular-contact classifier")
    p.add_argument("--mode", choices=["2d", "2.5d"], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_until", choices=["all", "layer3_layer4", "layer4", "none"], default="layer4")
    p.add_argument("--patience", type=int, default=10,
                    help="Early stopping: stop if val AUPRC doesn't improve")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--balance_method", choices=["loss_weight", "sampler", "none"], default="loss_weight",
                    help="How to handle class imbalance: reweight the loss (original "
                         "default behavior), oversample the minority class per-batch via "
                         "a WeightedRandomSampler, or neither. Cant run loss_weight and "
                         "sampler together as stacking both tends to overcorrect (recall "
                         "goes up, precision collapses).")
    p.add_argument("--out_dir", type=str, default=None,
                    help="Where to save checkpoints/logs (default: outputs/classification_runs/<crop_version>/<mode>/fold{N}/seed{S})")
    p.add_argument("--fold", type=int, default=None,
                    help="If set, train on split_train_fold{N}.csv / split_val_fold{N}.csv "
                         "instead of split_train.csv / split_val.csv (for cross-validation). "
                         "split_test.csv is never touched by this flag.")
    p.add_argument("--crop_version", type=str, default="fov_mm_v4",
                    help="Which classification_inputs/<crop_version>/ folder to train on, "
                         "e.g. 'fov_mm_v4' or 'bbox_mm_v1'.")
    return p.parse_args()


def evaluate(model, loader, device, mode, criterion):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for crops, labels, _ in loader:
            crops, labels = crops.to(device), labels.to(device)
            if mode == "2d":
                crops = prepare_2d_batch(crops)

            outputs = model(crops)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * crops.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # specificity = true negative rate = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    majority_class_baseline = max(np.mean(np.array(all_labels) == 0), np.mean(np.array(all_labels) == 1))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,  ## This woont do much tbh
        "auroc": auroc, ## Literature points at relying more on using this
        "auprc": auprc, ## And this
        "f1": f1,  ## And this
        "majority_baseline": majority_class_baseline,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "confusion_matrix": cm,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    print(f"EFFECTIVE CONFIG: {vars(args)}")

    set_seed(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "outputs" / "classification_runs" / "auroc_training_best" / args.crop_version / args.mode /
        (f"fold{args.fold}" if args.fold is not None else "holdout") /
        f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")


    crop_dir_name = "2d" if args.mode == "2d" else "2.5d"
    ## I'm testing different crops
    ## Original crops
    ## Fov_mm_v3 ==> mm-based crop-then-resize
    crop_dir = DERIVED_ANGLES_DIR.parent / "classification_inputs" / args.crop_version / crop_dir_name
    print(f"Crop dir: {crop_dir}")

    if args.fold is not None:
        train_csv = DERIVED_ANGLES_DIR / f"split_train_fold{args.fold}.csv"
        val_csv = DERIVED_ANGLES_DIR / f"split_val_fold{args.fold}.csv"
        print(f"Using CV fold {args.fold}: {train_csv.name} / {val_csv.name}")
    else:
        train_csv = DERIVED_ANGLES_DIR / "split_train.csv"
        val_csv = DERIVED_ANGLES_DIR / "split_val.csv"

    train_ds = CTVascularContactDataset(
        train_csv, crop_dir, mode=args.mode, transform=train_augment
    )
    val_ds = CTVascularContactDataset(
        val_csv, crop_dir, mode=args.mode, transform=None
    )

    train_labels_arr = train_ds.df["derived_label"].map({"low_vascular_contact": 0, "high_vascular_contact": 1}).values

    train_sampler = None
    shuffle_train = True

    if args.balance_method == "sampler":
        n_neg = (train_labels_arr == 0).sum()
        n_pos = (train_labels_arr == 1).sum()
        sample_weights = np.where(train_labels_arr == 1, 1.0 / n_pos, 1.0 / n_neg)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False
        print(f"Using WeightedRandomSampler (neg={n_neg}, pos={n_pos}) - "
              f"loss will be UNweighted.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle_train,
        sampler=train_sampler, num_workers=4,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {len(train_ds)} cases, Val: {len(val_ds)} cases")

    if args.balance_method == "loss_weight":
        n_neg = (train_labels_arr == 0).sum()
        n_pos = (train_labels_arr == 1).sum()
        class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
        print(f"Class weights (neg, pos): {class_weights.tolist()}")
    else:
        class_weights = None
        if args.balance_method == "none":
            print("No class imbalance handling (loss unweighted, no sampler).")

    # ------------------------- model -------------------------

    input_channels = 3 if args.mode == "2d" else 5
    model = build_resnet50(input_channels=input_channels, num_classes=2, freeze_until=args.freeze_until)
    model = model.to(device)

    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    # ------------------------- training loop -------------------------
    best_val_auroc = -1.0
    best_val_confusion_matrix = None
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for crops, labels, _ in train_loader:
            crops, labels = crops.to(device), labels.to(device)
            if args.mode == "2d":
                crops = prepare_2d_batch(crops)

            optimizer.zero_grad()
            outputs = model(crops)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * crops.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = running_loss / len(train_ds)
        train_acc = train_correct / train_total
        val_metrics = evaluate(model, val_loader, device, args.mode, criterion)
        scheduler.step(val_metrics["auroc"])
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | "
              f"lr={current_lr:.2e} | "
              f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} (baseline={val_metrics['majority_baseline']:.4f}) | "
              f"val_auroc={val_metrics['auroc']:.4f} | val_auprc={val_metrics['auprc']:.4f} | "
              f"val_f1={val_metrics['f1']:.4f} | "
              f"val_precision={val_metrics['precision']:.4f} | val_recall(sens)={val_metrics['recall']:.4f} | "
              f"val_specificity={val_metrics['specificity']:.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
            "val_majority_baseline": val_metrics["majority_baseline"],
            "val_auroc": val_metrics["auroc"], "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
        })

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_val_confusion_matrix = val_metrics["confusion_matrix"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> New best val AUROC ({best_val_auroc:.4f}), checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
                break

    import json
    history_with_cm = {
        "epochs": history,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "best_val_auprc": history[best_epoch - 1]["val_auprc"] if best_epoch else None,
        "best_val_confusion_matrix": best_val_confusion_matrix.tolist() if best_val_confusion_matrix is not None else None,
    }
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_with_cm, f, indent=2)

    print(f"\n[DONE] Best val AUROC: {best_val_auroc:.4f} (epoch {best_epoch})")
    print(f"\nConfusion matrix at best epoch (rows=true, cols=predicted):")
    print(f"                  pred_low  pred_high")
    print(f"  true_low        {best_val_confusion_matrix[0,0]:>8}  {best_val_confusion_matrix[0,1]:>9}")
    print(f"  true_high       {best_val_confusion_matrix[1,0]:>8}  {best_val_confusion_matrix[1,1]:>9}")
    print(f"Checkpoint + history saved to {out_dir}")


if __name__ == "__main__":
    main()

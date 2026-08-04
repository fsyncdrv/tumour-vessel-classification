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
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score


CLASSIFICATION_DIR = Path(__file__).resolve().parent
LABEL_DERIVATION_DIR = CLASSIFICATION_DIR.parent / "label_derivation"
PROJECT_ROOT = CLASSIFICATION_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))
sys.path.insert(0, str(CLASSIFICATION_DIR))

from config import DERIVED_ANGLES_DIR
from dataset import CTVascularContactDataset, train_augment
from model import build_resnet50, prepare_2d_batch, count_trainable_params


def parse_args():
    p = argparse.ArgumentParser(description="Train CT vascular-contact classifier")
    p.add_argument("--mode", choices=["2d", "2.5d"], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze_until", choices=["all", "layer4", "none"], default="layer4")
    p.add_argument("--patience", type=int, default=10,
                    help="Early stopping: stop if val AUROC doesn't improve for this many epochs")
    p.add_argument("--out_dir", type=str, default=None,
                    help="Where to save checkpoints/logs (default: outputs/classification_runs/<mode>)")
    return p.parse_args()


def evaluate(model, loader, device, mode, criterion):
    """Run the model on a loader (val or test), no gradient updates."""
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
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    majority_class_baseline = max(np.mean(np.array(all_labels) == 0), np.mean(np.array(all_labels) == 1))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,  ## This woont do much tbh
        "auroc": auroc, ## Literature points at relying more on using this
        "f1": f1,  ## And this
        "majority_baseline": majority_class_baseline,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "confusion_matrix": cm,
    }


def main():
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "outputs" / "classification_runs" / args.mode
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")


    crop_dir_name = "2d" if args.mode == "2d" else "2.5d"
    crop_dir = DERIVED_ANGLES_DIR.parent / "classification_inputs" / crop_dir_name

    train_ds = CTVascularContactDataset(
        DERIVED_ANGLES_DIR / "split_train.csv", crop_dir, mode=args.mode, transform=train_augment
    )
    val_ds = CTVascularContactDataset(
        DERIVED_ANGLES_DIR / "split_val.csv", crop_dir, mode=args.mode, transform=None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {len(train_ds)} cases, Val: {len(val_ds)} cases")


    train_labels = train_ds.df["derived_label"].map({"low_vascular_contact": 0, "high_vascular_contact": 1})
    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"Class weights (neg, pos): {class_weights.tolist()}")

    # ------------------------- model -------------------------

    input_channels = 3 if args.mode == "2d" else 5
    model = build_resnet50(input_channels=input_channels, num_classes=2, freeze_until=args.freeze_until)
    model = model.to(device)

    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # ------------------------- training loop -------------------------

    best_val_recall = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

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

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, args.mode, criterion)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | "
              f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} (baseline={val_metrics['majority_baseline']:.4f}) | "
              f"val_auroc={val_metrics['auroc']:.4f} | val_f1={val_metrics['f1']:.4f} | "
              f"val_precision={val_metrics['precision']:.4f} | val_recall(sens)={val_metrics['recall']:.4f} | "
              f"val_specificity={val_metrics['specificity']:.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
            "val_majority_baseline": val_metrics["majority_baseline"],
            "val_auroc": val_metrics["auroc"], "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
        })

        if val_metrics["recall"] > best_val_recall:
            best_val_recall = val_metrics["recall"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> New best val recall ({best_val_recall:.4f}), checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
                break

    # ------------------------- saving training history -------------------------

    import json
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[DONE] Best val recall (sensitivity): {best_val_recall:.4f}")
    print(f"Checkpoint + history saved to {out_dir}")


if __name__ == "__main__":
    main()

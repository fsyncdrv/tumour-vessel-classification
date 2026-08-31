"""
this script evaluates the final bbox_size_capped_v1 models (2D and 2.5D) on the held-out test set (split_test.csv) which was
never touched during cross-validation or in final training
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CLASSIFICATION_DIR = Path(__file__).resolve().parent
LABEL_DERIVATION_DIR = CLASSIFICATION_DIR.parent / "label_derivation"
PROJECT_ROOT = CLASSIFICATION_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LABEL_DERIVATION_DIR))
sys.path.insert(0, str(CLASSIFICATION_DIR))

from config import DERIVED_ANGLES_DIR
from dataset import CTVascularContactDataset
from model import build_resnet50, prepare_2d_batch
from train import evaluate

CROP_VERSION = "bbox_size_capped_v1"
FREEZE_UNTIL = "layer3_layer4"
BATCH_SIZE = 16


def run_test_eval(mode):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    crop_dir_name = "2d" if mode == "2d" else "2.5d"
    crop_dir = DERIVED_ANGLES_DIR.parent / "classification_inputs" / CROP_VERSION / crop_dir_name
    test_csv = DERIVED_ANGLES_DIR / "split_test.csv"

    test_ds = CTVascularContactDataset(test_csv, crop_dir, mode=mode, transform=None)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"[{mode}] Test set: {len(test_ds)} cases")

    input_channels = 3 if mode == "2d" else 5
    model = build_resnet50(input_channels=input_channels, num_classes=2, freeze_until=FREEZE_UNTIL)

    ckpt_path = PROJECT_ROOT / "outputs" / "classification_runs" / "final_model" / CROP_VERSION / mode / "final_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, device, mode, criterion)

    metrics_clean = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    metrics_clean["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    print(f"[{mode}] TEST RESULTS: AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
          f"Recall={metrics['recall']:.4f}  Precision={metrics['precision']:.4f}  F1={metrics['f1']:.4f}")

    out_path = ckpt_path.parent / f"test_results_{CROP_VERSION}_{mode}.json"
    with open(out_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"[{mode}] Saved to {out_path}")

    return metrics_clean


if __name__ == "__main__":
    for mode in ["2d", "2.5d"]:
        run_test_eval(mode)

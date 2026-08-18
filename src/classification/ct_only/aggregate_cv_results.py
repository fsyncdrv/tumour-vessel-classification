"""
Aggregates results across all CV runs (5 folds x N seeds) into a single
summary: mean +/- std for AUPRC, recall, precision, F1, AUROC at each
run's best epoch, plus per-run detail so you can spot outlier folds.

"""

import argparse
import json
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["2d", "2.5d"], default="2d")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 456])
    p.add_argument("--crop_version", type=str, default="fov_mm_v4")
    p.add_argument("--base_dir", type=str,
                    default="../../../outputs/classification_runs",
                    help="Base dir containing {crop_version}/{mode}/fold{N}/seed{S}/training_history.json")
    return p.parse_args()


def confusion_matrix_metrics(cm):
    """cm is [[tn, fp], [fn, tp]] as saved in training_history.json."""
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def main():
    args = parse_args()
    base = Path(args.base_dir) / args.crop_version / args.mode

    rows = []
    missing = []

    for seed in args.seeds:
        for fold in range(args.n_folds):
            path = base / f"fold{fold}" / f"seed{seed}" / "training_history.json"
            if not path.exists():
                missing.append(str(path))
                continue

            with open(path) as f:
                hist = json.load(f)

            best_epoch = hist["best_epoch"]
            best_auprc = hist["best_val_auprc"]
            cm = hist["best_val_confusion_matrix"]

            # Pull the full metrics row for the best epoch from the epoch list,
            # so we report AUROC/F1 etc consistently with what was logged,
            # not just recomputed from the confusion matrix.
            epoch_row = next((e for e in hist["epochs"] if e["epoch"] == best_epoch), None)

            derived = confusion_matrix_metrics(cm)

            rows.append({
                "seed": seed,
                "fold": fold,
                "best_epoch": best_epoch,
                "auprc": best_auprc,
                "auroc": epoch_row["val_auroc"] if epoch_row else None,
                "precision": derived["precision"],
                "recall": derived["recall"],
                "specificity": derived["specificity"],
                "f1": derived["f1"],
                "confusion_matrix": cm,
            })

    if missing:
        print(f"WARNING: {len(missing)} expected run(s) not found:")
        for m in missing:
            print(f"  {m}")
        print()

    if not rows:
        print("No runs found. Check --base_dir and that training_history.json "
              "files exist under fold{N}/seed{S}/.")
        return

    print(f"=== Per-run results ({len(rows)} runs found) ===\n")
    print(f"{'seed':>6} {'fold':>5} {'epoch':>6} {'auprc':>7} {'auroc':>7} "
          f"{'prec':>6} {'recall':>7} {'f1':>6}")
    for r in rows:
        print(f"{r['seed']:>6} {r['fold']:>5} {r['best_epoch']:>6} "
              f"{r['auprc']:>7.4f} {r['auroc']:>7.4f} "
              f"{r['precision']:>6.3f} {r['recall']:>7.3f} {r['f1']:>6.3f}")

    print(f"\n=== Aggregate (mean +/- std across {len(rows)} runs) ===\n")
    for metric in ["auprc", "auroc", "precision", "recall", "f1"]:
        values = np.array([r[metric] for r in rows])
        print(f"{metric:>10}: {values.mean():.4f} +/- {values.std():.4f}  "
              f"(min={values.min():.4f}, max={values.max():.4f})")

    # Per-seed breakdown too, in case one seed is systematically different
    print(f"\n=== Per-seed breakdown (mean across folds) ===\n")
    for seed in args.seeds:
        seed_rows = [r for r in rows if r["seed"] == seed]
        if not seed_rows:
            continue
        auprc_vals = np.array([r["auprc"] for r in seed_rows])
        recall_vals = np.array([r["recall"] for r in seed_rows])
        print(f"  seed {seed}: AUPRC {auprc_vals.mean():.4f} +/- {auprc_vals.std():.4f}  |  "
              f"Recall {recall_vals.mean():.4f} +/- {recall_vals.std():.4f}  "
              f"({len(seed_rows)} folds)")

    summary = {
        "n_runs": len(rows),
        "runs": rows,
        "aggregate": {
            metric: {
                "mean": float(np.array([r[metric] for r in rows]).mean()),
                "std": float(np.array([r[metric] for r in rows]).std()),
            }
            for metric in ["auprc", "auroc", "precision", "recall", "f1"]
        },
    }
    out_path = Path(f"cv_aggregate_summary_{args.crop_version}_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved machine-readable summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()

"""Step 8: run the experiment matrix for a chosen backbone and save results.

Trains augmented x unaugmented data on scratch / full_finetune / last_layer
strategies (6 runs total) for a single --backbone, evaluates each on the
held-out test set, and saves every model checkpoint plus a summary CSV to
MODEL_DIR. Checkpoints are named `<backbone>_<data>_<strategy>.pth` — drop
them (or any checkpoint matching this pattern) into MODEL_DIR and app.py
will pick them up automatically.

Run once per backbone to populate the model dir with several architectures,
e.g.:
    python train.py --backbone mobilenet_v2
    python train.py --backbone resnet18
    python train.py --backbone efficientnet_b0
"""
import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from config import METADATA_PARQUET, MODEL_DIR, SEED
from dataloaders import build_dataloaders
from engine import evaluate_model, train_model
from model import BACKBONES, build_model
from utils import set_seed

EXPERIMENT_DATASETS = ["augmented", "unaugmented"]
STRATEGIES = ["scratch", "full_finetune", "last_layer"]


def run_experiments(backbone="mobilenet_v2", fold=0, epochs=10, lr=1e-4):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Backbone:", backbone)

    if not os.path.exists(METADATA_PARQUET):
        raise FileNotFoundError(
            f"{METADATA_PARQUET} not found. Run download_data.py, preprocess_data.py, "
            "and split_data.py first."
        )

    df_metadata = pd.read_parquet(METADATA_PARQUET)
    num_classes = df_metadata["dx_numeric"].nunique()

    train_loader_aug, train_loader_noaug, val_loader, test_loader = build_dataloaders(df_metadata, fold)
    loaders_by_name = {"augmented": train_loader_aug, "unaugmented": train_loader_noaug}

    summary_rows = []

    for data_name in EXPERIMENT_DATASETS:
        train_loader = loaders_by_name[data_name]

        for strategy in STRATEGIES:
            run_name = f"{backbone}_{data_name}_{strategy}"
            print("\n" + "=" * 70)
            print(f"Running: {run_name}")
            print("=" * 70)

            start_time = time.time()

            model = build_model(backbone, strategy, num_classes, device)
            model, history = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)

            criterion = nn.CrossEntropyLoss()
            test_loss, test_acc, test_precision, test_recall, test_f1, _, _ = evaluate_model(
                model, test_loader, criterion, device
            )

            elapsed_time = time.time() - start_time

            model_path = os.path.join(MODEL_DIR, f"{run_name}.pth")
            torch.save(model.state_dict(), model_path)

            summary_rows.append({
                "experiment": run_name,
                "backbone": backbone,
                "data": data_name,
                "strategy": strategy,
                "best_val_acc": max(history["val_acc"]),
                "best_val_precision": max(history["val_precision"]),
                "best_val_recall": max(history["val_recall"]),
                "best_val_f1": max(history["val_f1"]),
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "time_seconds": elapsed_time,
                "model_path": model_path,
            })

            print(f"\nSaved model to: {model_path}")
            print(f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    results_df = pd.DataFrame(summary_rows).sort_values(by="test_f1", ascending=False).reset_index(drop=True)
    summary_path = os.path.join(MODEL_DIR, f"{backbone}_experiment_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print("\nSaved summary table to:", summary_path)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin-lesion classifiers for a chosen backbone")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="mobilenet_v2")
    parser.add_argument("--fold", type=int, default=0, help="Which CV fold to use as validation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_experiments(backbone=args.backbone, fold=args.fold, epochs=args.epochs, lr=args.lr)

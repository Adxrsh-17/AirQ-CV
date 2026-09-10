#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/evaluate.py
Evaluates the trained ST-ResUNet model on the holdout test dataset.
Saves quantitative metrics to evaluation/results/evaluation_metrics.csv.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent directory to sys.path to resolve imports cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from training.train import STResUNet, AtmosphericDataset, CONFIG, DEVICE
from evaluation.metrics import compute_all_metrics

def run_evaluation(
    model_path=None,
    results_dir=None,
    n_folds=5
):
    if model_path is None:
        model_path = os.path.join(parent_dir, "training", "checkpoints", "best_sharp_forecast_model.pt")
    if results_dir is None:
        results_dir = os.path.join(current_dir, "results")
    print("=" * 80)
    print("📊 RUNNING MODEL EVALUATION & K-FOLD CROSS-VALIDATION")
    print("=" * 80)

    os.makedirs(results_dir, exist_ok=True)
    device = DEVICE

    if not os.path.exists(model_path):
        if os.path.exists("best_sharp_forecast_model.pt"):
            model_path = "best_sharp_forecast_model.pt"
        else:
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    # 1. Load Model Checkpoint
    model = STResUNet(
        in_channels_s5p=CONFIG["S5P_CHANNELS"],
        in_channels_s2=CONFIG["S2_CHANNELS"],
        out_channels=CONFIG["S5P_CHANNELS"],
        base_channels=32
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✓ Model loaded from: {model_path}")

    real_data_dir = CONFIG.get("DATA_DIR", os.path.join(parent_dir, "data", "processed"))

    # =========================================================================
    # PART 1: K-Fold Cross-Validation Across Historical Sequences (<= 2022)
    # =========================================================================
    print(f"\n" + "-" * 80)
    print(f"🔄 PERFORMING {n_folds}-FOLD CROSS-VALIDATION ACROSS HISTORICAL SEQUENCES (<= 2022)")
    print("-" * 80)
    
    # Ingest historical dataset to extract historical sequences
    hist_ds = AtmosphericDataset(
        data_dir=real_data_dir, split="historical", t_in=CONFIG["T_IN"], h=CONFIG["IMG_H"], w=CONFIG["IMG_W"], augment=False
    )
    historical_sequences = hist_ds.train_val_seqs
    n_seqs = len(historical_sequences)
    print(f"Historical Pool: {n_seqs} temporal sequences ({n_seqs * 9} tiled 128x128 spatial patches)")

    # Partition sequences into n_folds (group k-fold by temporal sequence to avoid spatial leakage)
    fold_indices = np.array_split(np.arange(n_seqs), n_folds)
    
    pollutants = ["NO2 (mol/m²)", "CO (mol/m²)", "SO2 (mol/m²)"]
    fold_metrics = {p: {"MAE": [], "RMSE": [], "R2": [], "RelAcc": [], "SSIM": []} for p in pollutants}

    for fold_idx, test_indices in enumerate(fold_indices):
        fold_seqs = [historical_sequences[i] for i in test_indices]
        fold_ds = AtmosphericDataset(
            data_dir=real_data_dir, split="val", t_in=CONFIG["T_IN"], h=CONFIG["IMG_H"], w=CONFIG["IMG_W"],
            augment=False, custom_sequences=fold_seqs
        )
        loader = torch.utils.data.DataLoader(fold_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

        fold_preds, fold_trues = [], []
        with torch.no_grad():
            for s5p_in, s2_in, target in loader:
                s5p_in, s2_in = s5p_in.to(device), s2_in.to(device)
                pred = model(s5p_in, s2_in)
                fold_preds.append(pred.cpu().numpy())
                fold_trues.append(target.numpy())

        f_preds = np.concatenate(fold_preds, axis=0)
        f_trues = np.concatenate(fold_trues, axis=0)
        df_fold = compute_all_metrics(f_trues, f_preds, pollutant_names=pollutants)

        for _, row in df_fold.iterrows():
            pol = row["Target Pollutant"]
            fold_metrics[pol]["MAE"].append(float(row["MAE"]))
            fold_metrics[pol]["RMSE"].append(float(row["RMSE"]))
            fold_metrics[pol]["R2"].append(float(row["R² Score"]))
            fold_metrics[pol]["RelAcc"].append(float(row["Relative Accuracy"].replace("%", "")))
            fold_metrics[pol]["SSIM"].append(float(row["Spatial SSIM"]))

        print(f"  Fold {fold_idx+1}/{n_folds} ({len(fold_seqs)} seqs, {len(fold_ds)} patches) -> "
              f"MAE: NO2={fold_metrics['NO2 (mol/m²)']['MAE'][-1]:.2e}, "
              f"CO={fold_metrics['CO (mol/m²)']['MAE'][-1]:.2e}, "
              f"SO2={fold_metrics['SO2 (mol/m²)']['MAE'][-1]:.2e}")

    # Compute summary Mean ± Std across folds
    kfold_summary_records = []
    for pol in pollutants:
        m = fold_metrics[pol]
        kfold_summary_records.append({
            "Target Pollutant": pol,
            "MAE (Mean ± Std)": f"{np.mean(m['MAE']):.3e} ± {np.std(m['MAE']):.2e}",
            "RMSE (Mean ± Std)": f"{np.mean(m['RMSE']):.3e} ± {np.std(m['RMSE']):.2e}",
            "R² Score (Mean ± Std)": f"{np.mean(m['R2']):.3f} ± {np.std(m['R2']):.3f}",
            "Relative Accuracy": f"{np.mean(m['RelAcc']):.2f}% ± {np.std(m['RelAcc']):.2f}%",
            "Spatial SSIM": f"{np.mean(m['SSIM']):.3f} ± {np.std(m['SSIM']):.3f}"
        })

    kfold_df = pd.DataFrame(kfold_summary_records)
    kfold_csv = os.path.join(results_dir, "kfold_cross_validation_metrics.csv")
    kfold_df.to_csv(kfold_csv, index=False)
    print(f"\n✓ K-Fold CV metrics saved to: {kfold_csv}")
    print("\n--- 5-Fold Cross-Validation Metrics Across Historical Dataset (<= 2022) ---")
    print(kfold_df.to_string(index=False))

    # =========================================================================
    # PART 2: Evaluation on Strictly Unseen 2023-2024 Holdout Test Set (774 Patches)
    # =========================================================================
    print(f"\n" + "-" * 80)
    print("🧪 EVALUATION ON STRICTLY UNSEEN 2023-2024 HOLDOUT TEST SET (774 Patches)")
    print("-" * 80)

    test_dataset = AtmosphericDataset(
        data_dir=real_data_dir, split="test", t_in=CONFIG["T_IN"], h=CONFIG["IMG_H"], w=CONFIG["IMG_W"], augment=False
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for s5p_in, s2_in, target in test_loader:
            s5p_in, s2_in = s5p_in.to(device), s2_in.to(device)
            pred = model(s5p_in, s2_in)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(target.numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)

    metrics_df = compute_all_metrics(trues, preds, pollutant_names=pollutants)
    out_csv = os.path.join(results_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(out_csv, index=False)

    print(f"\n✓ Holdout Test metrics saved to: {out_csv}")
    print("\n--- Physical & Statistical Metrics on Holdout Test Set (2023-2024) ---")
    print(metrics_df.to_string(index=False))
    print("=" * 80)
    return kfold_df, metrics_df

if __name__ == "__main__":
    run_evaluation()

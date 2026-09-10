#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_extended_plots.py
================================================================================
Generates publication-quality diagnostic and environmental risk plots for ST-ResUNet:
1. multi_sample_forecasts.png      - 3 diverse temporal test cases (True vs Pred vs Error)
2. parity_and_correlation_plots.png- Parity scatters (Obs vs Pred) & Error Distributions
3. plume_transect_profiles.png     - 1D spatial transect slices through peak plume cores
4. environmental_hazard_index.png  - Composite Multi-Pollutant Environmental Hazard Map
================================================================================
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.ndimage as ndimage

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from training.train import STResUNet, AtmosphericDataset, CONFIG, DEVICE

def generate_all_plots():
    print("=" * 75)
    print("🎨 GENERATING EXTENDED PUBLICATION-GRADE DIAGNOSTIC PLOTS")
    print("=" * 75)

    # 1. Load Model
    model = STResUNet(
        in_channels_s5p=3, in_channels_s2=12, out_channels=3, base_channels=32
    ).to(DEVICE)
    ckpt_path = os.path.join(parent_dir, "training", "checkpoints", "best_sharp_forecast_model.pt")
    if not os.path.exists(ckpt_path):
        if os.path.exists("training/checkpoints/best_sharp_forecast_model.pt"):
            ckpt_path = "training/checkpoints/best_sharp_forecast_model.pt"
        elif os.path.exists("best_sharp_forecast_model.pt"):
            ckpt_path = "best_sharp_forecast_model.pt"
        else:
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found!")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"✓ Model weights loaded successfully from: {ckpt_path}")

    # 2. Load Evaluation Dataset on strictly unseen 2023-2024 holdout test set
    real_data_dir = CONFIG.get("DATA_DIR", os.path.join(parent_dir, "data", "processed"))
    test_dataset = AtmosphericDataset(
        data_dir=real_data_dir, split="test", t_in=CONFIG["T_IN"], h=CONFIG["IMG_H"], w=CONFIG["IMG_W"]
    )
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for s5p_in, s2_in, target, *_ in val_loader:
            s5p_in, s2_in = s5p_in.to(DEVICE), s2_in.to(DEVICE)
            pred = model(s5p_in, s2_in)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(target.numpy())

    preds = np.concatenate(all_preds, axis=0) # (32, 3, H, W)
    trues = np.concatenate(all_trues, axis=0) # (32, 3, H, W)
    print(f"✓ Evaluated {len(preds)} test forecast sequences.")

    pollutants = [
        ("NO2", "Tropospheric NO₂", "mol/m²", "viridis"),
        ("CO", "Total Column CO", "mol/m²", "magma"),
        ("SO2", "Surface Column SO₂", "mol/m²", "plasma")
    ]

    # =========================================================================
    # PLOT 1: Multi-Sample Forecasting Generalization (3 Test Sequences)
    # =========================================================================
    print("\n[Plot 1/4] Generating Multi-Sample Forecasting Generalization Plot...")
    sample_indices = [2, 10, 24] # 3 distinct test scenarios
    fig, axes = plt.subplots(len(sample_indices), 6, figsize=(20, 10))
    plt.suptitle("ST-ResUNet Forecasting Generalization Across Diverse Plume Timesteps", fontsize=16, fontweight='bold', y=0.98)

    col_titles = [
        "True NO₂ (S5P)", "Pred NO₂ (Model)",
        "True CO (S5P)", "Pred CO (Model)",
        "True SO₂ (S5P)", "Pred SO₂ (Model)"
    ]
    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=12, fontweight='semibold')

    for r, s_idx in enumerate(sample_indices):
        true_s = trues[s_idx]
        pred_s = preds[s_idx]
        
        col = 0
        for p_idx, (p_code, p_name, unit, cmap) in enumerate(pollutants):
            t_map = true_s[p_idx]
            p_map = pred_s[p_idx]
            vmin = min(np.percentile(t_map, 1), np.percentile(p_map, 1))
            vmax = max(np.percentile(t_map, 99), np.percentile(p_map, 99))
            
            im_t = axes[r, col].imshow(t_map, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[r, col].axis('off')
            fig.colorbar(im_t, ax=axes[r, col], fraction=0.046, pad=0.04, format='%.1e')

            im_p = axes[r, col + 1].imshow(p_map, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[r, col + 1].axis('off')
            fig.colorbar(im_p, ax=axes[r, col + 1], fraction=0.046, pad=0.04, format='%.1e')

            col += 2

        axes[r, 0].set_ylabel(f"Test Step #{s_idx + 1}", fontsize=12, fontweight='bold', labelpad=10)
        axes[r, 0].axis('on')
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.join(current_dir, "forecast"), exist_ok=True)
    p1_path = os.path.join(current_dir, "forecast", "multi_sample_forecasts.png")
    plt.savefig(p1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {p1_path}")

    # =========================================================================
    # PLOT 2: Parity Scatter Plots & Error Distribution Histograms
    # =========================================================================
    print("\n[Plot 2/4] Generating Parity Scatter & Residual Distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle("ST-ResUNet Statistical Parity & Error Residual Distributions", fontsize=16, fontweight='bold', y=0.98)

    for c, (p_code, p_name, unit, cmap) in enumerate(pollutants):
        t_flat = trues[:, c].flatten()
        p_flat = preds[:, c].flatten()
        err_flat = p_flat - t_flat

        # Subsampling for crisp scatter
        idx_sub = np.random.choice(len(t_flat), size=min(15000, len(t_flat)), replace=False)
        t_sub = t_flat[idx_sub]
        p_sub = p_flat[idx_sub]

        # Top row: Parity Scatter
        ax_scatter = axes[0, c]
        hb = ax_scatter.hexbin(t_sub, p_sub, gridsize=40, cmap="Blues", mincnt=1, bins='log')
        
        min_val = min(t_sub.min(), p_sub.min())
        max_val = max(t_sub.max(), p_sub.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Parity (y = x)")
        
        # Linear fit
        slope, intercept = np.polyfit(t_sub, p_sub, 1)
        r2 = 1.0 - (np.sum((t_sub - p_sub)**2) / (np.sum((t_sub - np.mean(t_sub))**2) + 1e-10))
        corr = np.corrcoef(t_sub, p_sub)[0, 1]
        
        ax_scatter.set_title(f"{p_name}\nParity Scatter (R² = {r2:.4f}, r = {corr:.4f})", fontsize=12, fontweight='bold')
        ax_scatter.set_xlabel(f"True Observed ({unit})", fontsize=10)
        ax_scatter.set_ylabel(f"Predicted Forecast ({unit})", fontsize=10)
        ax_scatter.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax_scatter.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax_scatter.legend(loc="upper left")
        fig.colorbar(hb, ax=ax_scatter, label='Log10 Pixel Count')
        ax_scatter.grid(True, linestyle=':', alpha=0.5)

        # Bottom row: Error Residuals
        ax_hist = axes[1, c]
        mae = np.mean(np.abs(err_flat))
        std_err = np.std(err_flat)
        
        ax_hist.hist(err_flat, bins=60, color='teal', alpha=0.75, edgecolor='black', density=True)
        ax_hist.axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
        ax_hist.axvline(np.mean(err_flat), color='orange', linestyle='-', lw=1.5, label=f'Mean Bias: {np.mean(err_flat):.1e}')
        
        ax_hist.set_title(f"{p_name} Residual Error Distribution\nMAE: {mae:.2e} | Std: {std_err:.2e}", fontsize=11, fontweight='bold')
        ax_hist.set_xlabel(f"Prediction Residual (Pred - True) [{unit}]", fontsize=10)
        ax_hist.set_ylabel("Probability Density", fontsize=10)
        ax_hist.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax_hist.legend(loc="upper right")
        ax_hist.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.join(current_dir, "analysis"), exist_ok=True)
    p2_path = os.path.join(current_dir, "analysis", "parity_and_correlation_plots.png")
    plt.savefig(p2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {p2_path}")

    # =========================================================================
    # PLOT 3: 1D Spatial Transect Profiles through Industrial Hotspot Plumes
    # =========================================================================
    print("\n[Plot 3/4] Generating 1D Spatial Plume Transect Profiles...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    plt.suptitle("Plume Sharpness Verification: 1D Spatial Cross-Section Profiles Across Plume Cores", fontsize=15, fontweight='bold', y=0.98)

    sample_test_idx = 0
    test_true = trues[sample_test_idx]
    test_pred = preds[sample_test_idx]

    # Transect lines across key features:
    # Row transect through North stack (y = 30)
    # Diagonal transect through corridor
    y_slice = 30

    for c, (p_code, p_name, unit, cmap) in enumerate(pollutants):
        # Left col: 2D map with transect line
        ax_map = axes[c, 0]
        im = ax_map.imshow(test_true[c], cmap=cmap)
        ax_map.axhline(y_slice, color='cyan', linestyle='--', lw=2, label=f"Transect Line (y={y_slice})")
        ax_map.set_title(f"{p_name} Observation Map with Transect", fontsize=11, fontweight='bold')
        ax_map.axis('off')
        ax_map.legend(loc="upper right")
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, format='%.1e')

        # Right col: 1D Profile comparison (True vs Pred)
        ax_prof = axes[c, 1]
        x_coords = np.arange(CONFIG["IMG_W"])
        t_prof = test_true[c, y_slice, :]
        p_prof = test_pred[c, y_slice, :]

        ax_prof.plot(x_coords, t_prof, 'k-', lw=2.5, label="Ground Truth Profile (S5P)")
        ax_prof.plot(x_coords, p_prof, 'r--', lw=2.0, label="ST-ResUNet Forecast Profile")
        ax_prof.fill_between(x_coords, t_prof, p_prof, color='red', alpha=0.15, label="Absolute Discrepancy")

        # Peak recovery stats
        true_peak = np.max(t_prof)
        pred_peak = np.max(p_prof)
        peak_recovery = (pred_peak / (true_peak + 1e-10)) * 100

        ax_prof.set_title(f"1D Cross-Section at y={y_slice} | Peak Recovery: {peak_recovery:.1f}% (True: {true_peak:.1e}, Pred: {pred_peak:.1e})", fontsize=11, fontweight='bold')
        ax_prof.set_xlabel("Horizontal Pixel Coordinate (X)", fontsize=10)
        ax_prof.set_ylabel(f"Concentration ({unit})", fontsize=10)
        ax_prof.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax_prof.grid(True, linestyle=':', alpha=0.6)
        ax_prof.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.join(current_dir, "analysis"), exist_ok=True)
    p3_path = os.path.join(current_dir, "analysis", "plume_transect_profiles.png")
    plt.savefig(p3_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {p3_path}")

    # =========================================================================
    # PLOT 4: Multi-Pollutant Environmental Hazard Index (EHI) Map
    # =========================================================================
    print("\n[Plot 4/4] Generating Multi-Pollutant Environmental Hazard Index Map...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.suptitle("Composite Geospatial Environmental Hazard Index (EHI) [NO₂ + CO + SO₂]", fontsize=15, fontweight='bold', y=0.98)

    # Compute standardized Air Quality Hazard Index:
    # EHI = (NO2 / NO2_ref) + (CO / CO_ref) + (SO2 / SO2_ref)
    no2_ref = 4.0e-5  # WHO / CPCB reference guideline column proxy
    co_ref  = 2.5e-2
    so2_ref = 8.0e-5

    true_ehi = (test_true[0] / no2_ref) + (test_true[1] / co_ref) + (test_true[2] / so2_ref)
    pred_ehi = (test_pred[0] / no2_ref) + (test_pred[1] / co_ref) + (test_pred[2] / so2_ref)
    ehi_diff = np.abs(pred_ehi - true_ehi)

    v_max = max(np.percentile(true_ehi, 99), np.percentile(pred_ehi, 99))
    v_min = min(np.percentile(true_ehi, 1), np.percentile(pred_ehi, 1))

    # Panel 1: Observed EHI
    im1 = axes[0].imshow(true_ehi, cmap="YlOrRd", vmin=v_min, vmax=v_max)
    axes[0].set_title("Ground Truth Hazard Index\n[Sentinel-5P Composite]", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='EHI Severity (Arbitrary Units)')

    # Panel 2: Forecasted EHI
    im2 = axes[1].imshow(pred_ehi, cmap="YlOrRd", vmin=v_min, vmax=v_max)
    axes[1].set_title("Forecasted Hazard Index\n[ST-ResUNet Forecast]", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='EHI Severity (Arbitrary Units)')

    # Panel 3: Risk Discrepancy Map
    im3 = axes[2].imshow(ehi_diff, cmap="magma", vmin=0, vmax=np.percentile(ehi_diff, 98))
    axes[2].set_title(f"Hazard Risk Discrepancy |Pred - True|\nMean Absolute Bias: {np.mean(ehi_diff):.3f}", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='Absolute Error')

    plt.tight_layout()
    os.makedirs(os.path.join(current_dir, "environment"), exist_ok=True)
    p4_path = os.path.join(current_dir, "environment", "environmental_hazard_index.png")
    plt.savefig(p4_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {p4_path}")

    print("\n" + "=" * 75)
    print("🎉 ALL 4 DIAGNOSTIC & ENVIRONMENTAL PLOTS GENERATED SUCCESSFULLY!")
    print(f"1. {p1_path}")
    print(f"2. {p2_path}")
    print(f"3. {p3_path}")
    print(f"4. {p4_path}")
    print("=" * 75)

if __name__ == "__main__":
    generate_all_plots()

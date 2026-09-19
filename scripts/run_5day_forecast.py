import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from airq.models.v2_forecaster import V2Forecaster
from airq.data.v2_dataset import V2Dataset
from skimage.metrics import structural_similarity as ssim

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    checkpoint_path = Path("experiments/v2_baseline/best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    model = V2Forecaster(s5p_channels=3, s2_channels=12, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print("Model successfully loaded from checkpoint.")
    
    # 2. Load Normalization Stats
    norm_file = Path("data/processed/v2_cache/norm_stats.json")
    with open(norm_file, "r") as f:
        stats = json.load(f)
    s5p_mean = np.array(stats['s5p_mean']).reshape(3, 1, 1)
    s5p_std = np.array(stats['s5p_std']).reshape(3, 1, 1)
    
    # 3. Load Test Dataset
    test_ds = V2Dataset('data/raw_full438', split='test', preload=True)
    print(f"Loaded {len(test_ds)} test sequences.")
    
    # Select sequence (e.g. index 0 of test set)
    seq_idx = 0
    sample = test_ds[seq_idx]
    seq_meta = test_ds.sequences[seq_idx]
    
    # History files & target files
    hist_last_file = seq_meta["s5p_history"][-1]
    target_5day_file = seq_meta["s5p_target"][0]
    
    print(f"\nEvaluating Sequence #{seq_idx}:")
    print(f"  Last History Frame: {hist_last_file}")
    print(f"  5-Day Target Frame: {target_5day_file}")
    
    # Prepare batch
    p1_x = sample['p1_x'].unsqueeze(0).to(device) # (1, 12, 3, H, W)
    s2_x = sample['s2_x'].unsqueeze(0).to(device) # (1, 12, 12, H, W)
    y_true_norm = sample['y'].numpy()              # (30, 3, H, W)
    
    # 4. Forecast 5 Days (k=1)
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            pred_norm = model(p1_x, s2_x, k=1).squeeze(0).cpu().numpy() # (1, 3, H, W)
            
    # Step 0 corresponds to 5-day forecast
    pred_step0 = pred_norm[0] # (3, H, W)
    true_step0 = y_true_norm[0] # (3, H, W)
    hist_last = sample['p1_x'][-1].numpy() # (3, H, W)
    
    # Inverse scale to physical units
    pred_phys = pred_step0 * s5p_std + s5p_mean
    true_phys = true_step0 * s5p_std + s5p_mean
    hist_phys = hist_last * s5p_std + s5p_mean
    
    pollutants = [
        ("NO2", "mol/m²", 0),
        ("CO", "mol/m²", 1),
        ("SO2", "mol/m²", 2),
    ]
    
    results = []
    
    # 5. Output directory for plots
    out_dir = Path("evaluation/plots/5day_forecast")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"5-DAY AIR QUALITY FORECAST RESULTS (Lead Time: 5 Days)")
    print("="*70)
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    
    for row, (name, unit, c) in enumerate(pollutants):
        p_c = pred_phys[c]
        t_c = true_phys[c]
        h_c = hist_phys[c]
        
        # Metrics
        diff = p_c - t_c
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        
        var_t = np.var(t_c)
        r2 = 1 - (np.mean(diff**2) / (var_t + 1e-12))
        
        # Pearson
        t_flat = t_c.flatten()
        p_flat = p_c.flatten()
        if np.std(t_flat) > 1e-12 and np.std(p_flat) > 1e-12:
            pearson = np.corrcoef(t_flat, p_flat)[0, 1]
        else:
            pearson = 0.0
            
        # Mass conservation
        mass_true = np.sum(t_c)
        mass_pred = np.sum(p_c)
        mass_err_pct = ((mass_pred - mass_true) / (mass_true + 1e-12)) * 100
        
        # SSIM
        drange = max(t_c.max() - t_c.min(), 1e-6)
        ssim_val = ssim(t_c, p_c, data_range=drange)
        
        results.append({
            "pollutant": name,
            "unit": unit,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson": pearson,
            "mass_err_pct": mass_err_pct,
            "ssim": ssim_val,
            "pred_mean": np.mean(p_c),
            "true_mean": np.mean(t_c),
            "pred_max": np.max(p_c),
            "true_max": np.max(t_c),
            "pred_min": np.min(p_c),
            "true_min": np.min(t_c),
        })
        
        print(f"\n--- {name} ({unit}) ---")
        print(f"  R² Score:            {r2:.4f}")
        print(f"  Pearson Correlation: {pearson:.4f}")
        print(f"  SSIM (Structure):    {ssim_val:.4f}")
        print(f"  RMSE:                {rmse:.6e} {unit}")
        print(f"  MAE:                 {mae:.6e} {unit}")
        print(f"  Mass Bias Error:     {mass_err_pct:+.2f}%")
        print(f"  Ground Truth:        Mean = {np.mean(t_c):.6e} | Max = {np.max(t_c):.6e} | Min = {np.min(t_c):.6e}")
        print(f"  Model Forecast:      Mean = {np.mean(p_c):.6e} | Max = {np.max(p_c):.6e} | Min = {np.min(p_c):.6e}")
        
        # Plotting
        vmin = min(np.min(h_c), np.min(t_c), np.min(p_c))
        vmax = max(np.percentile(h_c, 98), np.percentile(t_c, 98), np.percentile(p_c, 98))
        
        im0 = axes[row, 0].imshow(h_c, cmap='magma', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"{name} (T=0: Last Observed)", fontsize=11)
        axes[row, 0].axis('off')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        
        im1 = axes[row, 1].imshow(t_c, cmap='magma', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"{name} (T=+5d: Ground Truth)", fontsize=11)
        axes[row, 1].axis('off')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        
        im2 = axes[row, 2].imshow(p_c, cmap='magma', vmin=vmin, vmax=vmax)
        axes[row, 2].set_title(f"{name} (T=+5d: Forecast)\nSSIM: {ssim_val:.3f} | r: {pearson:.3f}", fontsize=11)
        axes[row, 2].axis('off')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
        
        err_map = np.abs(diff)
        im3 = axes[row, 3].imshow(err_map, cmap='Reds')
        axes[row, 3].set_title(f"{name} Absolute Error\nMAE: {mae:.2e}", fontsize=11)
        axes[row, 3].axis('off')
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)
        
    plt.suptitle(f"5-Day Air Quality Forecast (Lead Time: 5 Days)\nTarget Scene: {target_5day_file}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_path = out_dir / "5day_forecast_spatial.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSpatial visualization saved to: {fig_path}")
    
    # Save metrics JSON
    with open(out_dir / "5day_forecast_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {out_dir / '5day_forecast_metrics.json'}")

if __name__ == "__main__":
    main()
